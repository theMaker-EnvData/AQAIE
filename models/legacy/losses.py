from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

OBS_POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]


# ---------------------------------------------------------------------------
# Horizon weighting
# ---------------------------------------------------------------------------

def _apply_horizon_weights(mask: torch.Tensor, horizon_weights: torch.Tensor | None) -> torch.Tensor:
    if horizon_weights is None:
        return mask
    if mask.dim() == 5:
        hw = horizon_weights.view(1, -1, 1, 1, 1)
    elif mask.dim() == 4:
        hw = horizon_weights.view(1, -1, 1, 1)
    else:
        raise ValueError(f"Unsupported mask dims for horizon weighting: {tuple(mask.shape)}")
    return mask * hw.to(device=mask.device, dtype=mask.dtype)


def _build_horizon_weight_tensor(
    horizons: list[int],
    enabled_horizons: set[int],
    horizon_weights: dict[int, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vals = []
    for h in horizons:
        hi = int(h)
        if hi not in enabled_horizons:
            vals.append(0.0)
            continue
        vals.append(float(horizon_weights.get(hi, 1.0)))
    return torch.tensor(vals, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Masked point losses / metrics
# ---------------------------------------------------------------------------

def _masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    eps: float,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # pred/target/mask: [B, H, P, Y, X]
    finite = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(mask)
    effective_mask = _apply_horizon_weights(mask, horizon_weights) * finite.to(dtype=mask.dtype)
    err = torch.where(finite, pred - target, torch.zeros_like(pred))
    abs_err = err.abs()
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    linear = abs_err - quadratic
    huber = 0.5 * quadratic * quadratic + delta * linear

    weighted = huber * effective_mask
    valid_count = effective_mask.sum()
    loss = weighted.sum() / (valid_count + eps)
    return loss, valid_count


def _masked_huber_loss_4d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _masked_huber_loss(
        pred.unsqueeze(1),
        target.unsqueeze(1),
        mask.unsqueeze(1),
        delta=delta,
        eps=eps,
        horizon_weights=None,
    )


def _masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(mask)
    effective_mask = mask * finite.to(dtype=mask.dtype)
    diff = torch.where(finite, pred - target, torch.zeros_like(pred))
    err2 = diff * diff
    weighted = err2 * effective_mask
    valid_count = effective_mask.sum()
    mse = weighted.sum() / (valid_count + eps)
    rmse = torch.sqrt(mse + eps)
    return rmse, valid_count


def _masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(mask)
    effective_mask = mask * finite.to(dtype=mask.dtype)
    abs_err = torch.where(finite, (pred - target).abs(), torch.zeros_like(pred))
    weighted = abs_err * effective_mask
    valid_count = effective_mask.sum()
    mae = weighted.sum() / (valid_count + eps)
    return mae, valid_count


def _masked_r2_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(mask)
    w = mask * finite.to(dtype=mask.dtype)
    y = torch.where(finite, target, torch.zeros_like(target))
    err = torch.where(finite, pred - target, torch.zeros_like(pred))

    sse = ((err * err) * w).sum()
    sw = w.sum()
    swy = (w * y).sum()
    swy2 = (w * y * y).sum()
    return sse, sw, swy, swy2


def _masked_negative_mae_penalty(
    pred: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(pred) & torch.isfinite(mask)
    effective_mask = _apply_horizon_weights(mask, horizon_weights) * finite.to(dtype=mask.dtype)
    negative_mag = torch.relu(-torch.where(finite, pred, torch.zeros_like(pred)))
    weighted = negative_mag * effective_mask
    valid_count = effective_mask.sum()
    penalty = weighted.sum() / (valid_count + eps)
    return penalty, valid_count


def _masked_tail_mae_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    thresholds_by_pollutant: torch.Tensor,
    eps: float,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # thresholds_by_pollutant: [P] in normalized target space.
    thr = thresholds_by_pollutant.view(1, 1, -1, 1, 1).to(device=target.device, dtype=target.dtype)
    finite = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(mask)
    tail = target >= thr
    effective_mask = (
        _apply_horizon_weights(mask, horizon_weights)
        * finite.to(dtype=mask.dtype)
        * tail.to(dtype=mask.dtype)
    )
    abs_err = torch.where(finite, (pred - target).abs(), torch.zeros_like(pred))
    weighted = abs_err * effective_mask
    valid_count = effective_mask.sum()
    penalty = weighted.sum() / (valid_count + eps)
    return penalty, valid_count


def _masked_unsupervised_bound_penalty(
    pred: torch.Tensor,
    mask: torch.Tensor,
    lower_bounds_by_pollutant: torch.Tensor,
    upper_bounds_by_pollutant: torch.Tensor,
    temperature: float,
    eps: float,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Applies only on non-observed locations (mask<=0) in normalized output space.
    lower = lower_bounds_by_pollutant.view(1, 1, -1, 1, 1).to(device=pred.device, dtype=pred.dtype)
    upper = upper_bounds_by_pollutant.view(1, 1, -1, 1, 1).to(device=pred.device, dtype=pred.dtype)
    tau = max(1.0e-6, float(temperature))

    finite = torch.isfinite(pred) & torch.isfinite(mask)
    unsupervised = (mask <= 0.0)
    effective_mask = (
        _apply_horizon_weights(unsupervised.to(dtype=mask.dtype), horizon_weights)
        * finite.to(dtype=mask.dtype)
    )

    upper_violation = F.softplus((pred - upper) / tau)
    lower_violation = F.softplus((lower - pred) / tau)
    penalty_map = upper_violation * upper_violation + lower_violation * lower_violation

    weighted = penalty_map * effective_mask
    valid_count = effective_mask.sum()
    penalty = weighted.sum() / (valid_count + eps)
    return penalty, valid_count


# ---------------------------------------------------------------------------
# Quantile (pinball) loss and coverage
# ---------------------------------------------------------------------------

def _masked_pinball_loss(
    pred_q: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    taus: torch.Tensor,
    eps: float,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantile regression loss.

    pred_q: [B, H, P, K, Y, X], target/mask: [B, H, P, Y, X], taus: [K].
    Returns (loss, valid_count) where valid_count counts target dots (not x K).
    """
    if pred_q.dim() != 6:
        raise ValueError(f"pred_q must be [B,H,P,K,Y,X], got {tuple(pred_q.shape)}")
    if taus.numel() != pred_q.shape[3]:
        raise ValueError(f"taus length mismatch: {taus.numel()} vs K={pred_q.shape[3]}")
    target_e = target.unsqueeze(3)
    mask_e = _apply_horizon_weights(mask, horizon_weights).unsqueeze(3)
    finite = torch.isfinite(pred_q) & torch.isfinite(target_e) & torch.isfinite(mask_e)
    eff = mask_e * finite.to(dtype=mask.dtype)
    err = torch.where(finite, target_e - pred_q, torch.zeros_like(pred_q))
    tau_view = taus.view(1, 1, 1, -1, 1, 1).to(device=pred_q.device, dtype=pred_q.dtype)
    pinball = torch.maximum(tau_view * err, (tau_view - 1.0) * err)
    weighted = pinball * eff
    valid_count = eff.sum() / float(max(1, pred_q.shape[3]))
    loss = weighted.sum() / (eff.sum() + eps)
    return loss, valid_count


def _masked_unsupervised_spread_penalty(
    pred_q: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean quantile interval width (q_hi - q_lo) on unsupervised pixels.

    Pinball loss constrains the interval only at observation dots; elsewhere
    softplus offsets are free to blow up (edge/corner spread blobs). This
    penalty applies a weak shrinkage prior on unobserved pixels only, leaving
    dot-supervised calibration untouched.

    pred_q: [B, H, P, K, Y, X] (K ordered lo..hi), mask: [B, H, P, Y, X].
    """
    if pred_q.dim() != 6:
        raise ValueError(f"pred_q must be [B,H,P,K,Y,X], got {tuple(pred_q.shape)}")
    spread = pred_q[:, :, :, -1] - pred_q[:, :, :, 0]  # [B, H, P, Y, X]
    finite = torch.isfinite(spread) & torch.isfinite(mask)
    unsupervised = mask <= 0.0
    effective_mask = (
        _apply_horizon_weights(unsupervised.to(dtype=mask.dtype), horizon_weights)
        * finite.to(dtype=mask.dtype)
    )
    spread_safe = torch.where(finite, spread, torch.zeros_like(spread))
    weighted = spread_safe.abs() * effective_mask
    valid_count = effective_mask.sum()
    penalty = weighted.sum() / (valid_count + eps)
    return penalty, valid_count


def _masked_quantile_coverage(
    pred_lo: torch.Tensor,
    pred_hi: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fraction of masked dots where target falls inside [lo, hi]."""
    finite = (
        torch.isfinite(pred_lo)
        & torch.isfinite(pred_hi)
        & torch.isfinite(target)
        & torch.isfinite(mask)
    )
    eff = (mask > 0.0) & finite
    inside = (target >= pred_lo) & (target <= pred_hi) & eff
    covered = inside.to(dtype=torch.float32).sum()
    valid = eff.to(dtype=torch.float32).sum()
    return covered, valid


# ---------------------------------------------------------------------------
# Loss-region (context-margin crop) diagnostics helpers
# ---------------------------------------------------------------------------

def _slice_loss_region(field: torch.Tensor, loss_bbox) -> torch.Tensor:
    """Slice the supervised loss region from a crop-local field [..., Y, X].

    loss_bbox: (y0, x0, h, w) crop-local or None (no-op). Diagnostics computed
    on this slice describe the scored area only, excluding the context halo.
    """
    if loss_bbox is None:
        return field
    y0, x0, h, w = (int(v) for v in loss_bbox)
    return field[..., y0 : y0 + h, x0 : x0 + w]


def _loss_region_edge_bands(
    shape_hw: tuple[int, int],
    loss_bbox,
    band: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(edge_band, interior) masks [Y, X] inside the loss region.

    edge_band = ring of `band` cells along the loss-region boundary; interior =
    the rest of the loss region. Both are 0 outside the loss region, so
    multiplying LKO masks with them stratifies hold-out dots by distance to the
    supervised edge (margin-junk intrusion diagnostic).
    """
    full_h, full_w = int(shape_hw[0]), int(shape_hw[1])
    if loss_bbox is None:
        y0, x0, h, w = 0, 0, full_h, full_w
    else:
        y0, x0, h, w = (int(v) for v in loss_bbox)
    b = max(1, min(int(band), (min(h, w) - 1) // 2))
    region = torch.zeros((full_h, full_w), device=device, dtype=dtype)
    region[y0 : y0 + h, x0 : x0 + w] = 1.0
    interior = torch.zeros((full_h, full_w), device=device, dtype=dtype)
    interior[y0 + b : y0 + h - b, x0 + b : x0 + w - b] = 1.0
    edge_band = region - interior
    return edge_band, interior


def _hf_power(field: torch.Tensor, eps: float = 1.0e-12) -> float:
    """High-frequency energy: mean squared residual after 3x3 box smoothing.

    field: [..., Y, X]. Used to compare texture levels of c0_inr vs pred —
    if pred HF tracks c0 HF 1:1 over epochs, the advection anchor (not the
    U-Net) is the texture transmission channel.
    """
    x = field
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[:, None]
    elif x.dim() > 4:
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
    x = x.to(dtype=torch.float32)
    finite = torch.isfinite(x)
    x_safe = torch.where(finite, x, torch.zeros_like(x))
    smooth = F.avg_pool2d(x_safe, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    norm = F.avg_pool2d(finite.to(torch.float32), kernel_size=3, stride=1, padding=1,
                        count_include_pad=False)
    smooth = smooth / norm.clamp_min(eps)
    hf = torch.where(finite, x - smooth, torch.zeros_like(x))
    denom = finite.to(torch.float32).sum().clamp_min(1.0)
    return float((hf.square().sum() / denom).item())


# ---------------------------------------------------------------------------
# Edge blob diagnostics / border regularization
# ---------------------------------------------------------------------------

def _edge_interior_ratio(field: torch.Tensor, ring: int, eps: float = 1.0e-6) -> float:
    """Diagnostic for border blobs: mean |field| over the outer ring band
    divided by mean |field| over the interior. ~1.0 is healthy; >>1 indicates
    edge artifacts. field: [..., Y, X]."""
    y, x = field.shape[-2], field.shape[-1]
    r = max(1, min(int(ring), (min(y, x) - 1) // 2))
    band = torch.zeros((y, x), dtype=torch.bool, device=field.device)
    band[:r, :] = True
    band[-r:, :] = True
    band[:, :r] = True
    band[:, -r:] = True
    a = field.abs()
    finite = torch.isfinite(a)
    a = torch.where(finite, a, torch.zeros_like(a))
    band_b = band.expand_as(a) & finite
    inner_b = (~band).expand_as(a) & finite
    border_mean = a[band_b].sum() / (band_b.sum().clamp_min(1).to(a.dtype))
    interior_mean = a[inner_b].sum() / (inner_b.sum().clamp_min(1).to(a.dtype))
    return float(border_mean.item()) / max(float(interior_mean.item()), eps)


def _border_inward_tv_penalty(
    pred: torch.Tensor,
    ring: int,
    eps: float,
    supervised_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-directional TV on the outer ring band: each border pixel is pulled
    toward its inward neighbor (detached), so unsupervised edges inherit
    interior statistics instead of drifting freely. pred: [..., Y, X]."""
    y, x = pred.shape[-2], pred.shape[-1]
    r = max(1, min(int(ring), (min(y, x) - 1) // 2))
    total = pred.new_zeros(())
    count = 0.0
    sup = None
    if supervised_mask is not None:
        sup = supervised_mask > 0.0

    def _accum(edge: torch.Tensor, inner: torch.Tensor, sup_edge: torch.Tensor | None) -> None:
        nonlocal total, count
        diff = (edge - inner.detach()).square()
        if sup_edge is not None:
            diff = diff * (~sup_edge).to(diff.dtype)
            count_local = float((~sup_edge).sum().item())
        else:
            count_local = float(diff.numel())
        total = total + diff.sum()
        count += count_local

    for i in range(r):
        _accum(pred[..., i, :], pred[..., i + 1, :], None if sup is None else sup[..., i, :])
        _accum(pred[..., -1 - i, :], pred[..., -2 - i, :], None if sup is None else sup[..., -1 - i, :])
        _accum(pred[..., :, i], pred[..., :, i + 1], None if sup is None else sup[..., :, i])
        _accum(pred[..., :, -1 - i], pred[..., :, -2 - i], None if sup is None else sup[..., :, -1 - i])
    return total / (count + eps)


# ---------------------------------------------------------------------------
# Spatial operators
# ---------------------------------------------------------------------------

def _spatial_gradients(field: torch.Tensor, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
    # field: [B, P, Y, X]
    if field.dim() != 4:
        raise ValueError(f"field must be [B,P,Y,X], got {tuple(field.shape)}")
    y = field.shape[-2]
    x = field.shape[-1]
    dx = max(float(dx), 1.0e-6)
    dy = max(float(dy), 1.0e-6)

    dcdx = torch.zeros_like(field)
    dcdy = torch.zeros_like(field)

    if x >= 2:
        dcdx[..., :, 0] = (field[..., :, 1] - field[..., :, 0]) / dx
        dcdx[..., :, -1] = (field[..., :, -1] - field[..., :, -2]) / dx
    if x >= 3:
        dcdx[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (2.0 * dx)

    if y >= 2:
        dcdy[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dy
        dcdy[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dy
    if y >= 3:
        dcdy[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dy)

    return dcdx, dcdy


def _spatial_laplacian(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    # field: [B, P, Y, X]
    if field.dim() != 4:
        raise ValueError(f"field must be [B,P,Y,X], got {tuple(field.shape)}")
    dx = max(float(dx), 1.0e-6)
    dy = max(float(dy), 1.0e-6)
    d2cdx2 = torch.zeros_like(field)
    d2cdy2 = torch.zeros_like(field)
    y = field.shape[-2]
    x = field.shape[-1]

    if x >= 2:
        d2cdx2[..., :, 0] = (field[..., :, 1] - 2.0 * field[..., :, 0]) / (dx * dx)
        d2cdx2[..., :, -1] = (field[..., :, -1] - 2.0 * field[..., :, -2]) / (dx * dx)
    if x >= 3:
        d2cdx2[..., :, 1:-1] = (
            field[..., :, 2:] - 2.0 * field[..., :, 1:-1] + field[..., :, :-2]
        ) / (dx * dx)

    if y >= 2:
        d2cdy2[..., 0, :] = (field[..., 1, :] - 2.0 * field[..., 0, :]) / (dy * dy)
        d2cdy2[..., -1, :] = (field[..., -1, :] - 2.0 * field[..., -2, :]) / (dy * dy)
    if y >= 3:
        d2cdy2[..., 1:-1, :] = (
            field[..., 2:, :] - 2.0 * field[..., 1:-1, :] + field[..., :-2, :]
        ) / (dy * dy)

    return d2cdx2 + d2cdy2


# ---------------------------------------------------------------------------
# Full-model advection penalty (horizon intervals)
# ---------------------------------------------------------------------------

def _build_advection_interval_weights(
    horizons: list[int],
    enabled_horizons: set[int],
    *,
    interval_power: float,
    interval_weight_overrides: Mapping[int, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vals: list[float] = []
    prev_h = 0
    p = max(0.0, float(interval_power))
    for h in horizons:
        hi = int(h)
        dt = max(1, hi - prev_h)
        if hi not in enabled_horizons:
            vals.append(0.0)
        elif hi in interval_weight_overrides:
            vals.append(float(interval_weight_overrides[hi]))
        else:
            vals.append(float(dt) ** (-p))
        prev_h = hi
    return torch.tensor(vals, device=device, dtype=dtype)


def _masked_interval_advection_penalty(
    pred: torch.Tensor,
    c0: torch.Tensor,
    wind_uv_h2: torch.Tensor,
    horizons: list[int],
    mask: torch.Tensor,
    eps: float,
    *,
    interval_weights: torch.Tensor,
    dx: float,
    dy: float,
    use_midpoint_gradient: bool,
    wind_to_cells_per_hour: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advection residual over horizon intervals.

    pred: [B,H,P,Y,X], c0: [B,P,Y,X], wind_uv_h2: [B,H,2,Y,X] (raw m/s),
    mask: [B,H,P,Y,X].

    wind_to_cells_per_hour: unit calibration. With 1 km cells and dt in hours,
    m/s -> cells/hour requires 3600/1000 = 3.6. Legacy (uncalibrated) runs
    used 1.0.
    """
    if pred.dim() != 5:
        raise ValueError(f"pred must be [B,H,P,Y,X], got {tuple(pred.shape)}")
    if c0.dim() != 4:
        raise ValueError(f"c0 must be [B,P,Y,X], got {tuple(c0.shape)}")
    if wind_uv_h2.dim() != 5 or wind_uv_h2.shape[2] != 2:
        raise ValueError(f"wind_uv_h2 must be [B,H,2,Y,X], got {tuple(wind_uv_h2.shape)}")
    if mask.dim() != 5:
        raise ValueError(f"mask must be [B,H,P,Y,X], got {tuple(mask.shape)}")
    if len(horizons) != pred.shape[1]:
        raise ValueError(f"horizons length mismatch: len(horizons)={len(horizons)}, pred_h={pred.shape[1]}")

    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    total_count = torch.zeros((), device=pred.device, dtype=pred.dtype)
    wind_scale = float(wind_to_cells_per_hour)

    prev_h = 0
    prev_c = c0
    for i, h in enumerate(horizons):
        dt = max(1, int(h) - int(prev_h))
        curr_c = pred[:, i]
        grad_field = 0.5 * (curr_c + prev_c) if use_midpoint_gradient else curr_c
        dcdx, dcdy = _spatial_gradients(grad_field, dx=dx, dy=dy)

        u = wind_uv_h2[:, i, 0].unsqueeze(1) * wind_scale
        v = wind_uv_h2[:, i, 1].unsqueeze(1) * wind_scale
        curr_safe = torch.where(torch.isfinite(curr_c), curr_c, torch.zeros_like(curr_c))
        prev_safe = torch.where(torch.isfinite(prev_c), prev_c, torch.zeros_like(prev_c))
        u_safe = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        v_safe = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        dcdx_safe = torch.where(torch.isfinite(dcdx), dcdx, torch.zeros_like(dcdx))
        dcdy_safe = torch.where(torch.isfinite(dcdy), dcdy, torch.zeros_like(dcdy))
        # Raster is NW-origin (row 0 = north) while v10 is positive NORTHWARD,
        # so physical dC/dy_north = -dcdy_row: the meridional term enters with
        # a minus sign. (Sign fix 2026-06-12; +v previously reversed N-S
        # transport and piled systematic error along the northern crop edge.)
        residual = (curr_safe - prev_safe) / float(dt) + (u_safe * dcdx_safe - v_safe * dcdy_safe)

        finite = (
            torch.isfinite(curr_c)
            & torch.isfinite(prev_c)
            & torch.isfinite(u)
            & torch.isfinite(v)
            & torch.isfinite(mask[:, i])
        )
        wk = float(interval_weights[i].item())
        if wk <= 0.0:
            prev_h = int(h)
            prev_c = curr_c
            continue

        eff_mask = mask[:, i] * finite.to(dtype=mask.dtype) * wk
        weighted = residual.abs() * eff_mask
        total = total + weighted.sum()
        total_count = total_count + eff_mask.sum()

        prev_h = int(h)
        prev_c = curr_c

    loss = total / (total_count + eps)
    return loss, total_count


def _advection_full_mask(pred: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(pred)


def _resolve_advection_wind_scale(adv_cfg: Mapping[str, Any]) -> float:
    """m/s -> cells/hour scale from config. wind_units: mps | normalized(legacy)."""
    units = str(adv_cfg.get("wind_units", "mps")).strip().lower()
    if units in ("normalized", "legacy", "raw"):
        return 1.0
    cell_size_m = max(float(adv_cfg.get("cell_size_m", 1000.0)), 1.0e-6)
    return 3600.0 / cell_size_m


# ---------------------------------------------------------------------------
# Lambda schedules
# ---------------------------------------------------------------------------

def _resolve_warmup_lambda(base_lambda: float, epoch: int, warmup_start_epoch: int, warmup_epochs: int) -> float:
    if base_lambda <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return float(base_lambda)
    if epoch < warmup_start_epoch:
        return 0.0
    progress = min(1.0, float(epoch - warmup_start_epoch + 1) / float(warmup_epochs))
    return float(base_lambda) * progress


def _spectral_notch_loss(
    pred: torch.Tensor,
    target_period_px: float,
    notch_width: float,
    eps: float,
) -> torch.Tensor:
    """Penalise spectral energy at a specific spatial period (e.g. 64px grid).

    Computes the 2D FFT of the prediction and suppresses magnitude in a
    Gaussian-centred notch around the target frequency bin(s).
    Works on any spatial size — target period in pixels.

    pred: [..., Y, X]  (any leading dims collapsed into batch)
    Returns scalar loss = notch_energy / total_energy.
    """
    if pred.dim() < 2:
        raise ValueError(f"_spectral_notch_loss expects at least 2 spatial dims, got {pred.dim()}")
    if pred.dim() > 2:
        pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])

    h, w = pred.shape[-2], pred.shape[-1]
    fft = torch.fft.rfft2(pred.float(), norm="ortho")
    power = fft.abs().square()

    target_fy = h / max(float(target_period_px), 1.0)
    target_fx = w / max(float(target_period_px), 1.0)

    fy_num = int(round(target_fy))
    fx_num = int(round(target_fx))
    if fy_num <= 0 and fx_num <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    yc = torch.arange(power.shape[-2], device=pred.device, dtype=torch.float32)
    xc = torch.arange(power.shape[-1], device=pred.device, dtype=torch.float32)
    dy = yc[:, None] - target_fy
    dx = xc[None, :] - target_fx
    dist_sq = dy * dy + dx * dx
    # Real FFT: negative fy appears at bin h - target_fy (Hermitian mirror).
    dy_m = yc[:, None] - (float(h) - target_fy)
    dist_sq_m = dy_m * dy_m + dx * dx

    sigma = max(float(notch_width), 0.5)
    mask = torch.exp(-dist_sq / (2.0 * sigma * sigma)) + torch.exp(-dist_sq_m / (2.0 * sigma * sigma))
    mask[0, 0] = 0.0

    total = power.sum(dim=(-2, -1)) + eps
    notch = (power * mask.unsqueeze(0)).sum(dim=(-2, -1))
    return (notch / total).mean()


def _build_loss_region_field_mask(
    batch_size: int,
    num_pollutants: int,
    height: int,
    width: int,
    loss_bbox,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform mask = 1 inside loss_bbox for all pollutants; shape [B,1,P,Y,X]."""
    mask = torch.zeros((batch_size, 1, num_pollutants, height, width), device=device, dtype=dtype)
    if loss_bbox is None:
        mask.fill_(1.0)
        return mask
    y0, x0, h, w = (int(v) for v in loss_bbox)
    mask[..., y0 : y0 + h, x0 : x0 + w] = 1.0
    return mask


def _horizon_anchor_field_consistency_loss(
    pred: torch.Tensor,
    teachers_by_horizon: Mapping[int, torch.Tensor],
    horizons: list[int],
    loss_bbox,
    delta: float,
    eps: float,
    *,
    horizon_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
    """Match main multi-horizon q50 fields to stopgrad anchor h01 teachers.

    pred: [B,H,P,Y,X] main forecast median (q50 when quantile head enabled).
    teachers_by_horizon: target_h -> [B,P,Y,X] detached anchor h01 fields.
    All six pollutants are weighted equally inside loss_bbox.
    """
    if pred.dim() != 5:
        raise ValueError(f"pred must be [B,H,P,Y,X], got {tuple(pred.shape)}")
    b, _, p, y, x = pred.shape
    if p != len(OBS_POLLUTANTS):
        raise ValueError(f"expected {len(OBS_POLLUTANTS)} pollutants, got {p}")

    mask = _build_loss_region_field_mask(b, p, y, x, loss_bbox, device=pred.device, dtype=pred.dtype)
    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    weight_sum = 0.0
    for hi, h in enumerate(horizons):
        target_h = int(h)
        teacher = teachers_by_horizon.get(target_h)
        if teacher is None:
            continue
        if teacher.shape != (b, p, y, x):
            raise ValueError(
                f"teacher shape mismatch for h={target_h}: expected {(b, p, y, x)}, got {tuple(teacher.shape)}"
            )
        pred_h = pred[:, hi : hi + 1]
        teacher_h = teacher.unsqueeze(1)
        pair_loss, _ = _masked_huber_loss(
            pred_h,
            teacher_h,
            mask,
            delta=delta,
            eps=eps,
            horizon_weights=None,
        )
        w_h = 1.0
        if horizon_weights is not None:
            w_h = float(horizon_weights.get(target_h, 1.0))
        if w_h <= 0.0:
            continue
        total = total + pair_loss * w_h
        weight_sum += w_h

    if weight_sum <= 0.0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return total / max(weight_sum, eps)


def _resolve_aux_lambda_multiplier(schedule_cfg: Mapping[str, Any] | None, epoch: int) -> float:
    """Linear decay multiplier for INR aux losses (recon/assimil/PDE/SL).

    Before decay_start_epoch -> start; linear over decay_epochs; after -> end.
    """
    if not isinstance(schedule_cfg, Mapping) or not bool(schedule_cfg.get("enabled", False)):
        return 1.0
    start = float(schedule_cfg.get("start", 1.0))
    end = float(schedule_cfg.get("end", 1.0))
    decay_start = int(schedule_cfg.get("decay_start_epoch", 1))
    decay_epochs = max(1, int(schedule_cfg.get("decay_epochs", 1)))
    if epoch < decay_start:
        return start
    progress = min(1.0, float(epoch - decay_start + 1) / float(decay_epochs))
    return start + (end - start) * progress


def _is_improved(current: float, best: float, mode: str) -> bool:
    mode_l = str(mode).strip().lower()
    if mode_l == "max":
        return current > best
    return current < best


__all__ = [
    "OBS_POLLUTANTS",
    "_advection_full_mask",
    "_apply_horizon_weights",
    "_build_advection_interval_weights",
    "_build_loss_region_field_mask",
    "_build_horizon_weight_tensor",
    "_horizon_anchor_field_consistency_loss",
    "_is_improved",
    "_masked_huber_loss",
    "_masked_huber_loss_4d",
    "_masked_interval_advection_penalty",
    "_masked_mae",
    "_masked_negative_mae_penalty",
    "_masked_pinball_loss",
    "_masked_quantile_coverage",
    "_masked_r2_stats",
    "_masked_rmse",
    "_masked_tail_mae_penalty",
    "_masked_unsupervised_bound_penalty",
    "_masked_unsupervised_spread_penalty",
    "_resolve_advection_wind_scale",
    "_resolve_aux_lambda_multiplier",
    "_resolve_warmup_lambda",
    "_slice_loss_region",
    "_loss_region_edge_bands",
    "_hf_power",
    "_edge_interior_ratio",
    "_border_inward_tv_penalty",
    "_spatial_gradients",
    "_spatial_laplacian",
    "_spectral_notch_loss",
]
