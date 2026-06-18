# AQAIE

**A physics-informed framework for atmospheric state reconstruction and air quality forecasting.**

---

## About This Project

AQAIE is a research project exploring atmospheric state reconstruction and air quality forecasting from sparse monitoring networks.

The project combines meteorological data, environmental covariates, and physics-informed learning to reconstruct high-resolution pollutant fields and predict their temporal evolution.

Unlike conventional approaches that rely on dense model-generated target fields, AQAIE is designed around sparse real-world observations and focuses on learning physically consistent spatial and temporal representations.

---

## The Problem

Air quality observations are inherently sparse. Ground monitoring stations provide accurate measurements, but only at a limited number of locations, while pollutant concentrations evolve continuously in space and time under the influence of meteorology, emissions, and atmospheric transport.

This creates an ill-posed reconstruction problem: how can a model infer a physically consistent high-resolution pollutant field from sparse observations and environmental data?

AQAIE explores whether neural assimilation and physics-informed learning can recover latent atmospheric structure and forecast its future evolution without relying on a full chemical transport simulation during inference.

---

## Architecture

```mermaid
flowchart TB

  subgraph INPUTS["1. Inputs"]
    STATIONS["Surface Stations<br/>sparse observations"]
    ERA5_T["ERA5 Temporal<br/>multi-lag meteorology"]
    ERA5_D["ERA5 Direct<br/>current meteorology"]
    CAMS["Optional Background Prior<br/>CAMS forecast"]
    STATIC["Static Features<br/>DEM, land use, population"]
  end

  subgraph ASSIM["2. Sparse Observation Assimilation (INR)"]
    CROSS["Cross-Attention INR<br/>Fourier PE + distance-aware top-k"]
    DENSE["Dense Observation Field<br/>(c₀ reconstruction)"]

    STATIONS --> CROSS
    ERA5_T --> CROSS
    ERA5_D --> CROSS
    CROSS --> DENSE
  end

  subgraph BACKBONE["3. Forecast Backbone"]
    OBS["Observation Features<br/>(from INR)"]
    MET["Meteorological Features"]
    FUSION["Feature Fusion"]
    UNET["U-Net Backbone<br/>FiLM conditioned"]

    DENSE --> OBS
    ERA5_T --> MET
    ERA5_D --> MET

    OBS --> FUSION
    MET --> FUSION

    FUSION --> UNET

    STATIC -. conditioning .-> UNET
    CAMS -. gated injection .-> UNET
  end

  subgraph OUTPUT["4. Multi-Horizon Forecast"]
    HEAD["Forecast Head<br/>base + cumulative delta"]
    PRED["6 pollutants × 6 horizons<br/>1h · 2h · 4h · 8h · 12h · 24h"]

    UNET --> HEAD --> PRED
  end

  subgraph REG["5. Physics-Informed Constraints"]
    PHYS["Assimilation<br/>Advection<br/>Semi-Lagrangian<br/>Quantile & Auxiliary Losses"]
  end

  DENSE --> PHYS
  HEAD --> PHYS
```

---

## Core Ideas

1. Reconstruct a dense concentration field from sparse monitoring stations.
2. Forecast the reconstructed field rather than forecasting stations directly.
3. Use physics-informed regularization only where observations are absent.
4. Treat coarse-scale model products as optional priors through learned gating.
5. Diagnose artifacts visually and suppress them with targeted architectural changes.

---

## Evaluation

Primary metric: Leave-K-Out (LKO) station validation.

LKO measures spatial generalization by withholding monitoring stations during validation and predicting concentrations at unseen locations.

Current experiments achieve:

* PM2.5 LKO RMSE: ~0.076–0.081 (normalized space)
* Overall R²: ~0.77 across pollutants and forecast horizons

---

## Repository Structure

Selected components are included to illustrate data pipeline and the architecture.

### Data Pipeline

| Component | Description |
|------------|-------------|
| `build_era5_31ch.py` | ERA5 / ERA5-Land processing and grid harmonization |
| `build_airkorea_parquet.py` | Sparse station dataset construction |
| `preprocess_pop_1km.py` | Population raster preprocessing and aggregation |

### Modeling

| Component | Description |
|------------|-------------|
| `obs_sparse_inr.py` | Sparse observation assimilation using Fourier features and cross-attention |
| `fusion.py` | Meteorological feature fusion and preprocessing |
| `backbone_unet.py` | U-Net forecasting backbone with FiLM conditioning |
| `head_multi_horizon.py` | Multi-horizon quantile forecasting head |
| `branch_cams.py` | Optional coarse-scale background prior |
| `losses.py` | Physics-informed and uncertainty-aware training losses |

---

## Design Evolution

The current architecture emerged through a series of revisions driven by a single constraint:

**forecasting dense air-quality fields from sparse monitoring stations.**

Unlike weather forecasting systems trained on dense reanalysis grids, supervision is available only at observation locations. Most architectural decisions were introduced to address limitations created by this sparse-observation setting.

---

### 1. Baseline: End-to-End Forecasting

The initial baseline was motivated by recent advances in end-to-end, data-driven weather forecasting systems. Aardvark Weather [1] demonstrated an observation-driven forecasting framework that reduces dependence on traditional numerical weather prediction components by directly learning from heterogeneous observational inputs. Pangu-Weather [2] introduced a hierarchical multi-step forecasting architecture based on temporal aggregation, where forecasts at multiple lead times are constructed through structured intermediate representations rather than naive step-by-step autoregressive rollouts.

Following these ideas, a U-Net backbone was designed to consume meteorological inputs, static features, and station observations, producing multi-horizon forecasts in a single forward pass. A direct multi-horizon decoding strategy was adopted instead of iterative temporal rollout, motivated by two considerations in the sparse-observation setting: (1) error propagation in iterative forecasting is amplified when supervision is available only at irregular station locations, and (2) computational efficiency is important for potential deployment under constrained hardware settings.

While this approach achieved reasonable station-level predictive performance, it struggled to generate physically consistent spatial fields away from observation locations due to limited spatial supervision and weak inductive bias over continuous fields.

---

### 2. Sparse Observations Require Explicit Assimilation

Early experiments revealed a fundamental mismatch between sparse monitoring stations and convolutional architectures.

A handful of station measurements distributed across a large spatial domain provide insufficient spatial support for dense field reconstruction.

To address this limitation, a dedicated sparse-to-dense assimilation stage was introduced before the forecasting backbone.

The resulting INR module reconstructs a dense concentration field from sparse station observations [11] using Fourier positional encoding [9] and implicit field representation principles [10].

This reconstructed field acts as an observation anchor for downstream forecasting.

---

### 3. Physics as a Constraint, Not a Simulator

Even after dense reconstruction, large portions of the spatial domain remain weakly supervised.

Rather than attempting to emulate a full chemical transport model, the project uses lightweight physics-informed constraints to regularize solutions in these regions.

The design follows the PINN framework [7] while accounting for known failure modes of advection-dominated systems [6].

Current constraints include:

* advection consistency
* semi-Lagrangian consistency
* observation assimilation
* uncertainty-aware quantile regression
* spatial smoothness and stability regularization

The objective is not numerical simulation, but reduction of physically implausible solutions under sparse supervision.

---

### 4. Artifact Analysis Became a Research Problem

Several training runs exhibited checkerboard patterns [3], lattice artifacts, and high-frequency texture amplification.

Rather than treating these as generic deep-learning failures, the transmission paths were analyzed directly.

This investigation led to:

* anti-alias filtering of meteorological inputs [4]
* skip-path filtering within the decoder [4]
* frequency-domain regularization inspired by spectral reconstruction methods [5]

These additions significantly reduced persistent grid-scale artifacts while preserving forecast skill.

---

### 5. Temporal Consistency Beyond Independent Horizons

Direct multi-horizon forecasting is computationally efficient, but neighboring forecast horizons can evolve inconsistently.

To encourage coherent temporal evolution, the training loop incorporates a time-shifted consistency objective inspired by Temporal Cycle Consistency Learning [8].

Random forecast horizons are replayed using preceding observations, allowing adjacent lead times to act as consistency constraints during training.

This mechanism encourages transport-consistent evolution without requiring fully autoregressive rollout.

---

### Current Direction

Ongoing work focuses on three areas:

* improving long-range transport consistency
* integrating geostationary satellite observations
* transitioning from direct multi-horizon prediction toward autoregressive refinement

The overall goal remains unchanged: reconstructing and forecasting high-resolution air-quality fields from sparse observational networks.

---

## Artifact Suppression

Early training runs developed persistent high-frequency spatial artifacts.
The issue was diagnosed through intermediate field visualizations and
resolved without sacrificing hold-out station performance.

| Metric | Before | After |
|----------|----------|----------|
| High-frequency ratio | **2.17** | **0.94** |
| LKO RMSE (PM2.5) | 0.081 | 0.082 |
| LKO R² (PM2.5) | 0.38 | 0.38 |
| Uncertainty spread | 0.13 | **0.11** |

Severe high-frequency artifacts were removed while predictive skill
remained effectively unchanged.

### Before

![artifact_before](assets/artifact_before.png)

### After

![artifact_after](assets/artifact_after.png)

*Diagnostic panel used during development. Each panel contains:*

- Predicted median field (q50)
- Predicted field with observation dots
- Error at observation locations
- Quantile spread (uncertainty)
- INR reconstruction (`c₀`) with observations
- Forecast delta relative to `c₀`

---

## Tech Stack

**Modeling**
`PyTorch` `U-Net` `INR` `Cross-Attention` `PINN`

**Data**
`ECMWF/ERA5` `CAMS/EAC4` `Parquet` `Zarr`

**Training**
`Azure ML` `MLflow`

**Development**
`Python`

---

## Why This Problem Is Interesting

Most operational air-quality forecasting systems depend on a long modeling chain:

**Emission Inventory → Meteorology → Chemical Transport Model → Forecast**

Each stage introduces its own assumptions and uncertainties. In practice, emission inventories are often incomplete, temporally aggregated, or outdated, while forecast quality can be strongly affected by errors propagated through the modeling chain.

This project explores a different question:

> Can dense air-quality fields be reconstructed and forecast directly from sparse observations, meteorology, and spatial context?

Key characteristics:

* **No emission inventories as direct model inputs.**
* **No online chemical transport simulation required at inference.**
* **Designed for sparse-monitoring environments**, where only a limited number of stations are available.
* **Physics-informed regularization** constrains transport behavior under weak supervision.
* **Multi-pollutant, multi-horizon forecasting** from a single model.
* **Probabilistic outputs** through quantile prediction (q10 / q50 / q90).
* **Inspectable intermediate representations** (`c₀`, forecast deltas, uncertainty fields), enabling visual diagnosis of failure modes and training behavior.

Rather than reproducing an existing CTM workflow, the project investigates whether observation-driven learning can recover useful spatial structure that is difficult to obtain from sparse monitoring networks alone.

---

## References

### Category 1: End-to-End Weather / Air-Quality Forecasting

| #   | Citation                                                                                                                | Relevance                                                                                                            |
| --- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [1] | Vaughan, A., Markou, S., et al. *Aardvark Weather: End-to-End Data-Driven Weather Forecasting.* Nature, 2024. | End-to-end observation-driven forecasting framework that integrates heterogeneous observational data into a unified learning system, reducing reliance on traditional numerical weather prediction pipelines. This motivated the use of direct observation-to-forecast modeling in AQAIE. |
| [2] | Bi, K., et al. *Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast.* Nature, 2023. | Hierarchical multi-step forecasting model based on temporal aggregation, where multi-lead-time predictions are generated through structured intermediate representations rather than naive autoregressive rollouts. This informed the multi-horizon forecasting design in the baseline. |

### Category 2: Artifact Analysis & Frequency-Domain Stabilization

| #   | Citation                                                                                     | Relevance                                                                                                   |
| --- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [3] | Odena, A., Dumoulin, V., Olah, C. *Deconvolution and Checkerboard Artifacts.* Distill, 2016. | Diagnostic reference for checkerboard artifacts observed during early training.                             |
| [4] | Karras, T., et al. *Alias-Free Generative Adversarial Networks.* NeurIPS, 2021.              | Nyquist-aware filtering and anti-aliasing principles. Basis for `met_pre_filter` and skip-path smoothing.   |
| [5] | Jiang, L., et al. *Focal Frequency Loss for Image Reconstruction and Synthesis.* ICCV, 2021. | Frequency-domain supervision philosophy. Motivated the spectral-notch loss used to suppress grid artifacts. |

### Category 3: Physics-Informed Learning & Temporal Consistency

| #   | Citation                                                                                                                  | Relevance                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [6] | Krishnapriyan, A., et al. *Characterizing Possible Failure Modes in Physics-Informed Neural Networks.* NeurIPS, 2021.     | PINN failure modes in advection-dominated systems. Reference for advection-loss tuning and diagnostics. |
| [7] | Raissi, M., Perdikaris, P., Karniadakis, G.E. *Physics-Informed Neural Networks.* Journal of Computational Physics, 2019. | Foundation for advection-diffusion regularization and physics-informed training objectives.             |
| [8] | Dwibedi, D., et al. *Temporal Cycle-Consistency Learning.* CVPR, 2019.                                                    | Conceptual reference for horizon-consistency training using neighboring observed states.                |

### Category 4: INR — Sparse-to-Dense Field Reconstruction

| #    | Citation                                                                                                  | Relevance                                                                                                                     |
| ---- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [9]  | Tancik, M., et al. *Fourier Features Let Networks Learn High-Frequency Functions.* NeurIPS, 2020.         | Fourier positional encoding used in the INR encoder.                                                                          |
| [10] | Sitzmann, V., et al. *Implicit Neural Representations with Periodic Activation Functions.* NeurIPS, 2020. | Coordinate-based neural field modeling and continuous spatial representation concepts.                                        |
| [11] | Mildenhall, B., et al. *NeRF: Representing Scenes as Neural Radiance Fields.* ECCV, 2020.                 | Implicit neural representation of continuous 3D scenes via coordinate-based MLPs, learning a volumetric radiance field from multi-view image supervision. This work established a foundational paradigm for neural continuous field representation. |
