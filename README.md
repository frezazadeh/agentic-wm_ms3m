# Agentic World Modeling for 6G O-RAN (WM–MS³M)

> **Repository status:** 🚧 The code will be made **public soon**.  
> Watch/star the repo to get notified when the implementation is released.

This repository contains the reference implementation of **WM–MS³M** from the paper:

> **Agentic World Modeling for 6G O-RAN: Near-Real-Time Generative State-Space Reasoning**  
> Farhad Rezazadeh, Hatim Chergui, Merouane Debbah, Houbing Song, Dusit Niyato, and Lingjia Liu.

WM–MS³M is an **agentic world model** for 6G O-RAN that combines:

- A **strictly causal multi-scale structured state-space mixture** (MS³M) backbone  
- A **compact stochastic latent** (world state)  
- **Dual decoders** (full KPI frame + heteroscedastic target head with AR skip)  
- An **MPC / Cross-Entropy Method (CEM)** planner operating on short horizons  
- **Action-conditioned what-if rollouts** with calibrated uncertainty (PRBs as first-class actions)

The goal is to support **near-real-time, counterfactual, and uncertainty-aware control** in 6G O-RAN, treating *prediction/imagination* and *decision-making* as two cleanly separated steps.

---

## Key Ideas

- **Agentic world modeling**  
  - Learn a generative state-space model that can **predict** and **imagine** KPI trajectories under hypothetical PRB sequences.  
  - Use those rollouts to **choose** actions via MPC/CEM in the Near-RT RIC.

- **WM–MS³M architecture**  
  - Multi-scale, strictly causal SSM front-end (HiPPO–LegS-based depthwise kernels).  
  - Latent world state \( \mathbf{z} \) with KL-annealed prior/posterior (VAE-style).  
  - Dual decoders:  
    - Full next-step KPI frame (for representation & cross-KPI structure).  
    - Heteroscedastic target head (e.g., RSRP) + bounded AR skip from last KPI.

- **Leakage-safe, unit-aware training**  
  - Chronological splits, train-only scalers, no bidirectional leakage.  
  - Feature-channel dropout + light Gaussian input noise.  
  - Posterior/prior mixing and KL annealing for stable latent usage.

- **Planning in standardized space**  
  - MPC/CEM over a short horizon \( H \), with PRBs constrained to train-derived bounds.  
  - Reward trades SINR/SE/RSRP vs BLER/Delay/PRB cost + smoothness penalty.  
  - Actions are optimized in scaled space then mapped back to physical PRB units.

---

## Performance (High-Level)

On realistic O-RAN traces, WM–MS³M:

- Improves **MAE by ~1.69%** vs. a strong MS³M baseline  
- Uses **~32% fewer parameters** with similar inference latency  
- Achieves **35–80% lower RMSE** than attention/hybrid baselines  
- Runs **2.3–4.1× faster** at inference under Near-RT constraints  

Beyond metrics, it enables:

- **What-if PRB analysis** (e.g., “what if we reduce PRBs by 20% for the next 8 steps?”)  
- **Offline policy screening** and rare-event simulation  
- **Agentic Near-RT control** via MPC/CEM over calibrated world-model predictions

---

## Repository Structure (Planned)

> 🧩 **Note:** The structure below reflects the planned layout once the code is public.

- `wm_ms3m/`
  - `models/` – WM–MS³M architecture (SSM backbone, latent world model, decoders)  
  - `training/` – training loop, losses, KL annealing, posterior/prior mixing  
  - `planning/` – MPC / CEM planner, PRB constraints, reward shaping  
  - `data/` – dataset utilities (windowing, scaling, splits)  
- `experiments/`
  - `configs/` – YAML/JSON configs for baselines and WM–MS³M  
  - `scripts/` – end-to-end train/eval/plan scripts  
- `notebooks/` – example notebooks: forecasting, what-if rollouts, PRB planning  
- `README.md` – (this file)  
- `LICENSE` – non-commercial research license (TBA)

---

## Getting Started

> ⏳ The code and full instructions (installation, training, and reproduction scripts)  
> will be added once the repository is made public.

Planned workflow:

1. **Install** the environment (PyTorch and common TS/SSM dependencies).  
2. **Prepare data**: O-RAN KPI traces with PRB, SINR, RSRP, BLER, Delay, SE, etc.  
3. **Train WM–MS³M** on chronological splits with leakage-safe preprocessing.  
4. **Run inference** for KPI forecasting and uncertainty estimation.  
5. **Run planner (MPC/CEM)** for near-real-time PRB control or offline policy evaluation.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{rezazadeh2025agenticWMMS3M,
  title   = {Agentic World Modeling for 6G O-RAN: Near-Real-Time Generative State-Space Reasoning},
  author  = {Rezazadeh, Farhad and Chergui, Hatim and Debbah, Merouane and
             Song, Houbing and Niyato, Dusit and Liu, Lingjia},
  journal = {arXiv preprint},
  year    = {2025}
}
```

(A final BibTeX entry will be updated once the official venue is available.)

---

## License

The code for this project will be released under a **non-commercial research license**.  
Details will be added once the repository goes public.

---

## Contact

For questions, collaborations, or feedback, please contact:

- **Farhad Rezazadeh** – `farhad.rezazadeh@upc.edu`

Or open an issue once the code is live.

