# QPPONet-BMS

**Optimizing Charge-Discharge Cycles Using a Hybrid DQN-PPO Reinforcement Learning Agent for Intelligent Battery Management in Electric Vehicles**

> MV Sujan Kumar, Ganesh Khekare  
> School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%20%7C%20Nature-darkgreen?logo=nature)](https://www.nature.com/srep/)
[![Status](https://img.shields.io/badge/Status-Accepted%2C%20In%20Press-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Citation%20Required-red)](LICENSE)
[![Related Repo](https://img.shields.io/badge/Related-BatteryHealthNet-orange)](https://github.com/KRYSTALM7/BatteryHealthNet)

> **Journal:** Accepted for publication in *Scientific Reports* (Nature Portfolio), March 2026.  
> DOI will be added upon publication. This repository accompanies the paper.

---

## Overview

Battery Management Systems in electric vehicles have historically relied on rule-based control strategies that cannot adapt to real-world variability in temperature, load profiles, and battery aging. This work introduces **QPPONet** — a novel hybrid reinforcement learning agent that combines Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) to learn adaptive charge-discharge policies directly from battery operating data.

QPPONet is the control layer of a larger modular BMS framework. The predictive health modules (SOH estimation, RUL forecasting, thermal anomaly detection) are documented in the companion repo: [BatteryHealthNet](https://github.com/KRYSTALM7/BatteryHealthNet).

**The key idea:** rather than treating battery health prediction and charge optimization as separate problems, QPPONet directly couples the RL reward signal to live SOH estimates — so the agent is penalized in real time for actions that accelerate battery degradation, not just those that waste energy.

---

## Results at a Glance

| Method | Cumulative Reward | Gain vs. Rule-Based | SOH Degradation |
|--------|:---:|:---:|:---:|
| Rule-based baseline | 125 ± 5.1 | — | High |
| DQN only | 131 ± 4.5 | +4.8% | Medium |
| PPO only | 134 ± 4.1 | +7.2% | Medium |
| DQN + TD3 | 138 ± 3.9 | +12.0% | Low |
| **QPPONet (ours)** | **155 ± 4.2** | **+24 ± 1.2%** | **Lowest** |

Reported as mean ± std across **5 independent random seeds**, chronological 70/15/15 split, to prevent look-ahead leakage.

---

## Why QPPONet?

Standard RL approaches to BMS control rely on either value-based or policy-based methods alone:

| Method | Strength | Failure Mode in BMS |
|--------|----------|---------------------|
| DQN | Efficient Q-value estimation | Overestimates Q-values; unstable under multi-objective reward |
| PPO | Stable clipped policy updates | Slow to exploit value signal; conservative in sparse reward regions |
| TD3/SAC | Continuous action, stable | High sample complexity; requires careful tuning |

**QPPONet's core innovation — the hybrid advantage:**

Instead of using PPO's standard advantage (returns minus value baseline), QPPONet augments it with Q-value information from the DQN critic:

```
hybrid_advantage = (returns - V(s)) + λ_q * (Q(s,a) - V(s))
```

This gives the policy gradient a more informative signal during early training when the value baseline is unreliable, and helps the agent escape local optima that arise from the competing objectives in the BMS reward landscape.

---

## Reward Function

$$R = -\alpha \cdot P_{\text{loss}} + \beta \cdot E_{\text{efficiency}} - \gamma \cdot \text{SOH}_{\text{degradation}}$$

| Term | Role | Weight |
|------|------|--------|
| P_loss | Penalizes resistive + thermal losses | α = 0.6 |
| E_efficiency | Rewards useful energy per cycle | β = 0.3 |
| SOH_degradation | Penalizes capacity loss (from GB estimator) | γ = 0.1 |

Weights were determined by sensitivity analysis over α ∈ {0.3, 0.5, 0.6, 0.8}, β ∈ {0.1, 0.3, 0.5}, γ ∈ {0.05, 0.1, 0.2}. Full derivation and sensitivity table in [`docs/reward_formulation.md`](docs/reward_formulation.md).

The SOH_degradation term is not a fixed heuristic — it is computed at each timestep by the pre-trained Gradient Boosting estimator from BatteryHealthNet. This live coupling between the RL agent and the predictive health module is the architectural contribution that distinguishes this work from prior RL-for-BMS approaches.

---

## Algorithm

See [`QPPONet/pseudocode.py`](QPPONet/pseudocode.py) for the complete algorithmic description.

The full implementation will be released in this repository after the paper's DOI is assigned and the article goes live on Scientific Reports.

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| DQN learning rate | 1×10⁻⁴ |
| PPO actor learning rate | 3×10⁻⁴ |
| PPO critic learning rate | 3×10⁻⁴ |
| Clip epsilon (PPO) | 0.2 |
| Entropy coefficient | 0.01 |
| Replay buffer capacity | 200,000 |
| Target network update | every 1,000 steps |
| PPO epochs per rollout | 4 |
| Q-value blend weight (λ_q) | 0.01 |
| Random seeds | 5 |

---

## Repository Structure

```
QPPONet-BMS/
│
├── QPPONet/
│   └── pseudocode.py              # Complete algorithm pseudocode
│
├── docs/
│   ├── reward_formulation.md      # Reward function derivation + sensitivity analysis
│   └── architecture.md            # System architecture and module interactions
│
├── notebooks/
│   └── qpponet_demo.ipynb         # Reproducible demo on the public eVTOL dataset
│
├── figures/
│   ├── architecture_overview.png  # Full BMS framework diagram
│   └── rl_training_curve.png      # Learning curves across 5 seeds
│
├── requirements.txt
├── LICENSE
└── README.md
```

> **Implementation note:** `bms.py` (full training implementation) will be released here once the paper is live on Scientific Reports. The pseudocode provides a complete algorithmic description in the meantime.

---

## Dataset

### Proprietary Dataset (not released)

34 lithium-ion cells (2.0 Ah rated, EOL at 1.4 Ah) tested under four temperature conditions: room (24°C), high (43°C), low (4°C), and alternating (24°C / 44°C).

- Charging: CC-CV protocol (1.5A to 4.2V, cutoff at 20mA)
- Discharge: 1A / 2A / 4A, cutoff 2.0V–2.7V
- Impedance: EIS across 0.1 Hz to 5 kHz
- Hardware: Arbin BT-2000 cycler, IT9000 software

### Public Dataset — eVTOL (used in demo notebook)

Sony-Murata 18650 VTC-6 cells under electric air-taxi mission profiles. Released by Carnegie Mellon University.

**Download:** [CMU eVTOL Battery Dataset](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830)

The demo notebook runs end-to-end on this public dataset. The QPPONet algorithm is dataset-agnostic.

---

## Experimental Setup

- Chronological 70/15/15 train/val/test splits (no shuffle — prevents future leakage)
- 5-fold time-aware cross-validation
- 5 independent random seeds — all metrics reported as mean ± std
- Hardware: Intel Core i5-9300H @ 2.40GHz · 16 GB RAM · NVIDIA RTX 1050 (3 GB)

---

## Installation

```bash
git clone https://github.com/KRYSTALM7/QPPONet-BMS.git
cd QPPONet-BMS
pip install -r requirements.txt
```

**Quick start (public dataset demo):**
```bash
jupyter notebook notebooks/qpponet_demo.ipynb
```

---

## Related Work

This repository is the **RL control module** of a broader BMS framework. For the predictive health modules (SOH estimation, RUL forecasting, thermal anomaly detection):

→ [**BatteryHealthNet**](https://github.com/KRYSTALM7/BatteryHealthNet) — Conference paper (ICSPER 2025)

---

## Citation

If you use this work, please cite:

```bibtex
@article{sujankumar2026qpponet,
  title   = {Optimizing Charge Discharge Cycles Using QPPONet-Enabled Hybrid Learning
             Framework for Energy Management and Safety in Electric Vehicles},
  author  = {MV Sujan Kumar and Ganesh Khekare},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Accepted for publication},
  publisher = {Nature Portfolio}
}
```

*DOI will be added upon publication.*

---

## Authors

**MV Sujan Kumar** — [sujankumar7702@gmail.com](mailto:sujankumar7702@gmail.com) | [GitHub @KRYSTALM7](https://github.com/KRYSTALM7)

**Ganesh Khekare** — [ganesh.khekare@vit.ac.in](mailto:ganesh.khekare@vit.ac.in)

School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

*Supported by VIT Seed Grant (RGEMS) — Sanctioned Order No.: SPL/SG20230147*

---

## License

This work is released under a **custom Academic Use License** — see [LICENSE](LICENSE) for full terms.

**Key conditions:**
- Free for academic, research, and educational use
- Any publication or work using this code **must cite** the Scientific Reports paper (see Citation above)
- Commercial use is prohibited without written permission from the authors

> The proprietary EV battery dataset and full `bms.py` implementation are not included at this time. They will be released here upon journal publication. The pseudocode, reward formulation, and eVTOL demo are fully open.