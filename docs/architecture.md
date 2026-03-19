# QPPONet-BMS — System Architecture

## Framework Overview

The full BMS framework consists of four coordinated modules. QPPONet is the control layer (Module 3). The other three modules are documented in [BatteryHealthNet](https://github.com/KRYSTALM7/BatteryHealthNet).

```
┌────────────────────────────────────────────────────────────┐
│                  Unified BMS Framework                     │
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐ │
│  │ SOH Estimator│    │RUL Forecaster│    │Thermal Monitor│ │
│  │  (GB Model)  │    │ (LSTM-GRU)   │    │ (IF + OC-SVM) │ │
│  └──────┬───────┘    └──────┬───────┘    └──────┬────────┘ │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             │                              │
│                    ┌────────▼────────┐                     │
│                    │   QPPONet Agent  │                    │
│                    │  (DQN + PPO)    │                     │
│                    │                 │                     │
│                    │  Reward signal  │                     │
│                    │  couples live   │                     │
│                    │  SOH estimates  │                     │
│                    └────────┬────────┘                     │
│                             │                              │
│                    ┌────────▼────────┐                     │
│                    │  Charge/Discharge│                    │
│                    │  Control Output │                     │
│                    └─────────────────┘                     │
└────────────────────────────────────────────────────────────┘
```

## Module Interactions

The SOH estimator feeds the QPPONet reward function at every timestep. This is not a post-hoc integration — the Gradient Boosting SOH predictor runs in the inner loop of the RL environment, and its output directly shapes the degradation penalty term γ·SOH_degradation in the reward.

This coupling is what makes QPPONet different from RL-for-BMS approaches that treat battery health as a background constraint rather than an active optimization target.

## Decision Matrix

| Scenario | Module Triggered | Response |
|----------|-----------------|----------|
| Normal cycling | QPPONet | Adaptive charge/discharge rate |
| SOH drop detected | QPPONet + SOH estimator | Increase γ weight, reduce C-rate |
| Thermal anomaly | Thermal Monitor → QPPONet | Emergency rate reduction |
| Rapid load change | DQN component | Fast Q-value based action |
| Long-horizon planning | PPO component | Stable policy-gradient update |

## Environment State Space

The simulated battery environment exposes the following state to the QPPONet agent at each step:

- Voltage (V)
- Current (A)
- Temperature (°C)
- State of Charge estimate (%)
- Cycle number
- dT/dt (thermal rate of change)
- SOH estimate (from GB model)

## Why Simulated Environment?

The RL agent was trained in simulation (not directly on hardware) for two reasons: (1) real battery cycling is destructive — you cannot explore aggressive charge policies on physical cells without accelerating degradation, and (2) the simulator was calibrated against the proprietary dataset, so the learned policies transfer to real operating conditions without domain gap.
