# Reward Function — Derivation and Sensitivity Analysis

## Formulation

$$R = -\alpha \cdot P_{\text{loss}} + \beta \cdot E_{\text{efficiency}} - \gamma \cdot \text{SOH}_{\text{degradation}}$$

## Component Definitions

### P_loss — Power Loss
Resistive and thermal losses during the charge/discharge action. Computed from voltage sag and current draw. Penalizes aggressive current profiles that generate excess heat and reduce round-trip efficiency.

### E_efficiency — Energy Efficiency
Ratio of useful energy delivered to total energy drawn in the cycle. Rewards actions that minimize wasted energy, directly reflecting operational cost.

### SOH_degradation — State-of-Health Degradation
Estimated incremental capacity loss caused by the current action. **This term uses the pre-trained Gradient Boosting SOH estimator at inference time** — not a fixed electrochemical heuristic. The estimator takes the current (state, action, next_state) tuple and returns a predicted capacity fade value, which is used as the per-step degradation penalty.

This live coupling is the architectural mechanism that integrates the predictive health module into the RL control loop.

## Dense Reward Design

The reward is intentionally **dense** — all three terms provide feedback at every timestep. Sparse or episodic rewards (e.g., penalizing only at end-of-life) produced unstable training and high variance in preliminary experiments. Dense per-step signals were critical for QPPONet's convergence speed and consistency across seeds.

## Sensitivity Analysis

Grid search over:
- α ∈ {0.3, 0.5, 0.6, 0.8}
- β ∈ {0.1, 0.3, 0.5}
- γ ∈ {0.05, 0.1, 0.2}

| α | β | γ | Cum. Reward | SOH Loss | Notes |
|---|---|---|-------------|---------|-------|
| 0.3 | 0.5 | 0.1 | 141 ± 5.2 | Medium | Over-optimizes efficiency |
| 0.8 | 0.3 | 0.1 | 148 ± 4.8 | Low | Conservative, suboptimal reward |
| 0.6 | 0.1 | 0.1 | 143 ± 4.3 | Medium | Under-rewards efficiency |
| 0.5 | 0.3 | 0.2 | 146 ± 4.6 | Very low | Over-penalizes degradation early |
| **0.6** | **0.3** | **0.1** | **155 ± 4.2** | **Lowest** | **Best overall balance** |

## Design Rationale for Weight Values

**α = 0.6 (dominant):** Power loss is the most directly controllable signal per timestep and has the strongest immediate impact on cycle quality. Giving it the highest weight provides the densest gradient signal.

**β = 0.3:** Energy efficiency is important but partially captured by P_loss. A lower weight avoids redundancy between the two terms.

**γ = 0.1 (smallest):** SOH degradation accumulates slowly over hundreds of cycles. A small γ avoids over-penalizing the agent during early training when the SOH estimator's outputs are least reliable (low cycle count, less context for the GB model). As training progresses and the agent sees more cycles, this term becomes increasingly informative.
