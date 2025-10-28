# Proposed SDEs for Robust CBO on the Stiefel Manifold

[cite_start]This document details two proposed modifications to the "Stochastic Consensus Method for Nonconvex Optimization on the Stiefel Manifold" by Kim et al. [cite: 2-3].

The goal of these modifications is to address a critical limitation in the original SDE: its inability to guarantee global convergence. [cite_start]The original algorithm's exploration noise vanishes upon consensus [cite: 109, 1096-1097], causing it to become permanently trapped in any local minima, maxima, or saddle points, as demonstrated by our $V(2, 1)$ counterexample.

The following SDEs introduce a persistent, time-decaying noise source (simulated annealing) to ensure the algorithm is robust, can escape local traps, and can provably converge to the global optimum.

---

## Background: Kim's Original SDE

The SDE proposed by Kim et al. (Equation 5) [cite_start]is [cite: 103-104]:

$$
dX_t^i = \left(\lambda P_{X_t^i}(\overline{X}_t^*) - C_{n,k}\frac{\sigma^2|X_t^i-\overline{X}_t^*|_F^2}{2}X_t^i\right)dt + \sigma|X_t^i-\overline{X}_t^*|_F P_{X_t^i}(dW_t)
$$

* **Flaw:** The noise term $\sigma|X_t^i-\overline{X}_t^*|_F P_{X_t^i}(dW_t)$ is *state-dependent*. It goes to zero as consensus is reached ($|X_t^i-\overline{X}_t^*|_F \to 0$). This makes the algorithm efficient but "brittle," as it gets trapped in any critical point where it finds consensus.

---

## Cormac's 1st SDE (Annealed CBO)

This proposal completely replaces the state-dependent noise with a time-dependent annealing schedule, $\sigma(t)$.

### SDE Formulation

$$
dX_t^i = \left(\lambda P_{X_t^i}(\overline{X}_t^*) - C_{n,k}\frac{\sigma(t)^2}{2}X_t^i\right)dt + \sigma(t) P_{X_t^i}(dW_t)
$$

* **Consensus Drift:** $\lambda P_{X_t^i}(\overline{X}_t^*)dt$ (Unchanged)
* **Robust Noise:** $\sigma(t) P_{X_t^i}(dW_t)$. The noise is now independent of the consensus state and is controlled by a pre-defined "cooling schedule" $\sigma(t)$, where $\sigma(t) \to 0$ as $t \to \infty$.
* **Correction Term:** $C_{n,k}\frac{\sigma(t)^2}{2}X_t^i dt$. [cite_start]This term is mathematically required to replace Kim's original correction term, ensuring the SDE's solution remains on the manifold $V(n, k)$ under the new noise model [cite: 359-365].

### Analysis

* **Advantage:** This algorithm is **robust**. The persistent noise allows it to escape local traps. Its global convergence could be rigorously proven using standard simulated annealing theory.
* **Disadvantage:** This algorithm is **inefficient and "finicky"**. It has lost the adaptivity of the original. Its success now depends entirely on the careful, problem-specific tuning of the cooling schedule ($\sigma_{initial}$, $\sigma_{final}$, $T$). If cooled too fast, it gets trapped; if cooled too slow, it's computationally wasteful.

---

## Cormac's 2nd SDE (Hybrid CBO)

This proposal combines the "best of both worlds": the efficiency of Kim's adaptive noise and the robustness of the annealing schedule.

### SDE Formulation

First, we define a **hybrid noise strength** $S_{total}(i, t)$ that sums both noise sources:

$$
S_{total}(i, t) = \underbrace{\sigma_1 |X_t^i - \overline{X}_t^*|_F}_{\text{Kim's Adaptive Noise}} + \underbrace{\sigma_2(t)}_{\text{Annealing Noise Floor}}
$$

This combined strength is then used in the full SDE:

$$
dX_t^i = \left( \lambda P_{X_t^i}(\overline{X}_t^*) - C_{n,k}\frac{S_{total}(i, t)^2}{2}X_t^i \right)dt + S_{total}(i, t) P_{X_t^i}(dW_t)
$$

### Analysis

* **Advantage (Robustness):** If the algorithm gets stuck in a local trap, $|X_t^i - \overline{X}_t^*|_F \to 0$, but the $\sigma_2(t)$ term provides a **persistent noise floor** that allows the particles to escape.
* **Advantage (Efficiency):** When the algorithm finds the *global* minimum, $\sigma_2(t) \to 0$ (as the schedule ends) and $|X_t^i - \overline{X}_t^*|_F \to 0$ (as consensus is reached). **Both noise terms vanish automatically**, allowing the algorithm to "lock in" the final solution without inefficient, prolonged "jiggling."
* **Disadvantage:** The algorithm is now more complex, requiring the user to tune the hyperparameters for *both* the adaptive part ($\sigma_1$) and the annealing schedule ($\sigma_2(t)$). However, the schedule can likely be more aggressive (faster) than in the 1st SDE, as it only needs to provide an "escape" mechanism rather than being the sole source of exploration.

---

## Summary Comparison

| Algorithm | Noise Term Formula | Key Trade-off |
| :--- | :--- | :--- |
| **Kim's Original SDE** | $\sigma|X_t^i - \overline{X}_t^*|_F$ | **Simple & Efficient,** but **Brittle** (gets trapped) |
| **Cormac's 1st SDE** | $\sigma(t)$ | **Robust & Provable,** but **Finicky** (hard to tune) |
| **Cormac's 2nd SDE**| $\sigma_1 |X_t^i - \overline{X}_t^*|_F + \sigma_2(t)$ | **Robust & Efficient,** but **Complex** (more parameters) |