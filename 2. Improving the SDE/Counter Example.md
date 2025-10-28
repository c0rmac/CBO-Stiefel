# Counterexample: Failure of Manifold CBO under Euclidean "Mild Conditions"

[cite_start]This document demonstrates the hypothesis that the global convergence guarantees for Consensus-Based Optimization (CBO) in a Euclidean setting [cite: 6] [cite_start]do not directly apply to the Stiefel manifold CBO variant proposed by Kim et al.[cite: 995].

The fundamental reason is that the **constrained geometry of the manifold can introduce spurious critical points** (e.g., local maxima) that do not exist in the unconstrained Euclidean space. The specific SDE design in the Kim paper, which links the noise term to the consensus distance, causes the algorithm to get permanently trapped at these spurious points.

---

## 1. The Counterexample Setup

### The Manifold: $V(2, 1)$

We use the simplest non-trivial Stiefel manifold: $V(2, 1)$.
* This is the set of $2 \times 1$ matrices (vectors) $X$ such that $X^\top X = I_1$.
* In $\mathbb{R}^2$, this is simply the unit circle $S^1$.
* $X = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$ such that $x_1^2 + x_2^2 = 1$.

### The Objective Function: $\mathcal{E}(X)$

We will use an objective function that is perfectly "mild" in the ambient Euclidean space $\mathbb{R}^2$. [cite_start]The canonical example from the Fornasier paper is the squared Euclidean distance to the minimizer[cite: 142].

* **Global Minimizer:** Let $X^* = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$. This point is on the manifold $V(2, 1)$.
* **Objective Function:** $\mathcal{E}(X) = \frac{1}{2} ||X - X^*||_F^2 = \frac{1}{2} \left( (x_1 - 1)^2 + x_2^2 \right)$.

This function is convex on all of $\mathbb{R}^2$ and thus satisfies the "mild conditions" for global convergence in a Euclidean setting.

---

## 2. The "Trap": A Spurious Critical Point

When we restrict $\mathcal{E}(X)$ to the manifold $V(2, 1)$, its properties change. Using the constraint $x_1^2 + x_2^2 = 1$, we can simplify the objective:

$$
\mathcal{E}(X) = \frac{1}{2} (x_1^2 - 2x_1 + 1 + x_2^2) = \frac{1}{2} ( (x_1^2 + x_2^2) - 2x_1 + 1) = \frac{1}{2} (1 - 2x_1 + 1) = 1 - x_1
$$

On the manifold $V(2, 1)$, $\mathcal{E}(X) = 1 - x_1$. This function has two critical points:

1.  **Global Minimizer $X^*$:** The function is minimized when $x_1$ is maximized. This occurs at $x_1 = 1$, giving $X^* = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$. Here, $\mathcal{E}(X^*) = 0$.
2.  **The "Trap" Point $X_a$:** The function is maximized when $x_1$ is minimized. This occurs at $x_1 = -1$, giving the antipodal point $X_a = \begin{pmatrix} -1 \\ 0 \end{pmatrix}$. This is a **local maximum** on the manifold, and $\mathcal{E}(X_a) = 2$.

---

## 3. Demonstration of Failure

We now show that if the particle system in the Kim et al. paper converges to the "trap" point $X_a$, it will be permanently stuck.

Assume that at time $t$, all $N$ particles have reached consensus at $X_a$:
$$X_t^i = X_a = \begin{pmatrix} -1 \\ 0 \end{pmatrix} \quad \text{for all } i = 1, \dots, N$$

[cite_start]We now analyze the SDE from the Kim paper (Equation 5) [cite: 1096-1097]:
$$dX_t^i = \left(\lambda P_{X_t^i}(\overline{X}_t^*) - C_{n,k}\frac{\sigma^2|X_t^i-\overline{X}_t^*|_F^2}{2}X_t^i\right)dt + \sigma|X_t^i-\overline{X}_t^*|_F P_{X_t^i}(dW_t)$$

**Step 1: Calculate the Consensus Point $\overline{X}_t^*$**
The consensus point $\overline{X}_t^*$ is the weighted average $\frac{\sum \omega_f^\beta(X_t^i) X_t^i}{\sum \omega_f^\beta(X_t^i)}$.
Since all $X_t^i = X_a$, their function values $\mathcal{E}(X_t^i) = 2$ are identical. Thus, all weights $\omega_f^\beta(X_t^i) = \exp(-2\beta)$ are identical.
The weighted average is:
$$\overline{X}_t^* = \frac{\sum w_i X_a}{\sum w_i} = \frac{X_a \sum w_i}{\sum w_i} = X_a$$

**Step 2: Calculate the Distance Term**
The distance term $|X_t^i - \overline{X}_t^*|_F$ is:
$$|X_t^i - \overline{X}_t^*|_F = |X_a - X_a|_F = 0$$

**Step 3: Analyze the Collapsed SDE**
Because the distance term is 0, this value multiplies both the correction term and the entire noise term, setting them to zero. The SDE for every particle $i$ collapses to:
$$dX_t^i = \left(\lambda P_{X_a}(\overline{X}_t^*) - 0\right)dt + 0$$
Substituting $\overline{X}_t^* = X_a$:
$$dX_t^i = \lambda P_{X_a}(X_a) dt$$

**Step 4: Final State**
[cite_start]The Kim paper notes that the projection of a point $X$ onto its own tangent space $T_X V(n,k)$ is zero [cite: 1093-1094].
Therefore, $P_{X_a}(X_a) = 0$.

The SDE becomes:
$$dX_t^i = \mathbf{0}$$

The particles are permanently trapped at $X_a$, the global maximum on the manifold. The algorithm has failed to find the global minimum $X^*$.

---

## 4. Conclusion

> [cite_start]**Why this fails:** Fornasier's proof of global convergence in $\mathbb{R}^d$ relies on the CBO dynamics, on average, following the gradient flow of the convex function $v \mapsto ||v - v^*||_2^2$[cite: 142]. In the Euclidean space $\mathbb{R}^d$, this convex function has **only one** critical point ($v^*$), so a consensus-seeking algorithm is guaranteed to find it.

> **The Manifold Mismatch:** On the Stiefel manifold, our simple objective function $\mathcal{E}(X)$ has **multiple** critical points (a minimum at $X^*$ and a maximum at $X_a$). [cite_start]Kim's CBO algorithm, by design, causes its own exploration (noise) term to vanish as consensus is reached [cite: 1096-1097]. If consensus forms at *any* critical point, the distance term becomes zero, the noise disappears, and the particles are trapped.

The "mild conditions" from the Fornasier paper are insufficient because they do not account for the new topology: the manifold's geometry creates spurious critical points, and the Kim SDE's design lacks a mechanism (like a constant noise term) to escape them.