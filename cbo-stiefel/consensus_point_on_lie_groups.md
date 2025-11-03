# Analysis of Consensus Points for Project Tessera

This document outlines the distinction between the extrinsic and intrinsic consensus points for stochastic optimization on Lie groups, building upon the framework presented in the paper "Stochastic consensus dynamics for nonconvex optimization on the Stiefel manifold."

---

## The Extrinsic Mean: A Computationally Simple Approach

[cite_start]The paper's consensus dynamics model (1.2) [cite: 70] is built around a **weighted extrinsic mean**, $M_t$. [cite_start]This consensus point is defined as the weighted average of the particle positions $X_t^i$ in the ambient Euclidean space $\mathbb{R}^{n \times k}$[cite: 55]:

$$M_{t} = \sum_{i=1}^{N} w_{t}^{i} X_{t}^{i}, \quad \text{where} \quad w_{t}^{i} = \frac{\exp(-\beta f(X_{t}^{i}))}{\sum_{j=1}^{N} \exp(-\beta f(X_{t}^{j}))}$$

### Advantages:
* **Computationally Simple:** $M_t$ is calculated with a single, non-iterative summation. This is extremely fast and efficient.

### Disadvantages:
* [cite_start]**Geometrically Inaccurate:** $M_t$ is an average in $\mathbb{R}^{n \times k}$ and is not guaranteed to lie on the manifold $V(k,n)$[cite: 624]. For our case $SO(d)$, this means $M_t$ is generally *not* a rotation matrix.
* [cite_start]**Reliance on Projection:** The SDE must "correct" for this by projecting the drift vector $(M_t - X_t^i)$ onto the tangent space of each particle using the operator $P_X$[cite: 70].

---

## The Intrinsic Mean: A Geometrically Correct Alternative

For a Lie group like $SO(d)$, a geometrically superior alternative is the **intrinsic mean**, also known as the **Riemannian center of mass** or **Karcher mean**.

This mean, $\bar{X}_t$, is defined as the point on the manifold $SO(d)$ that minimizes the weighted sum of squared *Riemannian distances* to all particles:

$$\bar{X}_t = \underset{Y \in SO(d)}{\arg\min} \sum_{i=1}^{N} w_{t}^{i} \cdot d_{R}(Y, X_{t}^{i})^{2}$$

Where $d_R(Y, X_t^i)$ is the true geodesic distance on the $SO(d)$ manifold.

### Advantages:
* **Geometrically Correct:** $\bar{X}_t$ is, by definition, a point on $SO(d)$. It represents the "true" center of mass of the particle distribution on the manifold.
* **Structurally Appropriate:** On a Lie group, this computation is well-defined using the matrix exponential and logarithm, which respect the group's algebraic structure.

### Disadvantages:
* **Computationally Expensive:** Finding $\bar{X}_t$ requires solving a non-linear optimization problem on the manifold at *every time step* of the SDE, typically using an iterative Riemannian gradient descent algorithm.

This computational bottleneck makes the direct use of the intrinsic mean impractical for a large-scale particle simulation. However, we can design fast, non-iterative algorithms that *approximate* it.

---

## Practical Alternatives for Intrinsic Consensus

Given the computational cost, we propose two practical hybrid algorithms that can be analyzed for Project Tessera.

### 1. The Hybrid Approach: A "Warm-Start" Approximation

This method combines the speed of the extrinsic mean with a single correction step toward the intrinsic mean.

1.  [cite_start]**Fast Extrinsic Guess:** Compute the computationally cheap extrinsic mean $M_t$[cite: 55].
2.  **Project onto Manifold:** Project $M_t$ onto $SO(d)$ to get a valid initial guess $Y_0$. [cite_start]This uses the SVD-based projection method from Algorithm 1 in the paper: $Y_0 = M_t (M_t^\top M_t)^{-1/2}$[cite: 1277, 1294].
3.  **One-Step Intrinsic Update:** Perform *only one iteration* of the Riemannian gradient descent algorithm. We compute the gradient at $Y_0$ and take a single step along the geodesic:
    * **Gradient:** $V_0 = \sum_{i=1}^N w_t^i \log_{Y_0}(X_t^i)$
    * **Final Mean:** $\bar{X}_t \approx \text{Exp}_{Y_0}(-\alpha V_0)$

This algorithm is non-iterative and fast, yet its final consensus point is "pulled" from the simple projection toward the true geometric center of mass.

### 2. The Stochastic Approach: One-Step Update from Previous Mean

This method re-frames the problem as a continuous, online stochastic optimization.

1.  **Assume Slow Change:** We assume the true intrinsic mean $\bar{X}_t$ evolves slowly and continuously, so $\bar{X}_{t_{n-1}}$ is a good guess for $\bar{X}_{t_n}$.
2.  **Use Previous Mean as Guess:** At time $t_n$, we use the computed mean from the previous step, $\bar{X}_{t_{n-1}}$, as our initial guess $Y_0$.
3.  **One-Step Stochastic Update:** We compute the Riemannian gradient at this *old* mean using the *new* particle positions $\{X_{t_n}^i\}$ and take a single step:
    * **Gradient:** $V_n = \sum_{i=1}^N w_{t_n}^i \log_{\bar{X}_{t_{n-1}}}(X_{t_n}^i)$
    * **New Mean:** $\bar{X}_{t_n} = \text{Exp}_{\bar{X}_{t_{n-1}}}(-\alpha_n V_n)$

This effectively creates a **Riemannian Stochastic Gradient Descent (SGD)** algorithm for finding the consensus point, where the "stochasticity" in the gradient comes from the movement of the SDE particles.