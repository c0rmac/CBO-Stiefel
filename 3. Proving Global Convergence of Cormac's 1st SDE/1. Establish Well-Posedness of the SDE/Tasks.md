## Step 1: Well-Posedness Proof Tasks

| Task ID | Task Description | Method / Justification |
| :--- | :--- | :--- |
| **1.1** | **Prove Lipschitz Continuity of Objective Function** | Assume the objective function $f: V(n, k) \to \mathbb{R}$ is Lipschitz continuous. This is a reasonable assumption as the domain $V(n, k)$ is compact. |
| **1.2** | **Prove Lipschitz Continuity of Consensus Point** | Show that the weighted average $\overline{X}_t^*$ is a Lipschitz continuous function of the particle positions $\{X_t^i\}_{i=1}^N$. This follows from Task 1.1. |
| **1.3** | **Prove Lipschitz Continuity of Projection Operator** | Demonstrate that the projection operator $P_X(Z)$ is a smooth (and therefore locally Lipschitz) function of $X$. |
| **1.4** | **Establish Local Lipschitz Continuity of SDE Coefficients** | Combine Tasks 1.2 and 1.3 to conclude that both the drift and diffusion coefficients are locally Lipschitz continuous. |
| **1.5** | **Verify Itô Correction Term** | Apply Itô's lemma to the constraint function $g(X) = X^\top X - I_k$. Show that $d(g(X_t)) = 0$ is satisfied *if and only if* the drift includes the term $-C_{n,k}\frac{\sigma(t)^2}{2}X_t^i$. |
| **1.6** | **Prove Existence of Unique Local Solution** | Invoke classical SDE existence theorems (e.g., Theorem of Itô) using the local Lipschitz property from Task 1.4. |
| **1.7** | **Prove Global Existence (Non-Explosion)** | Argue that the solution is confined to the compact manifold $V(n, k)$ (from Task 1.5). A solution on a compact set cannot explode in finite time, allowing the local solution (Task 1.6) to be extended to a unique global strong solution. |