# Definition of the Intrinsic Mean (Karcher Mean)

The intrinsic mean, denoted $\bar{X}_t$, for a set of particles $\{X_t^i\}_{i=1}^N$ with corresponding weights $\{w_t^i\}_{i=1}^N$ on a Riemannian manifold $M$ (such as the Stiefel manifold $V(k,n)$ or the Lie group $SO(d)$) is the solution to the following minimization problem:

$$\bar{X}_t = \underset{Y \in M}{\arg\min} \left\{ \mathcal{F}(Y) \right\}$$

where the objective function $\mathcal{F}(Y)$ is the **weighted intrinsic variance**:

$$\mathcal{F}(Y) = \sum_{i=1}^{N} w_{t}^{i} \cdot d_{R}(Y, X_{t}^{i})^{2}$$

---

## Terminology

| Term | Description |
| :--- | :--- |
| $\bar{X}_t$ | The **Intrinsic Mean** (Riemannian Center of Mass). |
| $Y$ | A candidate point on the manifold, $Y \in M$. |
| $X_t^i$ | The position of the $i$-th particle at time $t$, $X_t^i \in M$. |
| $w_t^i$ | [cite_start]The weight of the $i$-th particle, typically a Boltzmann-type factor: $$w_{t}^{i} = \frac{\exp(-\beta f(X_{t}^{i}))}{\sum_{j=1}^{N} \exp(-\beta f(X_{t}^{j}))}$$ [cite: 55] |
| $d_{R}(Y, X_{t}^{i})$ | The **Riemannian distance** (or geodesic distance) between the points $Y$ and $X_t^i$. |