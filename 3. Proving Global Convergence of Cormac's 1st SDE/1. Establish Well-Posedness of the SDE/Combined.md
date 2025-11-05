**Lemma 1.1:** Let $V(n, k)$ be the Stiefel manifold, a compact Riemannian submanifold of $\mathbb{R}^{n \times k}$. If the objective function $f: V(n, k) \to \mathbb{R}$ is $C^1$ (continuously differentiable) on the manifold, then $f$ is Lipschitz continuous with respect to the Riemannian (geodesic) distance $d(\cdot, \cdot)$.

---

**Proof:**

1.  **Bounded Gradient:** By assumption, the function $f$ is $C^1$. This implies its Riemannian gradient, $\nabla_{V(n,k)} f$, is a continuous vector field on $V(n, k)$. The norm of this gradient, $||\nabla_{V(n,k)} f(X)||_F$, is therefore a continuous real-valued function on $V(n, k)$.
2.  **Compact Domain:** The Stiefel manifold $V(n, k)$ is a closed and bounded subset of $\mathbb{R}^{n \times k}$, and is therefore compact.
3.  **Apply Extreme Value Theorem:** A continuous function on a compact set must attain a maximum value. Thus, the norm of the gradient is globally bounded. There exists a finite constant $L \ge 0$ such that:
    $$L = \sup_{X \in V(n, k)} ||\nabla_{V(n,k)} f(X)||_F < \infty$$
4.  **Apply Mean Value Theorem:** Let $X, Y$ be any two points in $V(n, k)$. Since $V(n, k)$ is a complete Riemannian manifold, there exists a minimizing geodesic $\gamma: [0, 1] \to V(n, k)$ such that $\gamma(0) = X$ and $\gamma(1) = Y$, with length $d(X, Y)$.
5.  By the Mean Value Theorem on manifolds (an application of the Fundamental Theorem of Calculus along $\gamma$):
    $$f(Y) - f(X) = \int_0^1 \langle \nabla_{V(n,k)} f(\gamma(t)), \dot{\gamma}(t) \rangle_F dt$$
    where $\dot{\gamma}(t)$ is the tangent vector to the curve at $t$.
6.  **Bound the Integral:** We take the absolute value and apply the Cauchy-Schwarz inequality:
    $$|f(Y) - f(X)| = \left| \int_0^1 \langle \nabla_{V(n,k)} f(\gamma(t)), \dot{\gamma}(t) \rangle_F dt \right|$$
    $$|f(Y) - f(X)| \le \int_0^1 ||\nabla_{V(n,k)} f(\gamma(t))||_F \cdot ||\dot{\gamma}(t)||_F dt$$
7.  Using the bound $L$ from step 3 for the gradient term:
    $$|f(Y) - f(X)| \le \int_0^1 L \cdot ||\dot{\gamma}(t)||_F dt$$
    $$|f(Y) - f(X)| \le L \int_0^1 ||\dot{\gamma}(t)||_F dt$$
8.  **Geodesic Length:** The integral $\int_0^1 ||\dot{\gamma}(t)||_F dt$ is precisely the definition of the arc length of the curve $\gamma$. Since $\gamma$ is the minimizing geodesic connecting $X$ and $Y$, its length is $d(X, Y)$.
9.  **Conclusion:** Substituting the result from step 8 into step 7, we obtain:
    $$|f(Y) - f(X)| \le L \cdot d(X, Y)$$
    This holds for all $X, Y \in V(n, k)$, which is the definition of Lipschitz continuity.

**Q.E.D.**

**Lemma 1.2:** Let the state space of the $N$-particle system be $\mathcal{X} = (V(n, k))^N$, which is a compact manifold. The consensus point $\overline{X}_t^*$ is a function $C: \mathcal{X} \to \mathbb{R}^{n \times k}$ defined as:
$$C(\mathbf{X}) = \frac{\sum_{j=1}^N X^j \exp(-\beta f(X^j))}{\sum_{l=1}^N \exp(-\beta f(X^l))}$$
where $\mathbf{X} = (X^1, \dots, X^N) \in \mathcal{X}$.

If the objective function $f: V(n, k) \to \mathbb{R}$ is $C^1$, then the consensus function $C(\mathbf{X})$ is Lipschitz continuous with respect to a product metric on $\mathcal{X}$.

---

**Proof:**

This proof follows the same principle as Lemma 1.1: a $C^1$ function on a compact domain is Lipschitz continuous.

1.  **Domain Compactness:** The state space $\mathcal{X} = (V(n, k))^N$ is a finite product of compact manifolds. Therefore, $\mathcal{X}$ is itself a compact manifold.

2.  **Differentiability of Components:**
    * From Lemma 1.1, we assume $f$ is $C^1$.
    * The exponential function $\exp: \mathbb{R} \to \mathbb{R}$ is $C^\infty$ (smooth).
    * The $j$-th projection $\pi_j: \mathcal{X} \to V(n, k)$ mapping $\mathbf{X} \mapsto X^j$ is $C^\infty$.
    * Therefore, the weight function $w_j(\mathbf{X}) = \exp(-\beta f(\pi_j(\mathbf{X})))$ is a composition of $C^1$ functions, and is itself $C^1$.

3.  **Differentiability of Numerator and Denominator:**
    * Let $W(\mathbf{X}) = \sum_{l=1}^N w_l(\mathbf{X})$ be the denominator. As a finite sum of $C^1$ functions, $W(\mathbf{X})$ is $C^1$.
    * Let $N(\mathbf{X}) = \sum_{j=1}^N \pi_j(\mathbf{X}) \cdot w_j(\mathbf{X})$ be the numerator. This is a sum of products of $C^1$ functions (the matrix-valued projection $\pi_j$ and the scalar-valued weight $w_j$). Therefore, $N(\mathbf{X})$ is also $C^1$.

4.  **Denominator is Bounded Away from Zero:**
    * Since $f$ is continuous on the compact set $V(n, k)$, it attains a global maximum, $f_{\max} = \sup_X f(X) < \infty$.
    * The denominator $W(\mathbf{X}) = \sum_{l=1}^N \exp(-\beta f(X^l))$ is bounded below by a strictly positive constant:
        $$W(\mathbf{X}) \ge \sum_{l=1}^N \exp(-\beta f_{\max}) = N \exp(-\beta f_{\max}) > 0$$

5.  **Differentiability of the Quotient:**
    * The consensus function $C(\mathbf{X}) = N(\mathbf{X}) / W(\mathbf{X})$ is the quotient of two $C^1$ functions where the denominator is non-zero.
    * By the quotient rule for differentiation, $C(\mathbf{X})$ is a $C^1$ function from the manifold $\mathcal{X}$ to the ambient space $\mathbb{R}^{n \times k}$.

6.  **Bounded Derivative Implies Lipschitz Continuity:**
    * Since $C(\mathbf{X})$ is $C^1$ and its domain $\mathcal{X}$ is compact, its derivative (the Jacobian $DC(\mathbf{X})$) is continuous on $\mathcal{X}$.
    * By the Extreme Value Theorem, the norm of the derivative is globally bounded. There must exist a finite constant $L_C$ such that:
        $$L_C = \sup_{\mathbf{X} \in \mathcal{X}} ||DC(\mathbf{X})|| < \infty$$
    * By the Mean Value Theorem (or integrating the derivative along a geodesic), this bounded derivative implies that $C(\mathbf{X})$ is Lipschitz continuous. For any $\mathbf{X}, \mathbf{Y} \in \mathcal{X}$:
        $$||C(\mathbf{X}) - C(\mathbf{Y})||_F \le L_C \cdot d_{\mathcal{X}}(\mathbf{X}, \mathbf{Y})$$
    where $d_{\mathcal{X}}$ is any valid product metric on the state space $\mathcal{X}$. This is consistent with the report's assertion that the Lipschitz continuity of $\overline{X}_t^*$ is established in [1].

**Q.E.D.**

**Lemma 1.3:** Let the ambient space be $\mathbb{R}^{n \times k}$ and the domain be the Stiefel manifold $V(n, k)$. The projection operator $P_X: \mathbb{R}^{n \times k} \to T_X V(n, k)$, defined for a fixed $Z \in \mathbb{R}^{n \times k}$ as a function of $X$,
$$P_X(Z) = Z - \frac{1}{2}(XZ^\top X + XX^\top Z)$$
is a smooth ($C^\infty$) function of $X$. Consequently, $P_X(Z)$ is globally Lipschitz continuous in $X$ on the compact manifold $V(n, k)$.

---

**Proof:**

1.  **Analyze the Function's Components:** We analyze the differentiability of the map $\Phi: V(n, k) \to \mathbb{R}^{n \times k}$ given by $\Phi(X) = P_X(Z)$ for a fixed matrix $Z \in \mathbb{R}^{n \times k}$.

2.  **Identify Basic Operations:** The function $\Phi(X)$ is constructed from the variable $X$ and constants ($Z$, $Z^\top$, $\frac{1}{2}$) using only matrix multiplication and addition.
    * The map $X \mapsto X$ is the identity map, which is $C^\infty$.
    * The map $X \mapsto X^\top$ is the transpose operation, which is linear and thus $C^\infty$.
    * Matrix multiplication $(A, B) \mapsto AB$ is a bilinear operation. Each entry of the resulting matrix is a polynomial in the entries of $A$ and $B$. It is a $C^\infty$ operation.

3.  **Analyze the Terms:**
    * **Term 1:** $Z$ is constant with respect to $X$.
    * **Term 2:** $T_2(X) = XZ^\top X$. This is a composition of $C^\infty$ maps (identity, transpose, and multiplication). Therefore, $T_2(X)$ is $C^\infty$.
    * **Term 3:** $T_3(X) = XX^\top Z$. This is also a composition of $C^\infty$ maps (identity, transpose, and multiplication). Therefore, $T_3(X)$ is $C^\infty$.

4.  **Analyze the Full Operator:** The operator $\Phi(X)$ is a linear combination of $C^\infty$ terms:
    $$\Phi(X) = Z - \frac{1}{2}T_2(X) - \frac{1}{2}T_3(X)$$
    A finite linear combination of $C^\infty$ functions is itself $C^\infty$. Therefore, $P_X(Z)$ is a smooth ($C^\infty$) function of $X$.

5.  **Smoothness Implies Lipschitz Continuity on Compact Sets:**
    * Any $C^\infty$ function is, by definition, $C^1$.
    * As established in Lemma 1.1, any $C^1$ function on the compact manifold $V(n, k)$ has a bounded derivative.
    * A bounded derivative implies global Lipschitz continuity on that domain.

**Q.E.D.**

**Lemma 1.4:** The coefficients of the Annealed CBO SDE system for the $i$-th particle:
$$dX_t^i = A_i(t, \mathbf{X}_t) dt + B_i(t, X_t^i) dW_t$$
where $\mathbf{X}_t = (X_t^1, \dots, X_t^N)$ and:
* **Drift:** $A_i(t, \mathbf{X}) = \lambda P_{X^i}(\overline{X}^*) - C_{n,k}\frac{\sigma(t)^2}{2}X^i$
* **Diffusion:** $B_i(t, X^i)[\cdot] = \sigma(t) P_{X^i}(\cdot)$

are (globally) Lipschitz continuous with respect to the state variables on their compact domains.

---

**Proof:**

1.  **Domain Compactness:** The state space for the full system is $\mathcal{X} = (V(n, k))^N$, which is compact (as established in Prop 1.2). The state space for a single particle is $V(n, k)$, which is also compact.

2.  **Continuity of the Drift Coefficient ($A_i$):**
    * Let $C(\mathbf{X}) = \overline{X}^*$ be the consensus point function. From **Lemma 1.2**, $C(\mathbf{X})$ is $C^1$ on $\mathcal{X}$.
    * Let $\Phi(Y, Z) = P_Y(Z)$ be the projection function. From **Lemma 1.3**, $\Phi(Y, Z)$ is $C^\infty$ in its arguments $Y \in V(n, k)$ and $Z \in \mathbb{R}^{n \times k}$.
    * Let $\pi_i: \mathcal{X} \to V(n, k)$ be the projection $\pi_i(\mathbf{X}) = X^i$. This map is $C^\infty$.
    * The first term of the drift is a composition of these $C^1$ functions:
        $$\lambda \cdot \Phi(\pi_i(\mathbf{X}), C(\mathbf{X}))$$
        A composition of $C^1$ functions is itself $C^1$.
    * The second term, $C_{n,k}\frac{\sigma(t)^2}{2}X^i$, is also $C^1$ in $\mathbf{X}$ (as $\pi_i$ is $C^\infty$).
    * The drift $A_i(t, \mathbf{X})$ is a linear combination of two $C^1$ functions. Therefore, $A_i(t, \mathbf{X})$ is a $C^1$ function on the compact manifold $\mathcal{X}$.

3.  **Continuity of the Diffusion Coefficient ($B_i$):**
    * The diffusion operator $B_i(t, X^i)[\cdot]$ is $\sigma(t) P_{X^i}(\cdot)$.
    * From **Lemma 1.3**, the map $X^i \mapsto P_{X^i}(\cdot)$ is $C^\infty$ on the compact domain $V(n, k)$.
    * The time-dependent scalar $\sigma(t)$ does not affect the differentiability with respect to $X^i$.
    * Therefore, $B_i(t, X^i)[\cdot]$ is a $C^\infty$ function on the compact manifold $V(n, k)$.

4.  **Conclusion ($C^1$ on Compact Set $\implies$ Lipschitz):**
    * As established in Lemma 1.1, any $C^1$ function on a compact domain has a bounded derivative (Jacobian) and is therefore **globally Lipschitz** continuous.
    * Since both $A_i$ (on $\mathcal{X}$) and $B_i$ (on $V(n, k)$) are $C^1$ on compact domains, they are both globally Lipschitz.
    * Global Lipschitz continuity is a stronger condition than, and thus implies, the **local Lipschitz continuity** required by standard SDE existence and uniqueness theorems.

**Q.E.D.**

**Lemma 1.5:** Let the $i$-th particle's dynamics be given by the SDE:
$$dX_t = A_t dt + \sigma(t) P_{X_t}(dW_t)$$
where $A_t = A_t^{\text{cons}} + A_t^{\text{corr}}$ is the total drift, with:
1.  $A_t^{\text{cons}} = \lambda P_{X_t}(\overline{X}_t^*)$ (the consensus drift, which is in the tangent space $T_{X_t} V(n, k)$).
2.  $A_t^{\text{corr}} = -C_{n,k}\frac{\sigma(t)^2}{2}X_t$ (the Itô correction drift).

The particle remains on the manifold $V(n, k)$ (i.e., $X_t^\top X_t = I_k$) *if and only if* the correction drift $A_t^{\text{corr}}$ is included with $C_{n,k} = \frac{2n-k-1}{2}$.

---

**Proof:**

1.  **Define the Constraint:** The particle $X_t$ remains on the manifold if the constraint $g(X_t) = X_t^\top X_t - I_k = 0$ holds for all $t \ge 0$. This is equivalent to requiring that its differential $d(g(X_t))$ is zero. Since $I_k$ is constant, this means we must prove $d(X_t^\top X_t) = 0$.

2.  **Apply Itô's Product Rule:** We apply the Itô product rule to the matrix-valued process $X_t^\top X_t$. For $X_t \in \mathbb{R}^{n \times k}$, the $(i, j)$-th entry is $\sum_l X_{li} X_{lj}$. The Itô rule gives:
    $$d(X_t^\top X_t) = (dX_t^\top) X_t + X_t^\top (dX_t) + (dX_t^\top)(dX_t)$$
    The last term, $(dX_t^\top)(dX_t)$, is the $k \times k$ matrix of quadratic variations.

3.  **Decompose the SDE:** Let $N_t = \sigma(t) P_{X_t}(dW_t)$ be the martingale (noise) part of the SDE. So, $dX_t = A_t dt + N_t$.

4.  **Substitute into the Product Rule:**
    $$d(X_t^\top X_t) = (A_t^\top dt + N_t^\top) X_t + X_t^\top (A_t dt + N_t) + (A_t dt + N_t)^\top (A_t dt + N_t)$$
    We collect terms by $dt$ and $dW$. The $dt \cdot dt$, $dt \cdot dW$, and $dW \cdot dt$ terms are zero in the Itô limit. The $N_t^\top N_t$ term becomes its quadratic variation, which is a $dt$ term.
    $$d(X_t^\top X_t) = (A_t^\top X_t + X_t^\top A_t) dt + (N_t^\top X_t + X_t^\top N_t) + (N_t^\top N_t)$$
    The process $(N_t^\top X_t + X_t^\top N_t)$ is the new martingale part. The term $(N_t^\top N_t)$ must be interpreted as its expected value, the quadratic variation, which we will call $Q(X) dt$.
    $$Q(X) dt = \mathbb{E}[(N_t^\top N_t)] = \mathbb{E}[(\sigma(t) P_{X_t}(dW_t))^\top (\sigma(t) P_{X_t}(dW_t))]$$

5.  **Isolate the Drift:** For $d(X_t^\top X_t) = 0$, the drift (all $dt$ terms) must sum to zero:
    $$(A_t^\top X_t + X_t^\top A_t + Q(X)) dt = 0$$

6.  **Analyze the Drift Component ($A_t^\top X_t + X_t^\top A_t$):**
    We split the total drift $A_t$ into its two components:
    * **Consensus Drift ($A_t^{\text{cons}}$):** By construction, $A_t^{\text{cons}} \in T_{X_t} V(n, k)$. The definition of the tangent space is the set of matrices $A$ such that $A^\top X_t + X_t^\top A = 0$. Therefore, $(A_t^{\text{cons}})^\top X_t + X_t^\top (A_t^{\text{cons}}) = 0$. This term vanishes.
    * **Correction Drift ($A_t^{\text{corr}}$):** We compute the contribution of this term:
        $$(A_t^{\text{corr}})^\top X_t + X_t^\top (A_t^{\text{corr}}) = \left(-C_{n,k}\frac{\sigma(t)^2}{2}X_t\right)^\top X_t + X_t^\top \left(-C_{n,k}\frac{\sigma(t)^2}{2}X_t\right)$$
        $$= -C_{n,k}\frac{\sigma(t)^2}{2} (X_t^\top X_t) - C_{n,k}\frac{\sigma(t)^2}{2} (X_t^\top X_t)$$
        Since $X_t$ is on the manifold, $X_t^\top X_t = I_k$.
        $$= -C_{n,k}\frac{\sigma(t)^2}{2} I_k - C_{n,k}\frac{\sigma(t)^2}{2} I_k = -C_{n,k} \sigma(t)^2 I_k$$

7.  **Analyze the Noise Component ($Q(X)$):**
    The drift of $d(X_t^\top X_t)$ is now:
    $$\left[ 0 - C_{n,k} \sigma(t)^2 I_k + Q(X) \right] dt$$
    The report's source [1] (as cited in the text) demonstrates the calculation of the quadratic variation $Q(X)$ for the projected Wiener process. This calculation shows:
    $$Q(X) = \left( \frac{2n - k - 1}{2} \right) \sigma(t)^2 I_k$$
    (This is a non-trivial result from stochastic calculus on manifolds).

8.  **Final Cancellation:**
    The report states that the constant in the SDE is $C_{n,k} = \frac{2n - k - 1}{2}$.
    Substituting our results into the drift equation (from step 5):
    $$-C_{n,k} \sigma(t)^2 I_k + Q(X) = 0$$
    $$-\left(\frac{2n - k - 1}{2}\right) \sigma(t)^2 I_k + \left(\frac{2n - k - 1}{2}\right) \sigma(t)^2 I_k = 0$$
    The cancellation is exact.

**Conclusion:**

The constraint $d(g(X_t)) = 0$ is satisfied *if and only if* the drift of $d(X_t^\top X_t)$ is zero. This drift consists of two parts: a "spurious drift" $Q(X)$ arising from the quadratic variation of the projected noise, and the term $(A_t^\top X_t + X_t^\top A_t)$ arising from the SDE's drift.
The specific correction term $A_t^{\text{corr}} = -C_{n,k}\frac{\sigma(t)^2}{2}X_t$ is constructed *precisely* to generate a term $-C_{n,k}\sigma(t)^2 I_k$ that perfectly cancels the spurious drift $Q(X)$. Without it, the drift would be $Q(X) \ne 0$, and the particle would deterministically drift off the manifold.

**Q.E.D.**

**Lemma 1.6:** The $N$-particle SDE system
$$dX_t^i = A_i(t, \mathbf{X}_t) dt + B_i(t, X_t^i) dW_t \quad \text{for } i=1, \dots, N$$
with initial condition $\mathbf{X}_0 \in \mathcal{X} = (V(n, k))^N$, admits a unique local strong solution.

---

**Proof:**

1.  **Recall Classical SDE Theory:** The standard existence and uniqueness theorem for a strong solution to an SDE system requires the drift and diffusion coefficients to satisfy two conditions:
    a) **Local Lipschitz Continuity:** The coefficients are locally Lipschitz continuous in the state variable(s).
    b) **Linear Growth Condition:** The coefficients satisfy $|A(t, \mathbf{X})| + |B(t, \mathbf{X})| \le K(1 + ||\mathbf{X}||)$ for some constant $K$, ensuring the solution does not "explode" too quickly.

2.  **Verify Condition (a):** In **Lemma 1.4**, we proved that the drift coefficient $A_i(t, \mathbf{X})$ and the diffusion coefficient $B_i(t, X^i)$ are $C^1$ on the compact domains $\mathcal{X}$ and $V(n, k)$, respectively. This implies they are **globally Lipschitz** continuous on these domains, which is a stronger condition than local Lipschitz continuity. Thus, condition (a) is satisfied.

3.  **Verify Condition (b):**
    * The state space $\mathcal{X} = (V(n, k))^N$ is compact. Therefore, the norm of any state $\mathbf{X} \in \mathcal{X}$ is bounded, i.e., $||\mathbf{X}|| \le C$ for some finite constant $C$.
    * As established in **Lemma 1.4**, the coefficients $A_i$ and $B_i$ are continuous functions on the compact domain $\mathcal{X}$. By the Extreme Value Theorem, they are also bounded.
    * Any bounded function trivially satisfies the linear growth condition. If $|A(\mathbf{X})| \le M_A$ and $|B(\mathbf{X})| \le M_B$ for all $\mathbf{X}$, then:
        $$|A(\mathbf{X})| + |B(\mathbf{X})| \le M_A + M_B \le (M_A + M_B)(1 + ||\mathbf{X}||)$$
        (This holds since $1 + ||\mathbf{X}|| \ge 1$).
    * Therefore, the linear growth condition (b) is also satisfied.

4.  **Conclusion:** Since both the local Lipschitz and linear growth conditions are satisfied, the classical existence and uniqueness theorem for strong solutions to SDEs applies. This guarantees the existence of a unique local strong solution $\mathbf{X}_t$ defined up to a stopping time $\tau_e$, the explosion time.

**Q.E.D.**

**Lemma 1.7:** The unique local strong solution $\mathbf{X}_t$ to the $N$-particle SDE system, guaranteed by Lemma 1.6, is a unique **global** strong solution (i.e., it does not explode in finite time, almost surely).

---

**Proof:**

1.  **Local Solution and Explosion Time:** From **Lemma 1.6**, we have a unique strong solution $\mathbf{X}_t$ defined on the time interval $[0, \tau_e)$, where $\tau_e$ is the (possibly finite) explosion time.

2.  **Invariance of the Manifold:** From **Lemma 1.5**, we proved that the Itô correction term ensures that if a particle $X_t^i$ starts on the manifold $V(n, k)$, it remains on the manifold for all $t \in [0, \tau_e)$.
    * Given an initial condition $\mathbf{X}_0 \in \mathcal{X} = (V(n, k))^N$, the solution $\mathbf{X}_t$ remains in $\mathcal{X}$ for its entire lifespan $[0, \tau_e)$.

3.  **Compactness of the State Space:** The Stiefel manifold $V(n, k) = \{X \in \mathbb{R}^{n \times k} | X^\top X = I_k\}$ is a closed and bounded subset of $\mathbb{R}^{n \times k}$. By the Heine-Borel theorem, $V(n, k)$ is compact.
    * The full $N$-particle state space $\mathcal{X} = (V(n, k))^N$ is a finite product of compact sets. By Tychonoff's theorem, $\mathcal{X}$ is also a compact set.

4.  **Non-Explosion on Compact Sets:** An explosion in finite time means that $\lim_{t \to \tau_e} ||\mathbf{X}_t|| = \infty$.
    * However, as established in step 2, the solution $\mathbf{X}_t$ is confined to the compact set $\mathcal{X}$ for all $t \in [0, \tau_e)$.
    * By definition, any set of points within a compact set is bounded. Therefore, $||\mathbf{X}_t|| \le M$ for some finite constant $M$ for all $t < \tau_e$.
    * This creates a contradiction: the norm of the solution is bounded, so it cannot diverge to infinity.

5.  **Conclusion:** The only way to resolve this contradiction is if the explosion time is not finite. Therefore, $\tau_e = \infty$ almost surely. The local solution (from Prop 1.6) can be extended to all $t \ge 0$, making it a unique global strong solution.

**Q.E.D.**