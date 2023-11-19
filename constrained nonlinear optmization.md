# Constrained Non-Linear Optimization
Below is the standard formulation of a constrained optimzation problem $\mathcal{P}$
$$
\begin{align*}
\min \quad f(x) \\
\text{s.t}\quad g_i(x)&\leq 0 \quad \forall i = 1,2,...m \\
h_i(x) &= 0 \quad \forall i = 1,2,...,l
\end{align*}
$$

If $f(x)$ and $g_i(x)$ are convex, and $h_i(x)$ are linear, then $\mathcal{P}$ is a convex optimization problem. Linear optimization is a special case of convex optimization, where $f(x)$ and $g_i(x)$ are linear.

In general, we believe that an unconstrained optimization problem is much easier to solve than than constrained (see Unconstrained Non-Linear Optimization). To transform a constrained problem to unconstrained, the most direct way is to penalize the breach of constraints with infinite cost.

Consider the example:
$$
\begin{align*}
\min \quad x_1+x_2 \\
\text{s.t}\quad x_1^2+x_2^2-1\leq 0
\end{align*}
$$

We can transform it to an equivalent problem using a characteristic function
$$
\min \quad x_1+x_2+\chi(x)
$$
$$\chi(x) = \begin{cases}
    \infty \quad \text{if } x_1+x_2-1 > 0\\
    0 \quad \text{if } x_1+x_2-1\leq 0
\end{cases}
$$

The problem with this transformation is that $\chi$ is discontinuous, while our algorithms to solve unconstrained problems such as gradient descent and Newton's method rely on differentiating the objective function. Therefore, we need to come up with ways to approximate the function $\chi$: 1) a linear approximation, which leads to the Lagrangian 2) a non-linear approximation, which leads to the barrier function.

## Lagrangian Duality

### Walk through the formulations
One particular way to approximate the characteristic function $\chi$ is to use a linear function with slope $\lambda$ for each inequality constraint and $v$ for each equality constraint, this is known as the *Lagrangian function*:
$$\mathcal{L}(x, \lambda, v) = f(x) + \sum_{i=1}^m\lambda_i g_i(x) + \sum_{j=1}^l v_j h_j(x)$$

This is obviously not a great approximation since the slope of the original penalty function is infinite, but the slopes of the linear approximation is constant. However, if we take the maximum of the linear penalty over all possible slopes, we can recover the original penalty function, using the same example:
$$\max_{\lambda\geq 0} \quad \lambda(x_1+x_2-1) = \begin{cases}
     \infty \quad \text{if } x_1+x_2-1 > 0\\
    0 \quad \text{if } x_1+x_2-1\leq 0
\end{cases}$$

This is true because when $x_1+x_2-1>0$, the maximum is taken by letting $\lambda = \infty$, and when $x_1+x_2-1\leq0$, the maximum is taken by letting $\lambda = 0$!

Therefore, we can now transform the original problem to the following:
$$\min_{x\in \mathbb{R}^n} \quad f(x) + \max_{\lambda_i\geq 0, v_j\in \mathbb{R}} \quad \sum_{i=1}^m\lambda_i g_i(x) + \sum_{j=1}^l v_j h_j(x)$$

Notice that we have to take $\lambda_i\geq 0$ for the inequality constraints because $g_i(x)\leq 0$ can only be violated on one side (by having $g_i(x) > 0$) and the cost of violating a constraint must be non-negative. On the other hand, we can have $v$ to be positive or negative since the equality constraint can be violated in either direction.

The maximization operation could be brought to the front since $f(x)$ does not involve $\lambda, v$, that is,
$$\min_{x\in \mathbb{R}^n} \quad \{\max_{\lambda_i\geq 0, v_j\in \mathbb{R}} \quad f(x) + \sum_{i=1}^m\lambda_i g_i(x) + \sum_{j=1}^l v_j h_j(x)\} \\
= \min_{x\in \mathbb{R}^n} \quad \{\max_{\lambda_i\geq 0, v_j\in \mathbb{R}} \quad \mathcal{L}(x, \lambda, v)\}$$

Notice that so far, we have not changed anything about the original problem but merely reformulated it. The above formulation is still the *primal problem*. The optimal solution of the primal problem is denoted as $p^*$. Here comes the trick: when we switch the order of the min and the max, we obtain the following *dual problem*:
$$\max_{\lambda_i\geq 0, v_j\in \mathbb{R}} \quad \{\min_{x\in \mathbb{R}^n} \quad \mathcal{L}(x, \lambda, v)\}$$

This is called the *Lagrange dual problem* with $\min_{x\in \mathbb{R}^n} \mathcal{L}(x, \lambda, v)$ being the *Lagrange dual function*. The optimal value of the dual problem is denoted as $d^*$.

### Interpretations & Weak Duality
Known as "sensitivity analysis", the multipliers $\lambda_i$ and $v_i$ are sometimes called "dual prices" as they can represent the cost of violating each constraint.

The optimal value of the dual problem provides an lower bound to the optimal value of the primal problem. This is because when we are allowed to break the constraints, we can possibily achieve a lower value in the minimization, and the worst case is we don't break any constraint then the value is the same as the primal. This is known as the *weak duality*:
$$d^*\leq p^*$$

### Strong Duality & KKT Conditions
When strong duality holds, we have 
$$d^*=p^*$$
which would allow us to solve whichever is easier to solve (the primal and the dual).

The *KKT conditions* specify a set of requirements for $x^*$ and $\lambda^*$ such that strong duality holds:
 - primal feasibility: $g_i(x)\leq 0, \forall i = 1,...,m; h_i(x)=0 \forall i = 1,...,l$
 - dual feasibility: $\lambda_i \geq 0, \forall i = 1,...,m$
 - complementary slackness: $\lambda_i g_i(x) = 0, \forall i = 1,....,m$
 - vanishing gradient: $\nabla f(x)+\sum_{i=1}^m\lambda_i\nabla g_i(x) + \sum_{i=1}^l v_i\nabla h_i(x) = 0$

The KKT conditions are useful because it reduces solving an optimization problem to solveing a bunch of equations and inequalities.

For constrained optimization, we usually have to distinguish between two cases: $\lambda^* = 0$ and $\lambda^* > 0$. 
- $\lambda^* = 0$ corresponds to the case when we have the shallow price being $0$, meaning that the optimal solution is within the constraint. So we can just find the global optimal to the objective function directly (essentially ignoring the constraint).
- $\lambda^* > 0$ corresponds to a positive shallow price. In this case, we can solve the system of complementary slackness condition (since we now know that $g_i(x) = 0$ for constraints that have a positive shallow price) for the optimal $x^*$.

## Interior Point Method
Interior point methods are designed to solve the KKT conditions more systematically and much faster. In particular, it solves perturbed KKT conditions with a parameter $t$. Let's denote the solution to the original KKT conditions as $x^*$, and the solution to the perturbed KKT conditions as $x(t)$, then the perturbed KKT conditions have exactly the same formulation as the original, except for the complementary slackness condition, we write:
$$ \lambda_i g_i(x) = -t$$
It's clear that $x(t)$ goes to $x^*$ when $t$ goes to $0$. This actually makes the KKT conditions much easier to solve, since now we can write:
$$\lambda_i = \frac{-t}{g_i(x)}$$
When we plug this into the vanishing gradient condition, it becomes:
$$\nabla f(x) + \sum_{i=1}^m\frac{-t}{g_i(x)}\nabla g_i(x) = \nabla(f(x)-t\sum_{i=1}^m\log(-g_i(x)))$$

At this point, we have reduced the problem to 
$$\min_{x\in\mathbb{R}^n} \quad f(x) - t\sum_{i=1}^m\log(-g_i(x))$$
which is a unconstrained optimization that can be solved by Newton's Method. This is sometimes called the barrier problem. Even if we have equality constraints, we can conviniently add them to this problem and still solve it much easier. Notice that all KKT conditions are satisfied by the new optimization problem, except for the complementary slackness, which is approximated by $t=0$.

### The Barrier algorithm
The barrier problem with linear constraints can be written as:
$$\min_{x\in\mathbb{R}^n}\quad f(x)-t\sum_{i=1}^n\log(-g_i(x)) \\ \text{s.t } Ax = b$$

The intuition is that if we start at $t$ very large, then the log term dominates, and the minimization problem becomes minimizing $g_i(x)$ such that $Ax = b$, the solution to this problem $\bar{x}$ is called the *analytic center* of the feasible region. So when $t$ is large, our initial point of $\bar{x}$ is usually close to the analytic center.

We have a solution $x(t)$ to the barrier problem for every fixed $t$. We can try to solve the next problem using a slightly smaller $t$, such as $t' = \rho t$ where $0<\rho<1$. When we solve the next barrier problem for $x(t')$, we initialize Newton's method at $x(t)$, and since we know that $x(t)$ is close to $x(t')$, we know Newton's Method would converge really fast. This iterative process is described as the barrier algorithm.

- Intialization. We start at the interior point $x^S\in \mathbb{R}^n$, set iteration counter $k = 0$ and tolerance $\epsilon>0$, $t_0>0, 0<\rho<1$.
- Optimization. Update iteration counter from $k$ to $k+1$, solve the barrier problem using Newton's method for linearly constrained optimization starting in $x^S$, store the optimal solution in $x(t_k)$.
- Termination. If $mt_k\leq \epsilon$, stop, otherwise proceed.
- Iteration. Update $t_{k+1}\leftarrow t_k$ and $x^S\leftarrow x(t_k)$, and return to step 2.

From this process, we obtain a sequence of optimal solutions $x(t_0), x_(t_1),...$, as $0 < t_{k+1} < t_k$ and $t_k\to 0$, the *central path* is defined as the sequence of $x$:
$$x(t_k)\in \argmin \{f(x) - t_k\sum_{i=1}^m\log(-g_i(x))\}$$

Geometrically, the central path moves from the analytic center of the feasible region to the boundary (theoretically it's never on the boundary since $t_k$ is never exactly $0$).