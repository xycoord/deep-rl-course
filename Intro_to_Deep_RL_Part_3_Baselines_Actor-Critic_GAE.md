# Intro to Deep RL - Baselines, Actor-Critic and GAE

**Author: Logan Thomson**

[![GitHub](https://img.shields.io/badge/GitHub-Connect-181717?logo=github&logoColor=white)](https://github.com/xycoord)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/logan-thomson-01a4942ab)

[![GitHub](https://img.shields.io/badge/Star_Repository-gold?logo=github&logoColor=white)](https://github.com/xycoord/deep-rl-course)
[![GitHub Issues](https://img.shields.io/badge/Submit_Issue-red?logo=github)](https://github.com/xycoord/deep-rl-course/issues)

A central challenge in deep RL is the high variance in gradient estimates, leading to unstable training and poor sample efficiency. This blog post explores how baselines, actor-critic methods, and Generalised Advantage Estimation (GAE) tackle this problem.

A surprising result underpins these methods: we can subtract arbitrary baselines from returns without biasing gradient estimates—yet this modification may dramatically reduce variance. We'll progress from simple constant baselines through to state-dependent baselines (actor-critic), culminating in GAE, which allows precise control of the bias-variance trade-off. Along the way, we'll examine the effects and guarantees of each method with respect to bias and variance.

These techniques form the foundation of modern algorithms like PPO and TRPO. We'll also explore how GRPO adapts these concepts for LLM post-training, where environment structure enables more efficient training without a critic network.

This is the third part of my [Deep RL course](https://github.com/xycoord/deep-rl-course), assuming parts [1](https://colab.research.google.com/drive/1Lm_TI-Vrzai-WZQeZL3o7US07vVKWXlQ) and [2](https://colab.research.google.com/drive/1UULTQYnymQOpa7nuaw6mDXnvWRV9R_2y) as prerequisites (policy gradients, reward-to-go, and discounting). Beginners should start there, though readers already familiar with these foundations will find this post self-contained.

# Motivation 

Recall the simple sum-of-rewards surrogate objective from Part 1:

$$
L_{\text{surrogate}}(\theta) = \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T-1} \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot \left(\sum_{k=0}^{T-1} r_{i,k}\right)
$$

When we maximise this objective, we're adjusting the policy's action probabilities based on the rewards received. For each state-action pair in a trajectory:
- If the total reward for that trajectory is positive, the probability of taking action $a_{i,t}$ in state $s_{i,t}$ increases
- If the total reward is negative, this probability decreases
- The magnitude of this probability change is proportional to the total reward

Consider the case when all the rewards are positive: the policy optimisation will try to increase the probabilities of *all* actions, including both good and suboptimal ones. To understand why the policy nevertheless improves, let's examine the expected gradient over all possible trajectories $(N\rightarrow\infty)$.

Gradients arising from high-reward trajectories push the policy parameters $\theta$ towards increasing the probabilities of actions taken in those trajectories. Since probabilities for all actions available in a state must sum to one, this increase in probability for the chosen action necessarily leads to a decrease in the probabilities of the alternative actions in that same state. Even though gradients from lower-reward trajectories provide a push for their actions, this is weaker because the pushes are scaled by the trajectory's total reward.

When these effects are averaged in expectation, the stronger updates from high-reward trajectories dominate. The net effect is that the policy shifts to increase the probability of actions leading to higher-than-average rewards and decrease the probability of actions leading to lower-than-average rewards.

In practice, we sample only a small number of trajectories (small $N$). With limited samples, this expectation property breaks down, leading to high variance in gradient estimates and unstable training. The core idea we will discuss in this blog post is how to ensure that good actions have positive gradient weights and bad actions have negative gradient weights - decreasing variance and stabilising training.

> A **baseline** is the threshold for judging whether an action is deemed good or bad according to its return (either sum-of-rewards or reward-to-go). Typically, this is chosen to be some sort of average return over different actions/trajectories such that actions which perform above average are *good* and those which perform below average are *bad*.

By subtracting the baseline from the returns, we get a gradient weighting with the desired property.

# Constant Baseline

Let's first consider the simplest, but very common, baseline: a constant value applied to the sum-of-rewards return. Intuitively, we use the expected (average) sum-of-rewards over all trajectories:

$$ b = \mathbb E_{\tau\sim p_\theta}[r(\tau)]$$

Where $\tau$ is a trajectory and $r(\tau) = \sum_{t=0}^{T-1}r_t$

Although in practice we approximate this with the sampled trajectories $\tau_i$:

$$ b \approx \frac{1}{N}\sum_{i=1}^N r(\tau_i) $$

This gives the surrogate gradient:

$$ \nabla_\theta J_b(\theta) \approx \frac{1}{N}\sum_{i=1}^N \nabla_\theta\log p_\theta(\tau_i)[r(\tau_i)-b] $$

This is legitimate because, for any constant $b$ (including the baseline we just defined), the new surrogate gradient is unbiased in expectation.

Let's write out the idealised expectation of the gradient:

$$
\begin{align*}
\nabla_\theta J_b(\theta)
&= \mathbb E_{\tau\sim p_\theta}\big[\nabla_\theta\log p_\theta(\tau)(r(\tau)-b)\big]
\\&= \mathbb E_{\tau\sim p_\theta}\big[\nabla_\theta\log p_\theta(\tau)r(\tau)\big] - \mathbb E_{\tau\sim p_\theta}\big[\nabla_\theta\log p_\theta(\tau)b\big]
\end{align*}
$$

The first term is just $\nabla_\theta J(\theta)$. Hence, subtracting the baseline is unbiased if:

$$\mathbb E_{\tau\sim p_\theta} [\nabla_\theta\log p_\theta(\tau)b]=0$$

So let's analyse this term on its own.

Expand the expectation to write it as an integral:

$$\mathbb E_{\tau\sim p_\theta} [\nabla_\theta\log p_\theta(\tau)b] = \int p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)b \, \mathrm{d}\tau$$

Recall the convenient identity from part 1:

$$p_\theta(\tau)\nabla_\theta\log p_\theta(\tau) = \nabla_\theta p_\theta(\tau)$$

Substituting this in gives:

$$\int p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)b \, \mathrm{d}\tau = \int \nabla_\theta p_\theta(\tau) b \, \mathrm{d}\tau$$

Take $b$ and the gradient outside the integral:

$$= b\nabla_\theta\int  p_\theta(\tau) \, \mathrm{d}\tau$$

The probability distribution $p_\theta$ integrates to $1$.

$$= b\nabla_\theta1=b\cdot0=0$$

Therefore, using a constant baseline is unbiased.

## Does the average reward baseline reduce variance?

Our motivation for using baselines is to reduce the variance of the gradient estimate. Therefore, we should consider if and when this occurs.

Recall from probability theory that the variance of a random variable $x$ is defined as:

$$\mathrm{Var}(x) = \mathbb E[x^2]-\mathbb E[x]^2 $$

For notational conciseness we'll use the shorthand:

$$ g(\tau) = \nabla_\theta\log p_\theta(\tau)$$

Therefore, the variance of the gradient estimate with baseline $b$ is:

$$\mathrm{Var}(g(\tau)(r(\tau)-b)) = \mathbb E[(g(\tau)(r(\tau)-b))^2]-\mathbb E[g(\tau)(r(\tau)-b)]^2$$

In the previous section, we proved that the second term:

$$\mathbb E[g(\tau)(r(\tau)-b)]^2 = \mathbb E[g(\tau)r(\tau)]^2$$

And multiplying out the first term we get:

$$ \begin{align*}
\mathbb E[g(\tau)^2(r(\tau)-b)^2]
&= \mathbb E[g(\tau)^2r(\tau)^2 -2g(\tau)^2r(\tau)b+g(\tau)^2b^2]\\
&= \mathbb E[(g(\tau)r(\tau))^2] + \mathbb E[g(\tau)^2(b^2 -2br(\tau))]
\end{align*}$$

Putting these back together gives:

$$
\mathrm{Var}(g(\tau)(r(\tau)-b))
= \mathbb E[(g(\tau)r(\tau))^2] - \mathbb E[g(\tau)r(\tau)]^2 + \mathbb E[g(\tau)^2(b^2 -2br(\tau))]
$$

Notice that the first two terms make up the variance for the gradient estimate without the baseline. Hence,

$$
\mathrm{Var}(g(\tau)(r(\tau)-b))
= \mathrm{Var}(g(\tau)r(\tau)) + \mathbb E[g(\tau)^2(b^2 -2br(\tau))]
$$

Since both variances are positive, the variance with the baseline is smaller if:

$$ \mathbb E[g(\tau)^2(b^2 -2br(\tau))] < 0 $$

Unfortunately, this inequality doesn't hold in general for $b \approx \mathbb E[r(\tau)]$ so we're not guaranteed to reduce variance. For instance, if $g(\tau)$ is large when $r(\tau)$ is small, positive terms would dominate the expectation causing the inequality to fail. With further analysis we can show that the optimal constant baseline $b^* = \frac{\mathbb E[g(\tau)^2r(\tau)]}{\mathbb E[g(\tau)^2]}$, which has at least as low variance as no baseline. (The derivation is a good mathematical exercise).

However, in practice, using the average reward baseline $(b \approx \mathbb E[r(\tau)])$ does often reduce variance and is simple to implement. For large batch sizes, this average can be taken across just the batch. However, for small batch-sizes, this baseline would be highly variable across batches so a moving average (e.g. EMA) can be used to stabilise this.

To build intuition for this empirical effectiveness, let's analyse the impact of subtracting the average reward baseline, $b \approx \mathbb{E}[r(\tau)]$, in the case when all trajectory returns $r(\tau)$ are positive. Consider the distinct problems arising from trajectories with different return levels, and how subtracting the baseline addresses them:

1.  **Below Average Returns $(0 < r(\tau) < b)$:**  
    *Problem (Sample Noise / High Variance):*  
    These trajectories contribute a positive term $g(\tau)r(\tau)$ to the gradient estimate. In a *finite sample* $\frac{1}{N}\sum g(\tau_i)r(\tau_i)$, the positive contributions from sampled low-return trajectories may not be sufficiently counteracted by high-return samples within that batch. This discrepancy makes the gradient estimate noisy; these suboptimal trajectories introduce high variance by pushing the estimate away from the true expected gradient direction.  
        
    *Solution:*  
    Subtracting the baseline $b$ results in a negative weight. This creates a direct negative gradient contribution $g(\tau)(r(\tau)-b)$ which not only removes the noise but correctly provides signal to decrease the probability of these trajectories. This corrective signal is also proportional to how much the return fell below average.

2.  **Above Average Returns $(r(\tau) \ge b)$:**  
    *Problem (Sample Noise):*  
    Trajectories yielding average returns provide little information about policy improvement direction; ideally, their net effect on action probabilities should be minimal. However, without a baseline, each average sample $\tau_i$ contributes a potentially large positive term $g(\tau_i)r(\tau_i)$ to the gradient estimate. This individual contribution often significantly differs from the desired near-zero net effect, acting as noise that must be cancelled out by other samples in the batch, thus increasing the variance of the estimate. Similarly, even though above average trajectories yield correctly positive gradients, their contributions are relatively disproportionate, with trajectories closer to the average having their probabilities disproportionately increased. This excess push in these cases can be considered a continuation of the noise in the average case.  
        
    *Solution:*  
    Subtracting the baseline $b$  reduces the offset noise present in the raw return $r(\tau)$, resulting in a gradient contribution $g(\tau)(r(\tau)-b)$ that more accurately reflects relative performance. This noise reduction manifests across the range:
      * For returns near average ($r(\tau) \approx b$), where the advantage is near zero, this removes noise by silencing non-informative updates.
      * For returns significantly above average $(r(\tau) > b)$, this removes noise by ensuring the positive reinforcement signal is proportionally accurate to the actual advantage gained, rather than being distorted by the baseline offset $b$.

3.  **Highest Returns $(r(\tau) \gg b)$:**  
    *Problem (High Variance & Instability):*  
    While the raw return $r(\tau)$ provides the correct signal direction for these very successful trajectories, its large magnitude in the gradient term $g(\tau)r(\tau)$ can dominate the gradient estimate, cause high variance, and lead to training instability.  
    
    *Solution:*  
    Subtracting the baseline $b$ dampens the magnitude of these gradient terms compared to the raw return $(0 < r(\tau)-b < r(\tau))$. This reduces the influence of these extreme returns, lowering variance and promoting more stable updates.

By addressing these issues across the return spectrum, the baseline improves the overall quality and stability of the gradient estimate. In addition, dampening the large updates from the highest-return trajectories often makes training significantly more stable, which may allow for the use of a higher learning rate, potentially leading to faster convergence.

## Constant Baseline Implementation

We can implement the batch mean baseline simply as:
```python
def apply_baseline(returns):
   baseline = np.mean(returns)
   return returns - baseline
```

For the EMA version, the baseline is stateful so the example is written as part of an agent class:

```python
class Agent:
    def __init__(..., decay: float,...):
        self.decay = decay
        self.baseline = 0
        ...

    def apply_baseline_EMA(self, returns: np.ndarray) -> np.ndarray:
        self.baseline = self.decay * self.baseline + 
                        (1 - self.decay) * np.mean(returns)
        return returns - self.baseline
    
    ...
```

## Limitations of Constant Baselines

So far we've considered the average sum-of-rewards baseline with the sum-of-rewards accumulation. Can we do the same with reward-to-go? If we use the average sum-of-rewards baseline, we run into the problem that the values of reward-to-go differ greatly with timestep---typically they get smaller as the timestep increases. This might lead us to consider a timestep-dependent baseline which uses the average reward-to-go for the given timestep across trajectories. While better, this is far from optimal.

# State-dependent baselines

The key is to make the baseline dependent on the state ($b(s_t)$). It turns out that such a baseline is also unbiased.

Since $\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\|s_t)$, we can consider each timestep separately. At timestep $t$, the baseline is unbiased if:

$$\mathbb E_{\tau\sim p_\theta} [\nabla_\theta\log \pi_\theta(a_t|s_t)b(s_t)]=0$$

Using the law of total expectation, we can condition first on the state $s_t$:

$$=\mathbb E_{s_t}\big[\mathbb E_{a_t\sim\pi_\theta(\cdot|s_t)} [\nabla_\theta\log \pi_\theta(a_t|s_t)b(s_t)|s_t]\big]$$

Now the term $b(s_t)$ is constant with respect to the inner expectation so we can bring it out.

$$=\mathbb E_{s_t}\big[b(s_t)\mathbb E_{a_t\sim\pi_\theta(\cdot|s_t)} [\nabla_\theta\log \pi_\theta(a_t|s_t)]\big]$$

Following the same logic as before, this inner expectation is 0, thus state dependent baselines are unbiased.

# Q, V and A

When using sum-of-rewards, we're looking at the whole trajectory to give the measure of good and bad. Is the action part of a sequence of actions that were better than average? However, we really want to know if each policy decision was good - was it a good action to take given the state? If we can answer that question perfectly, then every summand will contribute high signal and low noise to the gradient estimate.

We can start to formalise this as:

> How much better was the selected action $a$ than the other actions available in that state?

To express this mathematically, we can decompose it into two functions: the *Q-function* ($Q^\pi$) and the *Value-function* ($V^\pi$), each specific to the policy $\pi$. Using $G_t$ to notate the reward-to-go:

$$
\begin{align*}
Q^\pi(s, a) &= \mathbb{E}_{\tau \sim \pi}[G_t \mid S_t = s, A_t = a]
\\
V^\pi(s) &= \mathbb{E}_{\tau \sim \pi}[G_t \mid S_t = s]
\\&= \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^\pi(s,a)]
\end{align*}
$$

Putting them together, we get the  *Advantage function* ($A^\pi$), which is the advantage of choosing $a$ over the other actions given we're in state $s$.

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

This function is positive for actions better than average ($Q^\pi(s, a) > V^\pi(s)$) and negative for those worse than average, giving it the desired property.

Unfortunately, computing $Q^\pi$ and $V^\pi$ exactly is intractable for most environments (requiring enumeration of all states/actions) and impossible in real-world environments with continuous state spaces that can only be sampled. However, we can estimate them!

# Estimating Advantages

Estimates of $Q^\pi$, $V^\pi$ and $A^\pi$ are notated using a hat.

The simplest estimate for $Q^\pi$ is a single sample estimate: the discounted reward-to-go. Here the discount is being used for variance reduction (see part 2).

$$Q^\pi(s_t,a_t)\approx G_t = \sum_{l=0}^{T-t}\gamma^l r_{t+l}$$

Since most real-world environments cannot be rolled back to the same state, we cannot collect multiple samples to improve this estimate. We'll discuss later how techniques like GAE can reduce variance regardless, but for now we'll stick with $G_t$.

This same limitation creates a problem for estimating $V^\pi$ directly with sampling. If we only have a single sample from state $s_t$, and we use that sample's return $G_t$ to estimate both $Q^\pi(s_t,a_t)$ and $V^\pi(s_t)$, the estimated advantage would be zero in all cases!

Fortunately, in continuous state spaces, the $V^\pi$ is typically continuous and different sampled trajectories may visit nearby states. This allows us to *learn* an estimate using all the collected data, not just the single sample from the current trajectory.

## Actor-Critic

The model trained to estimate the value function is called the *Critic*, notated $\hat V^\pi$. It is trained simultaneously with the actor using the same samples. It typically uses a similar architecture to the actor since both take observations as input. However, it predicts a scalar rather than a probability distribution.

For each batch, the critic is typically trained before the actor in order to maximise its performance when used. We use the MSE loss:

```python
def update(observations: Tensor, discounted_rtg: Tensor ...):
    ...
    values = critic(observations)
    critic_loss = F.mse_loss(values, discounted_rtg)
    ...
```

The particular training algorithm varies across implementations which make different design decisions such as how many gradient steps to take per batch.

We compute advantages with new values from the updated critic:

```python
def compute_advantages(observations: Tensor, discounted_rtg: Tensor) -> Tensor:
    values = critic(observations).detach() # Don't backprop through critic here
    advantages = discounted_rtg - values
    return advantages
```

# Aside: Group-dependent baselines

For the class of environments in which we can roll out multiple trajectories from the same initial state, but where there are multiple possible initial states, a third type of baseline, called a *Group-dependent baseline*, is particularly useful. Trajectories with the same initial state form a *group*, and the baseline is calculated as the average reward across all trajectories in each group.

A major limitation of this is that because the baseline is constant across the whole trajectory, it may not effectively reduce variance when different timesteps have different returns (reward-to-go). However, when reward is given only at the end of each trajectory, this is not a problem. Moreover, since such a reward signal creates high-variance returns, training a critic to accurately estimate values at earlier timesteps becomes particularly challenging. In these cases, a state-dependent baseline (such as in actor-critic) provides little benefit whilst requiring the overhead of training a second network. Therefore, a group baseline is preferred.

The best example of such environments are those used in LLM post-training. With these, the initial state is a prompt. At each step of generation, the state consists of all tokens produced so far. Thus a trajectory is just a completion. A group is a set of completions of the same prompt, and the reward is given per completion, either as a verifiable reward (e.g. for a maths or coding problem) or as human feedback when the reward is a matter of taste.

The group-dependent baseline is the key innovation of the GRPO (Group Relative Policy Optimization) algorithm introduced by DeepSeek for this exact scenario ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)). Notably, previous attempts to use RL for post-training LLMs used critics with the same architecture as the actor (the LLM). So removing the critic halved memory and training compute requirements!

# Value Functions Beyond Baselines

It turns out that having a good estimate of the value-function is useful beyond providing a state-dependent baseline.

## Correcting raw reward-to-go

In infinite time-horizon environments we have to truncate the trajectories to make sample collection tractable. This means that we are missing the potential rewards from the timesteps after truncation. This incentivises the agent to take actions which maximise reward within the time-limit of training, no matter how that affects later timesteps in unconstrained deployment.

Moreover, the truncation can lead to problems with the critic. Suppose that the agent visits similar states at multiple points in the same trajectory. This is common in locomotion environments such as Walker and HalfCheetah. The reward-to-go is notably different in these similar states since one has had more of its future rewards truncated. As such, the critic has two options:

1. It treats these similar states as quite different, assigning different values to them. This means it fails to learn the true value function and fails to generalise properly, making it a poor baseline for variance reduction.
2. It compromises between the states which creates an unbiased but sub-optimal baseline (from a variance-reduction perspective).

In both cases, the advantages are biased due to the bias in the reward-to-go caused by the truncation.

We can fix this by using the value-function to estimate the rewards from future timesteps. Let $T$ be the episode length limit; therefore $s_{T}$ is the final state of the episode. For a theoretical infinite episode, the reward-to-go would be:

$$
G_{t} = \sum_{l=0}^\infty \gamma^{l}r_{t+l}
= \sum_{l=0}^{T-t-1} \gamma^{l}r_{t+l} + \gamma^{T - t}\sum_{l=0}^\infty \gamma^l r_{T+l}
$$

Since

$$
\hat V^\pi(s_T) \approx \sum_{l=0}^\infty \gamma^l r_{T+l}
$$

Then

$$
G_{t} \approx \sum_{l=0}^{T-t-1} \gamma^{l}r_{t+l} + \gamma^{T - t}\hat V^\pi(s_T)
$$

Which can be written recursively as:

$$\begin{align*}
G_{T} &= \hat V^\pi(s_T)
\\
G_{t} &= r_t + \gamma G_{t+1}
\end{align*}
$$

This is the formula typically used for efficient computation:

```python
def compute_corrected_returns(rewards: Tensor, 
                              gamma: float,
                              final_observation: Tensor,
                              terminal: bool) -> Tensor:
    T = len(rewards)
    returns = torch.zeros(T + 1) # Extra slot for bootstrap value

    # Bootstrap 
    returns[T] = 0.0 if terminal else critic(final_observation).detach()

    # Work backwards from last timestep
    for t in reversed(range(T)):
        returns[t] = rewards[t] + gamma * returns[t+1]

    return returns[:T]  # Return only the T actual timesteps
```

Note that these new returns are used to train the value-function estimate.

## Reducing variance in the Q-value estimation

Interestingly, this same technique can also be applied more generally to reduce variance.

As previously noted, using the reward-to-go directly as an estimate for the Q-value has high variance. This is because the reward-to-go is a Monte Carlo estimate from a single trajectory. Later timesteps contribute more variance to the reward-to-go because there's more uncertainty in which states and rewards will be encountered.

We've just seen how we can use the value-function estimate to synthesise the missing rewards from truncation. Since this estimate is low-variance---it was trained on many trajectories and generalises across similar states---we can use it to replace high variance sampled rewards to reduce the variance of the Q-value estimate.

An extreme version of this is 1-step bootstrapping, where we use just the current reward and estimate the rest:

$$\hat Q^\pi_1(s_{t}, a_{t}) = r_{t} + \gamma \hat V^\pi(s_{t+1})$$


Or more generally, we can switch after $n$ timesteps:

$$\hat Q^\pi_n(s_t, a_t) = \sum_{l=0}^{n-1} \gamma^{l}r_{t+l} + \gamma^{n}\hat V^\pi(s_{t+n})$$

This is $n$-step bootstrapping. Unlike truncation correction, every timestep consistently uses $n$ sampled rewards before bootstrapping with the value function.

The fewer sampled rewards used, the lower the variance of the estimate. However, this, of course, comes at the cost of increasing bias from the inaccuracy of $\hat V^\pi$. A classic bias-variance trade-off.

# Generalised Advantage Estimation

Generalised Advantage Estimation is a technique developed to address this bias-variance trade-off by mixing different $\hat Q^\pi_n$ for different values of $n$. This is done by taking a weighted average, weighted exponentially with the hyper-parameter $\lambda$:

$$
\hat Q^\pi_{\text{GAE}_{n\text{-step}}}(s_t,a_t) = \frac{1-\lambda}{1-\lambda^{T-t}}\sum_{n=1}^{T-t}\lambda^{n-1}\hat Q^\pi_n(s_t,a_t)
$$

The fraction outside the sum is just to rescale the weights to sum to 1.

By subtracting the critic, we get the Generalised *Advantage* Estimation:

$$\hat A^\pi_{\text{GAE}_{n\text{-step}}}(s_t,a_t) = \hat Q^\pi_{\text{GAE}_{n\text{-step}}}(s_t,a_t) - \hat V^\pi(s_t)$$

## Efficient Reformulation

While this arrangement of the formula fits nicely into the baselines framework, making the variance reduction techniques explicit, it is computationally expensive ($O(T^2)$ due to the nested sums). In practice, an alternative formulation is used which we construct using the 1-step advantage estimators:

$$ \delta^V_t = r_{t} + \gamma \hat V^\pi(s_{t+1}) - \hat V^\pi(s_t) $$

We can use these to express $n$-step advantage estimators:

$$ \hat A^\pi_n(s_t,a_t) = \sum_{l=0}^{n-1} \gamma^l \delta^V_{t+l}$$

Which by telescoping (cancelling terms) is equal to:

$$\hat A^\pi_n(s_t,a_t) = \hat Q^\pi_n(s_t,a_t) - \hat V^\pi(s_t)$$

Since the baseline $\hat V^\pi(s_t)$ is constant across all $n$, taking the 
exponentially-weighted average of these $n$-step advantages gives us the same 
GAE estimate as before:

$$\hat A^\pi_{\text{GAE}_{n\text{-step}}}(s_t,a_t) = \frac{1-\lambda}{1-\lambda^{T-t}}\sum_{n=1}^{T-t}\lambda^{n-1}\hat A^\pi_n(s_t,a_t)$$

To simplify this expression into a computationally cheap form, we must look at the mathematically idealised infinite-time horizon case:

$$\hat A^\pi_{\text{GAE}_\infty}(s_t,a_t) = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A^\pi_n(s_t,a_t)$$

While this move, and the subsequent truncation we will do, give approximations of the original finite formulation, the error incurred is negligible for typical hyper-parameters due to exponential decay.

Expanding out gives:

$$= (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{l=0}^{n-1} \gamma^l \delta^V_{t+l}$$

The key insight is to regroup by timestep rather than by $n$-step return - like 
transposing a table where rows are $n$-step returns and columns are timesteps.

Each $\delta^V_{t+l}$ appears in all $n$-step returns where $n>l$, each time with coefficient $\gamma^l \lambda^{n-1}$. Regrouping gives an infinite geometric sum we can simplify:

$$
\begin{align*}
&= (1-\lambda)\sum_{l=0}^{\infty}\gamma^l \delta^V_{t+l}\sum_{n=l+1}^{\infty}\lambda^{n-1} 
\\
&= (1-\lambda)\sum_{l=0}^{\infty}\gamma^l \delta^V_{t+l} \left(\frac{\lambda^l}{1-\lambda}\right)
\\
&= \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta^V_{t+l}
\end{align*}
$$

This final equation is the canonical form of the GAE estimate as described in the paper. However, we can neither sample infinite time steps nor compute the infinite sum, so in practice we truncate the sum to stop at $l=T-t-1$.

$$\hat{A}^\pi_{\text{GAE}}(s_t,a_t) = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta^V_{t+l}$$

## The Effect of $\lambda$

Before moving to the practical implementation, let's examine how $\lambda$
controls the bias-variance trade-off in this formulation.

**The general pattern:**
The parameter $\lambda$ controls the exponential decay rate, determining how quickly distant prediction errors are downweighted. Note that each $\delta^V_{t+l}$ contributes exactly one sampled reward $r_{t+l}$, with the rest coming from the value function. Thus $\lambda$ exponentially controls the trade-off between relying on sampled rewards (unbiased but high variance) versus the value function (biased but low variance):

- **Lower $\lambda$** (0.8-0.92): Rapid decay means only nearby $\delta^V$ terms contribute significantly. This gives low variance (fewer random samples matter) but higher bias (more reliance on $\hat V^\pi$).
- **Higher $\lambda$** (0.95-0.99): Slower decay allows more future $\delta^V$ terms to contribute. This reduces bias (more sampled rewards are used) but increases variance (more sampling randomness).

Typical values are $\lambda \in [0.92, 0.99]$, balancing these competing concerns.

**The extreme case $\lambda = 0$:**
For completeness, when $\lambda = 0$ only the immediate prediction error contributes: $\hat{A}^\pi_{\text{GAE}(\lambda=0)}(s_t,a_t)= \delta^V_t$. This is the lowest-variance but highest-bias estimator.

**Very high $\lambda$ values:**
For $\lambda$ values approaching 1, the two formulations diverge more significantly. In particular, at $\lambda=1$, the truncated infinite formulation becomes:

$$\hat{A}^\pi_{\text{GAE}(\lambda=1)}(s_t,a_t) = \sum_{l=0}^{T-t-1}\gamma^l \delta^V_{t+l} = \left(\sum_{l=0}^{T-t-1}\gamma^l r_{t+l}\right) - \hat V^\pi(s_t)$$

This is the Monte Carlo estimate used in the basic actor-critic algorithm. It's unbiased with respect to the value function's accuracy, but has maximum variance. In contrast, the finite weighted-average formulation at $\lambda=1$ gives an arithmetic mean of all $n$-step returns, which is a different estimator entirely.

## The Dynamic Programming Implementation

The practical advantage of this formulation comes from its $O(T)$ complexity when computed iteratively for all timesteps, compared to $O(T^2)$ for the nested sum in the weighted-average formulation.

The algorithm proceeds as follows, with each step being $O(T)$:

1. Estimate $\hat V^\pi(s_t)$ for all timesteps using the Critic.  
   GAE natively implements the truncation correction from earlier since $\hat V^\pi(s_T)$ estimates any future rewards not sampled. However, if $s_T$ is a natural terminal state, set $\hat V^\pi(s_T) = 0$ since no further reward can be collected.
  
2. Compute all prediction errors: For $t = 0, \ldots, T-1$:  
   
   $$\delta^V_t = r_t + \gamma \hat V^\pi(s_{t+1}) - \hat V^\pi(s_t)$$
   
3. Compute advantages backwards from the episode end:  
   **Initialisation:** $t=T$  

   $$\hat A^\pi_\text{GAE}[T] = 0$$
   
   **Recurrence relation:** For $t = T-1, T-2, ..., 0$:  

   $$\hat A^\pi_\text{GAE}[t] = δ^V_t + γλ · \hat A^\pi_\text{GAE}[t+1]$$
   
   Note: For $t < T$, the array notation $\hat A^\pi_\text{GAE}[t]$ denotes $\hat A^\pi_\text{GAE}(s_t, a_t)$, but $\hat A^\pi_\text{GAE}[T]$ is purely a boundary value for the recursion.  

   Unrolling this recursion recovers our sum formula: 

   $$\hat A^\pi_\text{GAE}(s_t,a_t) = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta^V_{t+l}$$

```python
def compute_gae(rewards: Tensor,
                observations: Tensor,
                gamma: float,
                lambda_: float,
                terminal: bool) -> Tensor:
    T = len(rewards)
    
    # Step 1: Compute value estimates for all states
    values = critic(observations).detach()  # Shape: (T+1,)
    if terminal:
        values[T] = 0.0
    
    # Step 2: Compute TD errors (deltas)
    deltas = rewards + gamma * values[1:] - values[:-1]  # Shape: (T,)
    
    # Step 3: Compute advantages backwards
    advantages = torch.zeros(T + 1) # advantages[T] = 0
    
    for t in reversed(range(T)):
        advantages[t] = deltas[t] + gamma * lambda_ * advantages[t+1]
    
    return advantages[:T]  # Return only the T actual timesteps
```

# Conclusion and Next Steps

The techniques explored in this post are all fundamentally about variance reduction. This falls under the broader theme of extracting more information from the collected data. Rather than using each trajectory sample only once, we train critics on the full dataset, use value estimates to synthesise missing rewards, and control the balance through GAE. This data efficiency is crucial in RL, where environment interaction is often the computational bottleneck.

The same principle—squeezing as much learning as possible from each batch of experience—motivates the next major development in policy gradient methods: trust region algorithms. TRPO and PPO ask: how can we safely take multiple gradient steps on the same batch without overfitting to it? These algorithms combine the variance reduction techniques from this post with constraints on how much the policy can change, enabling even more efficient use of collected data.

For next steps, I recommend reading the PPO paper ([Schulman et al., 2017](https://arxiv.org/pdf/1707.06347)) to understand the core principles of the algorithm. Follow it with ["The 37 Implementation Details of Proximal Policy Optimization"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) which covers many crucial implementation details absent from the paper.

# Feedback and Contact Information

If you found this article useful, please consider:
- Sharing it with friends and colleagues who might benefit
- Starring the repository on GitHub
- Connecting with me on LinkedIn

[![GitHub](https://img.shields.io/badge/Star_Repository-gold?logo=github&logoColor=white)](https://github.com/xycoord/deep-rl-course)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/logan-thomson-01a4942ab)

If you found anything confusing, I'd be happy to answer any questions:

[![GitHub Issues](https://img.shields.io/badge/Submit_Issue-red?logo=github)](https://github.com/xycoord/deep-rl-course/issues)
[![GitHub Discussions](https://img.shields.io/badge/Join_Discussion-green?logo=github)](https://github.com/xycoord/deep-rl-course/discussions)

I'm continuously working to improve this material and extend it with additional topics. Your feedback helps make this resource better for everyone!