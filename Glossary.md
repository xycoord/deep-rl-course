# Glossary

> **Reinforcement Learning** is a framework in which an *agent* learns to perform tasks through trial-and-error interaction with its *environment*, receiving *rewards/penalties* as feedback from actions.

> An **environment** is a 5-tuple $(\mathcal S, \mathcal A, p, p_0, r)$ where:
> - $\mathcal S$ is the State Space, the set of all states. A state $s\in\mathcal S$ is a complete description of the environment
> - $\mathcal A$ is the Action Space, the set of all actions an agent can take in the environment
> - $p: \mathcal S\times\mathcal A\times\mathcal S\to[0,1]$ is the transition distribution s.t. $p(s_{t+1} | s_t,a_t)$ is the probability that the environment transitions from state $s_t$ to state $s_{t+1}$ when action $a_t$ is taken
> - $p_0:\mathcal S\to[0,1]$ is the initial state distribution s.t. $p_0(s)$ is the probability that $s_0 = s$
> - $r:\mathcal S\times\mathcal A\to\mathbb R$ is the reward function s.t. $r(s_t,a_t)$ is the reward given when the environment is in state $s_t$ and action $a_t$ is taken

> The **Markov Property**: the probability of transitioning to the next state $s_{t+1}$ depends only on the current state $s_t$ and $a_t$, not on the history of previous states or actions.

> $\mathcal O$ is the **observation space**, the set of all possible observations. An **observation** $o\in\mathcal O$ is a partial description of the environment.

> A **Terminal State** is one that cannot be left (such as winning, losing, dying etc.)

> An **Episodic Task** has a *terminal state*
> e.g. a game of chess, a level in a video game, landing a rocket

> A **Continuing Task** has no *terminal state* 
> e.g. trading on the stock market, walking, keeping the pole upright

> **Truncation** is the act of forcibly ending interaction with the environment before reaching a natural conclusion

> An **agent** is an entity that interacts with an environment by taking actions according to its policy.

> A **policy**, $\pi:\mathcal O\to P(\mathcal A)$, is a function from observations to distributions from which an action may be sampled (selected) given an observation:
> $a_t \sim \pi(\cdot|o_t)$
> Equivalently, $\pi(a_t|o_t)$ is the probability that action $a_t$ is selected given observation $o_t$

> An **experience** is the triple $(o_t, s_t, r_t)$

> A **trajectory** $\tau$ is a sequence of experiences from an agent interacting with its environment according to its policy until it reaches a terminal state or is truncated.

> A **reward**, $r_t$, is a numerical feedback signal provided by the reward function at timestep $t$

> A **reward function**, $r: \mathcal S \times \mathcal A \to \mathbb R$, is a function such that the accumulation of rewards over a trajectory corresponds to performance on the task in that trajectory.  
