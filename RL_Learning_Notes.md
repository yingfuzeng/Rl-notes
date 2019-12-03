# Notes of me learning RL

### Meta table of symbols

| Symbol | Meaning |
| ----------------------------- | ------------- |
| $$s \in \mathcal{S}$$ | States. |
| $$a \in \mathcal{A}$$ | Actions. |
| $$r \in \mathcal{R}$$ | Rewards. |
| $$S_t, A_t, R_t$$ | State, action, and reward at time step t of one trajectory. I may occasionally use $$s_t, a_t, r_t$$ as well. |
| $$\gamma$$ | Discount factor; penalty to uncertainty of future rewards; $$0<\gamma \leq 1$$. |
| $$G_t$$ | Return; or discounted future reward; $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$. |
| $$P(s’, r \vert s, a)$$ | Transition probability of getting to the next state s’ from the current state s with action a and reward r. |
| $$\pi(a \vert s)$$ | Stochastic policy (agent behavior strategy); $$\pi_\theta(.)$$ is a policy parameterized by θ. |
| $$\mu(s)$$ | Deterministic policy; we can also label this as $$\pi(s)$$, but using a different letter gives better distinction so that we can easily tell when the policy is stochastic or deterministic without further explanation. Either $$\pi$$ or $$\mu$$ is what a reinforcement learning algorithm aims to learn. |
| $$V(s)$$ | State-value function measures the expected return of state s; $$V_w(.)$$ is a value function parameterized by w.|
| $$V^\pi(s)$$ | The value of state s when we follow a policy π; $$V^\pi (s) = \mathbb{E}_{a\sim \pi} [G_t \vert S_t = s]$$. |
| $$Q(s, a)$$ | Action-value function is similar to $$V(s)$$, but it assesses the expected return of a pair of state and action (s, a); $$Q_w(.)$$ is a action value function parameterized by w. |
| $$Q^\pi(s, a)$$ | Similar to $$V^\pi(.)$$, the value of (state, action) pair when we follow a policy π; $$Q^\pi(s, a) = \mathbb{E}_{a\sim \pi} [G_t \vert S_t = s, A_t = a]$$. |
| $$A(s, a)$$ | Advantage function, $$A(s, a) = Q(s, a) - V(s)$$; it can be considered as another version of Q-value with lower variance by taking the state-value off as the baseline. |

#### Model: Predictions and reward
$$
P(s', r \vert s, a)  = \mathbb{P} [S_{t+1} = s', R_{t+1} = r \vert S_t = s, A_t = a]
$$

Thus the state-transition function can be defined as a function of $$P(s', r \vert s, a)$$:

$$
P_{ss'}^a = P(s' \vert s, a)  = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} P(s', r \vert s, a)
$$

The reward function R predicts the next reward triggered by one action:

$$
R(s, a) = \mathbb{E} [R_{t+1} \vert S_t = s, A_t = a] = \sum_{r\in\mathcal{R}} r \sum_{s' \in \mathcal{S}} P(s', r \vert s, a)
$$

#### Policy

Policy, as the agent's behavior function $$\pi$$, tells us which action to take in state s. It is a mapping from state s to action a and can be either deterministic or stochastic:
- Deterministic: $$\pi(s) = a$$.
- Stochastic: $$\pi(a \vert s) = \mathbb{P}_\pi [A=a \vert S=s]$$.


#### Value Function

Value function measures the goodness of a state or how rewarding a state or an action is by a prediction of future reward. The future reward, also known as **return**, is a total sum of discounted rewards going forward. Let's compute the return $$G_t$$ starting from time t:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The discounting factor $$\gamma \in [0, 1]$$ penalize the rewards in the future, because:
- The future rewards may have higher uncertainty; i.e. stock market.
- The future rewards do not provide immediate benefits; i.e. As human beings, we might prefer to have fun today rather than 5 years later ;).
- Discounting provides mathematical convenience; i.e., we don't need to track future steps forever to compute return.
- We don't need to worry about the infinite loops in the state transition graph.
The **state-value** of a state s is the expected return if we are in this state at time t, $$S_t = s$$:

$$
V_{\pi}(s) = \mathbb{E}_{\pi}[G_t \vert S_t = s]
$$

Similarly, we define the **action-value** ("Q-value"; Q as "Quality" I believe?) of a state-action pair as:

$$
Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \vert S_t = s, A_t = a]
$$

**Q-Value: my understanding is that measures how good the pair(state, action) together is.** 

Additionally, since we follow the target policy $$\pi$$, we can make use of the probility distribution over possible actions and the Q-values to recover the state-value:

$$
V_{\pi}(s) = \sum_{a \in \mathcal{A}} Q_{\pi}(s, a) \pi(a \vert s)
$$

**State Value: $$V_{\pi}(s)$$, is the goodness of the pair (state, police)**



The difference between action-value and state-value is the action **advantage** function ("A-value"):

$$
A_{\pi}(s, a) = Q_{\pi}(s, a) - V_{\pi}(s)
$$

#### Optimal Value and Policy

The optimal value function produces the maximum return:

$$
V_{*}(s) = \max_{\pi} V_{\pi}(s),
Q_{*}(s, a) = \max_{\pi} Q_{\pi}(s, a)
$$

The optimal policy achieves optimal value functions:

$$
\pi_{*} = \arg\max_{\pi} V_{\pi}(s),
\pi_{*} = \arg\max_{\pi} Q_{\pi}(s, a)
$$

And of course, we have $$V_{\pi_{*}}(s)=V_{*}(s)$$ and $$Q_{\pi_{*}}(s, a) = Q_{*}(s, a)$$.


### Markov Decision Processes
"Markov" property, referring to the fact that the future only depends on the current state, not the history:

$$
\mathbb{P}[ S_{t+1} \vert S_t ] = \mathbb{P} [S_{t+1} \vert S_1, \dots, S_t]
$$

Or in other words, the future and the past are **conditionally independent** given the present, as the current state encapsulates all the statistics we need to decide the future.

### Bellman Equations

Bellman equations refer to a set of equations that decompose the value function into the immediate reward plus the discounted future values.

$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$

Similarly for Q-value,

$$
\begin{aligned}
Q(s, a) 
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \mid S_t = s, A_t = a] \\
&= \mathbb{E} [R_{t+1} + \gamma \mathbb{E}_{a\sim\pi} Q(S_{t+1}, a) \mid S_t = s, A_t = a]
\end{aligned}
$$


#### Bellman Expectation Equations

The recursive update process can be further decomposed to be equations built on both state-value and action-value functions. As we go further in future action steps, we extend V and Q alternatively by following the policy $$\pi$$.




$$
V_{\pi}(s) &= \sum_{a \in \mathcal{A}} \pi(a \vert s) Q_{\pi}(s, a) 
$$

Value of $$s$$, is equal to the summation of all values of (state, action) pair, under policy $$\pi$$. (easy to understand)

$$
Q_{\pi}(s, a) &= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi} (s')
$$

$$
V_{\pi}(s) &= \sum_{a \in \mathcal{A}} \pi(a \vert s) \big( R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi} (s') \big)
$$

Q value euqals to the current reward plus the discont* future reward, which is the summation of all possibles **state value** under policy.


$$
Q_{\pi}(s, a) &= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' \vert s') Q_{\pi} (s', a')
$$


#### Deep Q-Network
Thus people use functions (i.e. a machine learning model) to approximate Q values and this is called **function approximation**. For example, if we use a function with parameter $$\theta$$ to calculate Q values, we can label Q value function as $$Q(s, a; \theta)$$.

 Q-learning may suffer from **instability and divergence** when combined with an nonlinear Q-value function approximation and [bootstrapping](#bootstrapping) (See [Problems #2](#deadly-triad-issue)).

### Solution:
- **Experience Replay**: All the episode steps $$e_t = (S_t, A_t, R_t, S_{t+1})$$ are stored in one replay memory $$D_t = \{ e_1, \dots, e_t \}$$. $$D_t$$ has experience tuples over many episodes. During Q-learning updates, samples are drawn at random from the replay memory and thus one sample could be used multiple times. Experience replay improves data efficiency, removes correlations in the observation sequences, and smooths over changes in the data distribution.
- **Periodically Updated Target**: Q is optimized towards target values that are only periodically updated. The Q network is cloned and kept frozen as the optimization target every C steps (C is a hyperparameter). This modification makes the training more stable as it overcomes the short-term oscillations. 



## Monte Carlo Methods

Although I've heard about it many times, it is my first time to formally define this method.

Monte Carlo (MC) methods are a subset of computational algorithms that use the process of repeated random sampling to make numerical estimations of unknown parameters.

It is based on this important theory: **Law of Large Numbers**

**As the number of identically distributed, randomly generated variables increases, their sample mean (average) approaches their theoretical mean.”**

A simple example: estimate the value of pi, area of a circle with radius $$R is -> \pi*R^2$$
Now, we randomly generate points in a box [2R,2R], the chance of it lands inside the circle is $$\pi*R^2 /(2*R)^2$$, by counting the numbers and sample mean is getting closer to the true value of $$\pi$$ as we increase the sample size.

Now to Monte Carlo in RL(MC method):

First, let's recall that $$V(s) = \mathbb{E}[ G_t \vert S_t=s]$$. Monte-Carlo (MC) methods uses a simple idea: It learns from episodes of raw experience without modeling the environmental dynamics and computes the observed mean return as an approximation of the expected return. To compute the empirical return $$G_t$$, MC methods need to learn from <span style="color: #e01f1f;">**complete**</span> episodes $$S_1, A_1, R_2, \dots, S_T$$ to compute $$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$ and all the episodes must eventually terminate.

The empirical mean return for state s is:

$$
V(s) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s]}
$$

*So here the sampled value is calculated*

where $$\mathbb{1}[S_t = s]$$ is a binary indicator function. We may count the visit of state s every time so that there could exist multiple visits of one state in one episode ("every-visit"), or only count it the first time we encounter a state in one episode ("first-visit"). This way of approximation can be easily extended to action-value functions by counting (s, a) pair.

$$
Q(s, a) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}
$$

*The same way we calculate the sample value of Q*


To learn the optimal policy by MC, we iterate it by following a similar idea to [GPI](#policy-iteration).

![Policy Iteration by MC]({{ '/assets/images/MC_control.png' | relative_url }})
{: style="width: 50%;" class="center"}

1. Improve the policy greedily with respect to the current value function: $$\pi(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$.
2. Generate a new episode with the new policy $$\pi$$ (i.e. using algorithms like [ε-greedy] helps us balance between exploitation and exploration.)
3. Estimate Q using the new episode: $$q_\pi(s, a) = \frac{\sum_{t=1}^T \big( \mathbb{1}[S_t = s, A_t = a] \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \big)}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}$$

**One thing to remember about MC is that it requires <span style="color: #e01f1f;">**Complete**</span>  episode to work because to calculate sample Value function, we need G_t, which requires the return value R_t for all  the timestep.**
### Temporal-Difference Learning

Similar to Monte-Carlo methods, Temporal-Difference (TD) Learning is model-free and learns from episodes of experience. However, TD learning can learn from <span style="color: #e01f1f;">**incomplete**</span> episodes and hence we don't need to track the episode up to termination. TD learning is so important that Sutton & Barto (2017) in their RL book describes it as "one idea … central and novel to reinforcement learning".


#### Bootstrapping

TD learning methods update targets with regard to existing estimates rather than exclusively relying on actual rewards and complete returns as in MC methods. This approach is known as **bootstrapping**.


#### Value Estimation

The key idea in TD learning is to update the value function $$V(S_t)$$ towards an estimated return $$R_{t+1} + \gamma V(S_{t+1})$$ (known as "**TD target**"). To what extent we want to update the value function is controlled by the learning rate hyperparameter α:

$$
\begin{aligned}
V(S_t) &\leftarrow (1- \alpha) V(S_t) + \alpha G_t \\
V(S_t) &\leftarrow V(S_t) + \alpha (G_t - V(S_t)) \\
V(S_t) &\leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
\end{aligned}
$$

Similarly, for action-value estimation:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

Next, let's dig into the fun part on how to learn optimal policy in TD learning (aka "TD control"). Be prepared, you are gonna see many famous names of classic algorithms in this section.


#### SARSA: On-Policy TD control

"SARSA" refers to the procedure of updaing Q-value by following a sequence of $$\dots, S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, \dots$$. The idea follows the same route of [GPI](#policy-iteration):
1. At time step t, we start from state $$S_t$$ and pick action according to Q values, $$A_t = \arg\max_{a \in \mathcal{A}} Q(S_t, a)$$; ε-greedy is commonly applied.
2. With action $$A_t$$, we observe reward $$R_{t+1}$$ and get into the next state $$S_{t+1}$$.
3. Then pick the next action in the same way as in step 1.: $$A_{t+1} = \arg\max_{a \in \mathcal{A}} Q(S_{t+1}, a)$$.
4. Update the action-value function: $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)) $$.
5. t = t+1 and repeat from step 1.

In each update of SARSA, we need to choose actions for two steps by following the current policy twice (in Step 1. & 3.).

### Combining TD and MC Learning

In the previous [section](#value-estimation) on value estimation in TD learning, we only trace one step further down the action chain when calculating the TD target. One can easily extend it to take multiple steps to estimate the return. 

Let's label the estimated return following n steps as $$G_t^{(n)}, n=1, \dots, \infty$$, then:

{: class="info"}
| $$n$$        | $$G_t$$           | Notes  |
| ------------- | ------------- | ------------- |
| $$n=1$$ | $$G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$$ | TD learning |
| $$n=2$$ | $$G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$$ | |
| ... | | |
| $$n=n$$ | $$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$ | |
| ... | | |
| $$n=\infty$$ | $$G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-t-1} R_T + \gamma^{T-t} V(S_T) $$ | MC estimation |

The generalized n-step TD learning still has the [same](#value-estimation) form for updating the value function:

$$
V(S_t) \leftarrow V(S_t) + \alpha (G_t^{(n)} - V(S_t))
$$

![TD lambda]({{ '/assets/images/TD_lambda.png' | relative_url }})
{: style="width: 70%;" class="center"}


We are free to pick any $$n$$ in TD learning as we like. Now the question becomes what is the best $$n$$? Which $$G_t^{(n)}$$ gives us the best return approximation? A common yet smart solution is to apply a weighted sum of all possible n-step TD targets rather than to pick a single best n. The weights decay by a factor λ with n, $$\lambda^{n-1}$$; the intuition is similar to [why](#value-estimation) we want to discount future rewards when computing the return: the more future we look into the less confident we would be. To make all the weight (n → ∞) sum up to 1, we multiply every weight by (1-λ), because:

$$
\begin{aligned}
\text{let } S &= 1 + \lambda + \lambda^2 + \dots \\
S &= 1 + \lambda(1 + \lambda + \lambda^2 + \dots) \\
S &= 1 + \lambda S \\
S &= 1 / (1-\lambda)
\end{aligned}
$$

This weighted sum of many n-step returns is called λ-return $$G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$. TD learning that adopts λ-return for value updating is labeled as **TD(λ)**. The original version we introduced [above](#value-estimation) is equivalent to **TD(0)**.

#### Q-Learning: Off-policy TD control

The development of Q-learning ([Watkins & Dayan, 1992](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)) is a big breakout in the early days of Reinforcement Learning.
1. At time step t, we start from state $$S_t$$ and pick action according to Q values, $$A_t = \arg\max_{a \in \mathcal{A}} Q(S_t, a)$$; ε-greedy is commonly applied.
2. With action $$A_t$$, we observe reward $$R_{t+1}$$ and get into the next state $$S_{t+1}$$.
3. Update the action-value function: $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t)) $$.
4. t = t+1 and repeat from step 1.

The first two steps are same as in SARSA. In step 3., Q-learning does not follow the current policy to pick the second action but rather estimate $$Q_*$$ out of the best Q values independently of the current policy.

![SARSA and Q-learning]({{ '/assets/images/sarsa_vs_q_learning.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 6. The backup diagrams for Q-learning and SARSA. (Image source: Replotted based on Figure 6.5 in Sutton & Barto (2017))*


#### Deep Q-Network

Theoretically, we can memorize $$Q_*(.)$$ for all state-action pairs in Q-learning, like in a gigantic table. However, it quickly becomes computationally infeasible when the state and action space are large. Thus people use functions (i.e. a machine learning model) to approximate Q values and this is called **function approximation**. For example, if we use a function with parameter $$\theta$$ to calculate Q values, we can label Q value function as $$Q(s, a; \theta)$$.

Unfortunately Q-learning may suffer from instability and divergence when combined with an nonlinear Q-value function approximation and [bootstrapping](#bootstrapping) (See [Problems #2](#deadly-triad-issue)).

Deep Q-Network ("DQN"; Mnih et al. 2015) aims to greatly improve and stabilize the training procedure of Q-learning by two innovative mechanisms:
- **Experience Replay**: All the episode steps $$e_t = (S_t, A_t, R_t, S_{t+1})$$ are stored in one replay memory $$D_t = \{ e_1, \dots, e_t \}$$. $$D_t$$ has experience tuples over many episodes. During Q-learning updates, samples are drawn at random from the replay memory and thus one sample could be used multiple times. Experience replay improves data efficiency, removes correlations in the observation sequences, and smooths over changes in the data distribution.
- **Periodically Updated Target**: Q is optimized towards target values that are only periodically updated. The Q network is cloned and kept frozen as the optimization target every C steps (C is a hyperparameter). This modification makes the training more stable as it overcomes the short-term oscillations. 

The loss function looks like this:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim U(D)} \Big[ \big( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \big)^2 \Big]
$$

where $$U(D)$$ is a uniform distribution over the replay memory D; $$\theta^{-}$$ is the parameters of the frozen target Q-network.

In addition, it is also found to be helpful to clip the error term to be between [-1, 1]. (I always get mixed feeling with parameter clipping, as many studies have shown that it works empirically but it makes the math much less pretty. :/)

## Exploitation versus Exploration
The exploration vs exploitation dilemma exists in many aspects of our life. Say, your favorite restaurant is right around the corner. If you go there every day, you would be confident of what you will get, but miss the chances of discovering an even better option. If you try new places all the time, very likely you are gonna have to eat unpleasant food from time to time. Similarly, online advisors try to balance between the known most attractive ads and the new ads that might be even more successful.


## ε-Greedy Algorithm

The ε-greedy algorithm takes the best action most of the time, but does random exploration occasionally. The action value is estimated according to the past experience by averaging the rewards associated with the target action a that we have observed so far (up to the current time step t):

$$
\hat{Q}_t(a) = \frac{1}{N_t(a)} \sum_{\tau=1}^t r_\tau \mathbb{1}[a_\tau = a]
$$

where $$\mathbb{1}$$ is a binary indicator function and $$N_t(a)$$ is how many times the action a has been selected so far, $$N_t(a) = \sum_{\tau=1}^t \mathbb{1}[a_\tau = a]$$.


**According to the ε-greedy algorithm, with a small probability $$\epsilon$$ we take a random action, but otherwise (which should be the most of the time, probability 1-$$\epsilon$$) we pick the best action that we have learned so far: $$\hat{a}^{*}_t = \arg\max_{a \in \mathcal{A}} \hat{Q}_t(a)$$.**

We need exploration because information is valuable. In terms of the exploration strategies, we can do no exploration at all, focusing on short-term returns. Or we occasionally explore at random. Or even further, we explore and we are picky about which options to explore — actions with higher uncertainty are favored because they can provide higher information gain.

**Summery: Greedy(no exploration) -> e-greedy(random exploration) -> UCB(Upper confidence bound, smart exploration)**


## Policy Gradient Algorithms
The basic idea is:

The goal of reinforcement learning is to find an optimal behavior strategy for the agent to obtain optimal rewards. The **policy gradient** methods target at modeling and optimizing the policy directly. The policy is usually modeled with a parameterized function respect to θ, $$\pi_\theta(a \vert s)$$. The value of the reward (objective) function depends on this policy and then various algorithms can be applied to optimize θ for the best reward.

The reward function is defined as:


$$
J(\theta) 
= \sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s) 
= \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \vert s) Q^\pi(s, a)
$$

*TO easily understand, d(s) ->  the probability of being in the 's' state under policy $$\pi$$, V(s)->value of this state, which equals to the  value of state,action pair, aka, Q value times the $\p$ of choosing a with this policy*

Ideally, now we have $$J(\theta)$$, we can just do gradient ascend  $$\nabla_\theta J(\theta)$$, and we are gold!

### Policy Gradient Theorem

Computing the gradient $$\nabla_\theta J(\theta)$$ is tricky because it depends on both the action selection (directly determined by $$\pi_\theta$$) and the stationary distribution of states following the target selection behavior (indirectly determined by $$\pi_\theta$$). Given that the environment is generally unknown, it is difficult to estimate the effect on the state distribution by a policy update.

Luckily, the **policy gradient theorem** comes to save the world! Woohoo! It provides a nice reformation of the derivative of the objective function to not involve the derivative of the state distribution $$d^\pi(.)$$ and simplify the gradient computation $$\nabla_\theta J(\theta)$$ a lot.


$$
\begin{aligned}
\nabla_\theta J(\theta) 
&= \nabla_\theta \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \pi_\theta(a \vert s) \\
&\propto \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^\pi(s, a) \nabla_\theta \pi_\theta(a \vert s) 
\end{aligned}
$$

#### Policy Gradient Theorem

Computing the gradient *numerically* can be done by perturbing θ by a small amount ε in the k-th dimension. It works even when $$J(\theta)$$ is not differentiable (nice!), but unsurprisingly very slow.

$$
\frac{\partial \mathcal{J}(\theta)}{\partial \theta_k} \approx \frac{\mathcal{J}(\theta + \epsilon u_k) - \mathcal{J}(\theta)}{\epsilon}
$$

Or *analytically*,

$$
\mathcal{J}(\theta) = \mathbb{E}_{\pi_\theta} [r] = \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) R(s, a)
$$

Actually we have nice theoretical support for (replacing $$d(.)$$ with $$d_\pi(.)$$):

$$
\mathcal{J}(\theta) = \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) Q_\pi(s, a) \propto \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) Q_\pi(s, a)
$$

Check Sec 13.1 in Sutton & Barto (2017) for why this is the case.

Then,

$$
\begin{aligned}
\mathcal{J}(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) Q_\pi(s, a) \\
\nabla \mathcal{J}(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \nabla \pi(a \vert s; \theta) Q_\pi(s, a) \\
&= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) \frac{\nabla \pi(a \vert s; \theta)}{\pi(a \vert s; \theta)} Q_\pi(s, a) \\
& = \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \vert s; \theta) \nabla \ln \pi(a \vert s; \theta) Q_\pi(s, a) \\
& = \mathbb{E}_{\pi_\theta} [\nabla \ln \pi(a \vert s; \theta) Q_\pi(s, a)]
\end{aligned}
$$

This result is named "Policy Gradient Theorem" which lays the theoretical foundation for various policy gradient algorithms:

$$
\nabla \mathcal{J}(\theta) = \mathbb{E}_{\pi_\theta} [\nabla \ln \pi(a \vert s, \theta) Q_\pi(s, a)]
$$

#### REINFORCE

REINFORCE, also known as Monte-Carlo policy gradient, relies on $$Q_\pi(s, a)$$, an estimated return by [MC](#monte-carlo-methods) methods using episode samples, to update the policy parameter $$\theta$$.

A commonly used variation of REINFORCE is to subtract a baseline value from the return $$G_t$$ to reduce the variance of gradient estimation while keeping the bias unchanged. For example, a common baseline is state-value, and if applied, we would use $$A(s, a) = Q(s, a) - V(s)$$ in the gradient ascent update.

1. Initialize θ at random
2. Generate one episode $$S_1, A_1, R_2, S_2, A_2, \dots, S_T$$
3. For t=1, 2, ... , T:
	1. Estimate the the return G_t since the time step t.
	2. $$\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla \ln \pi(A_t \vert S_t, \theta)$$.


#### Actor-Critic

**If the value function is learned in addition to the policy, we would get Actor-Critic algorithm.**
- **Critic**: updates value function parameters w and depending on the algorithm it could be action-value $$Q(a \vert s; w)$$ or state-value $$V(s; w)$$.
- **Actor**: updates policy parameters θ, in the direction suggested by the critic, $$\pi(a \vert s; \theta)$$.

Let's see how it works in an action-value actor-critic algorithm. 

1. Initialize s, θ, w at random; sample $$a \sim \pi(a \vert s; \theta)$$.
2. For t = 1… T:
	1. Sample reward $$r_t  \sim R(s, a)$$ and next state $$s' \sim P(s' \vert s, a)$$.
	2. Then sample the next action $$a' \sim \pi(s', a'; \theta)$$.
	3. Update policy parameters: $$\theta \leftarrow \theta + \alpha_\theta Q(s, a; w) \nabla_\theta \ln \pi(a \vert s; \theta)$$.
	4. Compute the correction for action-value at time t: <br/>
	$$G_{t:t+1} = r_t + \gamma Q(s', a'; w) - Q(s, a; w)$$ <br/>
	and use it to update action function parameters: <br/>
	$$w \leftarrow w + \alpha_w G_{t:t+1} \nabla_w Q(s, a; w) $$.
	5. Update $$a \leftarrow a'$$ and $$s \leftarrow s'$$.

$$\alpha_\theta$$ and $$\alpha_w$$ are two learning rates for policy and value function parameter updates, respectively.


#### A3C

**Asynchronous Advantage Actor-Critic** (Mnih et al., 2016), short for A3C, is a classic policy gradient method with the special focus on parallel training. 

In A3C, the critics learn the state-value function, $$V(s; w)$$, while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is good for parallel training by default, i.e. on one machine with multi-core CPU.

The loss function for state-value is to minimize the mean squared error, $$\mathcal{J}_v (w) = (G_t - V(s; w))^2$$ and we use gradient descent to find the optimal w. This state-value function is used as the baseline in the policy gradient update.

Here is the algorithm outline:
1. We have global parameters, θ and w; similar thread-specific parameters, θ' and w'.
2. Initialize the time step t = 1
3. While T <= T_MAX:
	1. Reset gradient: dθ = 0 and dw = 0.
	2. Synchronize thread-specific parameters with global ones: θ' = θ and w' = w.
	3. $$t_\text{start}$$ = t and get $$s_t$$.
	4. While ($$s_t \neq \text{TERMINAL}$$) and ($$t - t_\text{start} <= t_\text{max}$$):
		1. Pick the action $$a_t \sim \pi(a_t \vert s_t; \theta')$$ and receive a new reward $$r_t$$ and a new state $$s_{t+1}$$.
		2. Update t = t + 1 and T = T + 1.
	5. Initialize the variable that holds the return estimation $$R = \begin{cases} 
		0 & \text{if } s_t \text{ is TERMINAL} \\
		V(s_t; w') & \text{otherwise}
		\end{cases}$$.
	6. For $$i = t-1, \dots, t_\text{start}$$:
		1. $$R \leftarrow r_i + \gamma R$$; here R is a MC measure of $$G_i$$.
		2. Accumulate gradients w.r.t. θ': $$d\theta \leftarrow d\theta + \nabla_{\theta'} \log \pi(a_i \vert s_i; \theta')(R - V(s_i; w'))$$;<br/>
		Accumulate gradients w.r.t. w': $$dw \leftarrow dw + \nabla_{w'} (R - V(s_i; w'))^2$$.
	7. Update synchronously θ using dθ, and w using dw.

A3C enables the parallelism in multiple agent training. The gradient accumulation step (6.2) can be considered as a reformation of minibatch-based stochastic gradient update: the values of w or θ get corrected by a little bit in the direction of each training thread independently.























