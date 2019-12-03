# DQN: Playing Atari with Deep Reinforcement Learning
#### by Volodymyr Mnih

### Main take away messages regarding DQN
- DQN is model-free and off-policy
- The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards **Q(s,a)**.
- The action space is discrete, original usage is for Atari games
-  Experience replay: All the episode steps are stored in one replay memory $$D_t$$. This is a commonly used method to avoid the problem with original Q-learning, its instability and divergence when combined with a nonlinear Q-value function approximation 

  

### Why RL is more challenging than supervised learning?

- Most successful deep learning applications to date have required large amounts of hand-labeled training data.  RL algorithms, on the other hand, must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed. For example, robotic hand picking up an object, you only get a reward at the very end of a successfully grab, all the timesteps prior to that has a zero reward. 

- Another issue is that most deep
learning algorithms assume the data samples to be independent, while in reinforcement learning one
typically encounters sequences of highly correlated states.

### Background about Q-learning theory
Optimal action-value function $$Q^*(s,a)$$ obeys an important identity known as *Bellman equation*. why? The definition of Bellman equations refers to a set of equations that decompose the value function into the immediate reward plus the discounted future rewards.
$$V(S) = E[R_{t+1} + \gamma V(S_{t+1})|S_t = s]$$

For $$Q^*$$, we have the same property:

$$Q^*(s,a) = E[r + \gamma max_{a'}Q^*(s',a')|s,a]$$

What does it mean? if the optimal value $$Q^*(s',a')$$ of the sequence $$s'$$ at the next
time-step was known for all possible actions $$a'$$, then the optimal strategy is to select the action $$a'$$
maximizing the expected value of $$r + \gamma max_{a'}Q^*(s',a')$$. Easy!

Now we use a neural network $$\theta$$ to do function estimation of this optimal value function $$Q^*$$, that is:

$$Q(s, a; \theta) \approx Q^*(s, a).$$

We refer to a neural network function approximator with weights $$\theta$$ as a Q-network. A
Q-network can be trained by minimizing a sequence of loss functions $$L_i({\theta_i})$$ that changes at each
iteration i.

$$L_i({\theta_i}) = E[(y_i - Q(s,a;\theta_i))^2]$$

where $$y_i$$ is the target for iteration i, which is :

$$y_i = E[r + \gamma max_{a'}Q(s',a';\theta_{i-1})]$$

$$y_i$$ is the target value for step $$i$$, and is calculated from the bellman equation above using the previously estimated $$\theta_{i-1}$$  

Also as contrary to supervised learning, where the target value is actually depending on the parameters $$\theta$$, not a constant!

Differentiating the loss function with respect to the weights we arrive at the following gradient,

$$\nabla L_i(\theta_i) = E[(r + \gamma max_{a'}Q(s',a';\theta_{i-1}) - Q(s,a;\theta_i))\nabla_{\theta_i}Q(s,a;\theta_i)]$$

#### Environment setup
We consider tasks in which an agent interacts with an environment E, in this case the Atari emulator,
in a sequence of actions, observations and rewards. At each time-step the agent selects an action
$$a_t$$ from the set of legal game actions, $$A = {1, . . . , K}.$$ The action is passed to the emulator and
modifies its internal state and the game score. The emulator’s
internal state is not observed by the agent; instead, it observes an image $$x_t \in R^d$$ from the emulator,
which is a vector of raw pixel values representing the current screen. In addition, it receives a reward
$$r_t$$ representing the change in-game score. Note that in general the game score may depend on the
whole prior sequence of actions and observations; feedback about an action may only be received
after many thousands of time-steps have elapsed.

Since the agent only observes images of the current screen, the task is partially observed and many
emulator states are perceptually aliased, i.e. it is impossible to fully understand the current situation
from only the current screen $$x_t$$. We, therefore, consider sequences of actions and observations, $$s_t =
x_1 , a_1 , x_2 , ..., a_{t-1} , x_t$$, and learn game strategies that depend upon these sequences. **So one state is actually an episode**

**About model-free and off-policy**
Note that this algorithm is model-free: it solves the reinforcement learning task directly using samples from the emulator E, without explicitly constructing an estimate of E. It is also off-policy: it
learns about the greedy strategy a = max a Q(s, a; θ), while following a behavior distribution that
ensures the adequate exploration of the state space. In practice, the behavior distribution is often selected by an $$\epsilon$$-greedy strategy that follows the greedy strategy with probability 1 − $$\epsilon$$ and selects a
random action with probability $$\epsilon$$.

