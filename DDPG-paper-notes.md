# Continuous Control with Deep Reinforcement Learning
#### by Lillicrap
#### Openai spinning up tutorial:https://spinningup.openai.com/en/latest/algorithms/ddpg.html

### Quick facts about DDPG
- DQN for continuous action domain
- Model-free, off-policy, and actor-critic algorithm


### Backgourds
#### Why DQN is bad for continuous action space?
Remember DQN aims for finding the optimal action-value function $$Q^*(s,a)$$, where the action is chosen by 

$$a(s) = \arg\max_{a} \ Q^*(s,a)$$

When in continuous space, for example, 7-DOF action space, now even for the simplest discretization $$a_i = \{-k, 0,k\}$$, we have 3^7 = 2187 different cases to consider. Another route would be using a normal optimization algorithm, but this would make calculating $$max_{a}Q^*(s,a)$$ a very expensive subroutine.

#### Action-critic to save the day!
actor $$\mu(s)$$ that use a NN to approximate the optimal actor and the critic is $$Q^*(s,a)$$. now we have 

$$ \arg\max_{a} \ Q^*(s,a) = Q(s,\mu(s))$$

So two networks $$\theta, \phi$$ are used.

#### Mean-squared Bellman error (MSBE)
Now the neural network $$Q_\phi(s,a)$$ and replays D with transitions $$(s,a,r,s',d)$$ where d indicates whether $$s'$$ is terminal.  The MSBE is as follows:

$$L(\phi, D) = E_{(s,a,r,s',d) \sim D}[(Q_\phi(s,a) - (r + \gamma(1-d)\arg\max_{a'} \ Q_{\phi}(s',a')))^2]$$ 

Q-learning based algorithms like DQN or DDPG are based on minimizing this MSBE loss function.

#### Replay buffers D, why and how?
when using neural networks for reinforcement learning is that most optimization algorithms assume that the samples are **independently and identically distributed**. It is obvious that when generating samples directly from interacting with the environment sequentially, they are highly dependable on each other. That is why we use replay buffer D.

**how**:  If only very most recent data $$=>$$ overfit and break

If too much D => slower learning 

#### Why Q-learning is off-policy
Because the optimal Q-function should satisfy the Bellman equation for all possible transitions. So any transitions that we’ve ever experienced are fair game when trying to fit a Q-function approximator via MSBE minimization.

#### Target network, why and how?
In the original MSBE equation above, notice the target 

$$(r + \gamma(1-d)\arg\max_{a'} \ Q_{\phi}(s',a'))$$

are also based on the same parameters we are trying to train $$\phi$$, thus making the minimization unstable.
The solution is to use a set of parameters which comes close to $$\phi$$, but with a time delay—that is to say, a second network, called the target network, which lags the first. The parameters of the target network are denoted $$\phi_{\text{targ}}$$.

In DDPG-style algorithms, the target network is updated once per main network update by polyak averaging:

$$\phi_{\text{targ}} \leftarrow \rho\phi_{\text{targ}} + (1-\rho)\phi $$


 

![testimg](https://github.com/yingfuzeng/Rl-notes/blob/master/images/DDPG-code.png?raw=true)