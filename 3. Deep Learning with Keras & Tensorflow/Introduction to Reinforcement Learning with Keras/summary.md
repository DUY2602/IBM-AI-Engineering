# Reinforcement Learning with Keras — Summary

## What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a machine learning paradigm where an **agent** learns to make sequential decisions by interacting with an **environment** in order to maximize cumulative reward.

Unlike supervised learning (labeled data) or unsupervised learning (unlabeled patterns), RL learns through trial and error guided by rewards and penalties.

---

## Core Concepts

| Term            | Definition                                                 |
| --------------- | ---------------------------------------------------------- |
| **Agent**       | The learner / decision-maker that takes actions            |
| **Environment** | The world the agent interacts with (e.g., CartPole)        |
| **State**       | A representation of the current situation                  |
| **Action**      | A choice the agent can make                                |
| **Reward**      | Feedback signal indicating how good an action was          |
| **Policy**      | Strategy that maps states to actions                       |
| **Episode**     | A complete sequence of interactions until a terminal state |

---

## Key RL Hyperparameters

| Symbol / Term   | Meaning                                                                                |
| --------------- | -------------------------------------------------------------------------------------- |
| **α (alpha)**   | Learning rate — how strongly new information updates existing knowledge                |
| **γ (gamma)**   | Discount factor — importance of future rewards vs. immediate rewards (0 ≤ γ ≤ 1)       |
| **ε (epsilon)** | Exploration rate — probability of taking a random action instead of the best-known one |

---

## Q-Learning

**Q-Learning** is an **off-policy** algorithm that learns the value of taking a specific action in a given state.

### Q-Value

The **Q-value** estimates the expected cumulative future reward of taking action _a_ in state _s_ and then following the optimal policy:

$$Q(s, a) = \mathbb{E}\left[R_t + \gamma \max_{a'} Q(s', a')\right]$$

### Q-Table

A lookup table where each entry stores the estimated Q-value for a state-action pair. Works well only for small, discrete state spaces.

### Bellman Equation

The foundation of Q-learning. It expresses the relationship between the value of a state and the values of its successor states, enabling dynamic programming solutions.

---

## Deep Q-Networks (DQN)

**Deep Q-Networks** extend classic Q-learning by using a **neural network** (the **Q-network**) to approximate the Q-value function.

This allows RL to scale to environments with large or continuous state spaces where a Q-table is impractical.

### Typical DQN Architecture (Keras example)

```python
model = Sequential([
    Dense(24, activation='relu', input_dim=state_size),
    Dense(24, activation='relu'),
    Dense(action_size, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
```

### Key Components of a DQN Agent

| Component                   | Purpose                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------- |
| **Replay Buffer (Memory)**  | Stores past experiences `(state, action, reward, next_state, done)` for stable training |
| **ε-greedy Policy**         | Balances exploration (random actions) and exploitation (best known actions)             |
| **Target Network / Update** | Uses the Bellman equation to create training targets                                    |
| **Experience Replay**       | Samples random minibatches from memory to break correlation between consecutive samples |

---

## Training Loop Overview

1. **Observe** the current state
2. **Select** an action using ε-greedy policy
3. **Execute** the action and receive reward + next state
4. **Store** the experience in the replay buffer
5. **Sample** a minibatch and update the Q-network via the Bellman target
6. **Decay** ε over time to reduce exploration
7. Repeat until the agent reaches a desired performance level

---

## CartPole-v1 Environment

A classic benchmark environment (from OpenAI Gym / Gymnasium):

- A pole is balanced on a moving cart
- Goal: keep the pole upright as long as possible
- State: cart position, cart velocity, pole angle, pole angular velocity
- Actions: push cart left or right
- Episode ends when the pole falls beyond a threshold or the cart moves too far

---

## Practical Tips

- Start with a high **ε** (e.g., 1.0) and decay it gradually toward a minimum (e.g., 0.01)
- Use a **replay buffer** of reasonable size (e.g., 2,000 experiences)
- Train with **minibatches** rather than single experiences for more stable learning
- Monitor average episode length / total reward to judge progress
- Experiment with network architecture, learning rate, and reward shaping
- Consider early stopping when the agent consistently reaches the maximum episode length

---

## Glossary Quick Reference

| Term                       | Short Definition                                                          |
| -------------------------- | ------------------------------------------------------------------------- |
| **Alpha (α)**              | Learning rate                                                             |
| **Bellman Equation**       | Optimality condition used in dynamic programming & Q-learning             |
| **Deep Q-Network (DQN)**   | Q-learning + deep neural network for large state spaces                   |
| **Epsilon (ε)**            | Exploration rate                                                          |
| **Gamma (γ)**              | Discount factor for future rewards                                        |
| **Hyperparameters**        | Settings chosen before training (learning rate, batch size, layers, etc.) |
| **OpenAI Gym / Gymnasium** | Toolkit for developing and testing RL agents                              |
| **Q-Learning**             | Off-policy algorithm that learns action values                            |
| **Q-Network**              | Neural network approximating the Q-function                               |
| **Q-Table**                | Lookup table of Q-values (small discrete problems)                        |
| **Q-Value**                | Expected future reward for a state-action pair                            |
| **Reinforcement Learning** | Learning by maximizing cumulative reward through interaction              |
| **Supervised Learning**    | Learning from labeled data                                                |
| **Unsupervised Learning**  | Learning patterns from unlabeled data                                     |
