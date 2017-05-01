#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""agent.py -- reinforcement learning agent with function approximation.

This module is based on David Silver's lectures from
http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/FA.pdf and on Mat
Leonard's jupyter notebook
https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb

In essence, we implement stochastic gradient descent with experience replay.
So, given experience consisting of (state, value) pairs

```latex
\mathcal{D}=\{(s_1, v_1^\pi), \ldots, (s_T, v_T^\pi)\}
```

We:
    1. Sample state, value from experience (s, v) ~ D
    2. Apply stochastic gradient descent update


The module uses the OpenAI gym environments and has been tested with
CartPole-v0 and MountainCar-v0.
"""

from collections import deque

import numpy as np
import tensorflow as tf


class Memory(object):
    "Memory for experience replay."
    def __init__(self, max_size=1024):
        self.memory = deque(maxlen=max_size)

    def add(self, sample):
        "Adds a new sample to the memory."
        self.memory.append(sample)

    def sample(self, batch_size):
        "Samples a mini batch from experience."
        idx = np.random.choice(
            np.arange(len(self.memory)),
            size=batch_size,
            replace=False
        )
        return [self.memory[i] for i in idx]


class FullyConnectedQNetwork(object):
    "A fully connected Q-learning approximation neural network."
    def __init__(self, learning_rate, observation_size, action_size,
                 hidden_size, n_layers, name='FullyConnectedQNetwork',
                 clip_delta=False):

        self.layers = []
        tf.reset_default_graph()

        with tf.variable_scope(name):
            # Inputs coming from the environment. In principle, we could
            # abstract the state obtained from the environment by doing some
            # kind of feature extraction. In this implementation we don't do
            # anything like that, and that's why we explicitly use the term
            # "observation" instead of state here
            self.inputs_ = tf.placeholder(
                tf.float32,
                [None, observation_size],
                name='inputs'
            )

            # The actions we can execute
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')

            # Target Q values for training
            self.q_targets_ = tf.placeholder(tf.float32, [None], name='target')

            # Build each hidden layer
            prev = self.inputs_
            for _ in range(n_layers):
                prev = self._build_layer(prev, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(prev, action_size,
                                                            activation_fn=None)
            # Train with loss (targetQ - Q)^2 {{{
            self.q = tf.reduce_sum(
                # This multiplication allows us to select which output
                # corresponds to which action. Other values are useless.
                tf.multiply(
                    self.output,
                    tf.one_hot(self.actions_, action_size)
                ),
            axis=1)


            delta = self.q - self.q_targets_
            if clip_delta:
                delta = tf.clip_by_value(
                    delta,
                    -1.0,
                    +1.0
                )

            # The loss function
            self.loss = tf.reduce_mean(
                tf.square(delta)
            )
            # }}}

            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def _build_layer(self, previous, size, activation=tf.nn.relu):
        layer = tf.contrib.layers.fully_connected(
            previous,
            size,
            activation_fn=activation
        )
        self.layers.append(layer)
        return layer

    def fit(self, session, observations, actions, rewards, next_observations,
            gamma):
        """Perform a gradient descent update on a minibatch.

        Uses a minibatch sampled from memory to update the weights. Returns the
        loss of the mini batch.

        Params:
            :param session: The TensorFlow session
            :param observations: The observations (states) from the memory
            :param actions: The actions performed
            :param rewards: The rewards obtained when actions were performed
            :param next_observations: The observation of the world after
                   performing action a from state s
            :param gamma: the discount factor to use for next reward

        :return: the gradient descent loss
        """
        # These are the Q targets computed with the current weights in the
        # neural network function approximator. In other words, this gives
        # \hat{Q}(s, a) for all actions available at state s
        q_targets = session.run(
            self.output,
            feed_dict={self.inputs_: next_observations}
        )

        # Whenever an episode ends, we set next_observations to all zeros.
        # (check Agent._pretrain and Agent.train)
        # Hence, here we have to find these observations. We also set
        # Q(s, a) to zero in this case.
        episode_ends = (
            next_observations == np.zeros(env.observation_space.shape[0])
        ).all(axis=1)
        q_targets[episode_ends] = np.zeros(q_targets[0].shape)

        # Compute the Q function
        # FIXME: This breaks the abstraction between network and agent
        # Perhaps this should be computed by a callback?
        targets = rewards + gamma * np.max(q_targets, axis=1)

        # Now we update the weights by minimizing the squared error
        # between predicted Q-network and Q-learning targets
        loss, _ = session.run(
            [self.loss, self.opt],
            feed_dict={
                self.inputs_: observations,
                self.q_targets_: targets,
                self.actions_: actions
            }
        )

        return loss

    def predict(self, session, observation):
        "Given an observation, outputs an action by computing argmax(Q(s, ·))."
        shape = [1] + list(observation.shape)
        feed_dict = {
            self.inputs_: observation.reshape(*shape)
        }
        return np.argmax(
            session.run(
                self.output,
                feed_dict=feed_dict
            )
        )

    @staticmethod
    def save(session, checkpointfn):
        "Checkpoints the current neural network configuration."
        saver = tf.train.Saver()
        saver.save(session, checkpointfn)

    @staticmethod
    def restore(session, checkpointfn):
        "Restores the neural network from disk."
        saver = tf.train.Saver()
        saver.restore(session, checkpointfn)


class Agent(object):
    "Q-Learning agent."
    def __init__(self, memory, qnetwork, environment, pretrain_length,
                 exploration_start, exploration_stop, decay_rate, batch_size,
                 gamma):
        """Builds a new Q-learning agent.

        :param memory: Memory to where experience will be added and minibatches
                       will be sampled.
        :param qnetwork: The neural network used for learning.
        :param environment: An environment with the same methods of OpenAI gym
                            env objects.
        :param pretrain_length: How many samples to add to memory before
                                training begins
        :param exploration_start: The initial value for exploration exponential
                                  decay
        :param exploration_stop: A lower bound for exploration exponential decay
        :param decay_rate: The rate of decay for the exponential function
        :param batch_size: The size of the training batch sampled from memory
        :param gamma: The Q-learning discount factor
        """
        self.memory = memory
        self.qnetwork = qnetwork
        self.env = environment
        self.pretrain_length = pretrain_length
        self.pretrained = False

        self.exploration_start = exploration_start
        self.exploration_stop = exploration_stop
        self.decay_rate = decay_rate

        self.batch_size = batch_size
        self.gamma = gamma
        self.rewards = []

        self.session = tf.Session()

    def random_action(self):
        "Samples a random action and executes it in the environment."
        return self.env.action_space.sample()

    def _pretrain(self):
        "Pre trains the agent adding random experience to the memory."
        if self.pretrained:
            return

        action, observation, reward, done = self.env_reset()

        for _ in range(self.pretrain_length):
            action = self.random_action()
            next_observation, reward, done, _ = self.env.step(action)

            if done:
                # Episode ended
                next_observation = np.zeros(observation.shape)
                self.memory.add((observation, action, reward, next_observation))

                self.env.reset()
            else:
                self.memory.add((observation, action, reward, next_observation))
                observation = next_observation

        self.pretrained = True
        self.env.reset()

    def env_reset(self):
        "Resets the environment and takes a random action."
        self.env.reset()

        action = self.random_action()
        observation, reward, done, _ = self.env.step(action)

        return action, observation, reward, done

    def close(self):
        "Closes the environment"
        self.session.close()

    def train(self, episodes, max_steps, checkpointfn=None, render=False):
        """Trains the agent.
        
        :param episodes: The number of episodes to use.
        :param max_steps: The maximum number of steps to take within an episode.
        :param checkpointfn: The name of the file where to persist the network.
        :param render: Whether training should be rendered.
        """
        if not self.pretrained:
            self._pretrain()

        loss = 0.0
        iteration = 0
        exploration_diff = self.exploration_start - self.exploration_stop
        self.session.run(tf.global_variables_initializer())
        for episode in range(episodes):
            # Reinitialize environment
            total_reward = 0

            action, observation, reward, done = self.env_reset()

            for step in range(max_steps):
                iteration += 1
                if render:
                    self.env.render()

                explore_p = exploration_diff * np.exp(
                    -self.decay_rate * iteration
                )
                explore_p += self.exploration_stop

                if explore_p > np.random.rand():
                    action = self.random_action()
                else:
                    action = self.qnetwork.predict(self.session, observation)

                next_observation, reward, done, _ = self.env.step(action)

                total_reward += reward

                if done:
                    next_observation = np.zeros(observation.shape)

                    print('Episode:', episode, 'Total reward:', total_reward,
                          'Loss:', loss, 'P(exploration):', explore_p)

                    self.rewards.append((episode, total_reward))

                self.memory.add(
                    (observation, action, reward, next_observation)
                )

                batch = self.memory.sample(self.batch_size)
                obs, actions, rewards, next_obs = zip(*batch)
                loss = self.qnetwork.fit(self.session, obs, actions, rewards, next_obs, self.gamma)

                if done:
                    break
                else:
                    observation = next_observation

        if checkpointfn:
            self.qnetwork.save(self.session, checkpointfn)

    def evaluate(self, episodes, max_steps, render=False):
        """Evaluates the agent in the environment.
        
        :param episodes: On how many episodes to evaluate the agent.
        :param max_steps: Maximum number of steps within an episode.
        :param render: Whether evaluation should be rendered.
        """
        rewards = []
        for episode in range(episodes):
            total_reward = 0
            action, observation, reward, done = self.env_reset()

            for step in range(max_steps):
                action = self.qnetwork.predict(self.session, observation)
                next_observation, reward, done, _ = env.step(action)

                if render:
                    env.render()

                total_reward += reward

                if done:
                    rewards.append(total_reward)
                    break
                else:
                    observation = next_observation

        return rewards


if __name__ == '__main__':
    import gym

    ENVNAME = 'CartPole-v0'
    #ENVNAME = 'MountainCar-v0'

    env = gym.make(ENVNAME)

    TRAIN_EPISODES = 256          # max number of episodes to learn from
    MAX_STEPS = 200                # max steps in an episode
    GAMMA = 0.99                   # future reward discount

    # exploration parameters
    EXPLORE_START = 1.0            # exploration probability at start
    EXPLORE_STOP = 0.01            # minimum exploration probability
    DECAY_RATE = 0.0001            # exponential decay rate for exploration prob

    # network parameters
    HIDDEN_SIZE = 64               # number of units in each Q-network hidden layer
    LEARNING_RATE = 0.0001         # Q-network learning rate

    # memory parameters
    MEMORY_SIZE = 10000            # memory capacity
    BATCH_SIZE = 128               # experience mini-batch size
    PRETRAIN_LENGTH = BATCH_SIZE   # number experiences to pretrain the memory

    TRAIN_EPISODES = 600           # max number of episodes to learn from
    GAMMA = 0.99                   # future reward discount

    memory = Memory(MEMORY_SIZE)

    OBSERVATION_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n

    qnetwork = FullyConnectedQNetwork(
        LEARNING_RATE,
        OBSERVATION_SIZE,
        ACTION_SIZE,
        HIDDEN_SIZE,
        3
    )

    agent = Agent(
        memory,
        qnetwork,
        env,
        PRETRAIN_LENGTH,
        EXPLORE_START,
        EXPLORE_STOP,
        DECAY_RATE,
        BATCH_SIZE,
        GAMMA
    )
    agent.train(
        TRAIN_EPISODES,
        MAX_STEPS,
        checkpointfn=ENVNAME + '.ckpt',
        render=False
    )
    print('Agent performance:', agent.evaluate(3, MAX_STEPS, render=True))
