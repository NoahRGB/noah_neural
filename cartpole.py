import numpy as np
import gymnasium as gym
from collections import deque

from noah_neural.tensors import Tensor
from noah_neural.layers import Network, LinearLayer, ReLU

class QNet:

    def __init__(self):
        ...

LR = 0.001
REPLAY_SIZE = 1000
MINIBATCH_SIZE = 32
TIMESTEPS = 10000
UPDATE_FREQ = 4
GRADIENT_STEPS = 1
EPSILON_START = 1.0
EPSILON_END = 0.04
EPSILON_DECAY_STEPS = 10000
ROLLOUTS = TIMESTEPS / UPDATE_FREQ

env = gymnasium.make("CartPole-v1", render_mode=None)
states = env.reset()
replay_memory = deque(maxlen=REPLAY_SIZE)

epsilon = EPSILON_START
completed_timesteps = 0

for rollout in range(ROLLOUTS):

    for timestep in range(UPDATE_FREQ):

        if np.random.random() >= epsilon:
            qvals = network(Tensor(data=states))
            actions = qvals.vals.argmax(dim=-1)
        else:
            actions = env.action_space.sample()

        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)
        
        replay_memory.append((states, actions, rewards, sprimes, is_terms, is_truncs))

        states = sprimes
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * (completed_timesteps / EPSILON_DECAY_STEPS)

        completed_timesteps += 1

    for grad_update in range(GRADIENT_STEPS):
        ...
        
