import numpy as np
import gymnasium as gym
import random
from collections import deque

from noah_neural.tensors import Tensor
from noah_neural.layers import Network, LinearLayer, ReLU

class QNet:

    def __init__(self, in_dim, out_dim):
        self.net = Network([
            LinearLayer(in_dim, 256),
            ReLU(),
            LinearLayer(256, 256),
            ReLU(),
            LinearLayer(256, out_dim),
        ])

    def copy(self, other_qnet):
        for i in range(len(self.net.layers)):
            if isinstance(self.net.layers[i], LinearLayer):
                self.net.layers[i].weights = other_qnet.net.layers[i].weights
                self.net.layers[i].biases = other_qnet.net.layers[i].biases
            

    def gradient_descent(self, lr):
        self.net.gradient_descent(lr)

    def forward(self, inp):
        return self.net(inp)

LR = 0.001
REPLAY_SIZE = 10000
MINIBATCH_SIZE = 32
TIMESTEPS = 100000
GAMMA = 0.99
UPDATE_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 500
EPSILON_START = 1.0
EPSILON_END = 0.04
EPSILON_DECAY_STEPS = 100000
ROLLOUTS = TIMESTEPS // UPDATE_FREQ

env = gym.make("CartPole-v1", render_mode=None)
env = gym.wrappers.RecordEpisodeStatistics(env)

states, _ = env.reset()
replay_memory = deque(maxlen=REPLAY_SIZE)
network = QNet(4, 2)
target_network = QNet(4, 2)

epsilon = EPSILON_START
completed_timesteps = 0; completed_episodes = 0

for rollout in range(ROLLOUTS):

    for timestep in range(UPDATE_FREQ):

        if np.random.random() >= epsilon:
            qvals = network.forward(Tensor(data=states))
            actions = qvals.val.argmax(axis=-1)
        else:
            actions = env.action_space.sample()

        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)

        replay_memory.append((states, actions, rewards, sprimes, is_terms, is_truncs))

        states = sprimes

        if "episode" in info:
            completed_episodes += 1
            print(f"episode {completed_episodes}, timestep {completed_timesteps}, reward {info['episode']['r']}")
            states, _ = env.reset()

        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * (completed_timesteps / EPSILON_DECAY_STEPS)

        if completed_timesteps % TARGET_UPDATE_INTERVAL == 0:
            target_network.copy(network)

        completed_timesteps += 1

    for grad_update in range(GRADIENT_STEPS):
        if len(replay_memory) >= MINIBATCH_SIZE:
            minibatch = random.sample(list(replay_memory), MINIBATCH_SIZE)
            s = Tensor(data=np.array([transition[0] for transition in minibatch]))
            a = Tensor(data=np.array([transition[1] for transition in minibatch]))
            r = Tensor(data=np.array([transition[2] for transition in minibatch]))
            sprimes = Tensor(data=np.array([transition[3] for transition in minibatch]))
            masks = 1 - r.val
            
            qvals = network.forward(s)
            chosen_qvals = Tensor(data=qvals.val[np.arange(qvals.shape[0]), a.val.astype(np.int64)])

            next_qvals = target_network.forward(sprimes)
            targets = r + GAMMA * next_qvals.val.max(-1)[0] * masks

            loss = ((chosen_qvals - targets)**2).mean()
            loss.bprop()

            network.gradient_descent(LR)
