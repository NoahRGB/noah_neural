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
        for mine, other in zip(self.net.params(), other_qnet.net.params()):
            mine.val[...] = other.val

    def gradient_descent(self, lr):
        self.net.gradient_descent(lr)

    def forward(self, inp):
        return self.net(inp)

# hyperparams
# (currently using params from RLzoo)
LR = 0.0023
REPLAY_SIZE = 100_000
MINIBATCH_SIZE = 64
TIMESTEPS = 100_000
GAMMA = 0.99
UPDATE_FREQ = 256 
GRADIENT_STEPS = 128
TARGET_UPDATE_INTERVAL = 10 
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 8000 
ROLLOUTS = TIMESTEPS // UPDATE_FREQ

env = gym.make("CartPole-v1", render_mode=None)
env = gym.wrappers.RecordEpisodeStatistics(env)

states, _ = env.reset()
replay_memory = deque(maxlen=REPLAY_SIZE)
network = QNet(4, 2) # state is shape (4,) and there are 2 actions (push left, push right)
target_network = QNet(4, 2)

epsilon = EPSILON_START
completed_timesteps = 0; completed_episodes = 0

for rollout in range(ROLLOUTS):

    for timestep in range(UPDATE_FREQ):
        
        # epsilon-greedy action selection
        if np.random.random() >= epsilon:
            qvals = network.forward(Tensor(data=states))
            actions = qvals.val.argmax(axis=-1)
        else:
            actions = env.action_space.sample()

        # step environment, save transition in replay memory
        sprimes, rewards, is_terms, is_truncs, info = env.step(actions)
        replay_memory.append((states, actions, rewards, sprimes, is_terms, is_truncs))
        states = sprimes

        # if the episode terminated, reset the states and print reward 
        if "episode" in info:
            completed_episodes += 1
            print(f"episode {completed_episodes}, reward {info['episode']['r']}, train {round((completed_timesteps/TIMESTEPS)*100, 2)}% complete")
            states, _ = env.reset()

        # step epsilon decay, copy target network every C steps
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * min(1.0, completed_timesteps / EPSILON_DECAY_STEPS)

        if completed_timesteps % TARGET_UPDATE_INTERVAL == 0:
            target_network.copy(network)

        completed_timesteps += 1

    # after a rollout of UPDATE_FREQ steps, complete GRADIENT_STEPS gradient updates
    for grad_update in range(GRADIENT_STEPS):

        if len(replay_memory) >= MINIBATCH_SIZE:

            # sample random minibatch and unpack the transitions
            minibatch = random.sample(list(replay_memory), MINIBATCH_SIZE)
            s = Tensor(data=np.array([transition[0] for transition in minibatch]))
            a = np.array([transition[1] for transition in minibatch], dtype=np.int64)
            r = np.array([transition[2] for transition in minibatch])
            sprimes = Tensor(data=np.array([transition[3] for transition in minibatch]))
            terms = np.array([transition[4] for transition in minibatch])
            truncs = np.array([transition[5] for transition in minibatch])
            masks = 1 - (terms)
            
            qvals = network.forward(s)

            # numpy fancy indexing that uses the actions stored in 'a' as an index
            # into the second dimension of 'qvals' to take out the relevant Q(s,a)
            chosen_qvals = qvals[np.arange(qvals.shape[0]), a.astype(np.int64)]

            # y = r + γ Q^(s')
            next_qvals = target_network.forward(sprimes).detach()
            targets = Tensor(data=r + GAMMA * next_qvals.val.max(-1) * masks)

            # MSE between chosen qvals and the new targets
            loss = ((chosen_qvals - targets)**2).mean()
            loss.bprop()

            network.gradient_descent(LR)


# training is done, so render some full episodes
env = gym.make("CartPole-v1", render_mode="human")
done = False; episodes = 0
states, _ = env.reset()
while not done:
    actions = network.forward(Tensor(data=states)).val.argmax(axis=-1)
    states, rewards, is_terms, is_truncs, info = env.step(actions)
    if is_terms or is_truncs:
        episodes += 1 
        if episodes == 10: done = True
        states, _ = env.reset()
