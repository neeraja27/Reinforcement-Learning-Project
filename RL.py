from random import random, randint
import numpy as np
import torch
import torch.nn as nn
from tetris import Tetris
import matplotlib.pyplot as plt

# Global Hyperparameters

learning_rate = 1e-3
gamma = 0.99
initial_epsilon = 1
epsilon_min = 1e-3
decay_epoch = 700
episodes = 1000
memory_size = 10000
batch_size = 512
warm_up = 1000
random_rate = 0.3
expert_rate = 0.7
update_frequency = 20
prioritise_alpha = 0.4


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)

        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class PrioritizedReplayMemory:
    def __init__(self, memory_size, alpha):
        self.memory_size = memory_size
        self.buffer = []
        self.index = 0
        self.priorities = np.zeros((memory_size,), dtype=np.float32)
        self.alpha = alpha

    def push(self, memory):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.memory_size:
            self.buffer.append(memory)
        else:
            self.buffer[self.index] = memory

        self.priorities[self.index] = max_priority
        self.index = (self.index + 1) % self.memory_size

    def sample(self, batch_size, beta = 0.4):
        if len(self.buffer) == self.memory_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.index]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = list(zip(*samples))
        states = torch.stack(batch[0])
        scores = torch.tensor(batch[1], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack(batch[2])
        dones = batch[3]

        return states, scores, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

def heuristic_agent(next_states):

    best_result = -np.inf
    index = 0

    for i in range(len(next_states)):
        result = heuristic_function(next_states[i])
        if result > best_result:
            best_result = result
            index = i

    return index

def heuristic_function(state):
    lines_cleared = state[0].item()
    holes = state[1].item()
    height = state[2].item()
    bumpiness = state[3].item()
    return -0.51 * height - 0.36 * holes - 0.18 * bumpiness + 0.76 * lines_cleared


class DQNagent():
    def __init__(self):
        self.model = QNetwork()
        self.target_model = QNetwork()
        self.epsilon = initial_epsilon
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, next_states, epoch):

        self.epsilon = epsilon_min + (max(decay_epoch - epoch, 0) * (
                initial_epsilon - epsilon_min) / decay_epoch)
        p = random()

        if p < self.epsilon * random_rate:
            index = randint(0, len(next_states) - 1)
        elif p < self.epsilon * (random_rate + expert_rate):
            index = heuristic_agent(next_states)
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(next_states)[:, 0]
            self.model.train()
            index = torch.argmax(predictions).item()

        return index

    def replay(self, replay_memory, beta=0.4):

        states, scores, next_states, dones, indices, weights = replay_memory.sample(batch_size, beta)

        q_values = self.model(states)
        self.model.eval()
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
        self.model.train()

        targets = torch.cat([
            score if done else score + gamma * next_q
            for score, done, next_q in zip(scores, dones, next_q_values)
        ])[:, None]

        td_errors = (q_values - targets).squeeze()
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        replay_memory.update_priorities(indices, new_priorities)

        print('loss:', loss.item())

def train():
    env = Tetris()
    agent = DQNagent()
    state = env.reset()
    replay_memory = PrioritizedReplayMemory(memory_size, prioritise_alpha)
    epoch = 0
    lines = []
    scores = []
    mean_lines = []
    mean_scores = []
    while epoch < episodes:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        index = agent.act(next_states, epoch)

        next_state = next_states[index,:]
        action = next_actions[index]

        score, done = env.step(action, render=True)
        replay_memory.push([state, score, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines

            if len(replay_memory) > warm_up:
                lines.append(final_cleared_lines)
                scores.append(final_score)
                mean_lines.append(np.mean(lines))
                mean_scores.append(np.mean(scores))
            state = env.reset()
        else:
            state = next_state
            continue

        if len(replay_memory) < warm_up:
            continue

        epoch += 1
        agent.replay(replay_memory)
        if epoch % update_frequency == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        print("Epoch: {}/{}, Score: {}, Piece: {}, Cleared lines: {}, Epsilon: {}, Average Lines: {}, Average Score: {}".format(
            epoch,
            episodes,
            final_score,
            final_tetrominoes,
            final_cleared_lines,
            agent.epsilon,
            np.mean(lines),
            np.mean(scores)))

    x = np.arange(episodes)
    plt.plot(x, mean_scores)
    plt.xlabel('Episodes')
    plt.ylabel('Average Game Score')
    plt.show()
    plt.plot(x, mean_lines)
    plt.xlabel('Episodes')
    plt.ylabel('Average Lines Cleared')
    plt.show()

train()