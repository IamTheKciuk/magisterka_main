import torch
import gym
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from random import sample, random
import wandb
from tqdm import tqdm
import numpy as np
import argh
from models import Model, ConvModel
from utils import FrameStackingAndResizingEnv

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

#pamiec gry, wykonane akcje itd /// state>action>reward>next state
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = [None]*buffer_size
        self.idx = 0

    def insert(self, sarsd):
        self.buffer[self.idx % self.buffer_size] = sarsd # jesli index bedzie na koncu tablicy to rolujemy go do 0 // lol jakie to madre
        self.idx += 1

    def sample(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        if self.idx < self.buffer_size: # nie mozemy samplowac wartosci None wiec jesli idx jest mniejszy niz wielkosc buffera to samplujemy do indexu
            return sample(self.buffer[:self.idx], num_samples)
        return sample(self.buffer, num_samples)

#kopiowanie wag polaczen z jednej sieci do drugiej
def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transitions, tgt, num_actions, device, gamma=0.99):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]

    with torch.no_grad(): # nie zapisuje do sieci tgt
        qvals_next = tgt(next_states).max(-1)[0] # shape -> (N, num_actions) /obliczenie q vals dla next action

    model.opt.zero_grad()
    qvals = model(cur_states) # oblicza q val
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)


    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals*one_hot_actions, -1), rewards + mask[:, 0] * qvals_next * gamma)
    #loss = ((rewards + mask[:, 0] * qvals_next * gamma - torch.sum(qvals*one_hot_actions, -1))**2).mean()
    loss.backward()
    model.opt.step()
    return loss

def main(name, test = False, chkpt = None, device = 'cpu'):
    if not test:
        wandb.init(project='dqn-tutorial', name=name)

    memory_size = 100000 # pamięć gry
    min_rb_size = 20000
    sample_size = 100 #batch size <----
    lr = 0.0001

    eps_min = 0.01
    eps_decay = 0.999999

    env_steps_before_train = 100 # srodowisko wykonuje tyle stepów przed trenowaniem
    tgt_model_update = 500

    env = gym.make("Breakout-v0")
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)
    last_observation = env.reset()

    m = ConvModel(env.observation_space.shape, env.action_space.n, lr=lr).to(device) #stworzenie modelu sieci trenujacej -> odpalenie go na gpu
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = ConvModel(env.observation_space.shape, env.action_space.n).to(device) # stworzenie modelu sieci docelowej
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.01)
            tq.update(1)
            eps = eps_decay**(step_num)
            if test:
                eps = 0

            if random() < eps:
                action = env.action_space.sample()
            else:
                action = m(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += reward
            reward = reward*0.1

            rb.insert(Sarsd(last_observation, action, reward, observation, done))
            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                observation = env.reset()

            steps_since_train += 1
            step_num += 1

            if (not test) and rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
                loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n, device)
                wandb.log({'loss': loss.detach().cpu().item(), 'eps': eps,'avg_reward': np.mean(episode_rewards)}) # wysylanie logow do wandb app
                episode_rewards = []
                #print(step_num, loss.detach().item())
                epochs_since_tgt += 1
                if epochs_since_tgt > tgt_model_update:
                    print('updating tgt model')
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"/home/karol/Dokumenty/Magisterka badania/cartpole_test/trained_models/{step_num}.pth")
                steps_since_train = 0
    except KeyboardInterrupt:
        pass

    env.close()

if __name__ == '__main__':
    #main(True, "/home/karol/Dokumenty/Magisterka badania/cartpole_test/trained_models/trained.pth")
    main("breakout_test")