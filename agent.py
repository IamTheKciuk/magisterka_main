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
    loss = loss_fn(torch.sum(qvals*one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * qvals_next * gamma)
    #loss = ((rewards + mask[:, 0] * qvals_next * gamma - torch.sum(qvals*one_hot_actions, -1))**2).mean()
    loss.backward()
    model.opt.step()
    return loss

def main(name, test = False, chkpt = None, device = 'cpu'):
    if not test:
        wandb.init(project='dqn-tutorial', name=name)

    do_boltzman_exploration = True
    memory_size = 100000 # pamięć gry, ilosc zapisywanych obserwacji
    min_rb_size = 20000 # minimalna ilosc wpisow w Sarsd do samplowania
    sample_size = 128 #batch size <----
    lr = 0.0001

    eps_min = 0.05
    eps_decay = 0.999999

    env_steps_before_train = 16 # srodowisko wykonuje tyle stepów przed trenowaniem
    tgt_model_update = 1000 # epochs before target model update
    epochs_before_test = 500 # epochs before testing model


    env = gym.make("Breakout-v0")
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)

    test_env = gym.make("Breakout-v0")
    test_env = FrameStackingAndResizingEnv(test_env, 84, 84, 4)

    last_observation = env.reset()

    m = ConvModel(env.observation_space.shape, env.action_space.n, lr=lr).to(device) #stworzenie modelu sieci trenujacej -> odpalenie go na gpu
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = ConvModel(env.observation_space.shape, env.action_space.n).to(device) # stworzenie modelu sieci docelowej
    update_tgt_model(m, tgt)

    rb = ReplayBuffer(memory_size)
    steps_since_train = 0
    epochs_since_tgt = 0
    epochs_since_test = 0

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

            if do_boltzman_exploration:
                logits = m(torch.Tensor(last_observation).unsqueeze(0).to(device))[0]
                action = torch.distributions.Categorical(logits = logits).sample().item()
            else:
                if random() < eps:
                    action = (env.action_space.sample())
                else:
                    action = m(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += reward

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
                epochs_since_test += 1

                if epochs_since_test > epochs_before_test:
                    rew, frames = run_test_episode(m, test_env, device)
                    wandb.log({'test_reward': rew, f'test_video{step_num}': wandb.Video(frames.transpose(0, 3, 1, 2), str(rew), fps=25, format='mp4')})
                    epochs_since_test = 0

                if epochs_since_tgt > tgt_model_update:
                    print('updating tgt model')
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"/home/karol/Dokumenty/Magisterka badania/test_br/{step_num}.pth")
                steps_since_train = 0
    except KeyboardInterrupt:
        pass

    env.close()

def run_test_episode(model, env, device,  max_steps=1000):
    frames = []
    obs = env.reset()
    frames.append(env.frame)

    idx = 0
    done = False
    reward = 0

    while not done and idx < max_steps:
        action = model(torch.Tensor(obs).unsqueeze(0).to(device)).max(-1)[-1].item()
        obs, r, done, _ = env.step(action)
        reward += r
        frames.append(env.frame)
        idx += 1

    return reward, np.stack(frames, 0)


if __name__ == '__main__':
    #main(True, "/home/karol/Dokumenty/Magisterka badania/cartpole_test/trained_models/trained.pth")
    main("breakout_test_first")