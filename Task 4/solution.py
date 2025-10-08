import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode

TRAIN_STEPS_PER_EPISODE = 200
TEST_STEPS_PER_EPISODE = 200

num_initializations = 0


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    Simple MLP Q-function.
    '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # DONE: add components as needed (if needed)

        self.net_1 = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])
        self.net_2 = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # DONE: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)

        xa = torch.cat([x, a], dim=-1)
        return self.net_1(xa).squeeze(-1), self.net_2(xa).squeeze(-1)

        #####################################################################


class Actor(nn.Module):
    '''
    Simple Tanh deterministic actor.
    '''
    def __init__(self, action_low, action_high, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # DONE: add components as needed (if needed)

        self.log_std_min = -20
        self.log_std_max = 2

        self.net = MLP([obs_size] + [num_units] * num_layers + [action_size * 2])

        #####################################################################
        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x, deterministic=False, with_log_prob=False):
        #####################################################################
        # DONE: code the forward pass
        # the actor will receive a batch of observations of shape (batch_size x obs_size)
        # and output a batch of actions of shape (batch_size x action_size)

        x = self.net(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        u = dist.mean if deterministic else dist.rsample()

        action = torch.tanh(u)
        if with_log_prob:
            log_prob = dist.log_prob(u).sum(axis=-1) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=-1)
            return action, log_prob

        #####################################################################
        return action


class Agent:

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # DONE: store and tune hyperparameters here

    batch_size: int = 256
    gamma: float = 0.9999
    tau: float = 0.0053
    actor_lr: float = 1e-3
    actor_num_layers: int = 3
    actor_num_units: int = 256
    critic_lr: float = 1e-3
    critic_num_layers: int = 3
    critic_num_units: int = 256
    temp_init: float = 2.0
    temp_lr: float = 3e-4
    update_freq: int = 1
    total_runs: int = 10

    #########################################################################

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float().to(self.device)
        self.action_high = torch.tensor(env.action_space.high).float().to(self.device)

        #####################################################################
        # DONE: initialize actor, critic and attributes

        self.target_entropy = float(-self.action_size.item())

        self.actor = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.actor_num_layers, self.actor_num_units).to(self.device)
        self.critic = Critic(self.obs_size, self.action_size, self.critic_num_layers, self.critic_num_units).to(self.device)
        self.critic_target = Critic(self.obs_size, self.action_size, self.critic_num_layers, self.critic_num_units).to(self.device)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(param.data)
        self.log_temp = torch.tensor(math.log(self.temp_init), requires_grad=True, device=self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.temp_optimizer = optim.Adam([self.log_temp], lr=self.temp_lr)

        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
        self.test_step = 0

        global num_initializations
        print(f"Run {num_initializations + 1}/{self.total_runs}")
        num_initializations = (num_initializations + 1) % self.total_runs

    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        #####################################################################
        # DONE: code training logic

        if self.train_step % TRAIN_STEPS_PER_EPISODE == 0:
            n_ep = self.train_step // TRAIN_STEPS_PER_EPISODE
            print(n_ep, ': ', end = '')

        # Normalize reward
        reward = (reward - reward.mean()) / reward.std()

        # Update critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs, deterministic=False, with_log_prob=True)
            target_Q_1, target_Q_2 = self.critic_target(next_obs, next_action)
        target_Q = torch.min(target_Q_1, target_Q_2) - torch.exp(self.log_temp) * next_log_prob
        target_Q = reward + self.gamma * (1 - done) * target_Q
        current_Q_1, current_Q_2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q_1, target_Q) + F.mse_loss(current_Q_2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        action, log_prob = self.actor(obs, deterministic=False, with_log_prob=True)
        current_Q_1, current_Q_2 = self.critic(obs, action)
        current_Q = torch.min(current_Q_1, current_Q_2)
        actor_loss = -(current_Q - torch.exp(self.log_temp) * log_prob).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        with torch.no_grad():
            action, log_prob = self.actor(obs, deterministic=False, with_log_prob=True)
        temp_loss = torch.exp(self.log_temp) * (-log_prob - self.target_entropy).mean()
        self.temp_optimizer.zero_grad()
        temp_loss.backward()
        self.temp_optimizer.step()

        # Soft-update target
        if (self.train_step + 1) % self.update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_step += 1

        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # DONE: return the agent's action for an observation (np.array
        # of shape (obs_size, )). The action should be a np.array of
        # shape (act_size, )

        if not train and self.test_step % TEST_STEPS_PER_EPISODE == 0:
            n_ep = self.test_step // TEST_STEPS_PER_EPISODE
            print(n_ep, ': ', end = '')

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs, deterministic=not train, with_log_prob=False)

        if not train:
            self.test_step += 1

        #####################################################################
        return action.squeeze(0).detach().cpu().numpy()

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing.
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = True
    verbose = True
    seeds = np.arange(1)  # seeds for public evaluation

    Agent.total_runs = len(seeds)

    start = time.time()
    print(f'Running public evaluation.')
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for i in range(WARMUP_EPISODES):
            print(i, ': ', end = '')
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for i in range(TRAIN_EPISODES):
            # print(i, ': ', end = '')
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                # print(n_ep, ': ', end = '')
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
