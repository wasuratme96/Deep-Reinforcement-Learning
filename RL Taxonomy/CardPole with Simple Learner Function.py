import gym
import logging
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16 # Number of episode
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
''' 
Episode
- Single Episode, stored as total undiscounted reawrd and collection of episode
EpisodeStep
- Represent one single step that agent made in episode.
- Episode steps from elite episodes as training dat 
'''
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Action per batch loop
def iterate_batches(env, net, batch_size):
    batch = []  # Collect a list of Episode instance
    episode_reward  = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim = 1)

    while True:
        # Convert observation space into tensor object
        # In CartPole, it is 4 numbers
        obs_v = torch.FloatTensor([obs])   
        act_prob_v = sm(net(obs_v))   # Pass output from observation in to NeuralNet
        
        # Get vector probability of each actions
        act_probs = act_prob_v.data.numpy()[0] #
        
        # Randomly Select actions base on probability vectors
        # In our case, it would be left of right [-1, 1]
        action = np.random.choice(len(act_probs), p = act_probs)

        # Pass action to environments from selected action
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward # Collect reward for single episode

        # Collect obsvervation (that used to choose action)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        # How we handle when the current episode is over
        # Stick has fallen down despite our efforts
        if is_done:
            # Collect reward and steps from ended episode
            # steps in Episode in compose with observation and action in each steps
            e = Episode(reward= episode_reward, steps = episode_steps)
            batch.append(e)

            # Reset episode reward and episode step
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            # Sent out batch to outer iterations
            if len(batch) == batch_size:
                yield batch
                # Clear number of episode in batch
                batch = []

        obs = next_obs

# Training loop of cross-entropy
def filter_batch(batch, percentile):
    
    # batch is collection of episode 
    # Which contains reward of each episode and steps for each episode
    # Episode(reward= episode_reward, steps = episode_steps))
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = [] 
    train_act = []
    for reward, steps in batch:
        # Filter out episode with reward lower than reward_bound
        if reward < reward_bound:
            continue
        
        # Select observation and action only from episode that exceed reward_bound
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    # Conver train observation and action into tensor
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory= "Monitor", force = True)
    obs_size = env.observation_space.shape[0] # Total is 4 value (CartPole State)
    n_actions = env.action_space.n # Total is 2 value [-1, 1]

    # Initiate deep learner model with defined objective and optimizer
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr = 0.01)
    writer = SummaryWriter(comment ="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # get snapshot of train dataset (obs, acts, reward_bound, reward_mean)
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        
       # Reset gradient of optimizer
        optimizer.zero_grad()

         # Feed data into our learner
        action_scores_v = net(obs_v)

        # Get loss comparing with actual actions,
        loss_v = objective(action_scores_v, acts_v)
        # Perform back propagate
        loss_v.backward()

        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break

    writer.close()






