import numpy as np
import torch
import gym

from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim

gamma = 0.99 # Discounting rewards value over step in episode

# Policy network
class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__ ()

        # Leanable parameters
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        ]
        
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()       
        self.train()

    def onpolicy_reset(self) :
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)

        pd = Categorical(logits = pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        
        return action.item()

# Train function for policy
def train (pi, optimizer):
    # Inner gradient descent loop of REINFORCE algorithm
    T = len(pi.rewards) # get maximum step within episode
    rets = np.empty(T, dtype = np.float32)
    future_ret = 0.0

    # Compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)

    loss = -log_probs*rets
    loss = torch.sum(loss)

    optimizer.zero_grad() 
    loss.backward()  # Backpropagate, compute gradients
    optimizer.step() # Update weight to opimizer

    return loss

# main function is main loop
def main():
    env = gym.make('CartPole-v0')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr = 0.01)

    for epi in range(300): # number of total episode
        state = env.reset()
        for t in range(200): # number of time within episode
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            
            env.render()
            if done:
                break
        
        # After ending one episode
        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset() # onpolicy : clear memory after training
        print(f'Episode {epi}, loss: {loss}, total_reward : {total_reward}, solved : {solved}')

if __name__ == '__main__':
    main()