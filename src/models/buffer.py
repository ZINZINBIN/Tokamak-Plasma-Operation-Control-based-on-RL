import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, is_finite : bool = False):
        # buffers
        self.start_idx_of_episodes = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buffer = [] # if done, first store episode in tmp_episode_buffer and then store data to episodes

        # properties
        self.is_finite = is_finite
    
    def clear(self):
        self.start_idx_of_episodes.clear()
        self.idx_to_episode_idx.clear()
        self.episodes.clear()
        self.tmp_episode_buffer.clear()
    
    def __getitem__(self, idx):
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episodes[episode_idx]
        i = idx - start_idx

        states, actions, next_states, rewards = self.episodes[episode_idx]
        state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]

        return state, action, next_state, reward
    
    def __len__(self):
        return len(self.idx_to_episode_idx)
    
    def mean_reward(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.mean(reward) for reward in rewards])

    def mean_return(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.sum(reward) for reward in rewards])

    def store_step(self, state, action, next_state, reward, done = None):
        if done is None:
            self.tmp_episode_buffer.append(
                (state, action, next_state, reward)
            )
        else:
            self.tmp_episode_buffer.append(
                (state, action, next_state, reward, done)
            )

    def done_episode(self):
        if self.is_finite:
            states, actions, next_states, rewards, dones = zip(*self.tmp_episode_buffer)
        else:
            states, actions, next_states, rewards = zip(*self.tmp_episode_buffer)
        
        episode_len = len(states)
        usable_episode_len = episode_len - 1
        self.start_idx_of_episodes.append(len(self.idx_to_episode_idx))
        self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
        self.episodes.append((states, actions, next_states, rewards))
        self.tmp_episode_buffer = []

    def store_episodes(self, episodes):
        for episode in episodes:
            states, actions, next_states, rewards = zip(*episode)
            episode_len = len(states)
            usable_episode_len = episode_len - 1
            self.start_idx_of_episodes.append(len(self.idx_to_episode_idx))
            self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
            self.episodes.append((states, actions, next_states, rewards))