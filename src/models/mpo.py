'''
Maximum a Posteriori Policy optimisation
- see https://arxiv.org/abs/1806.06920
- use Expectation Maximum algorithm to solve "get the actions which maximize future rewards"
- reference code 1: https://github.com/daisatojp/mpo
- reference code 2: https://github.com/acyclics/MPO
'''
from typing_extensions import Self
import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.data import SubsetRandomSampler, BatchSampler
from scipy.optimize import minimize
import numpy as np
import random
import math
import gc
import os
from torch import long
import matplotlib.pyplot as plt
from itertools import count
from src.models.ActorCritic import Actor, Critic
from src.models.buffer import ReplayBuffer
from tqdm import tqdm
from typing import Optional, List, Tuple

def bt(m:torch.Tensor):
    return m.transpose(dim0 = -2, dim1 = -1)

def btr(m:torch.Tensor):
    return m.diagonal(dim1 = -2, dim2 = -1).sum(-1)

def gaussian_kl(mu_i:torch.Tensor, mu:torch.Tensor, Ai:torch.Tensor, A:torch.Tensor):
    # calculate the decoupled KL between two gaussian distribution
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1) # B,n,1
    mu = mu.unsqueeze(-1)

    sigma_i = Ai @ bt(Ai) # B,n,n
    sigma = A @ bt(A) # B,n,n
    sigma_i_det = sigma_i.det()
    sigma_det = sigma.det()

    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-8)
    sigma_det = torch.clamp_min(sigma_det, 1e-8)

    sigma_i_inv = sigma_i.inverse() # B,n,n
    sigma_inv = sigma.inverse() # B,n,n

    inner_mu = ((mu - mu_i).transpose(-2,-1) @ sigma_i_inv @ (mu - mu_i)).squeeze() # (B,)
    inner_sigma = torch.log(sigma_det / sigma_i_det) - n + btr(sigma_inv @ sigma_i) #(B,)

    c_mu = 0.5 * torch.mean(inner_mu)
    c_sigma = 0.5 * torch.mean(inner_sigma)

    return c_mu, c_sigma, torch.mean(sigma_i_det), torch.mean(sigma_det)

def categorical_kl(p1 : torch.Tensor, p2 : torch.Tensor):
    # calculate KL between two categorical distributions
    p1 = torch.clamp_min(p1, 1e-8)
    p2 = torch.clamp_min(p2, 1e-8)
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim = -1))

class MPO(object):
    def __init__(
        self,
        env,
        device : Optional[str] = "cpu",
        dual_constraint : float = 0.1,
        kl_mean_constraint : float = 0.01,
        kl_var_constraint : float = 0.0001,
        kl_constraint : float =0.01,
        discount_factor : float =0.99,
        alpha_mean_scale : float =1.0,
        alpha_var_scale : float =100.0,
        alpha_scale : float =10.0,
        alpha_mean_max : float =0.1,
        alpha_var_max : float =10.0,
        alpha_max : float =1.0,
        sample_episode_num : int =30,
        sample_episode_maxstep : int =200,
        sample_action_num : int =64,
        batch_size : int =256,
        episode_rerun_num : int =3,
        mstep_iteration_num : int =5,
        evaluate_period : int =10,
        evaluate_episode_num : int =100,
        evaluate_episode_maxstep : int = 200,
        hidden_dim : int = 256,
        lr : float = 1e-3
        ):

        self.device = device
        self.env = env

        if self.env.action_space.dtype == np.float32:
            self.continuous_action_space = True
            self.da = env.action_space.shape[0]
        else:
            self.continuous_action_space = False
            self.da = env.action_space.n
        
        self.ds = env.observation_space.shape[0]

        self.eps_dual = dual_constraint
        self.eps_kl_mu = kl_mean_constraint
        self.eps_kl_sigma = kl_var_constraint
        self.eps_kl = kl_constraint

        self.gamma = discount_factor

        self.alpha_mu_scale = alpha_mean_scale
        self.alpha_sigma_scale = alpha_var_scale
        self.alpha_scale = alpha_scale
        self.alpha_mu_max = alpha_mean_max
        self.alpha_sigma_max = alpha_var_max
        self.alpha_max = alpha_max

        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxstep = sample_episode_maxstep
        self.sample_action_num = sample_action_num

        self.batch_size = batch_size
        
        self.episode_rerun_num = episode_rerun_num
        self.mstep_iteration_num = mstep_iteration_num
        
        self.evaluate_period = evaluate_period
        self.evaluate_episode_num = evaluate_episode_num
        self.evaluate_episode_maxstep = evaluate_episode_maxstep

        if not self.continuous_action_space:
            self.A_eye = torch.eye(self.da).to(self.device)

        if self.continuous_action_space:
            self.actor = Actor(env, mode = "continuous", hidden_dim = hidden_dim)
            self.critic = Critic(env, mode = "coutinuous", hidden_dim = hidden_dim)
            self.target_actor = Actor(env, mode = "continuous", hidden_dim = hidden_dim)
            self.target_critic = Critic(env, mode = "coutinuous", hidden_dim = hidden_dim)
        else:
            self.actor = Actor(env, mode = "discrete", hidden_dim = hidden_dim)
            self.critic = Critic(env, mode = "discrete", hidden_dim = hidden_dim)
            self.target_actor = Actor(env, mode = "discrete", hidden_dim = hidden_dim)
            self.target_critic = Critic(env, mode = "discrete", hidden_dim = hidden_dim)

        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

        self.__update_param(set_requires_grad=False)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr = lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr = lr)

        self.norm_loss_q = nn.SmoothL1Loss()

        self.eta = np.random.rand()
        self.alpha_mu = 0.0
        self.alpha_sigma = 0.0
        self.alpha = 0.0

        self.replaybuffer = ReplayBuffer()

        self.max_return_eval = -np.inf
        self.iteration = 1 
        self.render = False

    def load_model(self, path : str):
        check_point = torch.load(path)
        self.iteration = check_point['iteration']

        self.critic.load_state_dict(check_point['critic_state_dict'])
        self.target_critic.load_state_dict(check_point['target_critic_state_dict'])
        self.actor.load_state_dict(check_point['actor_state_dict'])
        self.target_actor.load_state_dict(check_point['target_actor_state_dict'])

        self.critic_optimizer.load_state_dict(check_point['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(check_point['actor_optim_state_dict'])

        self.critic.train()
        self.target_critic.train()

        self.actor.train()
        self.target_actor.train()

    def save_model(self, path : str):
        data = {
            'iteration' : self.iteration,
            'actor_state_dict':self.actor.state_dict(),
            'critic_state_dict':self.critic.state_dict(),
            'target_actor_state_dict':self.target_actor.state_dict(),
            'target_critic_state_dict':self.target_critic.state_dict(),
            'actor_optim_state_dict':self.actor_optimizer.state_dict(),
            'critic_optim_state_dict':self.critic_optimizer.state_dict()
        }   
        torch.save(data, path)

    def __sample_trajectory_worker(self, i : int):
        buff = []
        state = self.env.reset()
        for steps in range(self.sample_episode_maxstep):
            action = self.target_actor.action(
                torch.from_numpy(state).to(self.device)
            ).cpu().numpy()

            next_state, reward, done, _ = self.env.step(action)
            buff.append((state, action, next_state, reward))

            if self.render and i == 0:
                self.env.render()
            
            if done:
                break
            else:
                state = next_state
        return buff
    
    def __sample_trajectory(self, sample_episode_num):
        self.replaybuffer.clear()
        episodes = [self.__sample_trajectory_worker(i) for i in range(sample_episode_num)]
        self.replaybuffer.store_episodes(episodes)

    def __evaluate(self):
        with torch.no_grad():
            total_rewards = []
            for epi in range(self.evaluate_episode_num):
                total_reward = 0.0
                state = self.env.reset()

                for s in range(self.evaluate_episode_maxstep):
                    state = torch.from_numpy(state).to(self.device)
                    action = self.target_actor.action(
                        state
                    )
                    action = action.cpu().numpy()
                    
                    state, reward, done, _ = self.env.step(action)
                    total_reward += reward

                    if done:
                        break
                
                total_rewards.append(total_reward)
            
        return np.mean(total_rewards)

    def __update_critic_td(
        self, 
        state_batch : torch.Tensor,
        action_batch : torch.Tensor,
        next_state_batch : torch.Tensor,
        reward_batch : torch.Tensor,
        sample_num : int = 64
        ):
        '''
        :param state_batch: (B, ds)
        :param action_batch: (B, da) or (B,)
        :param next_state_batch: (B, ds)
        :param reward_batch: (B,)
        :param sample_num:
        :return: loss : (B,), y : (B, )
        '''

        B = state_batch.size(0)
        ds = self.ds
        da = self.da

        with torch.no_grad():
            r = reward_batch # (B,)

            if self.continuous_action_space:
                pi_mu, pi_a = self.target_actor(next_state_batch)
                pi = MultivariateNormal(pi_mu, scale_tril = pi_a) # (b,)
                sampled_next_actions = pi.sample((sample_num,)).transpose(0, 1) #(B, smaple_num, da)
                expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1) # (B,sample_num, ds)
                expected_next_q = self.target_critic(
                    expanded_next_states.reshape(-1,ds), #(B * sample_num, ds)
                    sampled_next_actions.reshape(-1,da) #(B * smaple_num, da)
                ).reshape(B, sample_num).mean(dim = 1) # (B, )
            else:
                pi_p = self.target_actor(next_state_batch)
                pi = Categorical(probs = pi_p)
                pi_prob = pi.expand((da, B)).log_prob(
                    torch.arange(da)[..., None].expand(-1,B).to(self.device) #(da,B)
                ).exp().transpose(0,1)
                sampled_next_actions = self.A_eye[None, ...].expand(B,-1,-1) #(B,da,da)
                expanded_next_states = next_state_batch[:,None,:].expand(-1,da,-1) #(B,da,ds)
                expected_next_q = (
                    self.target_critic(
                        expanded_next_states.reshape(-1,ds), # (B * da, ds)
                        sampled_next_actions.reshape(-1,da) # (B * da, da)
                    ).reshape(B,da) * pi_prob
                ).sum(dim = -1) #(B,)

            y = r + self.gamma * expected_next_q
        
        self.critic_optimizer.zero_grad()

        if self.continuous_action_space:
            t = self.critic(
                state_batch,
                action_batch
            ).squeeze()
        else:
            t = self.critic(
                state_batch,
                self.A_eye[action_batch.long()]
            ).squeeze(-1)

        loss = self.norm_loss_q(y,t)
        loss.backward()
        self.critic_optimizer.step()

        return loss, y


    def __update_param(self, set_requires_grad : Optional[bool] = None):
        # set target parameters to trained parameter
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            if set_requires_grad is not None:
                target_param.requires_grad = set_requires_grad

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            if set_requires_grad is not None:
                target_param.requires_grad = set_requires_grad
    

    def train(self, iteration_num : int = 1024, model_save_period : int = 16, render = False, save_dir : str = "./results/log/model"):
        
        self.render =render
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for iter in tqdm(range(self.iteration, iteration_num + 1), desc = "MPO algorithm training process"):
            self.__sample_trajectory(self.sample_episode_num)
            buffer_size = len(self.replaybuffer)

            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []
            
            max_kl_mu = []
            max_kl_sigma = []
            max_kl = []
            mean_sigma_det = []

            for r in range(self.episode_rerun_num):
                total_indices = range(buffer_size)
                batch_sampler = BatchSampler(SubsetRandomSampler(total_indices), batch_size = self.batch_size, drop_last = True)

                for indices in batch_sampler:
                    K = len(indices) # the sample number of states
                    N = self.sample_action_num # the sample number of actions per state

                    ds = self.ds
                    da = self.da

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replaybuffer[index] for index in indices]
                    )
                    # state_batch : (K,ds)
                    # action_batch : (K,da) or (K,)
                    # next_state_batch : (K,ds)
                    # reward_batch : (K,)
                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(self.device)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(self.device)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)
                    
                    # policy evaluation : critic optimization
                    loss_q, q = self.__update_critic_td(
                        state_batch = state_batch,
                        action_batch = action_batch,
                        next_state_batch = next_state_batch,
                        reward_batch = reward_batch,
                        sample_num = self.sample_action_num
                    )

                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # E-step of Policy Improvement : actor optimization step 1
                    with torch.no_grad():
                        if self.continuous_action_space:
                            b_mu, b_a = self.target_actor(state_batch) # mean, cholesky
                            b = MultivariateNormal(b_mu, scale_tril=b_a)
                            sampled_actions = b.sample((N,)) # N,K,da
                            expanded_states = state_batch[None, ...].expand(N,-1,-1) # N, K, ds
                            target_q = self.target_critic(
                                expanded_states.reshape(-1,ds), # N * K, ds
                                sampled_actions.reshape(-1, da) # N * K, da
                            )
                            target_q_np = target_q.cpu().transpose(0,1).numpy()
                        else:
                            actions = torch.arange(da)[..., None].expand(da, K).to(self.device)
                            b_p = self.target_actor(state_batch) # K,da
                            b = Categorical(probs = b_p)
                            b_prob = b.expand((da,K)).log_prob(actions).exp()
                            expanded_actions = self.A_eye[None, ...].expand(K, -1, -1) # K,da,da
                            expanded_states = state_batch.reshape(K,1,ds).expand((K,da,ds)) # K,da,ds
                            target_q = (
                                self.target_critic(
                                    expanded_states.reshape(-1, ds), # K * da, ds
                                    expanded_actions.reshape(-1, da) # K * da, da
                                ).reshape(K,da) # (K,da)
                            ).transpose(0,1) # (da, K)
                            b_prob_np = b_prob.cpu().transpose(0,1).numpy()
                            target_q_np = target_q.cpu().transpose(0,1).numpy()
                    
                    if self.continuous_action_space:
                        def dual(eta):
                            return self.dual_function(eta, target_q_np)
                    else:
                        def dual(eta):
                            return self.dual_function(eta, target_q_np, b_prob_np)

                    # Finding action weights : step 2
                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.eta]), method = 'SLSQP', bounds = bounds)
                    self.eta = res.x[0]

                    q_ij = torch.softmax(target_q / self.eta, dim = 0) # (N,K) or (da,K)


                    # M-step of policy improvement

                    for _ in range(self.mstep_iteration_num):
                        if self.continuous_action_space:
                            mu, a = self.actor(state_batch) # mean, cholesky

                            # question : why two matrix b_a and a are changed?
                            pi1 = MultivariateNormal(loc = mu, scale_tril=b_a)
                            pi2 = MultivariateNormal(loc = b_mu, scale_tril=a)

                            loss_p = torch.mean(
                                q_ij * (
                                    pi1.expand((N,K)).log_prob(sampled_actions) + 
                                    pi2.expand((N,K)).log_prob(sampled_actions)
                                )
                            )

                            mean_loss_p.append((-loss_p).item())

                            kl_mu, kl_sigma, sigma_i_det, sigma_det = gaussian_kl(
                                mu_i = b_mu,
                                mu = mu,
                                Ai = b_a,
                                A = a
                            )

                            max_kl_mu.append(kl_mu.item())
                            max_kl_sigma.append(kl_sigma.item())
                            mean_sigma_det.append(sigma_det.item())

                            if np.isnan(kl_mu.item()):
                                raise RuntimeError('kl_mu is nan')
                            if np.isnan(kl_sigma.item()):
                                raise RuntimeError('kl_sigma is nan')
                            
                            self.alpha_mu -= self.alpha_mu_scale * (self.eps_kl_mu - kl_mu).detach().item()
                            self.alpha_sigma -= self.alpha_sigma_scale * (self.eps_kl_sigma - kl_sigma).detach().item()

                            self.alpha_mu = np.clip(0, self.alpha_mu, self.alpha_mu_max)
                            self.alpha_sigma = np.clip(0, self.alpha_sigma, self.alpha_sigma_max)

                            self.actor_optimizer.zero_grad()

                            loss_l = -(
                                loss_p + 
                                self.alpha_mu * (self.eps_kl_mu - kl_mu) + 
                                self.alpha_sigma * (self.eps_kl_sigma - kl_sigma)
                            )

                            mean_loss_l.append(loss_l.item())
                            loss_l.backward()
                            
                            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
                            self.actor_optimizer.step()

                        else:
                            pi_p = self.actor(state_batch)
                            pi = Categorical(probs = pi_p) # (K,)
                            loss_p = torch.mean(
                                q_ij * pi.expand((da,K)).log_prob(actions)
                            )

                            mean_loss_p.append((-loss_p).item())

                            kl = categorical_kl(
                                p1 = pi_p,
                                p2 = b_p
                            )
                            max_kl.append(kl.item())

                            if np.isnan(kl.item()):
                                raise RuntimeError('kl is nan')
                            
                            self.alpha -= self.alpha_scale * (self.eps_kl - kl).detach().item()
                            self.alpha = np.clip(self.alpha, 0.0, self.alpha_max)

                            self.actor_optimizer.zero_grad()

                            loss_l = -(loss_p + self.alpha * (self.eps_kl - kl))
                            mean_loss_l.append(loss_l.item())
                            loss_l.backward()
                            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
                            self.actor_optimizer.step()
        
            self.__update_param()

            return_eval = None

            if iter % self.evaluate_period == 0:
                self.actor.eval()
                return_eval = self.__evaluate()
                self.actor.train()
                self.max_return_eval = max(self.max_return_eval, return_eval)
            
            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)

            if self.continuous_action_space:
                max_kl_mu = np.max(max_kl_mu)
                max_kl_sigma = np.max(max_kl_sigma)
                mean_sigma_det = np.mean(mean_sigma_det)
            else:
                max_kl = np.max(max_kl)
            
            print("iteration : ", iter)

            if iter % self.evaluate_period == 0:
                print(" max_return_eval : ", self.max_return_eval)
                print(" return_eval : ", return_eval)
            print(" mean return : ", mean_return)
            print(" mean_reward : ", mean_reward)
            print(" mean_loss_q : ", mean_loss_q)
            print(" mean_loss_p : ", mean_loss_p)
            print(" mean_loss_l : ", mean_loss_l)
            print(" mean_est_q : ", mean_est_q)
            print(" eta : ", self.eta)

            if self.continuous_action_space:
                print(" max_kl_mu : ", max_kl_mu)
                print(" max_kl_sigma : ", max_kl_sigma)
                print(" mean_sigma_det : ", mean_sigma_det)
                print(" alpha_mu : ", self.alpha_mu)
                print(" alpha_sigma : ", self.alpha_sigma)
            else:
                print(" max_kl : ", max_kl)
                print(" alpha : ", self.alpha)

    def dual_function(self, eta, target_q_np, b_prob_np = None):
        max_q = np.max(target_q_np, 1)
        if self.continuous_action_space:
            return eta * self.eps_dual + np.mean(max_q) + eta * np.mean(np.log(np.mean(np.exp((target_q_np - max_q[:,None]) / eta), axis = 1)))
        else:
            return eta * self.eps_dual + np.mean(max_q) + eta * np.mean(np.log(np.sum(b_prob_np * np.exp((target_q_np - max_q[:,None]) / eta ), axis = 1)))
