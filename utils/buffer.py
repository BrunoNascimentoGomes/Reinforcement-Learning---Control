import tensorflow as tf
import numpy as np
import torch
import os
import torch.nn.functional as F


class Buffer:
    def __init__(self, buffer_capacity=10000, batch_size=64, state_dim=4, action_dim=1):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((buffer_capacity, state_dim))
        self.action_buffer = np.zeros((buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, state_dim))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = np.array(obs_tuple[0]).reshape(self.state_buffer.shape[1])
        self.next_state_buffer[index] = np.array(obs_tuple[3]).reshape(self.next_state_buffer.shape[1])
        self.action_buffer[index] = np.array(obs_tuple[1]).reshape(self.action_buffer.shape[1])
        self.reward_buffer[index] = np.array(obs_tuple[2]).reshape(1)


        self.buffer_counter += 1


    def sample_batch(self):
        """Retorna um batch aleatório de transições."""
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size, replace = False)

        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]

        return (
        torch.tensor(state_batch, dtype=torch.float32),
        torch.tensor(action_batch, dtype=torch.float32),
        torch.tensor(reward_batch, dtype=torch.float32),
        torch.tensor(next_state_batch, dtype=torch.float32)
        )

    def reset(self):
        """Limpa o buffer."""
        self.buffer_counter = 0
        self.state_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.next_state_buffer.fill(0)

class BufferMADDPG:
    def __init__(self, buffer_capacity=10000, batch_size=64,
                 state_dim=4, action_dim=1, num_agents=2):

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_buffer = np.zeros((buffer_capacity, num_agents, state_dim))
        self.action_buffer = np.zeros((buffer_capacity, num_agents, action_dim))
        self.reward_buffer = np.zeros((buffer_capacity, num_agents, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_agents, state_dim))


    def record(self, obs_tuple):
 

        idx = self.buffer_counter % self.buffer_capacity

        self.state_buffer[idx] = np.array(obs_tuple[0]).reshape(self.num_agents, self.state_dim)
        self.action_buffer[idx] = np.array(obs_tuple[1]).reshape(self.num_agents, self.action_dim)
        self.reward_buffer[idx] = np.array(obs_tuple[2]).reshape(self.num_agents, 1)
        self.next_state_buffer[idx] = np.array(obs_tuple[3]).reshape(self.num_agents, self.state_dim)

        self.buffer_counter += 1


    def sample_batch(self):
        max_samples = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(max_samples, self.batch_size, replace=False)

        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch


    def reset(self):
        self.buffer_counter = 0
        self.state_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.next_state_buffer.fill(0)
    

###########################################################################

import numpy as np
import torch

class BufferMADDPGPrio:
    def __init__(self, buffer_capacity=10000, batch_size=64,
                 state_dim=4, action_dim=1, num_agents=2,
                 alpha=0.6, beta=0.6):

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.alpha = alpha   # controle de quanta priorização usar
        self.beta = beta     # fator de importância IS
        self.eps = 1e-6      # para evitar prioridade zero

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # PRIORIDADES (1-D)
        self.priority = np.zeros(buffer_capacity)

        # BUFFERS
        self.state_buffer = np.zeros((buffer_capacity, num_agents, state_dim))
        self.action_buffer = np.zeros((buffer_capacity, num_agents, action_dim))
        self.reward_buffer = np.zeros((buffer_capacity, num_agents, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_agents, state_dim))


    # ---------------------------
    # GRAVA UMA EXPERIÊNCIA
    # ---------------------------
    def record(self, obs_tuple):

        idx = self.buffer_counter % self.buffer_capacity

        self.state_buffer[idx] = np.array(obs_tuple[0])
        self.action_buffer[idx] = np.array(obs_tuple[1])
        self.reward_buffer[idx] = np.array(obs_tuple[2])
        self.next_state_buffer[idx] = np.array(obs_tuple[3])

        # prioridade inicial = maior prioridade do buffer
        max_prio = self.priority.max() if self.buffer_counter > 0 else 1.0
        self.priority[idx] = max_prio

        self.buffer_counter += 1


    # ---------------------------
    # AMOSTRAGEM PRIORIZADA (RANK-BASED)
    # ---------------------------
    def sample_batch(self):
        max_samples = min(self.buffer_counter, self.buffer_capacity)

        # Ranking: maior prioridade → rank=1
        sorted_idx = np.argsort(-self.priority[:max_samples])
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, max_samples+1)

        # D_j = 1 / rank(j)
        D = 1.0 / ranks

        # Probabilidades normalizadas
        P = D ** self.alpha
        P /= P.sum()

        self.p = P

        # Amostragem segundo P
        batch_indices = np.random.choice(max_samples, self.batch_size, p=P)

        # Importance Sampling Weights
        weights = (1/(max_samples * P[batch_indices])) ** (self.beta)
        #weights /= weights.max()  # normalização

        # Convertendo para tensores
        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, batch_indices, weights


    # ---------------------------
    # ATUALIZA AS PRIORIDADES USANDO TD-ERROR
    # ---------------------------
    def update_priority(self, indices, td_errors):
        # prioridade = |TD-error| + eps
        new_prios = np.abs(td_errors) + self.eps
        for idx, prio in zip(indices, new_prios):
            self.priority[idx] = prio


    # ---------------------------
    # REINICIAR BUFFER
    # ---------------------------
    def reset(self):
        self.buffer_counter = 0
        self.priority.fill(0)
        self.state_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.next_state_buffer.fill(0)

        




        
####################################################################



    

if __name__ == "__main__":
    import random
    buffer1 = BufferMADDPGPrio(state_dim=3, batch_size=2, action_dim=1, num_agents=2, buffer_capacity=10)
    state1 = []
    state2 = []
    next_state1 = []
    next_state2 = []
    reward1 = 0
    reward2 = 0
    action1 = 0
    action2 = 0
    for i in range(5):
        state1 = np.random.rand(3,1)
        action1 = random.random()
        reward1 = random.random()
        next_state1 = np.random.rand(3,1)

        state2 = np.random.rand(3,1)
        action2 = random.random()
        reward2 = random.random()
        next_state2 = np.random.rand(3,1)

        state = np.array([state1, state2])
        action = np.array([action1, action2])
        reward = np.array([reward1, reward2])
        next_state = np.array([next_state1, next_state2])


        obs = (state, action, reward, next_state)
        buffer1.record(obs)
    batch = buffer1.sample_batch()
    print("Batch_State = ", batch[0])
    print("Batch_Action = ", batch[1])

    print("Batch_Reward = ", batch[2])
    print("Batch_NextSate = ", batch[3])
    print('------------------------------')
    print("Batch Action1: ",buffer1.action_buffer)