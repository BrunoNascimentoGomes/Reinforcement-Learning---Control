from utils.networks import ActorNetwork, CriticNetwork
from utils.buffer import Buffer, Buffer_Done
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action,
                 buffer_capacity=1000, batch_size=64,
                 actor_lr=0.001, critic_lr=0.002, gamma=0.99, 
                 tau=0.05, device="cpu"):
        self.device = device
        
        #Inicializando as redes e o buffer de replay
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        
        self.replay_buffer = Buffer(state_dim=state_dim, action_dim=action_dim,
                                    buffer_capacity=buffer_capacity, batch_size=batch_size)
        
        #Hiperparâmetros
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        #Criação das targets e copia dos pesos
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    

    def select_action(self, state, noise=0.0, deterministic=False):
        """
        Retorna ação a partir de um estado (array ou tensor). Adiciona ruído Gaussiano se noise>0.
        state: np.array  ou torch tensor (1D ou 2D batch)
        """
        self.actor.eval()
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).unsqueeze(0)  # adiciona batch dim
            state_t = state.to(self.device)
            action = self.actor(state_t)
            action = action.cpu().numpy().squeeze()
        
        if deterministic:
            action = action
        else:
            action = action + np.random.normal(0, noise, size=action.shape)

        # garantir limites
        if np.isscalar(self.max_action):
            low = -abs(self.max_action)
            high = abs(self.max_action)
            action = np.clip(action, low, high)
        else:
            # se max_action for vetor
            action = np.clip(action, -np.abs(self.max_action), np.abs(self.max_action))

        return action
    
    def train(self):
        # Sample a batch of transitions from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample_batch()

        # Convert to torch tensors
        '''
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        '''
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device) 
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        
        # Compute target actions and Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            target_Q = self.critic_target(next_state_batch, next_actions)
            target_Q = reward_batch + (self.gamma * target_Q)

        # Get current Q estimates
        current_Q = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def save(self, path_prefix):
        """
        Salva pesos das redes e otimizadores (prefixo de caminho).
        Ex: agent.save("checkpoints/ddpg")
        irá salvar:
         - checkpoints/ddpg_actor.pth
         - checkpoints/ddpg_critic.pth
         - checkpoints/ddpg_actor_optim.pth
         - checkpoints/ddpg_critic_optim.pth
        """
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{path_prefix}_actor_optim.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{path_prefix}_critic_optim.pth")
    

class DDPGAgent_Done:
    def __init__(self, state_dim, action_dim, max_action,
                 buffer_capacity=1000, batch_size=64,
                 actor_lr=0.001, critic_lr=0.002, gamma=0.99, 
                 tau=0.05, device="cpu"):
        self.device = device
        
        #Inicializando as redes e o buffer de replay
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        
        self.replay_buffer = Buffer_Done(state_dim=state_dim, action_dim=action_dim,
                                    buffer_capacity=buffer_capacity, batch_size=batch_size)
        
        #Hiperparâmetros
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        #Criação das targets e copia dos pesos
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    

    def select_action(self, state, noise=0.0, deterministic=False):
        """
        Retorna ação a partir de um estado (array ou tensor). Adiciona ruído Gaussiano se noise>0.
        state: np.array  ou torch tensor (1D ou 2D batch)
"""
        self.actor.eval()
        with torch.no_grad():
            # 1. Garante que é tensor
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state)
            
            # 2. Se for 1D (ex: [3]), transforma em 2D (ex: [1, 3]) para o BatchNorm aceitar
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            state_t = state.to(self.device)
            action = self.actor(state_t)
            action = action.cpu().numpy().squeeze()
        
        if deterministic:
            action = action
        else:
            action = action + np.random.normal(0, noise, size=action.shape)

        # garantir limites
        if np.isscalar(self.max_action):
            low = -abs(self.max_action)
            high = abs(self.max_action)
            action = np.clip(action, low, high)
        else:
            action = np.clip(action, -np.abs(self.max_action), np.abs(self.max_action))

        return action
    
    def train(self):
        # Sample a batch of transitions from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_batch()
        self.actor.train()
        self.critic.train()
        # Convert to torch tensors
        '''
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        '''
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device) 
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        # Compute target actions and Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            target_Q = self.critic_target(next_state_batch, next_actions)
            target_Q = reward_batch + ((1 - done_batch) * self.gamma * target_Q)

        # Get current Q estimates
        current_Q = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def save(self, path_prefix):
        """
        Salva pesos das redes e otimizadores (prefixo de caminho).
        Ex: agent.save("checkpoints/ddpg")
        irá salvar:
         - checkpoints/ddpg_actor.pth
         - checkpoints/ddpg_critic.pth
         - checkpoints/ddpg_actor_optim.pth
         - checkpoints/ddpg_critic_optim.pth
        """
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{path_prefix}_actor_optim.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{path_prefix}_critic_optim.pth")


if __name__ == "__main__":
    state_dim = 3
    action_dim = 1
    max_action = 1
    agent = DDPGAgent(state_dim, action_dim, max_action, device="cpu")
    agent.save("ddpg_test")