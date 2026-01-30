from utils.networks import ActorNetwork, CriticNetworkMADDPG
from utils.buffer import BufferMADDPG
from utils.noise import OUNoise
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys
import os


class Agente:
    def __init__(self, id, state_dim, action_dim, max_action, num_agents,
                 device="cpu", actor_lr=0.0001, critic_lr=0.0002):
        
        self.id = id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_agents = num_agents
        self.device = device

        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetworkMADDPG(state_dim, action_dim, num_agents).to(self.device)

        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = CriticNetworkMADDPG(state_dim, action_dim, num_agents).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
        # Parâmetros do ruído OU
        ou_theta=0.15 
        ou_sigma=0.20
        ou_dt=1.0


        self.ou_noise = OUNoise(
            action_dim=action_dim, theta=ou_theta,
            sigma=ou_sigma,
            dt=ou_dt)
        
    def select_action(self, state, noise=0.0, deterministic=False):
        """
        Retorna ação a partir de um estado. Suporta 1D ou 2D.
        Adiciona ruído gaussiano se deterministic=False.
        """
        self.actor.eval()
        with torch.no_grad():

            if not torch.is_tensor(state):
                state = torch.FloatTensor(state)

            # garante formato [batch, state_dim]
            if state.dim() == 1:
                state = state.unsqueeze(0)

            state_t = state.to(self.device)
            action = self.actor(state_t)
            action = action.cpu().numpy().squeeze()  # remove batch

        self.actor.train()

        # aplica ruído só quando NÃO é determinístico
        if not deterministic:
            #action = action + np.random.normal(0, noise, size=self.action_dim)
            action = action + self.ou_noise.sample()
        # limita ação ao intervalo permitido
        #Normal
        #action = np.clip(action, -self.max_action, self.max_action)

        #Para o PettingZoo
        action = np.clip(action, 0.0, 1)
        action = action.astype(np.float32)


        return action
    
    def select_action_target(self, state):
        """
        Retorna ação a partir de um estado usando a rede alvo do ator.
        state: np.array  ou torch tensor (1D ou 2D batch)
        """
        self.actor_target.eval()
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state)
            # garante formato [batch, state_dim]
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state_t = state.to(self.device)
            action = self.actor_target(state_t)
            action = action.cpu().numpy().squeeze()
        
        self.actor_target.train()

        return action

