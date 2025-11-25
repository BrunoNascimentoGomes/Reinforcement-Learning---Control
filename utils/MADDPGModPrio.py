from utils.agente import Agente
import torch
import torch.nn as nn
import numpy as np
import os


class MADDPGPrio:
    def __init__(self, num_agents, state_dim, action_dim, max_action,
                 buffer, actor_lr=0.0001, critic_lr=0.0002,
                 gamma=0.99, tau=0.005, device="cpu"):

        self.device = device
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = buffer
        self.batch_size = buffer.batch_size

        # criar agentes
        self.agents = []
        for i in range(num_agents):
            self.agents.append(
                Agente(i, state_dim, action_dim,
                       max_action, num_agents,
                       device=device,
                       actor_lr=actor_lr,
                       critic_lr=critic_lr)
            )

    # ---------------------------------------------------------
    # AÇÃO
    # ---------------------------------------------------------
    def select_action(self, states, noise=0.0, deterministic=False):
        actions = []
        for i, agent in enumerate(self.agents):
            a = agent.select_action(states[i], noise, deterministic)
            actions.append(np.array(a).reshape(self.action_dim))
        return np.array(actions)

    # ---------------------------------------------------------
    # TREINO
    # ---------------------------------------------------------
    def train(self):

        state_batch, action_batch, reward_batch, next_state_batch, indices, weights= \
            self.replay_buffer.sample_batch()

        state_batch = state_batch.to(self.device)               # [B, N, S]
        action_batch = action_batch.to(self.device)             # [B, N, A]
        reward_batch = reward_batch.to(self.device)             # [B, N, 1]
        next_state_batch = next_state_batch.to(self.device)     # [B, N, S]
        weights = weights.to(self.device).unsqueeze(1)

        B = state_batch.size(0)
        #print(B)

        # ---------------------------------------------------------
        # AÇÕES TARGET
        # ---------------------------------------------------------
        with torch.no_grad():
            next_actions = []
            for agent in self.agents:
                ns_i = next_state_batch[:, agent.id, :]         # [B, S]
                next_actions.append(agent.actor_target(ns_i))   # [B, A]

            next_actions = torch.stack(next_actions, dim=1)     # [B, N, A]

            next_states_flat = next_state_batch.view(B, -1)
            next_actions_flat = next_actions.view(B, -1)

        # ---------------------------------------------------------
        # ATUALIZAÇÃO POR AGENTE
        # ---------------------------------------------------------

        
        td_errors_list = []
         
        for agent in self.agents:
            agent_id = agent.id

            # ---------------- Critic ----------------
            with torch.no_grad():
                reward_i = reward_batch[:, agent_id, :]

                target_Q = agent.critic_target(next_states_flat,
                                               next_actions_flat)

                target_Q = reward_i + self.gamma * target_Q

            state_flat = state_batch.view(B, -1)
            action_flat = action_batch.view(B, -1)

            current_Q = agent.critic(state_flat, action_flat)

            td_error = current_Q - target_Q

            td_errors_list.append(td_error.abs())
            critic_loss = (weights * td_error.pow(2)).mean()


            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()



            # ---------------- Actor ----------------
            pred_actions = []

            for j, other_agent in enumerate(self.agents):
                s_j = state_batch[:, j, :]

                if j == agent_id:
                    a_j = other_agent.actor(s_j)
                else:
                    with torch.no_grad():
                        a_j = other_agent.actor(s_j)

                pred_actions.append(a_j)

            pred_actions_flat = torch.cat(pred_actions, dim=1)

            actor_loss = -agent.critic(state_flat,
                                       pred_actions_flat).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # ---------------- Soft Update ----------------
            with torch.no_grad():
                for p, tp in zip(agent.critic.parameters(),
                                 agent.critic_target.parameters()):
                    tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

                for p, tp in zip(agent.actor.parameters(),
                                 agent.actor_target.parameters()):
                    tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        
        td_errors_all = torch.stack(td_errors_list, dim=1)  # [B, N, 1]
        td_errors_mean = td_errors_all.mean(dim=1)
        td_errors_np = td_errors_mean.squeeze(1).cpu().detach().numpy()

        self.replay_buffer.update_priority(indices, td_errors_np)


        

    # ---------------------------------------------------------
    # SALVAR (por agente!)
    # ---------------------------------------------------------
    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)

        for agent in self.agents:
            torch.save(agent.actor.state_dict(),
                       f"{dir_path}/agent{agent.id}_actor.pth")

            torch.save(agent.critic.state_dict(),
                       f"{dir_path}/agent{agent.id}_critic.pth")

            torch.save(agent.actor_optimizer.state_dict(),
                       f"{dir_path}/agent{agent.id}_actor_optim.pth")

            torch.save(agent.critic_optimizer.state_dict(),
                       f"{dir_path}/agent{agent.id}_critic_optim.pth")
