from utils.networks import ActorNetwork, CriticNetworkMADDPG
from utils.buffer import BufferMADDPG
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class MADDPGAgent:
    def __init__(self, name,agent_id,state_dim, action_dim, num_agents,max_action,
                 buffer, batch_size=64,
                 actor_lr=0.0001, critic_lr=0.0002, gamma=0.99, 
                 tau=0.005, device="cpu"):
        self.device = device
        self.agent_id = agent_id
        self.name = name
        
        #Inicializando as redes e o buffer de replay
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetworkMADDPG(state_dim, action_dim, num_agents).to(self.device)
        
        self.replay_buffer = buffer
        
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
        self.critic_target = CriticNetworkMADDPG(state_dim, action_dim, num_agents).to(self.device)
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
        
        self.actor.train()
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
    
    def train(self, all_agents):
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
        

        next_actions_target_batch = torch.zeros_like(action_batch)
        
        # Critic UPDATE
        # Compute target actions and Q-values
        #print("Iniciando atualização do Crítico do agente", self.name)
        with torch.no_grad():
            for i, agent_i in enumerate(all_agents):
                #print("Estado do agente {}: ".format(i), state_batch[:, i, :])
                agent_next_state = next_state_batch[:, i, :]
                #print(f"Próximo estado do agente {i}: ", agent_next_state)
                agent_next_action = agent_i.actor_target(agent_next_state)


                next_actions_target_batch[:, i, :] = agent_next_action
                #print(f"Ação target do agente {i}: ", agent_next_action)

            next_state_batch_flat = next_state_batch.view(self.batch_size, -1)
            next_actions_target_batch_flat = next_actions_target_batch.view(self.batch_size, -1)
            
            target_Q = self.critic_target(next_state_batch_flat, next_actions_target_batch_flat)
            reward_i = reward_batch[:, self.agent_id, :]
            #print(f"Recompensa do agente {self.name}: ",reward_i)
            target_Q = reward_i + (self.gamma * target_Q)

        
        state_batch_flat = state_batch.view(self.batch_size, -1)
        action_batch_flat = action_batch.view(self.batch_size, -1)
        #print("Estado batch achatado: ", state_batch_flat.shape)
        #print("Ação batch achatada: ", action_batch_flat.shape)
        
        # Get current Q estimates
        # self.critic é o Crítico 'i'
        current_Q = self.critic(state_batch_flat, action_batch_flat)
        #print("Current Q calculado para o agente", current_Q)
        #print("Target Q para o agente", self.name, ": ", target_Q)

        self.td_errors = target_Q - current_Q
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #ACTOR UPDATE
        # Compute actor loss
# --- Compute actor loss (para o 'self' agent) ---
        
        # 1. Calcular as ações preditas de TODOS os agentes usando seus atores LOCAIS
        # (Precisamos que os gradientes fluam para o 'self.actor')
        
        # Criamos uma lista de ações preditas [a_1, a_2, ..., a_N]

        actions_pred_list = []
        for i, agent_i in enumerate(all_agents):
            agent_state_i = state_batch[:, i, :]
            if i == self.agent_id:
            # precisamos de gradientes apenas para o self
                action_pred_i = agent_i.actor(agent_state_i)
            else:
                with torch.no_grad():
                    action_pred_i = agent_i.actor(agent_state_i)
            actions_pred_list.append(action_pred_i)


        # 2. Criar a "visão" do crítico para este agente ('self')
        #    Queremos ∇_θi Q_i(x, a_1, ..., a_N)

        # 3. "Achatamos" a lista para o formato que o crítico espera: [B, N * A_dim]
        actions_pred_flat_for_self = torch.cat(actions_pred_list, dim=1)
        #actions_pred_flat_for_self = actions_pred_detached_list.view(self.batch_size, -1)

        # 4. Calcular a perda do ator
        # Queremos maximizar Q, então minimizamos -Q
        actor_loss = -self.critic(state_batch_flat, actions_pred_flat_for_self).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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
    print("--- Iniciando Teste de Fumaça (Smoke Test) do MADDPGAgent ---")

    # 1. Definir Hiperparâmetros
    STATE_DIM = 4
    ACTION_DIM = 1
    MAX_ACTION = 1.0
    NUM_AGENTS = 2
    BUFFER_CAPACITY = 10000
    BATCH_SIZE = 64
    DEVICE = "cpu"

    print(f"Agentes: {NUM_AGENTS}, Estado: {STATE_DIM}, Ação: {ACTION_DIM}, Batch: {BATCH_SIZE}")

    # 2. Criar o Buffer Compartilhado
    # (A classe BufferMADDPG já deve estar importada do 'utils.buffer')
    try:
        shared_buffer = BufferMADDPG(buffer_capacity=BUFFER_CAPACITY, 
                                     batch_size=BATCH_SIZE,
                                     state_dim=STATE_DIM, 
                                     action_dim=ACTION_DIM, 
                                     num_agents=NUM_AGENTS)
        print("Buffer compartilhado criado.")
    except NameError:
        print("\n[ERRO] A classe 'BufferMADDPG' não foi encontrada.")
        print("Certifique-se de que está importada de 'utils.buffer'.")
        sys.exit(1) # Sai se o buffer não puder ser criado

    # 3. Criar a lista de Agentes
    all_agents = []
    for i in range(NUM_AGENTS):
        agent = MADDPGAgent(agent_id=i,
                            state_dim=STATE_DIM,
                            action_dim=ACTION_DIM,
                            num_agents=NUM_AGENTS,
                            max_action=MAX_ACTION,
                            shared_buffer=shared_buffer, # Passa o MESMO buffer
                            batch_size=BATCH_SIZE,
                            device=DEVICE)
        all_agents.append(agent)
        print(f"Agente {i} criado.")

    # 4. Popular o Buffer com dados falsos (mínimo BATCH_SIZE)
    # O método .train() falhará se o buffer tiver menos de BATCH_SIZE amostras
    print(f"Populando o buffer com {BATCH_SIZE} amostras falsas...")
    for _ in range(BATCH_SIZE):
        # (states, actions, rewards, next_states)
        states = np.random.rand(NUM_AGENTS, STATE_DIM)
        actions = np.random.rand(NUM_AGENTS, ACTION_DIM)
        rewards = np.random.rand(NUM_AGENTS, 1)
        next_states = np.random.rand(NUM_AGENTS, STATE_DIM)
        
        # obs_tuple = (states, actions, rewards, next_states)
        shared_buffer.record((states, actions, rewards, next_states))
    
    print(f"Buffer populado. Contagem: {shared_buffer.buffer_counter}")

    try:
        print("\nExecutando all_agents[0].train(all_agents)...")
        all_agents[0].train(all_agents)
        print("[SUCESSO] O passo de 'train' (Critic e Actor Update) foi executado sem erros.")
    except Exception as e:
        print(f"\n[FALHA] Ocorreu um erro durante o 'train': {e}")
        import traceback
        traceback.print_exc()

    # 6. Testar o salvamento
    try:
        print("\nTestando salvamento do Agente 0...")
        prefix = "test_agent_0"
        all_agents[0].save(prefix)

        
        print("[SUCESSO] Salvamento e limpeza de arquivos OK.")
    except Exception as e:
        print(f"\n[FALHA] Ocorreu um erro durante o 'save': {e}")

    print("\n--- Teste de Fumaça Concluído ---")