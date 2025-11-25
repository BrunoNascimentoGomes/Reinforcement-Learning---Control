import torch
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from utils.buffer import BufferMADDPG
from utils.MADDPGMod import MADDPG # Importa sua classe MADDPG
import os
from tqdm import tqdm
import time # Para adicionar um pequeno atraso na renderização

# --- 1. Definição de Hiperparâmetros ---
# Ambiente
NUM_AGENTS = 3                         # Número padrão de agentes e landmarks no Simple Spread
MAX_CYCLES = 50                      # Passos máximos por episódio
EPISODES = 2000                       # Total de episódios para treino

# MADDPG
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.95
TAU = 0.0001
BUFFER_CAPACITY = 100000
BATCH_SIZE = 1024
LEARN_START = BATCH_SIZE * 5           # Começar o treino após um certo número de amostras
NOISE = 0.1                            # Ruído de exploração (OUNoise seria melhor, mas Gaussiano serve)
NOISE_DECAY = 0.9995

# Avaliação
TEST_EPISODES = 5                     # Número de episódios para avaliação visual

# Outros
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "maddpg_simple_spread_checkpoints"

# --- 2. Inicialização do Ambiente e Parâmetros ---
# Usamos 'None' para render_mode durante o treino.
env = simple_spread_v3.parallel_env(N=NUM_AGENTS, max_cycles=MAX_CYCLES, continuous_actions=True, render_mode=None)

AGENT_NAMES = env.possible_agents

n_agents = len(env.possible_agents)
state_dim = env.observation_space(env.possible_agents[0]).shape[0]
action_dim = env.action_space(env.possible_agents[0]).shape[0]
max_action = env.action_space(env.possible_agents[0]).high[0]

print(f"Número de Agentes: {n_agents}")
print(f"Dimensão do Estado Local (por agente): {state_dim}")
print(f"Dimensão da Ação (por agente): {action_dim}")
print(f"Ação Máxima: {max_action}")
print(f"Dispositivo: {DEVICE}")

# --- 3. Inicialização do MADDPG e Buffer ---
replay_buffer = BufferMADDPG(
    buffer_capacity=BUFFER_CAPACITY,
    batch_size=BATCH_SIZE,
    state_dim=state_dim,
    action_dim=action_dim,
    num_agents=n_agents
)

maddpg = MADDPG(
    num_agents=n_agents,
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    buffer=replay_buffer,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    gamma=GAMMA,
    tau=TAU,
    device=DEVICE
)

# --- 4. Laço Principal de Treino ---
## 🚀 Treino MADDPG
---
total_steps = 0
current_noise = NOISE

print("\nIniciando o Treino...")
for episode in tqdm(range(EPISODES)):
    # 4.1. Reset do Ambiente
    observations, infos = env.reset()
    
    current_states = np.array([observations[agent] for agent in AGENT_NAMES])
    
    episode_reward = 0
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        
        # 4.2. Seleção de Ação (Com Ruído de Exploração)
        actions_list = maddpg.select_action(current_states, noise=current_noise, deterministic=False)
        
        # 4.3. Conversão de Ações para o formato do Petting Zoo
        actions_dict = {
        agent_name: actions_list[i]
        for i, agent_name in enumerate(AGENT_NAMES)
        }
        
        # 4.4. Passo no Ambiente
        observations_new, rewards, terminations, truncations, infos = env.step(actions_dict)
        
        # 4.5. Prepara a Transição
        next_states = np.array([observations_new[agent] for agent in AGENT_NAMES])
        rewards_list = np.array([[rewards[agent_name]] for agent_name in AGENT_NAMES]) # Formato [N, 1]
        
        # 4.6. Grava no Buffer
        maddpg.replay_buffer.record((current_states, actions_list, rewards_list, next_states))

        # 4.7. Acumula Recompensa
        episode_reward += np.sum(rewards_list)

        # 4.8. Atualiza Estado e Flags
        current_states = next_states
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        total_steps += 1

        # 4.9. Treino
        if maddpg.replay_buffer.buffer_counter >= LEARN_START:
            maddpg.train()
    
    # 4.10. Decay do Ruído
    current_noise = max(0.01, current_noise * NOISE_DECAY)

    
    tqdm.write(f"Episódio: {episode}, Recompensa Total: {episode_reward:.2f}, Ruído: {current_noise:.4f}")

# --- Fim do Treino ---
env.close() # Fechar o ambiente de treino sem renderização
maddpg.save(SAVE_DIR)
print(f"Treino concluído. Modelos salvos em: {SAVE_DIR}")

# --- 5. Avaliação com Renderização ---
## 🤖 Avaliação Visual (Renderização)
---
print("\nIniciando Avaliação Visual...")

# Reabre o ambiente com render_mode='human'
# Nota: Você pode precisar fechar o ambiente de treino (env.close()) antes de criar um novo com render.
test_env = simple_spread_v3.parallel_env(N=NUM_AGENTS, max_cycles=MAX_CYCLES, continuous_actions=True, render_mode='human')
AGENT_NAMES_TEST = test_env.possible_agents

for episode in range(TEST_EPISODES):
    # 5.1. Reset do Ambiente de Teste
    observations, infos = test_env.reset()
    
    current_states = np.array([observations[agent] for agent in AGENT_NAMES_TEST])
    
    episode_reward = 0
    terminated = False
    truncated = False
    step_count = 0
    
    while not terminated and not truncated:
        
        # 5.2. Seleção de Ação (Determinística - Sem Ruído)
        # Passamos 0.0 de ruído e deterministic=True para usar a política treinada
        actions_list = maddpg.select_action(current_states, noise=0.0, deterministic=True)
        
        # 5.3. Conversão de Ações
        actions_dict = {
        agent_name: actions_list[i]
        for i, agent_name in enumerate(AGENT_NAMES_TEST)
        }
        
        # 5.4. Passo no Ambiente
        observations_new, rewards, terminations, truncations, infos = test_env.step(actions_dict)
        
        # 5.5. Atualiza Estado
        next_states = np.array([observations_new[agent] for agent in AGENT_NAMES_TEST])
        
        # 5.6. Acumula Recompensa
        episode_reward += np.sum(list(rewards.values()))
        
        # 5.7. Atualiza Estado e Flags
        current_states = next_states
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        step_count += 1
        
        # Opcional: Adiciona um pequeno delay para melhor visualização
        time.sleep(0.05) 

    print(f"Avaliação - Episódio: {episode}, Recompensa Total: {episode_reward:.2f}, Passos: {step_count}")

# Fechar o ambiente de teste
test_env.close()
print("Avaliação concluída.")