import torch
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from utils.buffer import BufferMADDPG
from utils.MADDPGMod import MADDPG # Importa sua classe MADDPG
import os
from tqdm import tqdm # Para barras de progresso

# --- 1. Definição de Hiperparâmetros ---
# Ambiente
NUM_AGENTS = 3                         # Número padrão de agentes e landmarks no Simple Spread
MAX_CYCLES = 50                      # Passos máximos por episódio
EPISODES = 2000                       # Total de episódios para treino

# MADDPG
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.95
TAU = 0.01
BUFFER_CAPACITY = 100000
BATCH_SIZE = 1024
LEARN_START = BATCH_SIZE * 5           # Começar o treino após um certo número de amostras
NOISE = 0.1                            # Ruído de exploração (OUNoise seria melhor, mas Gaussiano serve)
NOISE_DECAY = 0.9995

# Outros
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "maddpg_simple_spread_checkpoints"

# --- 2. Inicialização do Ambiente e Parâmetros ---
# O Simple Spread usa ações contínuas por padrão no Simple_Spread_v3, o que é ideal para DDPG/MADDPG.
# Usamos a API 'parallel_env' para simplificar a coleta de ações/observações.
env = simple_spread_v3.parallel_env(N=NUM_AGENTS, max_cycles=MAX_CYCLES, continuous_actions=True, render_mode=None)

AGENT_NAMES = env.possible_agents

# O Simple Spread usa o espaço Box, compatível com o seu código DDPG.
n_agents = len(env.possible_agents)
# Assume que todos os agentes têm o mesmo espaço de observação e ação.
# Para N=3, o Simple Spread tem state_dim=18 (segundo a documentação, embora o valor possa variar dependendo da versão).
state_dim = env.observation_space(env.possible_agents[0]).shape[0]
action_dim = env.action_space(env.possible_agents[0]).shape[0]
max_action = env.action_space(env.possible_agents[0]).high[0] # Valor máximo do espaço de ação (Box)

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
total_steps = 0
current_noise = NOISE

for episode in tqdm(range(EPISODES)):
    # 4.1. Reset do Ambiente
    observations, infos = env.reset()
    
    # Converte as observações de dicionário para uma lista/array ordenado.
    # A ordem é importante, e o Petting Zoo garante a ordem em `env.agents`.
    # O MADDPG deve processar na ordem `agent_0, agent_1, ...`
    current_states = np.array([observations[agent] for agent in AGENT_NAMES])
    
    episode_reward = 0
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        
        # 4.2. Seleção de Ação
        actions_list = maddpg.select_action(current_states, noise=current_noise, deterministic=False)
        #print(actions_list)
        # 4.3. Conversão de Ações para o formato do Petting Zoo
        # O Petting Zoo espera um dicionário mapeando nome do agente para a ação.
        # `actions_list` é um array numpy [N_agents, A_dim]
        actions_dict = {
        agent_name: actions_list[i]
        for i, agent_name in enumerate(AGENT_NAMES) # USAR AGENT_NAMES (constante)
        }
        
        # 4.4. Passo no Ambiente
        observations_new, rewards, terminations, truncations, infos = env.step(actions_dict)
        
        # 4.5. Prepara a Transição
        next_states = np.array([observations_new[agent] for agent in AGENT_NAMES])
        rewards_list = np.array([[rewards[agent_name]] for agent_name in AGENT_NAMES]) # Formato [N, 1]
        
        # 4.6. Grava no Buffer
        maddpg.replay_buffer.record((current_states, actions_list, rewards_list, next_states))

        # 4.7. Acumula Recompensa (geralmente a soma das recompensas)
        episode_reward += np.sum(rewards_list)

        # 4.8. Atualiza Estado
        current_states = next_states
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        total_steps += 1

        # 4.9. Treino (se o buffer estiver cheio o suficiente)
        if maddpg.replay_buffer.buffer_counter >= LEARN_START:
            maddpg.train()
    
    # 4.10. Decay do Ruído
    current_noise = max(0.01, current_noise * NOISE_DECAY)

    
    tqdm.write(f"Episódio: {episode}, Recompensa Total: {episode_reward:.2f}, Ruído: {current_noise:.4f}")



    
maddpg.save(SAVE_DIR)
print(f"Treino concluído. Modelos salvos em: {SAVE_DIR}")