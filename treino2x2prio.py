import numpy as np
import torch
import random
from modelo_2x2 import TITOSystem
from utils.buffer import BufferMADDPGPrio
from utils.MADDPGModPrio import MADDPGPrio

# -----------------------------------------------------
# 1. Parâmetros do Ambiente
# -----------------------------------------------------

# Parâmetros do sistema TITO 2x2


# Configurações de RL (baseadas na análise de dimensões)
NUM_AGENTS = 2
STATE_DIM = 4      # [error, y, u_prev, setpoint]
ACTION_DIM = 1     # Cada agente controla uma entrada (u1 ou u2)
MAX_ACTION = 5.0   # Exemplo: Limite a entrada de controle
NOISE = 1.0        # Ruído inicial para exploração (Ornstein-Uhlenbeck ou Gaussiano)

# -----------------------------------------------------
# 2. Parâmetros do Treinamento
# -----------------------------------------------------

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.01

BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 64
MAX_EPISODES = 10000
MAX_STEPS = 1000



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------
# 3. Inicialização
# -----------------------------------------------------
env = TITOSystem(K=np.array([[3,2],[2,5]]), wn1=1.0, wn2=0.5, zeta1=0.3, zeta2=0.4,
                 max_steps=MAX_STEPS, setpoint=[1.0, 1.0], render_mode="human")

buffer = BufferMADDPGPrio(
    buffer_capacity=BUFFER_CAPACITY,
    batch_size=BATCH_SIZE,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    num_agents=NUM_AGENTS
)

maddpg = MADDPGPrio(
    num_agents=NUM_AGENTS,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    max_action=MAX_ACTION,
    buffer=buffer,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    gamma=GAMMA,
    tau=TAU,
    device=DEVICE
)

# -----------------------------------------------------
# 4. Loop de Treinamento
# -----------------------------------------------------

print(f"Iniciando treinamento em {DEVICE}...")

for episode in range(MAX_EPISODES):
    # Reset do ambiente: obs será [2, 4]
    obs, info = env.reset()
    
    # Reduzir ruído (simples decay linear)
    current_noise = max(NOISE * (1.0 - episode / MAX_EPISODES), 0.1)
    
    episode_reward = np.zeros(NUM_AGENTS)
    reward1 = 0
    reward2 = 0
    for step in range(MAX_STEPS):
        # 1. Seleção de Ação (Descentralizada)
        # obs[i] é a observação local do agente i
        actions = maddpg.select_action(obs, noise=current_noise, deterministic=False) # [2, 1]
        
        # 2. Interação com o Ambiente (Executar Ação)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        
        # 3. Armazenamento da Transição (Centralizada no Buffer)
        # obs_tuple = (state, action, reward, next_state)
        reward1 += reward[0]
        reward2 += reward[1]
        obs_tuple = (obs, actions, reward.reshape(NUM_AGENTS, 1), next_obs)
        buffer.record(obs_tuple)
        
        # 4. Treinamento
        if buffer.buffer_counter >= BATCH_SIZE:
            maddpg.train()
        
        episode_reward += reward
        obs = next_obs
        
        if truncated:
            break
            
    avg_reward = np.mean(episode_reward)
    total_reward = reward1 + reward2
    if reward1 > 0 and reward2 > 0:
        env.render()
        break
      
    
    print(f"Episódio: {episode+1} | Recompensa Total: {total_reward} | Recompensa 1: {reward1}| Recompensa 2: {reward2}|Ruído: {current_noise:.2f}")

# Salvar o modelo após o treinamento (opcional)
maddpg.save("./maddpg_tito_models")
print("Treinamento concluído e modelos salvos.")