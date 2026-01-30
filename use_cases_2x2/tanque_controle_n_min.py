import numpy as np
import torch
import random
from tanque.tanques_n_min import Quatro_Tanques
from utils.buffer import BufferMADDPG_Done
from utils.MADDPGMod import MADDPG_Done
import matplotlib.pyplot as plt
import math


# -----------------------------------------------------
# 1. Parâmetros do Ambiente
# -----------------------------------------------------

# Parâmetros do sistema TITO 2x2


# Configurações de RL (baseadas na análise de dimensões)
NUM_AGENTS = 2
STATE_DIM = 6      # [erro,y, setpoint, delta_u]
ACTION_DIM = 1     # Cada agente controla uma entrada (u1 ou u2)
MAX_ACTION = 1.0   # Exemplo: Limite a entrada de controle (Pois o processo recebe a ação normalizada)
NOISE = 0.1        # Ruído inicial para exploração (Ornstein-Uhlenbeck ou Gaussiano)
decay_rate = 0.995  #Quanto mais epi coloca mais 9 p diminuir o reuido mais devagar
# -----------------------------------------------------
# 2. Parâmetros do Treinamento
# -----------------------------------------------------

ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
GAMMA = 0.99
TAU = 0.001

BUFFER_CAPACITY = 50000
BATCH_SIZE = 64
MAX_EPISODES = 20
MAX_STEPS = 1200
max_steps = MAX_STEPS



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------
# 3. Inicialização
# -----------------------------------------------------
env = Quatro_Tanques(render_mode='human', max_steps=MAX_STEPS)

buffer = BufferMADDPG_Done(
    buffer_capacity=BUFFER_CAPACITY,
    batch_size=BATCH_SIZE,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    num_agents=NUM_AGENTS
)

maddpg = MADDPG_Done(
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

#maddpg.load("use_cases_2x2/maddpg_tito_parcial6")
maddpg.load("use_cases_2x2/Modelo_N_Min_Best")
         


print("\n" + "="*40)
print(" INICIANDO AVALIAÇÃO COM AÇÃO DETERMINÍSTICA")
print("="*40)

# Quantos episódios de teste você quer rodar
NUM_EVAL_EPISODES = 5

for eval_ep in range(NUM_EVAL_EPISODES):
    setpoint = np.array([6.2, 6.35], dtype=np.float32)
    obs, info = env.reset(setpoint=setpoint)
    
    # Se você implementou o setpoint aleatório no reset, aqui ele já mudou.
    # Se quiser forçar um setpoint específico para teste visual (ex: degrau):
    # env.setpoint = [5.0, 7.0] 
    
    eval_reward = 0
    terminated = False
    truncated = False
    step_count = 0
    
    for i in range(MAX_STEPS):
        # AQUI ESTÁ A MUDANÇA: deterministic=True e noise=0.0
        actions = maddpg.select_action(obs, noise=0.0, deterministic=True)
        
        if i == 100:
            env.setpoint = [9.2,6.35]
        if i == 300:
            env.setpoint=[6.2,6.35]
        if i == 500:
            env.setpoint = [6.2, 4.35]
        if i == 700:
            env.setpoint=[6.2,6.35]
        
        obs, reward, terminated, truncated, info = env.step(actions)
        
        eval_reward += np.sum(reward) # Soma recompensa dos dois agentes
        step_count += 1

    print(f"Avaliação {eval_ep+1}/{NUM_EVAL_EPISODES} | Steps: {step_count} | Recompensa Total: {eval_reward:.4f} | Terminated: {terminated}")

    # Renderiza o último episódio para você ver o gráfico de controle final
    if eval_ep == NUM_EVAL_EPISODES - 1:
        print("Gerando gráfico do último episódio de teste...")
        plt.figure(figsize=(10, 8)) # Garante uma nova figura
        env.render()
        #plt.show() # Bloqueia o script para mostrar a janela    
        plt.savefig('Controle_Teste_N_Min.png') # Salva em arquivo
        plt.show()                  # Mostra na tela