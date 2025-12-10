import numpy as np
import torch
import random
from tanques import Quatro_Tanques
from utils.buffer import BufferMADDPG_Done
from utils.MADDPGMod import MADDPG_Done
import matplotlib.pyplot as plt


# -----------------------------------------------------
# 1. Parâmetros do Ambiente
# -----------------------------------------------------

# Parâmetros do sistema TITO 2x2


# Configurações de RL (baseadas na análise de dimensões)
NUM_AGENTS = 2
STATE_DIM = 4      # [erro,y, setpoint, delta_u]
ACTION_DIM = 1     # Cada agente controla uma entrada (u1 ou u2)
MAX_ACTION = 1.0   # Exemplo: Limite a entrada de controle (Pois o processo recebe a ação normalizada)
NOISE = 0.5        # Ruído inicial para exploração (Ornstein-Uhlenbeck ou Gaussiano)
decay_rate = 0.995  #Quanto mais epi coloca mais 9 p diminuir o reuido mais devagar
# -----------------------------------------------------
# 2. Parâmetros do Treinamento
# -----------------------------------------------------

ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.99
TAU = 0.001

BUFFER_CAPACITY = 50000
BATCH_SIZE = 64
MAX_EPISODES = 10
MAX_STEPS = 700
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

# -----------------------------------------------------
# 4. Loop de Treinamento
# -----------------------------------------------------

print(f"Iniciando treinamento em {DEVICE}...")
history_reward_total = []
history_reward_ag1 = []
history_reward_ag2 = []

for episode in range(MAX_EPISODES):
    # Reset do ambiente: obs será [2, 4]
    obs, info = env.reset()
    
    
    current_noise = max(NOISE * (decay_rate ** episode), 0.001)
    
    episode_reward = np.zeros(NUM_AGENTS)
    reward1 = 0
    reward2 = 0
    for step in range(MAX_STEPS):
        # 1. Seleção de Ação (Descentralizada)
        # obs[i] é a observação local do agente i
        actions = maddpg.select_action(obs, noise=current_noise, deterministic=False) # [2, 1]
        if step == 100:
            env.setpoint = [8.2,6.35]
        if step == 200:
            env.setpoint=[6.2,6.35]
        if step == 400:
            env.setpoint = [6.2, 4.35]
         
        # 2. Interação com o Ambiente (Executar Ação)
        next_obs, reward, terminated, truncated, info = env.step(actions_norm=actions)
        
        # 3. Armazenamento da Transição (Centralizada no Buffer)
        # obs_tuple = (state, action, reward, next_state)
        reward1 += reward[0]
        reward2 += reward[1]

        done_sig = terminated
        obs_tuple = (obs, actions, reward.reshape(NUM_AGENTS, 1), next_obs, done_sig)
        buffer.record(obs_tuple)
        
        # 4. Treinamento
        if buffer.buffer_counter >= BATCH_SIZE:
            maddpg.train()
        
        episode_reward += reward
        obs = next_obs
        
        if truncated or terminated[0] or terminated[1]:
            break
            
    avg_reward = np.mean(episode_reward)

    total_reward = reward1 + reward2
    '''
    if total_reward < 0.01:
        env.render()
        break
    '''

      
    history_reward_total.append(total_reward)
    history_reward_ag1.append(reward1)
    history_reward_ag2.append(reward2)
    
    print(f"Episódio: {episode+1} | Recompensa Total: {total_reward} | Recompensa 1: {reward1}| Recompensa 2: {reward2}|Ruído: {current_noise:.2f} | Terminaed : {terminated}")
    '''
    plt.figure(figsize=(10, 8)) # Garante uma nova figura
    
    env.render()
    plt.show()
    '''
# Salvar o modelo após o treinamento (opcional)
#env.render()
maddpg.save("./maddpg_tito_models")
print("Treinamento concluído e modelos salvos.")

# -----------------------------------------------------
# 5. Plotagem dos Resultados
# -----------------------------------------------------
def plot_learning_curve(scores, label, color, window=50):
    # Função para calcular média móvel
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    
    plt.plot(scores, color=color, alpha=0.3)  # Linha original transparente
    plt.plot(running_avg, color=color, label=label, linewidth=2) # Média móvel sólida

plt.figure(figsize=(12, 6))

# --- Gráfico 1: Recompensa Total do Sistema ---
plt.subplot(1, 2, 1)
plt.title("Evolução da Recompensa Total (Cooperação)")
plot_learning_curve(history_reward_total, 'Total Reward', 'blue')
plt.xlabel('Episódios')
plt.ylabel('Recompensa Acumulada')
plt.grid(True, alpha=0.3)
plt.legend()


# --- Gráfico 2: Recompensa por Agente ---
plt.subplot(1, 2, 2)
plt.title("Desempenho por Agente")
plot_learning_curve(history_reward_ag1, 'Agente 1 (u1)', 'green')
plot_learning_curve(history_reward_ag2, 'Agente 2 (u2)', 'orange')
plt.xlabel('Episódios')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('historico_aprendizado.png') # Salva em arquivo
plt.show()          


print("\n" + "="*40)
print(" INICIANDO AVALIAÇÃO COM AÇÃO DETERMINÍSTICA")
print("="*40)

# Quantos episódios de teste você quer rodar
NUM_EVAL_EPISODES = 5

for eval_ep in range(NUM_EVAL_EPISODES):
    obs, info = env.reset()
    
    # Se você implementou o setpoint aleatório no reset, aqui ele já mudou.
    # Se quiser forçar um setpoint específico para teste visual (ex: degrau):
    # env.setpoint = [5.0, 7.0] 
    
    eval_reward = 0
    terminated = False
    truncated = False
    step_count = 0
    
    for i in range(500):
        # AQUI ESTÁ A MUDANÇA: deterministic=True e noise=0.0
        actions = maddpg.select_action(obs, noise=0.0, deterministic=True)
        if i == 100:
            env.setpoint = [8.2,6.35]
        if i == 200:
            env.setpoint=[6.2,6.35]
        if i == 400:
            env.setpoint = [6.2, 4.35]
        obs, reward, terminated, truncated, info = env.step(actions)
        
        eval_reward += np.sum(reward) # Soma recompensa dos dois agentes
        step_count += 1

    print(f"Avaliação {eval_ep+1}/{NUM_EVAL_EPISODES} | Steps: {step_count} | Recompensa Total: {eval_reward:.4f} | Setpoint: {env.setpoint} | Terminated: {terminated}")

    # Renderiza o último episódio para você ver o gráfico de controle final
    if eval_ep == NUM_EVAL_EPISODES - 1:
        print("Gerando gráfico do último episódio de teste...")
        plt.figure(figsize=(10, 8)) # Garante uma nova figura
        env.render()
        #plt.show() # Bloqueia o script para mostrar a janela    
        plt.savefig('Controle.png') # Salva em arquivo
        plt.show()                  # Mostra na tela