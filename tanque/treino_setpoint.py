import numpy as np
import torch
import random
from .tanques import Quatro_Tanques
from utils.buffer import BufferMADDPG_Done
from utils.MADDPGMod import MADDPG_Done
import matplotlib.pyplot as plt
import math
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)




NUM_AGENTS = 2
STATE_DIM = 6      # [erro,y, setpoint, delta_u, h3, h4]
ACTION_DIM = 1     # Cada agente controla uma entrada (u1 ou u2)
MAX_ACTION = 1.0   # Exemplo: Limite a entrada de controle (Pois o processo recebe a ação normalizada)
NOISE = 0.1        # Ruído inicial para exploração (Ornstein-Uhlenbeck ou Gaussiano)
decay_rate = 0.995  #Quanto mais epi coloca mais 9 p diminuir o reuido mais devagar


ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
GAMMA = 0.99
TAU = 0.001

BUFFER_CAPACITY = 50000
BATCH_SIZE = 64
MAX_EPISODES = 3000
MAX_STEPS = 800
max_steps = MAX_STEPS



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


print(f"Iniciando treinamento em {DEVICE}...")
history_reward_total = []
history_reward_ag1 = []
history_reward_ag2 = []
best_reward = -math.inf

SP_NOMINAL = np.array([6.2, 6.35], dtype=np.float32)

FASE1_EP = 500
FASE2_EP = 800
FASE3_EP = 1700

CURRICULUM = [
    dict(name="Fase 1 (easy)",   episodes=FASE1_EP, delta=0.25, max_steps=200, noise_start=0.15, noise_floor=0.02),
    dict(name="Fase 2 (medium)", episodes=FASE2_EP, delta=1.00, max_steps=350, noise_start=0.12, noise_floor=0.015),
    dict(name="Fase 3 (hard)",   episodes=FASE3_EP, delta=2.50, max_steps=500, noise_start=0.10, noise_floor=0.01),
]

SP_MIN = np.array([0.5, 0.5], dtype=np.float32)
SP_MAX = np.array([10.0, 10.0], dtype=np.float32)

def sample_setpoint(nominal, delta, sp_min, sp_max):
    """Amostra setpoint aleatório por episódio, em torno do nominal, com clipping."""
    sp = nominal + np.random.uniform(-delta, +delta, size=nominal.shape).astype(np.float32)
    sp = np.clip(sp, sp_min, sp_max)
    return sp.tolist()

def curriculum_phase(episode_idx):
    """Retorna (phase_dict, local_ep) para o episódio global episode_idx."""
    acc = 0
    for phase in CURRICULUM:
        if episode_idx < acc + phase["episodes"]:
            return phase, (episode_idx - acc)
        acc += phase["episodes"]
    # fallback: última fase
    return CURRICULUM[-1], episode_idx


global_episode = 0
TOTAL_EPISODES = sum(p["episodes"] for p in CURRICULUM)

for episode in range(TOTAL_EPISODES):
    phase, local_ep = curriculum_phase(episode)

    # ----- Ajusta o "comprimento" do episódio no ambiente (sem mudar setpoint no meio) -----
    env.max_steps = int(phase["max_steps"])     # usado pelo seu env.step para truncated
    max_steps_ep = int(phase["max_steps"])      # usado no for step

    # ----- Reset do ambiente -----
    obs, info = env.reset()

    # ----- Setpoint aleatório por episódio (Currículo) -----
    env.setpoint = sample_setpoint(SP_NOMINAL, phase["delta"], SP_MIN, SP_MAX)

    # ----- Ruído: decaimento dentro da fase -----
    # Ex: ruído decai com local_ep, mas nunca abaixo de noise_floor.
    # Você pode ajustar a taxa para mais/menos exploração.
    decay = 0.995
    current_noise = max(phase["noise_start"] * (decay ** local_ep), phase["noise_floor"])

    episode_reward = np.zeros(NUM_AGENTS, dtype=np.float32)
    reward1 = 0.0
    reward2 = 0.0

    # ----- Loop do episódio -----
    for step in range(max_steps_ep):
        actions = maddpg.select_action(obs, noise=current_noise, deterministic=False)  # [2,1]

        next_obs, reward, terminated, truncated, info = env.step(actions_norm=actions)

        # reward é shape (2,) no seu env; vamos manter como float
        reward1 += float(reward[0])
        reward2 += float(reward[1])

        # done correto: terminated OR truncated (para os 2 agentes)
        done_sig = np.logical_or(terminated, truncated).astype(np.float32).reshape(NUM_AGENTS, 1)

        obs_tuple = (obs, actions, reward.reshape(NUM_AGENTS, 1), next_obs, done_sig)
        buffer.record(obs_tuple)

        if buffer.buffer_counter >= BATCH_SIZE:
            maddpg.train()

        episode_reward += reward
        obs = next_obs

        if truncated or terminated[0] or terminated[1]:
            break

    total_reward = reward1 + reward2

    # Salva melhor modelo (pela soma)
    if total_reward > best_reward:
        maddpg.save("./maddpg_tito_best")
        best_reward = total_reward
        print(f"[SAVE] Melhor modelo salvo: total_reward={best_reward:.3f}")

    history_reward_total.append(total_reward)
    history_reward_ag1.append(reward1)
    history_reward_ag2.append(reward2)

    print(
        f"Episódio {episode+1}/{TOTAL_EPISODES} | {phase['name']} | "
        f"SP={env.setpoint} | Steps={step+1}/{max_steps_ep} | "
        f"R_total={total_reward:.3f} (R1={reward1:.3f}, R2={reward2:.3f}) | "
        f"noise={current_noise:.3f} | terminated={terminated} truncated={truncated}"
    )

maddpg.save("./Modelo_Pós_Treino")
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
plt.title("Evolução da Recompensa Total")
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

NUM_EVAL_EPISODES = 1

for eval_ep in range(NUM_EVAL_EPISODES):
    obs, info = env.reset()

    
    eval_reward = 0
    terminated = False
    truncated = False
    step_count = 0
    
    for i in range(MAX_STEPS):
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