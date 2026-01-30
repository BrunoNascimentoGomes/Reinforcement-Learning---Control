import numpy as np
import torch
import random
from .tanques_n_min import Quatro_Tanques
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
MAX_ACTION = 1.0   # Exemplo: Limite a entrada de controle (Pois o processo recebe a ação normalizada)      # Ruído inicial para exploração (Ornstein-Uhlenbeck ou Gaussiano)
decay_rate = 0.995  #Quanto mais epi coloca mais 9 p diminuir o reuido mais devagar


ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
GAMMA = 0.99
TAU = 0.001

BUFFER_CAPACITY = 50000
BATCH_SIZE = 128
MAX_STEPS = 800
max_steps = MAX_STEPS



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = Quatro_Tanques(render_mode=None, max_steps=MAX_STEPS)

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

maddpg.load("tanque/Modelo_N_Min_Best")

print(f"Iniciando treinamento em {DEVICE}...")
history_reward_total = []
history_reward_ag1 = []
history_reward_ag2 = []
best_reward = -math.inf

SP_NOMINAL = np.array([6.2, 6.35], dtype=np.float32)

CURRICULUM = [
    dict(name="Fase 1 (easy)",   episodes=500, delta=0.25, max_steps=200, noise_floor=0.02, tol = 0.2),
    dict(name="Fase 2 (medium)", episodes=1000, delta=1.00, max_steps=350, noise_floor=0.015, tol = 0.1),
    dict(name="Fase 3 (hard)",   episodes=2000, delta=2.50, max_steps=500, noise_floor=0.01, tol = 0.05),
]

SP_MIN = np.array([0.5, 0.5], dtype=np.float32)
SP_MAX = np.array([10.0, 10.0], dtype=np.float32)

TRAIN_INTERVAL = 1

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

    env.tol = float(phase["tol"])

    # ----- Reset do ambiente -----
    

    # ----- Setpoint aleatório por episódio (Currículo) -----
    setpoint = sample_setpoint(SP_NOMINAL, phase["delta"], SP_MIN, SP_MAX)
    obs, info = env.reset(setpoint=setpoint)

    episode_reward = np.zeros(NUM_AGENTS, dtype=np.float32)
    reward1 = 0.0
    reward2 = 0.0

    for ag in maddpg.agents:
        ag.ou_noise.reset()


    # ----- Loop do episódio -----
    for step in range(max_steps_ep):
        actions = maddpg.select_action(obs, deterministic=False)  # [2,1]

        next_obs, reward, terminated, truncated, info = env.step(actions_norm=actions)

        # reward é shape (2,) no seu env; vamos manter como float
        reward1 += float(reward[0])
        reward2 += float(reward[1])

        # done correto: terminated OR truncated (para os 2 agentes)
        done_sig = np.logical_or(terminated, truncated).astype(np.float32).reshape(NUM_AGENTS, 1)

        obs_tuple = (obs, actions, reward.reshape(NUM_AGENTS, 1), next_obs, done_sig)
        buffer.record(obs_tuple)

        if buffer.buffer_counter >= BATCH_SIZE and step % TRAIN_INTERVAL == 0:
            maddpg.train()

        episode_reward += reward
        obs = next_obs

        if truncated or terminated[0] or terminated[1]:
            break

    total_reward = reward1 + reward2

    # Salva melhor modelo (pela soma)
    if total_reward > best_reward:
        maddpg.save("./Modelo_N_Min_Best")
        best_reward = total_reward
        print(f"[SAVE] Melhor modelo salvo: total_reward={best_reward:.3f}")

    history_reward_total.append(total_reward)
    history_reward_ag1.append(reward1)
    history_reward_ag2.append(reward2)

    print(
        f"Episódio {episode+1}/{TOTAL_EPISODES} | {phase['name']} | "
        f"SP={env.setpoint} | Steps={step+1}/{max_steps_ep} | "
        f"R_total={total_reward:.3f} (R1={reward1:.3f}, R2={reward2:.3f}) | "
        f"| Sigma = {maddpg.agents[0].ou_noise.sigma:.3f} and {maddpg.agents[1].ou_noise.sigma:.3f} | terminated={terminated} truncated={truncated}"
    )
    sigma_min = float(phase.get("noise_floor", 0.05))
    decay = 0.995  # pode deixar fixo ou colocar no currículo também

    for ag in maddpg.agents:
        ag.ou_noise.sigma = max(sigma_min, ag.ou_noise.sigma * decay)


maddpg.save("./Modelo_Pós_Treino_N_Min")
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