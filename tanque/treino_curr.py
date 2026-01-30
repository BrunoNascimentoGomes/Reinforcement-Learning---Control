from collections import deque
import numpy as np
from collections import deque
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

TOTAL_EPISODES = 1000


SP_NOMINAL = np.array([6.2, 6.35], dtype=np.float32)

SP_MIN = np.array([0.5, 0.5], dtype=np.float32)
SP_MAX = np.array([10.0, 10.0], dtype=np.float32)

TRAIN_INTERVAL = 1

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
# =========================
# Curriculum por desempenho
# =========================
CURRICULUM = [
    dict(name="Fase 1 (easy)",   delta=0.25, max_steps=200, noise_floor=0.02,  tol=0.2,
         r_target=-25.0, max_viol_rate=0.10),
    dict(name="Fase 2 (medium)", delta=1.00, max_steps=350, noise_floor=0.015, tol=0.1,
         r_target=-35.0, max_viol_rate=0.08),
    dict(name="Fase 3 (hard)",   delta=2.50, max_steps=500, noise_floor=0.01,  tol=0.05,
         r_target=-45.0, max_viol_rate=0.05),
]

W = 40              # janela para média móvel
PATIENCE = 3        # quantas janelas seguidas precisa cumprir
SIGMA_DECAY = 0.995 # decay do OU sigma
BASE_SP = np.array([6.2, 6.35], dtype=np.float32)

def sample_setpoint(delta: float) -> list:
    sp = BASE_SP + np.random.uniform(-delta, +delta, size=(2,))
    sp = np.clip(sp, 0.0, 10.0)
    return [float(sp[0]), float(sp[1])]



# =========================
# TREINO (rodou o script -> treina)
# =========================
# Garanta que essas variáveis/objetos já existem no seu arquivo:
# env, maddpg, buffer, BATCH_SIZE, TRAIN_INTERVAL, TOTAL_EPISODES
# Ex.: TOTAL_EPISODES = 3000

np.random.seed(123)

phase_idx = 0
streak = 0

hist_reward = deque(maxlen=W)
hist_viol = deque(maxlen=W)

global_step = 0
phase_idx = 0
streak = 0

hist_reward = deque(maxlen=W)
hist_viol = deque(maxlen=W)

for episode in range(TOTAL_EPISODES):
    phase = CURRICULUM[phase_idx]

    # ----- Ajusta o tamanho do episódio e tolerância -----
    env.max_steps = int(phase["max_steps"])
    env.tol = float(phase["tol"])
    max_steps_ep = int(phase["max_steps"])

    # ----- Setpoint aleatório por episódio (Currículo) -----
    setpoint = sample_setpoint(SP_NOMINAL, phase["delta"], SP_MIN, SP_MAX)
    obs, info = env.reset(setpoint=setpoint)

    reward1 = 0.0
    reward2 = 0.0
    violated = False

    # reset OU noise por episódio
    for ag in maddpg.agents:
        ag.reset_noise()

    terminated = [False, False]
    truncated = False

    # ----- Loop do episódio -----
    for step in range(max_steps_ep):
        actions = maddpg.select_action(obs, deterministic=False)  # [2,1]
        next_obs, reward, terminated, truncated, info = env.step(actions_norm=actions)

        reward1 += float(reward[0])
        reward2 += float(reward[1])

        # violação = terminou por restrição (terminated)
        if terminated[0] or terminated[1]:
            violated = True

        # done correto: terminated OR truncated (para os 2 agentes)
        done_sig = np.logical_or(terminated, truncated).astype(np.float32).reshape(NUM_AGENTS, 1)

        # grava no replay (mantém seu padrão)
        obs_tuple = (obs, actions, reward.reshape(NUM_AGENTS, 1), next_obs, done_sig)
        buffer.record(obs_tuple)

        # treina (warm-up + intervalo)
        if buffer.buffer_counter >= BATCH_SIZE and step % TRAIN_INTERVAL == 0:
            maddpg.train()

        obs = next_obs

        if truncated or terminated[0] or terminated[1]:
            break

    total_reward = reward1 + reward2

    # --------------------------
    # Salva melhor modelo (igual seu treino)
    # --------------------------
    if total_reward > best_reward:
        maddpg.save("./Modelo_N_Min_Best")
        best_reward = total_reward
        print(f"[SAVE] Melhor modelo salvo: total_reward={best_reward:.3f}")

    history_reward_total.append(total_reward)
    history_reward_ag1.append(reward1)
    history_reward_ag2.append(reward2)

    # --------------------------
    # Decay do OU sigma com piso por fase
    # --------------------------
    sigma_min = float(phase["noise_floor"])
    for ag in maddpg.agents:
        ag.ou_noise.sigma = max(sigma_min, ag.ou_noise.sigma * SIGMA_DECAY)

    # --------------------------
    # Logs
    # --------------------------
    hist_reward.append(total_reward)
    hist_viol.append(1.0 if violated else 0.0)

    if (episode + 1) % 10 == 0:
        r_mean = float(np.mean(hist_reward)) if len(hist_reward) else total_reward
        v_rate = float(np.mean(hist_viol)) if len(hist_viol) else (1.0 if violated else 0.0)
        print(
            f"Episódio {episode+1}/{TOTAL_EPISODES} | {phase['name']} | "
            f"SP={env.setpoint} | Steps={step+1}/{max_steps_ep} | "
            f"R_total={total_reward:.3f} (R1={reward1:.3f}, R2={reward2:.3f}) | "
            f"Rmean{W}={r_mean:.3f} | viol_rate{W}={v_rate*100:.1f}% | "
            f"terminated={terminated} truncated={truncated}"
        )

    # --------------------------
    # Curriculum por desempenho (Critério 1)
    # --------------------------
    if len(hist_reward) == W:
        r_mean = float(np.mean(hist_reward))
        v_rate = float(np.mean(hist_viol))

        passed = (r_mean > float(phase["r_target"])) and (v_rate < float(phase["max_viol_rate"]))

        if passed:
            streak += 1
        else:
            streak = 0

        if (streak >= PATIENCE) and (phase_idx < len(CURRICULUM) - 1):
            old = CURRICULUM[phase_idx]["name"]
            phase_idx += 1
            new = CURRICULUM[phase_idx]["name"]

            streak = 0
            hist_reward.clear()
            hist_viol.clear()

            print(f"\n=== ADVANCE: {old} -> {new} (episode={episode+1}) ===\n")

# --------------------------
# Salva modelo final (igual seu treino)
# --------------------------
maddpg.save("./Modelo_Pós_Treino_N_Min")
print("Treinamento concluído e modelos salvos.")
