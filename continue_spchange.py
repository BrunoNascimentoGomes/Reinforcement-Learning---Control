import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# Importações dos seus arquivos locais
from tito_sys.modelo_2x2 import TITOSystem
from utils.buffer import BufferMADDPG
from utils.MADDPGMod import MADDPG

# -----------------------------------------------------
# 1. Configurações e Hiperparâmetros
# -----------------------------------------------------

# Configurações de Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensões (Atualizado para o novo vetor de estado com delta_u e setpoint)
NUM_AGENTS = 2
STATE_DIM = 5      # [erro, y, u, delta_u, setpoint]
ACTION_DIM = 1     # Controle u1 ou u2
MAX_ACTION = 1.0   

# Parâmetros de Exploração (Ruído)
          # Começa alto (3V físicos) para explorar bem
          # Termina baixo para ajuste fino
         # Decaimento suave para durar todo o treino

# Parâmetros de Aprendizado
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.99
TAU = 0.001

# Buffer e Treino
BUFFER_CAPACITY = 100000
BATCH_SIZE = 256
MAX_EPISODES = 100        # Quantidade de episódios
MAX_STEPS = 800            # Passos por episódio (tempo suficiente para estabilizar)

# -----------------------------------------------------
# 2. Inicialização do Sistema
# -----------------------------------------------------

# Inicializa o ambiente TITO
# Nota: setpoint inicial não importa muito pois será resetado
env = TITOSystem(
    K=np.array([[3,2],[2,5]]), 
    wn1=1.0, wn2=0.5, 
    zeta1=0.3, zeta2=0.4,
    max_steps=MAX_STEPS, 
    setpoint=[1.0, 1.0], 
    render_mode="human"
)

# Inicializa Buffer
buffer = BufferMADDPG(
    buffer_capacity=BUFFER_CAPACITY,
    batch_size=BATCH_SIZE,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    num_agents=NUM_AGENTS
)

# Inicializa Agentes MADDPG
maddpg = MADDPG(
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


CONTINUE_TRAINING = True

if CONTINUE_TRAINING:
    NOISE_START = 0.1     # Ruído menor, pois ele já sabe controlar +/-
else:
    NOISE_START = 0.5     # Exploração agressiva para treino novo

NOISE_END = 0.01
DECAY_RATE = 0.999        # Decaimento mais lento para ajuste fino

# ... (Instanciação do env e buffer igual) ...


if CONTINUE_TRAINING:
    maddpg.load("./Modelos_MADDPG/maddpg_tito_models6")


# Históricos para plotagem
history_reward_total = []
history_reward_ag1 = []
history_reward_ag2 = []

print(f"Iniciando treinamento em {DEVICE}...")
print(f"Estado: {STATE_DIM} vars | Ruído Inicial: {NOISE_START}")

# -----------------------------------------------------
# 3. Loop de Treinamento (Setpoint Dinâmico Aleatório)
# -----------------------------------------------------

for episode in range(MAX_EPISODES):
    obs, info = env.reset() # Reset já aleatoriza o setpoint inicial no modelo_2x2
    
    # Decaimento do ruído
    current_noise = max(NOISE_START * (DECAY_RATE ** episode), NOISE_END)
    
    episode_reward = np.zeros(NUM_AGENTS)
    reward1 = 0
    reward2 = 0
    
    # Lógica de mudança de Setpoint para evitar Overfitting
    steps_until_change = 400

    last_change_step = 0

    for step in range(MAX_STEPS):
        
        # --- MUDANÇA DINÂMICA DE SETPOINT ---
        if (step - last_change_step) > steps_until_change:
            # Gera novos alvos aleatórios entre 2.0 e 8.0 (dentro da faixa operável)
            new_sp = [np.random.uniform(1.0, 8.0), np.random.uniform(1.0, 8.0)]
            env.setpoint = new_sp
            
            # Reseta contadores
            last_change_step = step
            

        
        # 1. Seleção de Ação (com Ruído)
        actions = maddpg.select_action(obs, noise=current_noise, deterministic=False)
        
        # 2. Passo no Ambiente
        next_obs, reward, terminated, truncated, info = env.step(actions)
        
        # 3. Gravar no Buffer
        reward1 += reward[0]
        reward2 += reward[1]
        
        # obs_tuple = (state, action, reward, next_state)
        obs_tuple = (obs, actions, reward.reshape(NUM_AGENTS, 1), next_obs)
        buffer.record(obs_tuple)
        
        # 4. Treinar Rede
        if buffer.buffer_counter >= BATCH_SIZE:
            maddpg.train()
        
        # Atualiza estado e métricas
        episode_reward += reward
        obs = next_obs
        
        if truncated:
            break
            
    # Salva métricas do episódio
    total_reward = reward1 + reward2
    history_reward_total.append(total_reward)
    history_reward_ag1.append(reward1)
    history_reward_ag2.append(reward2)
    
    
    print(f"Episódio: {episode+1} | R.Total: {total_reward:.2f} | R.Ag1: {reward1:.2f} | R.Ag2: {reward2:.2f} | Noise: {current_noise:.3f}")
    #env.render()

# Salva os modelos finais
maddpg.save("./maddpg_tito_models")
print("Treinamento concluído. Modelos salvos.")

# -----------------------------------------------------
# 4. Plotagem do Aprendizado
# -----------------------------------------------------
def plot_learning_curve(scores, label, color, window=50):
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    plt.plot(scores, color=color, alpha=0.2)
    plt.plot(running_avg, color=color, label=label, linewidth=2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Recompensa Total (Cooperação)")
plot_learning_curve(history_reward_total, 'Total', 'blue')
plt.xlabel('Episódios'); plt.ylabel('Reward')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Desempenho por Agente")
plot_learning_curve(history_reward_ag1, 'Agente 1', 'green')
plot_learning_curve(history_reward_ag2, 'Agente 2', 'orange')
plt.xlabel('Episódios')
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('historico_aprendizado.png')
# plt.show() # Descomente se quiser ver na hora

# -----------------------------------------------------
# 5. Avaliação (Setpoint Fixo "Escadinha")
# -----------------------------------------------------
print("\n" + "="*40)
print(" INICIANDO AVALIAÇÃO (DETERMINÍSTICA)")
print(" Teste de rastreamento de trajetória (Servo)")
print("="*40)

# Cronograma de Setpoints para o teste visual
# Formato: (passo_onde_muda, [sp1, sp2])
# Isso cria o efeito "degrau" para validarmos visualmente
eval_schedule = [
    (0,   [2.0, 2.0]),  # Começa baixo
    (200, [6.0, 2.0]),  # Sobe apenas Y1
    (400, [6.0, 7.0]),  # Sobe Y2
    (600, [3.0, 3.0])   # Desce ambos
]

obs, info = env.reset()
env.setpoint = eval_schedule[0][1] # Aplica o primeiro setpoint

schedule_idx = 0
eval_reward = 0
steps_run = 0

# Loop manual até o fim dos passos
for step in range(MAX_STEPS):
    
    # Verifica o cronograma
    if schedule_idx < len(eval_schedule):
        trigger_step, new_sp = eval_schedule[schedule_idx]
        if step >= trigger_step:
            env.setpoint = new_sp
            schedule_idx += 1
            # Nota: Na vida real, o "step" seguinte já pegará a obs com o erro novo
    
    # Ação Determinística (Sem ruído)
    actions = maddpg.select_action(obs, noise=0.0, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(actions)
    eval_reward += np.sum(reward)
    steps_run += 1
    
    if truncated: break

print(f"Avaliação Finalizada. Steps: {steps_run} | Recompensa: {eval_reward:.2f}")

# Gera o gráfico de controle final
print("Gerando gráfico de controle...")
plt.figure(figsize=(10, 8))
env.render()
plt.savefig('Controle_Final_Servo.png')
plt.show()