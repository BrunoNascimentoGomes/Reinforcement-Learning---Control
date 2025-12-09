import numpy as np
import torch
from utils.DDPGAgent import DDPGAgent_Done  # seu agente
from modelos import ModeloSegundaOrdem  # sua planta
import matplotlib.pyplot as plt


# =============================
# Configurações do experimento
# =============================
state_dim = 6  # [error, y,du, setpoint, integral_erro, dy]
action_dim = 1
max_action = 1

n_episodes = 300
max_steps = 500
num_treino_per_step = 1

start_steps = 2000

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU") 

# Inicializa agente
agent = DDPGAgent_Done(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    buffer_capacity=100000,
    batch_size=128,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-3,
    critic_lr=1e-4, #Antes tava 2e-3
    device=device
)

# Inicializa planta


# Ruído para exploração
noise_std = 0.5
initial_noise_std = 0.5
final_noise_std = 0.05 
noise_decay = 0.995 # Taxa de decaimento por episódio

rewards = []
# =============================
# Loop de episódios
# =============================
#setpoint_schedule = [(0, 1.0), (250, 3), (500, 5), (750, 2), (1000,4), (1250,0), (1500, -2), (1750, -5)]
aux = 0
total_steps = 0
for ep in range(n_episodes):
    if ep%2 == 0:
        sp=1
    else:
        sp=2
    env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=sp, max_steps=max_steps)
    
    obs, _ = env.reset()
    total_reward = 0.0
    sp_index = 0
    
    aux+=1
    for step in range(max_steps):
        #print("Setpoint:", env.setpoint)
        '''
        if sp_index < len(setpoint_schedule) and step >= setpoint_schedule[sp_index][0]:
            env.setpoint = setpoint_schedule[sp_index][1]
            sp_index += 1
        # Convertendo obs para tensor batch (1, state_dim)
        '''
        total_steps+=1
        
        # Seleciona ação do agente
        action = agent.select_action(obs, noise=noise_std)

        # Executa ação na planta
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated

        # Armazena transição no buffer
        agent.replay_buffer.record((obs, action, reward, next_obs, float(done)))

        # Treina agente
        #if agent.replay_buffer.buffer_counter >= agent.batch_size:
        if total_steps >= start_steps:
            agent.train()

        # Próximo estado
        obs = next_obs
        if terminated or truncated:
            break

    print(f"Episode {ep+1} finished | Total reward: {total_reward:.2f} | Noise: {noise_std:.4f} | Terminated : {terminated}")
    noise_std = max(final_noise_std, initial_noise_std * (noise_decay ** ep)) # Atualização
    rewards.append(total_reward)


# =============================
# Render final
def plot_learning_curve(scores, label, color, window=50):
    # Função para calcular média móvel
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    
    plt.plot(scores, color=color, alpha=0.3)  # Linha original transparente
    plt.plot(running_avg, color=color, label=label, linewidth=2) # Média móvel sólida
agent.save('Treinamento_salvo')
env.render()

plt.figure(figsize=(12, 6))

# --- Gráfico 1: Recompensa Total do Sistema ---

plt.title("Evolução da Recompensa Total (Cooperação)")
plot_learning_curve(rewards, 'Total Reward', 'blue')
plt.xlabel('Episódios')
plt.ylabel('Recompensa Acumulada')
plt.grid(True, alpha=0.3)
plt.legend()

print('Avaliação')

eval_reward = 0
obs, _ = env.reset()
env.setpoint = 1.0
sp_index = 0

for step in range(max_steps):
        
        # Seleciona ação do agente
    action = agent.select_action(obs,deterministic=True, noise=0.0)

        # Executa ação na planta
    obs, reward, terminated, truncated, info = env.step(action)

    eval_reward += reward
    if terminated or truncated:
        break

print(f"Episodio Avaliacao finished | Total reward: {eval_reward:.2f}")

env.render()


