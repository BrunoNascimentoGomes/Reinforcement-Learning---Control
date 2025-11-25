import numpy as np
import torch
from utils.DDPGAgent import DDPGAgent  # seu agente
from modelos import ModeloSegundaOrdem  # sua planta

# =============================
# Configurações do experimento
# =============================
state_dim = 4  # [error, y, delta_e, setpoint]
action_dim = 1
max_action = 6

n_episodes = 1000
max_steps = 2000
num_treino_per_step = 1

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU") 

# Inicializa agente
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    buffer_capacity=100000,
    batch_size=1024,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-3,
    critic_lr=1e-3, #Antes tava 2e-3
    device=device
)

# Inicializa planta


# Ruído para exploração
noise_std = 0.1
initial_noise_std = 0.1
final_noise_std = 0.05 
noise_decay = 0.995 # Taxa de decaimento por episódio

# =============================
# Loop de episódios
# =============================
setpoint_schedule = [(0, 1.0), (250, 3), (500, 5), (750, 2), (1000,4), (1250,0), (1500, -2), (1750, -5)]
aux = 0
for ep in range(n_episodes):
    env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=1.0, max_steps=max_steps)
    obs, _ = env.reset()
    total_reward = 0.0
    sp_index = 0
    
    aux+=1
    for step in range(max_steps):
        #print("Setpoint:", env.setpoint)
        if sp_index < len(setpoint_schedule) and step >= setpoint_schedule[sp_index][0]:
            env.setpoint = setpoint_schedule[sp_index][1]
            sp_index += 1
        # Convertendo obs para tensor batch (1, state_dim)
        
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        # Seleciona ação do agente
        action = agent.select_action(state_tensor, noise=noise_std)

        # Executa ação na planta
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Armazena transição no buffer
        agent.replay_buffer.record((obs, action, reward, next_obs))

        # Treina agente
        if agent.replay_buffer.buffer_counter >= agent.batch_size:
            for i in range(num_treino_per_step):
                agent.train()

        # Próximo estado
        obs = next_obs

        if terminated or truncated:
            break

    print(f"Episode {ep+1} finished | Total reward: {total_reward:.2f}")
    noise_std = max(final_noise_std, initial_noise_std * (noise_decay ** ep)) # Atualização
    if aux %10 == 0 and aux>=10:
        pass
        #env.render()
    if total_reward > 7800:
        env.render()
        agent.save("ddpg_model_best")
        break

# =============================
# Render final
#
env.render()


