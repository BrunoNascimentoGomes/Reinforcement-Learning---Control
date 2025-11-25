from modelo_2x2 import TITOSystem
import numpy as np
import torch
from utils.MADDPGAgent import MADDPGAgent
from utils.buffer import BufferMADDPG
import matplotlib.pyplot as plt

def normalize_observation(obs):
    error = obs[0]
    y = obs[1]
    u_prev = obs[2]
    setpoint = obs[3]
    
    norm_error = error / NORM_ERROR
    norm_y = y / NORM_Y
    norm_u_prev = u_prev / NORM_U
    norm_setpoint = setpoint / NORM_SP
    
    normalized_obs = np.array([norm_error, norm_y, norm_u_prev, norm_setpoint], dtype=np.float32)
    return normalized_obs

state_dim = 4  # [error1, y1, u[-1], setpoint1]
action_dim = 1  # u1 e u2
max_action = 6

n_episodes = 1000
max_steps = 2000
num_treino_per_step = 1

#Normalização 
NORM_ERROR = 10.0   # Assumindo que o erro não passa muito disso
NORM_Y = 10.0       # Saída máxima do sistema
NORM_U = 6.0        # Ação máxima (definida no seu treino como 6)
NORM_SP = 6.0      # Setpoint máximo

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")
        
shared_buffer = BufferMADDPG(buffer_capacity=1000000, batch_size=2048,
                            state_dim=state_dim, action_dim=action_dim,
                            num_agents=2)
agent1 = MADDPGAgent(
    name = 'AGENTE UNO',
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    buffer=shared_buffer,
    agent_id=0, 
    device=device,
    batch_size=2048,
    num_agents=2)

agent2 = MADDPGAgent(
    name = 'AGENTE DOS',
    state_dim=state_dim,   
    action_dim=action_dim,
    max_action=max_action,
    buffer=shared_buffer,
    agent_id=1,
    device=device,
    batch_size=2048,
    num_agents=2)

all_agents = [agent1, agent2]
noise_std = 0.5
initial_noise_std = 0.5
final_noise_std = 0.05
noise_decay = 0.999 # Taxa de decaimento por episódio

'''
total_rewards_history = []
plt.ion()  # Ativa o modo interativo do matplotlib
fig_reward, ax_reward = plt.subplots(figsize=(10, 5))
ax_reward.set_title("Recompensa Total por Episódio")
ax_reward.set_xlabel("Episódio")
ax_reward.set_ylabel("Recompensa Total Acumulada")
'''
# --- Fim da Inicialização ---

env = TITOSystem(K= np.array([[3,2],[2,5]]), wn1=1.0, wn2=0.5, zeta1=0.3, zeta2=0.4,
                 max_steps=max_steps, setpoint=[1.0, 1.0], render_mode="human")



for ep in range(n_episodes):
    obs, _ = env.reset()
    obs1 = obs[0]
    obs2 = obs[1]
    total_reward = 0.0
    total_reward1 = 0.0
    total_reward2 = 0.0
    
    for step in range(max_steps):
        
        obs1_norm = normalize_observation(obs1)
        obs2_norm = normalize_observation(obs2)

        obs_norm = np.array([obs1_norm, obs2_norm])
        
        state_tensor1 = torch.FloatTensor(obs1_norm).unsqueeze(0)
        state_tensor2 = torch.FloatTensor(obs2_norm).unsqueeze(0)

        action1 = agent1.select_action(state_tensor1, noise_std)
        action2 = agent2.select_action(state_tensor2, noise_std)

        
        actions = [action1, action2]
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        next_obs1 = next_obs[0]
        next_obs2 = next_obs[1]

        next_obs1_norm = normalize_observation(next_obs1)
        next_obs2_norm = normalize_observation(next_obs2)
        next_obs_norm = np.array([next_obs1_norm, next_obs2_norm])

        reward1 = rewards[0]
        reward2 = rewards[1]

        
        shared_buffer.record((obs_norm, actions, rewards, next_obs_norm))
        
        obs = next_obs

        obs1 = next_obs1
        obs2 = next_obs2
        
        total_reward1 += reward1
        total_reward2 += reward2
        
        if shared_buffer.buffer_counter >= shared_buffer.batch_size:
            for _ in range(num_treino_per_step):
                agent1.train(all_agents)
                agent2.train(all_agents)
            
        if terminated or truncated:
            break
    
    total_reward = total_reward1 + total_reward2
    print(f"Episódio {ep+1}, Recompensa Total: {total_reward:.2f} | Agente 1: {total_reward1:.2f} | Agente 2: {total_reward2:.2f} ")
    
    '''
    ax_reward.cla() # Limpa o gráfico anterior
    ax_reward.plot(total_rewards_history) # Plota o histórico
    ax_reward.set_title("Recompensa Total por Episódio --- MADDPG")
    ax_reward.set_xlabel("Episódio")
    ax_reward.set_ylabel("Recompensa Total Acumulada")
    ax_reward.grid(True)
    plt.draw()
    plt.pause(0.01) # Pausa pequena para a GUI do gráfico atualizar
    '''
    # Decaimento do ruído
    noise_std = max(final_noise_std, initial_noise_std * (noise_decay ** ep))
    #env.render()
    if total_reward1 > -100 and total_reward2 > -100:
        env.render()
        agent1.save("maddpg_model_best1")
        agent2.save("maddpg_model_best2")
        break