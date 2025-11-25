from modelo_2x2 import TITOSystem
import numpy as np
import torch
from utils.DDPGAgent import DDPGAgent
from utils.buffer import Buffer
import matplotlib.pyplot as plt


state_dim = 4  # [error1, y1, u[-1], setpoint1]
action_dim = 1  # u1 e u2
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
        
agent1 = DDPGAgent(
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

agent2 = DDPGAgent(
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

'''
total_rewards_history = []
plt.ion()  # Ativa o modo interativo do matplotlib
fig_reward, ax_reward = plt.subplots(figsize=(10, 5))
ax_reward.set_title("Recompensa Total por Episódio")
ax_reward.set_xlabel("Episódio")
ax_reward.set_ylabel("Recompensa Total Acumulada")
'''
# --- Fim da Inicialização ---

noise_std = 0.1
initial_noise_std = 0.1
final_noise_std = 0.05
noise_decay = 0.995 # Taxa de decaimento por episódio

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
        state_tensor1 = torch.FloatTensor(obs1).unsqueeze(0)
        state_tensor2 = torch.FloatTensor(obs2).unsqueeze(0)

        action1 = agent1.select_action(state_tensor1, noise_std)
        action2 = agent2.select_action(state_tensor2, noise_std)
        
        actions = [action1, action2]
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        
        reward1 = rewards[0]
        reward2 = rewards[1]

        next_obs1 = next_obs[0]
        next_obs2 = next_obs[1]



        agent1.replay_buffer.record((obs1, action1, reward1, next_obs1))
        agent2.replay_buffer.record((obs2, action2, reward2, next_obs2))

        if agent1.replay_buffer.buffer_counter >= agent1.replay_buffer.batch_size:
            for _ in range(num_treino_per_step):
                agent1.train()
                agent2.train()
        
        obs = next_obs
        total_reward1 += reward1
        total_reward2 += reward2
        
            
        if terminated or truncated:
            break
    
    total_reward = total_reward1 + total_reward2
    print(f"Episódio {ep+1}, Recompensa Total: {total_reward:.2f} | Agente 1: {total_reward1:.2f} | Agente 2: {total_reward2:.2f} ")
    # --- NOVO: Bloco de atualização do gráfico ---
    '''
    ax_reward.cla() # Limpa o gráfico anterior
    ax_reward.plot(total_rewards_history) # Plota o histórico
    ax_reward.set_title("Recompensa Total por Episódio ----- DDPG")
    ax_reward.set_xlabel("Episódio")
    ax_reward.set_ylabel("Recompensa Total Acumulada")
    ax_reward.grid(True)
    plt.draw()
    plt.pause(0.01) # Pausa pequena para a GUI do gráfico atualizar
    '''
    # --- Fim do Bloco ---
    # Decaimento do ruído
    noise_std = max(final_noise_std, initial_noise_std * (noise_decay ** ep))
    #env.render()
    if total_reward > 500:
        env.render()
        agent1.save("maddpg_model_best1")
        agent2.save("maddpg_model_best2")
        break