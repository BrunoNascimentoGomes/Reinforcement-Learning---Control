import numpy as np
import torch
from utils.DDPGAgent import DDPGAgent  # seu agente
from modelos import ModeloSegundaOrdem  # sua planta
import torch.nn as nn


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
    critic_lr=1e-3,
    device=device
)

data_files = [
    "Machine_Teaching/dados_pid/pid_data1.npz",
    "Machine_Teaching/dados_pid/pid_data2.npz",
    "Machine_Teaching/dados_pid/pid_data3.npz",
    "Machine_Teaching/dados_pid/pid_data4.npz",
    "Machine_Teaching/dados_pid/pid_data5.npz"
]


for file in data_files:
    data = np.load(file)
    print(f"Carregando {file} -> {data.files}")

    data_obs = data['states']
    data_actions = data['actions']
    data_rewards = data['rewards']
    data_next_obs = data['next_states']
    print(f"  Observações: {data_obs.shape}, Ações: {data_actions.shape}, Recompensas: {data_rewards.shape}, Próximas Observações: {data_next_obs.shape}")

    for s, a, r, s_next in zip(data_obs, data_actions, data_rewards, data_next_obs):
        agent.replay_buffer.record((s, a, r, s_next))


# 2. Pré-treinamento supervisionado (Opcional, mas recomendado para Behavior Cloning)
n_pretrain_steps = 2000
print(f"Iniciando pré-treinamento do Ator por {n_pretrain_steps} passos...")
for i in range(n_pretrain_steps):
    # Amostrar um batch
    state_batch, action_batch, _, _ = agent.replay_buffer.sample_batch()
   
    # Converter para o dispositivo (se não estiver já)
    state_batch = state_batch.to(agent.device)
    action_batch = action_batch.to(agent.device)
    # Forward pass do ator
    predicted_actions = agent.actor(state_batch)
   
    # Loss: MSE entre a ação do Ator e a ação do Expert (PID)
    bc_loss = torch.nn.MSELoss()(predicted_actions, action_batch)
   
     # Otimização
    agent.actor_optimizer.zero_grad()
    bc_loss.backward()
    agent.actor_optimizer.step()

print("Pré-treinamento do Ator concluído.")

# 2. Pré-treinamento do CRÍTICO 
n_critic_pretrain_steps = 2000
print(f"Iniciando pré-treinamento do Crítico por {n_critic_pretrain_steps} passos...")
for i in range(n_critic_pretrain_steps):
    # Agora o Ator (e o actor_target) já imita o PID.
    # Podemos chamar agent.train() para treinar o Crítico
    # a aprender o valor (Q-value) da política do PID.
    
    # ATENÇÃO: NÃO ATUALIZE O ATOR AQUI
    # Congele o Ator temporariamente (ou use uma loss só do crítico)
    # A forma mais simples é chamar agent.train(), mas
    # o Ator vai sofrer um pouco.
    
    # A forma CORRETA é treinar SÓ o crítico:
    
    # Sample
    state_batch, action_batch, reward_batch, next_state_batch = agent.replay_buffer.sample_batch()
    state_batch = state_batch.to(agent.device)
    action_batch = action_batch.to(agent.device)
    reward_batch = reward_batch.to(agent.device)
    next_state_batch = next_state_batch.to(agent.device)

    # Calcule o target Q (usando o actor_target que já imita o PID)
    with torch.no_grad():
        next_actions = agent.actor_target(next_state_batch)
        target_Q = agent.critic_target(next_state_batch, next_actions)
        target_Q = reward_batch + (agent.gamma * target_Q)
    
    # Calcule o Q atual
    current_Q = agent.critic(state_batch, action_batch) # Usa a ação REAL do buffer
    
    # Loss SÓ DO CRÍTICO
    critic_loss = nn.MSELoss()(current_Q, target_Q)
    
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()
    
    # Atualize a target do crítico
    with torch.no_grad():
        for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
            target_param.data.copy_(agent.tau * param.data + (1 - agent.tau) * target_param.data)

print("Pré-treinamento do Crítico concluído.")

# Treino direto do agente com RL
'''
for i in range(2000):
    agent.train()
'''

print("Pré-treinamento concluído.")



#Teste para o pré-treinamento
env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=1.0, max_steps=max_steps)
obs, _ = env.reset()
total_reward = 0.0
sp_index = 0
setpoint_schedule = [(0, 1.0), (250, 3), (500, 5), (750, 2), (1000,4), (1250,0), (1500, -2), (1750, -5)]

for step in range(max_steps):
    #print("Setpoint:", env.setpoint)
    if sp_index < len(setpoint_schedule) and step >= setpoint_schedule[sp_index][0]:
        env.setpoint = setpoint_schedule[sp_index][1]
        sp_index += 1
        # Convertendo obs para tensor batch (1, state_dim)
        
    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        # Seleciona ação do agente
    action = agent.select_action(state_tensor, noise=0)

        # Executa ação na planta
    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

        # Próximo estado
    obs = next_obs

    if terminated or truncated:
        break
env.render()




# Ruído para exploração
noise_std = 0.1

initial_noise_std = 0.1 # Aumentei o ruído inicial para maior exploração
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
    if total_reward > 4000:
        env.render()
        agent.save("ddpg_model_best")
        break

# =============================
# Render final
#
env.render()
