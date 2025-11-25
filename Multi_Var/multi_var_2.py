# Usei os imports que você tinha no seu script original
from coluna_binary import DistilationColumn
from utils.pid import PID
from utils.DDPGAgent import DDPGAgent
import numpy as np
import torch


state_dim = 4  # [error, y, u passado do outro controlador, setpoint]
action_dim = 1
max_action = 3

n_episodes = 1500
max_steps = 6500
num_treino_per_step = 1

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU") 
# [MUDANÇA 1]:  
# Isso faz com que ela use os valores padrão (u=[2.706, 3.706]),
# o que define self.Vr e self.Ls com valores válidos desde o início.

#Estado inicial
x = np.array([0.989996,   0.98506869, 0.97890509, 0.97123076, 0.96173051, 0.95005371,
 0.93582724, 0.91867871, 0.89827236, 0.87435801, 0.84683005, 0.81578814,
 0.78158594, 0.74485109, 0.70646163, 0.66747398, 0.6290119,  0.59213988,
 0.55775034, 0.52648847, 0.49872563, 0.47417117, 0.4455474,  0.41300484,
 0.37704334, 0.33853025, 0.29864485, 0.25874749, 0.22020013, 0.18418648,
 0.15157992, 0.12288666, 0.09826274, 0.07758201, 0.0605254,  0.04666715,
 0.03554402, 0.02670329, 0.01973121, 0.01426654, 0.010004])

#Planta em estado estacionário
env = DistilationColumn(max_steps=max_steps, state = x)  

agent_D = DDPGAgent(
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

agent_B = DDPGAgent(
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



u_d_ss = 2.706
u_b_ss = 3.206
noise_std = 0.1

initial_noise_std = 0.1
final_noise_std = 0.05 
noise_decay = 0.995 # Taxa de decaimento por episódio
history = []

for ep in range(n_episodes):
    # Reset do ambiente
    env = DistilationColumn(max_steps=max_steps, state = x)  
    obs, _, _, _, _ = env.step(env.u)

    
    total_reward_D = 0.0
    total_reward_B = 0.0
    total_reward = 0

    
    # Reset do ruído ou estado interno do agente se necessário
    # agent_D.reset_noise() # Se seu agente tiver ruído Ornstein-Uhlenbeck

    for i in range(max_steps):
        if i == 0:
            print(env.u)
        if i == 100:
            # Step change no setpoint para forçar o agente a reagir
            env.setpoint = [0.996, 0.01] 
        
        # Extrai observação apenas do agente D
        obs_d = obs[0]
        obs_b = obs[1]
        
        state_tensor_d = torch.FloatTensor(obs_d).unsqueeze(0).to(device) # Mande pro device
        state_tensor_b = torch.FloatTensor(obs_b).unsqueeze(0).to(device)
        # Seleciona ação
        action_d = agent_D.select_action(state_tensor_d, noise=noise_std)
        action_b = agent_B.select_action(state_tensor_b, noise= noise_std)
        
        # [CORREÇÃO 3]: Clipping físico
        # A rede dá o delta. Somamos ao estacionário.
        u_d = action_d + u_d_ss
        u_b = action_b + u_b_ss
        history.append(u_d)
         
        
        # Monta vetor de controle [Lr, Vs]
        u = [float(u_d), float(u_b)] 

        # Passo no ambiente
        next_obs, reward, terminated, truncated, info = env.step(u)
        
        total_reward_D += reward[0]
        total_reward_B += reward[1]
        
        
        # Grava no buffer (Obs atual, Ação da rede, Reward, Prox Obs)
        # Importante: Grave 'action_d' (saída da rede), não 'u_d' (valor físico), 
        # a menos que seu agente espere o contrário. Normalmente grava-se a saída da rede.
        agent_D.replay_buffer.record((obs[0], action_d, reward[0], next_obs[0]))
        agent_B.replay_buffer.record((obs[1], action_b, reward[1], next_obs[1]))
        # Treinamento
        if agent_D.replay_buffer.buffer_counter >= agent_D.batch_size:
             for _ in range(num_treino_per_step):
                agent_D.train()
        if agent_B.replay_buffer.buffer_counter >= agent_B.batch_size:
            for _ in range(num_treino_per_step):
                agent_B.train()

        
        obs = next_obs 
        
        if terminated or truncated:
            break
         
    print(f"Episode {ep+1} finished | Total reward D: {total_reward_D:.2f} | Total reward B: {total_reward_B:.2f}")
    #print(history)
    total_reward = total_reward_B + total_reward_D
    env.render()
    
    # Decaimento do ruído
    noise_std = max(final_noise_std, initial_noise_std * (noise_decay ** ep))

    if total_reward > 500:
        env.render()
        agent_D.save("ddpg_model_bestD")
        agent_B.save("ddpg_model_bestB")
        break

print("Simulação concluída.")
print("Gerando gráficos...")
env.render()

print("Execução finalizada.")
