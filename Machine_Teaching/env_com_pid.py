import numpy as np
import torch
from modelos import ModeloSegundaOrdem  # sua planta
from utils.pid import PID
# =============================
# Configurações do experimento
# =============================
state_dim = 4  # [error, y, delta_e, setpoint]
action_dim = 1
max_action = 6

max_steps = 2000

env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=1.0, max_steps=max_steps)
pid = PID(Kp=5+0.5, Ki=4.852-1, Kd=1, dt=0.02, u_min=-max_action, u_max=max_action)

states_hist = []
actions_hist = []
rewards_hist = []
next_states_hist = []

# =============================
# Loop de episódios
# =============================
setpoint_schedule = [(0, 1.0), (250, 3), (500, 5), (750, 2), (1000,4), (1250,0), (1500, -2), (1750, -5)]



obs, _ = env.reset()
sp_index = 0
for step in range(max_steps):
    #print("Setpoint:", env.setpoint)
    if sp_index < len(setpoint_schedule) and step >= setpoint_schedule[sp_index][0]:
        env.setpoint = setpoint_schedule[sp_index][1]
        sp_index += 1
        # Convertendo obs para tensor batch (1, state_dim)
    error = obs[0]
    # Seleciona ação do agente
    action = pid.compute(error)

    # Executa ação na planta
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    states_hist.append(obs)
    actions_hist.append(action) 
    rewards_hist.append(reward)
    next_states_hist.append(next_obs)


    
    obs = next_obs

    if terminated or truncated:
         break

# =============================
# Render final
save = True
if save:
    np.savez("pid_data.npz",
        states=np.array(states_hist, dtype=np.float32),
        actions=np.array(actions_hist, dtype=np.float32),
        rewards=np.array(rewards_hist, dtype=np.float32),
        next_states=np.array(next_states_hist, dtype=np.float32)
    )

env.render()


