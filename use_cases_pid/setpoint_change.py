import numpy as np
import torch
from utils.DDPGAgent import DDPGAgent
from modelos import ModeloSegundaOrdem
from utils.networks import ActorNetwork_
from utils.pid import PID

# Configurações
state_dim = 4
action_dim = 1
max_action = 6
max_steps = 5000

setpoint_schedule = [
    (0, 1.0),
    (1069, 5),
    (2500, 1),
]

# Carrega a planta
env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=2.0, max_steps=max_steps)
pid = PID(Kp=5, Ki=4, Kd=2, dt=0.02, u_min=-max_action, u_max=max_action)


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

    next_obs, reward, terminated, truncated, info = env.step(action)
     
    obs = next_obs

    if terminated or truncated:
         break

# Visualização
env.render()
