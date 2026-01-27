from utils.pid import PID
import numpy as np
import matplotlib.pyplot as plt
import torch
from tito_sys.modelo_2x2 import TITOSystem

pid1 = PID(Kp=0.1, Ki=0.05, Kd=0, u_max=6, u_min=-6, dt=0.1)
pid2 = PID(Kp=1, Ki=0.1, Kd=0, u_max=6, u_min=-6, dt=0.1)

max_steps = 2000

env = TITOSystem(K= np.array([[3,2],[2,5]]), wn1=1.0, wn2=0.5, zeta1=0.3, zeta2=0.4,
                 max_steps=max_steps, setpoint=[1.0, 1.0], render_mode="human")

obs, _ = env.reset()
sp_index = 0
for step in range(max_steps):

    error1 = obs[0][0]
    error2 = obs[1][0]
    if step == 0:
         print(error1, error2)
    # Seleciona ação do agente
    action1 = pid1.compute(error1)
    action2 = pid2.compute(error2)
    action = [action1, action2]

    next_obs, reward, terminated, truncated, info = env.step(action)
     
    obs = next_obs

    if terminated or truncated:
         break

# Visualização
env.render()