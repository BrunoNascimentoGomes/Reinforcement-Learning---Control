import numpy as np
import torch
from utils.DDPGAgent import DDPGAgent
from modelos import ModeloSegundaOrdem
from utils.networks import ActorNetwork_

# Configurações
state_dim = 4
action_dim = 1
max_action = 6
max_steps = 5000

setpoint_schedule = [
    (0, 1.0),
    (1069, 5),
    (1600, 1),
    (2000, 4),
]

# Carrega a planta
env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=2.0, max_steps=max_steps)

# Carrega o ator treinado
actor = ActorNetwork_(state_dim, action_dim, max_action)
actor.load_state_dict(torch.load("use_cases/modelo_7500_com/ddpg_model_best_actor.pth", map_location="cpu"))
actor.eval()

# Loop de simulação
obs, _ = env.reset()
sp_index = 0
for step in range(max_steps):
    if sp_index < len(setpoint_schedule) and step >= setpoint_schedule[sp_index][0]:
        env.setpoint = setpoint_schedule[sp_index][1]
        sp_index += 1


    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action = actor(state_tensor)
    action = action.detach().cpu().numpy().flatten()
    
    obs, reward, terminated, truncated, info = env.step(action)

# Visualização
env.render()
