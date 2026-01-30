import numpy as np
import torch
from utils.DDPGAgent import DDPGAgent_Done
from modelos import ModeloSegundaOrdem
from utils.networks import ActorNetwork

# Configurações
state_dim = 6
action_dim = 1
max_action = 1
max_steps = 1000


# Carrega a planta
env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=1.5, max_steps=max_steps)

# Carrega o ator treinado
actor = ActorNetwork(state_dim, action_dim, max_action)
actor.load_state_dict(torch.load("use_cases/modelo/Treinamento_salvo_actor.pth", map_location="cpu"))
actor.eval()

# Loop de simulação
obs, _ = env.reset()
sp_index = 0
for step in range(max_steps):
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action = actor(state_tensor)
    action = action.detach().cpu().numpy().flatten()
    
    obs, reward, terminated, truncated, info = env.step(action)

# Visualização
env.render()
