from utils.buffer import BufferMADDPG
from utils.MADDPGAgent import MADDPGAgent
import torch.optim as optim
import random
import numpy as np
import torch

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
shared_buffer = BufferMADDPG(buffer_capacity=2, batch_size=1,
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
    batch_size=1,
    num_agents=2)

agent2 = MADDPGAgent(
    name = 'AGENTE DOS',
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    buffer=shared_buffer,
    agent_id=1, 
    device=device,
    batch_size=1,
    num_agents=2)      
all_agents = [agent1, agent2] 

for _ in range(2):
    state1 = np.random.rand(4,1)
    action1 = random.random()
    reward1 = random.random()
    next_state1 = np.random.rand(4,1)

    state2 = np.random.rand(4,1)
    action2 = random.random()
    reward2 = random.random()
    next_state2 = np.random.rand(4,1)

    state = np.array([state1, state2])
    action = np.array([action1, action2])
    reward = np.array([reward1, reward2])
    next_state = np.array([next_state1, next_state2])


    obs = (state, action, reward, next_state)
    shared_buffer.record(obs)

print("Buffer preenchido. Iniciando treino dos agentes...")
print('------------------------------')
print("States = ", shared_buffer.state_buffer)
print("Actions = ", shared_buffer.action_buffer)
print("Rewards = ", shared_buffer.reward_buffer)
print("Next States = ", shared_buffer.next_state_buffer)

cond = True
for agent in all_agents:
    agent.train(all_agents)
    while cond:
        cond = input("Continuar treinando o agente {}? (s/n) ".format(agent.name))
        if cond.lower() == 's':
            cond = False
        elif cond.lower() == 'n':   
            cond = True
            

