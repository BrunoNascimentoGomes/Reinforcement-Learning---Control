import torch
import torch.nn as nn

 
'''
state_dim: dimensão do vetor de estado (entrada da rede).

action_dim: dimensão do vetor de ação (saída da rede).

max_action: valor máximo permitido para a ação 
(ex: se o atuador físico vai de -1 a 1 ou 0 a 100).
'''
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.LayerNorm(400)
        
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.LayerNorm(300)
        
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)
    
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # Saída (sem BN aqui, pois usamos tanh e queremos o range da ação)
        x = self.max_action * torch.tanh(self.fc3(x))

        return x

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.bn1 = nn.LayerNorm(400) # <--- Adicionado
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        return x

class CriticNetworkMADDPG(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(CriticNetworkMADDPG, self).__init__()
        self.fc1 = nn.Linear(state_dim * num_agents+ action_dim*num_agents, 400)
        self.bn1 = nn.LayerNorm(400) # <--- Adicionado
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        '''
        
        O método forward espera dois tensores já "achatados" (flattened):

        state: [batch_size, state_dim * num_agents]

        action: [batch_size, action_dim * num_agents]'''
        x = torch.cat([state, action],1)
        '''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        '''
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        return x
        
    

