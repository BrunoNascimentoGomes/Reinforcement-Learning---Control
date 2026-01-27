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
<<<<<<< Updated upstream
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
=======
        self.fc1 = nn.Linear(state_dim, 64) #400
        self.bn1 = nn.LayerNorm(64) #400
        
        self.fc2 = nn.Linear(64, 64) #400,300
        self.bn2 = nn.LayerNorm(64) #300
        
        self.fc3 = nn.Linear(64, action_dim) #300,action_dim
>>>>>>> Stashed changes
        self.max_action = max_action
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        #Para o PettingZoo
        x = torch.sigmoid(self.fc3(x))
        
        #Para o normal
        #x = self.max_action * torch.tanh(self.fc3(x))

        return x

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
<<<<<<< Updated upstream
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
    
=======
        self.fc1 = nn.Linear(state_dim + action_dim, 128) #400
        self.bn1 = nn.LayerNorm(128) #400
        self.fc2 = nn.Linear(128, 128) #400,300
        self.bn2 = nn.LayerNorm(128) #300
        self.fc3 = nn.Linear(128, 1) #300

>>>>>>> Stashed changes
    def forward(self, state, action):
        x = torch.cat([state, action],1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CriticNetworkMADDPG(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(CriticNetworkMADDPG, self).__init__()
<<<<<<< Updated upstream
        self.fc1 = nn.Linear(state_dim * num_agents+ action_dim*num_agents, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
=======
        self.fc1 = nn.Linear(state_dim * num_agents+ action_dim*num_agents, 64) #400
        self.bn1 = nn.LayerNorm(64) #400
        self.fc2 = nn.Linear(64, 64) #400,300
        self.bn2 = nn.LayerNorm(64) #300
        self.fc3 = nn.Linear(64, 1) #300,1
>>>>>>> Stashed changes
    
    def forward(self, state, action):
        '''
        
        O método forward espera dois tensores já "achatados" (flattened):

        state: [batch_size, state_dim * num_agents]

        action: [batch_size, action_dim * num_agents]'''
        x = torch.cat([state, action],1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# ---------------------------------GITHUB--------------------------------


import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork_(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork_, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork_(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork_, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.max_action = 6
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.max_action * T.tanh(x)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

