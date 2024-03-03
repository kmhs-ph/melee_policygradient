from abc import ABC, abstractmethod
from melee import enums
import numpy as np
from melee_env.agents.util import *
import code
import threading
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import numpy as np
import math
from tqdm import tqdm
import random
from melee_env.agents.util import ObservationSpace, ActionSpace

from hyper_param import AHP

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.player_num = AHP["player_num"]
        self.pos_embed = nn.Linear(2 * self.player_num, 48)
        self.action_embed = nn.Linear(3 * self.player_num, 64)
        self.stock_embed = nn.Linear(self.player_num, 16)
        
        self.fc = nn.Linear(128, AHP["state_embed_size"])
        
    def forward(self, x):
        x = x.float()/10
        # x shape = batch_size x player_num x information
        batch_size = x.shape[0]
        pos = x[:,:,:2]
        pos = pos.flatten(1, -1)
        pos_embed = F.relu(self.pos_embed(pos))
        
        action = x[:,:,2:5]
        action = action.flatten(1, -1)
        action_embed = F.relu(self.action_embed(action))
        
        stock = x[:,:,5]
        stock = stock.flatten(1, -1)
        stock_embed = F.relu(self.stock_embed(stock))
        
        state = torch.cat([pos_embed, action_embed, stock_embed], axis=1)
        
        state_embed = F.relu(self.fc(state))
        
        return state_embed
        
    
class Core(nn.Module):
    def __init__(self):
        super(Core, self).__init__()
        self.n_layers = AHP["lstm_num_layer"]
        self.hidden_dim = AHP["lstm_hidden_size"]
        embedding_dim = AHP["state_embed_size"]

        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.n_layers, 
                            batch_first=True)
        
        self.hidden_state = None

    def forward(self, state_embed):
        batch_size = state_embed.shape[0]
                
        if self.hidden_state is None:
            self.hidden_state = self.init_hidden_state(batch_size=batch_size)

        lstm_output, self.hidden_state = self.lstm(state_embed, self.hidden_state)
        lstm_output = lstm_output.reshape(batch_size, self.hidden_dim)

        return lstm_output

    def init_hidden_state(self, batch_size=1):
        '''
        TODO: use learned hidden state ?
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        or 
        device = next(self.parameters()).device
        self.hidden = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        self.cell_state = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_dim))        
        nn.init.uniform_(self.hidden, b=1./ self.hidden_dim)
        nn.init.uniform_(self.cell_state, b=1./ self.hidden_dim)
        '''

        device = next(self.parameters()).device
        hidden = (torch.zeros(batch_size, self.hidden_dim).to(device), 
                  torch.zeros(batch_size, self.hidden_dim).to(device))

        return hidden
    
class Action(nn.Module):
    def __init__(self):
        super(Action, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(AHP["lstm_hidden_size"], 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, AHP["action_size"]),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.stack(x)
    
class Model(nn.Module):
    def __init__(self, obs_space):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.core = Core()
        self.action_head = Action()
        self.observation_space = obs_space
        
        self.Rt = []
        self.probs = []

        self.gamma=0.99
        self.lr = 0.002
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
    def forward(self, x):        
        state_embed = self.encoder(x)
        lstm_output = self.core(state_embed)
        action = self.action_head(lstm_output)
        return action
    
    def put_data(self, item):
        R, prob = item
        self.Rt.append(torch.tensor([R]).to(prob.device))
        self.probs.append(prob)

    def train_net(self):
        print("train has been started")
        self.training_ended = False
        R = torch.stack(self.Rt[:-1])
        prob = torch.stack(self.probs[:-1])

        self.optimizer.zero_grad()
        loss = -torch.sum(torch.log(prob) * R)
        loss.backward(retain_graph = True)
        self.optimizer.step()

        self.Rt = []
        self.probs = []
        loss = 0
        self.core.hidden_state = None

    def train_while_stepping(self, step):
        # Start thread for training
        train = threading.Thread(target=self.train_net)
        train.start()
        while train.is_alive():
            step()
        # Wait for thread a to finish
        train.join()
        

    
class Agent(ABC):
    def __init__(self):
        self.agent_type = "AI"
        self.controller = None
        self.port = None  # this is also in controller, maybe redundant?
        self.action = 0
        self.press_start = False
        self.self_observation = None
        self.current_frame = 0

    @abstractmethod
    def act(self):
        pass

class nnAgent(Agent):
    def __init__(self, obs_space):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Model(obs_space).to(self.device)
        self.character = enums.Character.FOX

        self.action_space = ActionSpace()
        self.observation_space = obs_space
        self.action = 0
        
    @from_observation_space  # convert gamestate to an observation
    def act(self, observation):
        observation, reward, done, info = observation
        obs = torch.tensor(np.array([observation])).to(self.device)

        action_logits = self.net(obs)
        action_sampled = int(torch.multinomial(action_logits, 1)[0])
        self.action = action_sampled
        control = self.action_space(self.action)
        control(self.controller)
        return action_logits[0][self.action]