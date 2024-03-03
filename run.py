from melee import enums
from melee_env.agents.util import ObservationSpace
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
import argparse
from arch_core import nnAgent
import torch
from torch.distributions import Categorical
import time
import os
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default='ssbm.iso', type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO")
parser.add_argument("--restore", default=False, type=bool)

args = parser.parse_args()

time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
MODEL_PATH = 'melee_PG/model/'
SAVE_PATH = os.path.join(MODEL_PATH, time_str)

obs_space = ObservationSpace()
AGENT = nnAgent(obs_space)
if args.restore:
    AGENT.net.load_state_dict(torch.load(os.listdir(MODEL_PATH)[-1]))
players = [AGENT, Rest()]

env = MeleeEnv(args.iso, players, fast_forward=False)

episodes = 10; reward = 0; score = 0.0
print_interval = 1
env.start()

torch.autograd.set_detect_anomaly(True)

for episode in range(episodes):
    gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    R = 0
    while not done: 
        prob = players[0].act(gamestate)
        players[1].act(gamestate)
        
        observation, r, done, info = obs_space(gamestate)
        R = r + AGENT.net.gamma * R
        AGENT.net.put_data((R, prob)) 
        score += float(r)

        gamestate, done = env.step()
    AGENT.net.train_while_stepping(env.console.step)
    
    save_path = SAVE_PATH + ".pth"
    print('Save model state_dict to', save_path)
    torch.save(AGENT.net.state_dict(), save_path)

    if episode % print_interval == 0 and episode != 0:
        print("# of episode :{}, avg score : {}".format(
            episode, score/print_interval))

        # 성공 조건 설정
        if score/print_interval > 500:
            print("Congratulations! solved :)")
            break

        score = 0.0