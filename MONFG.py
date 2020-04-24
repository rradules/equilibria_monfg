import random
import pandas as pd
import QLearnerSER as ql
import numpy as np
from QLearnerSER import QLearnerSER, calc_ser
from collections import Counter
import time
import argparse
from utils import *

def get_recommendations():
    # recommendations are set as a list of probabilities (rec_probs) and a list of recommendation pairs (recs)
    global recs, rec_probs  # these are specified next to the payoff matrices for the MONFG below
    dice = random.uniform(0.0, 1.0)
    sum_probs = 0.0
    selected_index = -1
    for rec in range(len(rec_probs)-1):
        sum_probs += rec_probs[rec]
        if dice < sum_probs:
            selected_index = rec
            break
    if selected_index < 0:
        selected_index = len(recs)-1
    recommendation = recs[selected_index].copy()
    return recommendation


def select_actions(states):
    selected = []
    for ag in range(num_agents):
        selected.append(agents[ag].select_action_mixed_nonlinear(states[ag]))
    return selected


def select_recommended_actions(states):
    selected = []
    for ag in range(num_agents):
        selected.append(agents[ag].select_recommended_action(states[ag]))
    return selected


def calc_payoffs():
    global payoffs
    payoffs.clear()
    for ag in range(num_agents):
        payoffs.append([payoffsObj1[selected_actions[0]][selected_actions[1]],
                        payoffsObj2[selected_actions[0]][selected_actions[1]]])


def decay_params():
    global alpha, epsilon
    alpha *= alpha_decay
    epsilon *= epsilon_decay
    for ag in range(num_agents):
        agents[ag].alpha = alpha
        agents[ag].epsilon = epsilon


def update():
    for ag in range(num_agents):
        agents[ag].update_q_table(prev_states[ag], selected_actions[ag], current_states[ag], payoffs[ag])


def do_episode(ep):
    global prev_states, selected_actions, payoffs, current_states
    if provide_recs:
        prev_states = get_recommendations()
    if ep < recommendation_time:
        selected_actions = select_recommended_actions(prev_states)
    else:
        selected_actions = select_actions(prev_states)
    calc_payoffs()
    update()
    if ep > recommendation_time:
        decay_params()

def reset(opt=False, rand_prob=False):
    global agents, current_states, selected_actions, alpha, epsilon
    agents.clear()
    for ag in range(num_agents):
        new_agent = QLearnerSER(ag, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt, multi_ce,
                                ce_ser[ag], single_ce, rand_prob, CE_sgn[ag])
        agents.append(new_agent)
    current_states = [0, 0]
    selected_actions = [-1, -1]
    alpha = alpha_start
    epsilon = epsilon_start


parser = argparse.ArgumentParser()


parser.add_argument('-opt_init', dest='opt_init', action='store_true', help="optimistic initialization")
parser.add_argument('-game', type=str, default='game1', help="which MONFG game to play")
parser.add_argument('-rand_prob', dest='rand_prob', action='store_true', help="rand init for optimization prob")

parser.add_argument('-mCE', dest='multi_ce', action='store_true', help="multi-signal CE")
parser.add_argument('-sCE', dest='single_ce', action='store_true', help="single-signal CE")
parser.add_argument('-rec_time', type=int, default=0, help="define the number of steps agents receive recommendation")

parser.add_argument('-runs', type=int, default=100, help="number of trials")
parser.add_argument('-episodes', type=int, default=10000, help="number of episodes")

'''
Game 1, run multi-signal CE, under SER
python MONFG.py -game game1 -mCE -rec_time 500

Game 1, run single-signal CE, under SER
python MONFG.py -game game1 -sCE -rec_time 500

Game 1, run NE, under SER
python MONFG.py -game game1
'''


args = parser.parse_args()

game = args.game

num_objectives = 2
ce_ser = None

if game == 'game1':
    # original (Im)balancing act game
    payoffsObj1 = np.array([[4, 3, 2],
                            [3, 2, 1],
                            [2, 1, 0]])
    payoffsObj2 = np.array([[0, 1, 2],
                            [1, 2, 3],
                            [2, 3, 4]])
    rec_probs = [0.75, 0.25]
    recs = [[0, 1], [2, 1]]
    CE_sgn = [[0.75, 0, 0.25], [0, 1, 0]]


elif game == 'game2noM':
    # 2 action game that has no NE, it is the original (Im)balancing act game without M
    payoffsObj1 = np.array([[4, 2],
                            [2, 0]])
    payoffsObj2 = np.array([[0, 2],
                            [2, 4]])
    rec_probs = [0.25, 0.25, 0.25, 0.25]
    recs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    CE_sgn = [[0.5, 0.5], [0.5, 0.5]]

elif game == 'game2noR':
    # 2 action game that has one pure strategy NE, it is the original (Im)balancing act game without R
    # the pure strategy NE is (L,M)
    payoffsObj1 = np.array([[4, 3],
                            [3, 2]])
    payoffsObj2 = np.array([[0, 1],
                            [1, 2]])
    rec_probs = [1.0]
    recs = [[0, 1]]
    CE_sgn = [[1.0, 0.0], [0.0, 1.0]]

elif game == 'game4':
    # 3 action game that has 3 pure strategy NE
    # The pure strategy NE are for the action combinations (L,L), (M,M) and (R,R)
    payoffsObj1 = np.array([[4, 1, 2],
                            [3, 3, 1],
                            [1, 2, 1]])
    payoffsObj2 = np.array([[1, 2, 1],
                            [1, 2, 2],
                            [2, 1, 3]])
    rec_probs = [0.5, 0.5]
    recs = [[0, 0], [1, 1]]
    #rec_probs = [1.0]
    #recs = [[0, 0]]
    #recs = [[1, 1]]
    CE_sgn = [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]

else:
    # 2 action game that has multiple pure strategy NE
    payoffsObj1 = np.array([[4, 1],
                            [3, 3]])
    payoffsObj2 = np.array([[1, 2],
                            [1, 2]])
    rec_probs = [0.5, 0.5]
    recs = [[0, 0], [1, 1]]
    CE_sgn = [[0.5, 0.5], [0.5, 0.5]]


ce_ser_o = np.zeros(num_objectives)

rec_obj1 = [payoffsObj1[el[0], el[1]] for el in recs]
ce_ser_o[0] = np.dot(rec_probs, rec_obj1)

rec_obj2 = [payoffsObj2[el[0], el[1]] for el in recs]
ce_ser_o[1] = np.dot(rec_probs, rec_obj2)

print("Expected return o1 and o2: ", ce_ser_o)
ce_ser = [calc_ser(i, ce_ser_o) for i in range(2)]
print("SER Agent 1 and 2: ", ce_ser)


num_agents = 2
num_actions = payoffsObj1.shape[0]
num_states = num_actions  # as the number of states is equal to the number of possible actions to be recommended
agents = []
prev_states = [0, 0]
selected_actions = [-1, -1]
payoffs = [-1, -1]
current_states = [0, 0]
alpha = 0.05
alpha_start = 0.05
alpha_decay = 1
epsilon = 0.1
epsilon_start = 0.1
epsilon_decay = 0.999
gamma = 0.0  # this should always be zero as a MONFG is a stateless one shot decision problem
payoff_log = []
rand_prob = args.rand_prob

# set provide_recs to True to try to get the agents to learn a correlated equilibrium
# the correlated equilibrium is defined using rec_probs and recs above, next to the MONFG payoffs
# the recommendation_time parameter was set to 500 when learning a CE
recommendation_time = args.rec_time
multi_ce = args.multi_ce
single_ce = args.single_ce
provide_recs = multi_ce or single_ce
opt_init = args.opt_init

num_runs = args.runs
num_episodes = args.episodes

payoff_episode_log1 = []
payoff_episode_log2 = []
state_distribution_log = np.zeros((num_actions, num_actions))
action_hist = [[], []]
act_hist_log = [[], []]
window = 100
final_policy_log = [[], []]

if provide_recs:
    equil = 'CE'
else:
    equil = 'NE'

path_data = f'data/{game}'


if opt_init:
    path_data +='/opt_init'
else:
    path_data +='/zero_init'

if rand_prob:
    path_data += '/opt_rand'
else:
    path_data += '/opt_eq'


print(path_data)
mkdir_p(path_data)

start = time.time()
for r in range(num_runs):
    print("Starting run ", r)
    reset(opt_init, rand_prob)
    action_hist = [[], []]
    for e in range(num_episodes):
        do_episode(e)
        payoff_episode_log1.append([e, r, ql.calc_ser(0, payoffs[0])])
        payoff_episode_log2.append([e, r, ql.calc_ser(1, payoffs[1])])
        for i in range(num_agents):
            action_hist[i].append(selected_actions[i])
        if e >= 0.9 * num_episodes:
            state_distribution_log[selected_actions[0], selected_actions[1]] += 1

        payoffs = []

    # transform action history into probabilities
    for a, el in enumerate(action_hist):
        for i in range(len(el)):
            if i+window < len(el):
                count = Counter(el[i:i+window])
            else:
                count = Counter(el[-window:])
            total = sum(count.values())
            act_probs = [0, 0, 0]
            for action in range(num_actions):
                act_probs[action] = count[action] / total
            act_hist_log[a].append([i, r, act_probs[0], act_probs[1], act_probs[2]])

end = time.time()
elapsed_mins = (end - start) / 60.0
print("Time elapsed: " + str(elapsed_mins))

info = ''
if provide_recs:
    info += f'CE_{recommendation_time}_'
    if single_ce:
        info += 'single'
    if multi_ce:
        info += 'multi'
else:
    info += 'NE'

columns = ['Episode', 'Trial', 'Payoff']
df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

df1.to_csv(f'{path_data}/agent1_{info}.csv', index=False)
df2.to_csv(f'{path_data}/agent2_{info}.csv', index=False)

columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
df1 = pd.DataFrame(act_hist_log[0], columns=columns)
df2 = pd.DataFrame(act_hist_log[1], columns=columns)

df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)


state_distribution_log /= num_runs * (0.1 * num_episodes)
df = pd.DataFrame(state_distribution_log)
df.to_csv(f'{path_data}/states_{info}.csv', index=False, header=None)