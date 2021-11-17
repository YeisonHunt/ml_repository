#import gym
import ql
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import itertools

def some_weights():
    ws = np.random.randint(30, 65, size = 16)
    return ws

# def step(s, a, possible_actions, new_net, steps, _s):
#     data = {}
#     impossibles = 0
#     for j in range(0,len(possible_actions)):
#         if(a == possible_actions[j]):
#             impossibles = 1
#     if(impossibles == 0):
#         reward = -55
#         s_ = s
#         done = False
#     else:
#         if(activator[s][0] == activator[s][1]):
#             reward = 100
#             s_ = s
#             done = True

#         else: 
#             done = False
#             suma = a + 1
#             for k in range(0,len(activator)):
#                 if (suma == activator[k][0] and activator[s][1] == activator[k][1]):
#                     s_ = k
#                     break
#             if (s_ == s):
#                 reward = -35
#             else:
#                 reward = -2 * steps
#     _s = s
#     return _s, s_, reward, done, data

def step(s, a, possible_actions, new_net, steps, _s):
    data = {}
    impossibles = 0
    for j in range(0,len(possible_actions)):
        if(a == possible_actions[j]):
            impossibles = 1
    
    if(impossibles == 0):
        reward = -100
        s_ = s
        done = False

    # if(a == activator[s][1]):
    #     reward = -100
    #     s_ = s
    #     done = False
    # if(activator[s][2] == 'L' and a == 2):
    # if((activator[s][2] == 'L' and a == 2) or (activator[s][2] == 'H' and a == 3)):
    else:
        if(activator[s][2] == 'L' and activator[s][0] == 2):
            reward = 100
            s_ = s
            done = True
        # elif(activator[s][2] == 'H' and a == 2):
        #     reward = -55
        #     s_ = s
        #     done = False
        elif(activator[s][2] == 'H' and activator[s][0] == 4):
            reward = 100
            s_ = s
            done = True

        else:
            done = False 
            suma = a + 1
            for x in range(0,len(activator)):
                    if (suma == activator[x][0] and activator[s][1] == activator[x][1] and activator[s][2] == activator[x][2]):
                        s_ = x
                        break
            if (s_ == _s):
                reward = -100
            else:
                reward = -10 * steps

    
    # elif(activator[s][2] == 'H' and a == 4):
    # # elif(activator[s][2] == 'H' and a == 3):
    #     reward = 100
    #     s_ = s
    #     # s_ = activator[4]
    #     done = True
    # else: 
    #     reward = -100
    #     s_ = s
    #     done = False
    _s = s
    return _s, s_, reward, done, data
    


def reset():
    initial_state = [j for j in range(0,5)]
    random_choice = np.random.choice(initial_state, size=1)
    return random_choice[0]

def action_x_inception(next_a, next_b, next_c, next_d, inception):
    if (inception == 1):
        return next_a
    elif (inception == 2):
        return next_b
    elif (inception == 3):
        return next_c
    else:
        return next_d

if __name__ =="__main__":
    new_net = nx.DiGraph()
    list_nodes = [i for i in range(1,5)]
    actions = list_nodes
    # Defining nodes neighbohrs 
    next_a = [2,3,4]
    next_b = [1]
    next_c = [1]
    next_d = [1]

    new_net.add_nodes_from(list_nodes)
    new_net.nodes()

    ws = [25, 32, 10, 7]

    list_links = [(1, 2, ws[0]), (1, 3, ws[1]), (1, 4, ws[2])]
    new_net.add_weighted_edges_from(list_links)
    new_net.edges()

    new_net.nodes[1]['pos'] = (2, 2)
    new_net.nodes[2]['pos'] = (1, 1)
    new_net.nodes[3]['pos'] = (3, 1)
    new_net.nodes[4]['pos'] = (2, 1)
    # new_net.nodes[5]['pos'] = (1, 0)
    # new_net.nodes[6]['pos'] = (3, 0)
    # new_net.nodes[7]['pos'] = (4, 0)

    node_posterior = nx.get_node_attributes(new_net, 'pos')
    nx.draw_networkx(new_net, node_posterior, node_size=450)
    # arc_weight = nx.get_edge_attributes(new_net, 'height')
    labels = nx.get_edge_attributes(new_net, 'weight')
    # nx.draw_networkx_edge_labels(new_net, node_posterior, edge_labels=arc_weight)
    nx.draw_networkx_edge_labels(new_net, node_posterior, edge_labels=labels)
    
    # NETWORK GRAPH
    plt.show()
    
    # container = [actions, [6,7]]
    # container = [actions, [1,1]]
    container = [actions, [1], ['H', 'L']]
    activator = list(itertools.product(*container))
    print(activator)
    """
        Original taxi's part.
    """

    t = time.time()
    #env = gym.make('Taxi-v3')
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 10000
    # episodes = 500
    max_steps = 2500
    n_tests = 2

    #n_states, n_actions = env.observation_space.n, env.action_space.n
    n_states, n_actions = 6, 3
    # n_states, n_actions = 42, 7
    agente = ql.QL_agent(alpha, gamma, epsilon, n_states,n_actions) #(alpha, gamma, epsilon, episodes, n_states, n_actions)
    
    episode_rewards = [[] for _ in range(episodes)]
    
    for episode in range(episodes):
        print("Episode: {0}".format(episode))
        #s = env.reset()
        s = reset()
        _s = s
        #a = agente.take_action(s,False)
        episode_reward = 0

        steps = 0
        done = False
        while steps < max_steps:
            steps += 1    
            a = agente.take_action(s,True)

            # My part
            #s_, reward, done, info = env.step(a)
            gamma = activator[s][0]
            azione = action_x_inception(next_a, next_b, next_c, next_d, gamma)
            _s, s_, reward, done, info = step(s, a, azione, new_net, steps, _s)
            # End of my part
            episode_reward += reward
            a_ = np.argmax(agente.Q[s_,:])
            agente.updateQ(reward,s,a,a_,s_,done) 
            s, a = s_ , a_
            if done:
                end_ep = time.time()
                episode_rewards[episode].append(episode_reward)
                break
    #Test model  
    for test in range(n_tests):
        print("Test #{0}".format(test))
        s = reset()
        _s = s
        done = False
        epsilon = 0
        st = 0
        steps = 0
        while True:
            time.sleep(1)
            origin = activator[s][0]
            azione = action_x_inception(next_a, next_b, next_c, next_d, gamma)
            #env.render()
            #a = agente.take_action(s,False)
            steps += 1

            if(st == 0):
                first_state = False
            else:
                first_state = True
            
            print("Estado actual: {0}".format(activator[s]))
            a = agente.take_action(s, first_state)
            print("Chose action {0} for state {1}".format(a,s))
            print(_s, s)
            first_state = True
            st += 1
            _s, s, reward, done, info = step(s,a,azione,new_net,steps,_s)
            print(azione, reward, done)
            if done:
                if reward > 0:
                    print("Reached goal!")
                else:
                    print("Shit! dead x_x")
                time.sleep(3)
                break   