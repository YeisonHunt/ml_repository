#import gym
import ql
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import itertools

# 1. Generate a random array of 16 integers between 30 and 65. 
# 2. Return the array.
def some_weights():
    ws = np.random.randint(30, 65, size = 16)
    return ws

# 1. First, it checks if the action is valid. If it is not, it returns a reward of -100 and the current state.
# 2. If the action is valid, it checks if the agent has reached the goal. If it has, it returns a reward of 100 and the current state.
# 3. If the agent hasn’t reached the goal, it checks if the agent has reached a pit. If it has, it returns a reward of -100 and the current state.
# 4. If the agent hasn’t reached the goal or a pit, it returns a reward of -10 times the number of steps taken so far.
# 5. The last step is to update the current state to the new state.

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
    else:
        if(activator[s][2] == 'L' and activator[s][0] == 2):
            reward = 100
            s_ = s
            done = True
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


    _s = s
    return _s, s_, reward, done, data
    


# It creates a random number between 0 and 4.
def reset():
    initial_state = [j for j in range(0,5)]
    random_choice = np.random.choice(initial_state, size=1)
    return random_choice[0]

# 1. The function action_x_inception takes in 5 parameters: next_a, next_b, next_c, next_d, and inception.
# 2. If inception is 1, then the function returns next_a.
# 3. If inception is 2, then the function returns next_b.
# 4. If inception is 3, then the function returns next_c.
# 5. If inception is anything else, then the function returns next_d.
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
    # 1. First, we create a new network with the following nodes: 1, 2, 3, and 4.
    # 2. Next, we define the actions that can be taken at each node.
    # 3. Then, we define the weights of each edge.
    # 4. Finally, we define the position of each node.
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

    node_posterior = nx.get_node_attributes(new_net, 'pos')
    nx.draw_networkx(new_net, node_posterior, node_size=450)

    labels = nx.get_edge_attributes(new_net, 'weight')

    nx.draw_networkx_edge_labels(new_net, node_posterior, edge_labels=labels)
    
    # NETWORK GRAPH
    plt.show()

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

    n_states, n_actions = 6, 3
    agente = ql.QL_agent(alpha, gamma, epsilon, n_states,n_actions) #(alpha, gamma, epsilon, episodes, n_states, n_actions)
    
    episode_rewards = [[] for _ in range(episodes)]
    
    # 1. The for loop is responsible for repeating the same action for a given number of episodes.
    # 2. The print statement is just for visualization.
    # 3. The agente.take_action() method is responsible for selecting an action using the epsilon-greedy policy.
    # 4. The step() method is responsible for taking a step in the environment and returning the next state, the reward, and other information.
    # 5. The agente.updateQ() method is responsible for updating the Q-table.
    # 6. The end_ep variable keeps track of the total time taken by the agent to complete the episode.
    # 7. The episode_rewards list is used to keep track of the rewards obtained by the agent in each episode.
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


    # 1. First, it creates a new environment and a new agent.
    # 2. It then runs the environment loop, which continues until the agent reaches the goal or falls into a hole.
    # 3. At each iteration, the agent chooses an action, and the environment returns an observation and a reward.
    # 4. The agent then stores this transition in its memory, and uses the new observation to choose the next action.
    # 5. After the agent has reached the goal or fallen into a hole, the loop resets the environment, clears its memory, and restarts.
    # 6. The loop continues until the agent has reached the goal 10 times.
    # 7. After the agent has reached the goal 10 times, the python code ends.
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