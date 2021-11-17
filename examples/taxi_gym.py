import gym
import ql
import numpy as np
import time

if __name__ =="__main__":
    t = time.time()
    env = gym.make('Taxi-v3')
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 10000
    max_steps = 2500
    n_tests = 2

    n_states, n_actions = env.observation_space.n, env.action_space.n
    agente = ql.QL_agent(alpha, gamma, epsilon, n_states,n_actions) #(alpha, gamma, epsilon, episodes, n_states, n_actions)
    
    episode_rewards = [[] for _ in range(episodes)]
    
    for episode in range(episodes):
        print("Episode: {0}".format(episode))
        s = env.reset()
        a = agente.take_action(s,False)
        episode_reward = 0
        steps = 0
        done = False
        while steps < max_steps:

            steps += 1    
            a = agente.take_action(s,True)
            s_, reward, done, info = env.step(a)
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
        s = env.reset()
        done = False
        epsilon = 0
        while True:
            time.sleep(1)
            env.render()
            a = agente.take_action(s,False)
            print("Chose action {0} for state {1}".format(a,s))
            s, reward, done, info = env.step(a)
            if done:
                if reward > 0:
                    print("Reached goal!")
                else:
                    print("Shit! dead x_x")
                time.sleep(3)
                break
         