import gym
import numpy as np
import matplotlib.pyplot as plt
import dqn_agent


def convert(state_): #convertir el estado de indice a formato de lista
    state = []
    for j in env.decode(state_):
        state.append(np.float32(j))
    return state


env = gym.make('Taxi-v3')
agente = dqn_agent.Agent(4,env.action_space.n)

env2 = gym.make('Taxi-v3')
episodes = 100
episode_rewards = [] #Lista para almacenar las recompensa de cada episodio
episode_rewards2 = []

for episode in range(episodes):
    print("Episode: {0}".format(episode))
    
    steps = 0
    done = False
    episode_reward = 0
    episode_reward2 = 0

    agente.handle_episode_start()    
    s = env.reset()
    a = agente.step(convert(s),0)
    
    s2 = env2.reset()    
    
    while True:
        steps += 1          
        s_, reward, done, info = env.step(int(a)) 
        episode_reward += reward
        a_ = agente.step(convert(s_),reward)
        s, a = s_ , a_

        #random actions
        a2 = env2.action_space.sample()
        s2, reward2, done2, info2 = env.step(a2)
        episode_reward2 += reward2
        
        if done:          
            episode_rewards.append(episode_reward)
            episode_rewards2.append(episode_reward2)
            break



#Graficar resultados:

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("")
plt.plot(episode_rewards,'b')
plt.plot(episode_rewards2,'r')
plt.legend()
plt.show()