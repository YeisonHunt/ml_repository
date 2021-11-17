import time
import main_agent
import ten_arm_env
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from rlglue.rl_glue import RLGlue
import ql 

def our_argmax(q_values):
    """
    Imput: list of q_values 
    Returns: Index of the item with the highest value. Breaks ties randomly.
    """
    value = float("-inf")
    breaks = []
    
    for i in range(len(q_values)):
        if q_values[i]>value:
            value=q_values[i]
            breaks = []
        if q_values[i]==value:
            breaks.append(i)
    return np.random.choice(breaks)


def main():
    # intialise data of lists. 
        data = {'NS-1':[1,np.nan,0,np.nan,np.nan,1],\
                'NS-2':[np.nan,0,np.nan,1,np.nan,np.nan],\
                "NS-3":[np.nan,np.nan,np.nan,np.nan,1,np.nan]} 

        # Create DataFrame 
        df = pd.DataFrame(data) 

        num_runs = 250                    # The number of times the experiment is runned
        num_steps = 1200                  # The number of times our agent eats each ice cream ("Number of pulls on each arm")
        env = ten_arm_env.Environment     # Ten K-armed bandir environment (RLglue)
        # agent = GreedyAgent               # Selection of the agent
        agent_info = {"num_actions": 10}  # Number of arms our agent has
        env_info = {}                     # We create the empty environment

        my_averages = []


        # QL AGENT 

        t = time.time()
        # env = gym.make('Taxi-v3')
        alpha = 0.4
        gamma = 0.999
        epsilon = 0.9
        episodes = 10000
        max_steps = 2500
        n_tests = 2

        # n_states, n_actions = env.observation_space.n, env.action_space.n
        n_states, n_actions = 4, 10
        agent = ql.QL_agent(alpha, gamma, epsilon, n_states,n_actions) #(alpha, gamma, epsilon, episodes, n_states, n_actions)
        
        episode_rewards = [[] for _ in range(episodes)]

    # END QL AGENT

        our_best_average = 0
        for run in tqdm(range(num_runs)):           # tqdm is what creates the progress bar below
                np.random.seed(run)
                
                rl_glue = RLGlue(env, agent)          # We create a new RLGlue experiment with the env and agent we chose above
                rl_glue.rl_init(agent_info, env_info) # We pass RLGlue its requirements to initialize the agent and environment
                rl_glue.rl_start()                    # Vamonos!!

                our_best_average += np.max(rl_glue.environment.arms)
                
                score = [0]
                average = []
                
                for i in range(num_steps):
                        reward, _, action, _ = rl_glue.rl_step() # The environment and agent take a step and returns the reward, and action taken.
                        score.append(score[-1] + reward)
                        average.append(score[-1] / (i + 1))
                my_averages.append(average)

        plt.figure(figsize=(17, 7))
        plt.plot([our_best_average / num_runs for _ in range(num_steps)], linestyle="--")
        plt.plot(np.mean(my_averages, axis=0))
        plt.legend(["Upper Bound", "Greedy Agent"])
        plt.title("Average Reward Greedy Agent")
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.show()
        greedy_average_reward = np.mean(my_averages, axis=0)

if __name__ == "__main__":
    main()