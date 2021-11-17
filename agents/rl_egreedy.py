import time
import main_agent
import ten_arm_env
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from rlglue.rl_glue import RLGlue

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

class GreedyAgent(main_agent.Agent):
        def agent_step(self, reward, observation):
                """
                The function agent_stpe takes in a reward and observation and 
                returns the action the agent chooses at that time step.

                Input
                reward: the reward the agent gets from the environment after the last action.
                observation: The state the agents obverves .
                Returns: The action chosen by the agent at the current time step.
                """
                #Number of times we visited each arm (action)
                self.arm_count[self.last_action]+=1

                #Step Size
                step_size=1/self.arm_count[self.last_action]

                #Incremental update rule
                self.q_values[self.last_action]=self.q_values[self.last_action]+step_size*(reward-self.q_values[self.last_action])

                present_action = our_argmax(self.q_values)

                self.last_action = present_action

                return present_action

class eGreedyAgent(main_agent.Agent):
        def agent_step(self, reward, observation):
                """
                The function agent_stpe takes in a reward and observation and 
                returns the action the agent chooses at that time step.
                
                Input: 
                reward: the reward the agent recieved from the environment after taking the last action.
                observation: the observed state the agent is in. 
                Returns:
                current_action: the action chosen by the agent at the current time step.
                """
                
                #Action value update
                self.arm_count[self.last_action]+=1
                
                step_size=1/self.arm_count[self.last_action]
                
                #Incremental update rule
                
                self.q_values[self.last_action]=self.q_values[self.last_action]+step_size*(reward-self.q_values[self.last_action])
                
                #Action Selection (e-greedy approach)
                our_prob = np.random.random()
                if our_prob < self.epsilon:
                        current_action = np.random.choice(self.num_actions)
                else:          
                        current_action = our_argmax(self.q_values)
                
                self.last_action = current_action
                
                return current_action

class QLearningAgent(main_agent.Agent):
        def agent_step(self, reward, observation):
                pass

def main():
        num_runs = 250                    # The number of times the experiment is runned
        num_steps = 1200                  # The number of times our agent eats each ice cream ("Number of pulls on each arm")
        env = ten_arm_env.Environment     # Ten K-armed bandir environment (RLglue)
        agent = GreedyAgent               # Selection of the agent
        agent_info = {"num_actions": 10}  # Number of arms ("Ice cream types") our agent has
        env_info = {}                     # We create the empty environment

        my_averages = []

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
        num_runs = 250
        num_steps = 1200
        epsilon = 0.1
        agent = eGreedyAgent
        env = ten_arm_env.Environment
        agent_info = {"num_actions": 10, "epsilon": epsilon}
        env_info = {}
        my_averages = []

        for run in tqdm(range(num_runs)):
                np.random.seed(run)
                
                rl_glue = RLGlue(env, agent)
                rl_glue.rl_init(agent_info, env_info)
                rl_glue.rl_start()

                scores = [0]
                averages = []
                for i in range(num_steps):
                        reward, _, action, _ = rl_glue.rl_step() 
                        scores.append(scores[-1] + reward)
                        averages.append(scores[-1] / (i + 1))
                my_averages.append(averages)

        plt.figure(figsize=(18, 7))
        plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
        plt.plot(greedy_average_reward)
        plt.title("Average Reward (Greedy Agent vs. epsilon-Greedy Agent)")
        plt.plot(np.mean(my_averages, axis=0))
        plt.legend(("Upper bound", "Greedy", "Epsilon: 0.1"))
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.show()

        epsilons = [0.0, 0.015,0.05, 0.1, 0.4]

        plt.figure(figsize=(18, 7))
        plt.plot([1.55 for _ in range(num_steps)], linestyle="--")

        n_q_values = []
        n_averages = []
        n_best_actions = []

        num_runs = 170

        for epsilon in epsilons:
            all_averages = []
            for run in tqdm(range(num_runs)):
                agent = eGreedyAgent
                agent_info = {"num_actions": 10, "epsilon": epsilon}
                env_info = {"random_seed": run}

                rl_glue = RLGlue(env, agent)
                rl_glue.rl_init(agent_info, env_info)
                rl_glue.rl_start()
                
                best_arm = np.argmax(rl_glue.environment.arms)

                scores = [0]
                averages = []
                my_best_action = []
                
                for i in range(num_steps):
                    reward, state, action, is_terminal = rl_glue.rl_step()
                    scores.append(scores[-1] + reward)
                    averages.append(scores[-1] / (i + 1))

                all_averages.append(averages)
                    
            plt.plot(np.mean(all_averages, axis=0))

        plt.title("Comparing different epsilons")
        plt.legend(["Best Possible"] + epsilons)
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.show()

if __name__ == "__main__":
        main()