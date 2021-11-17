from rlglue.agent import BaseAgent

import numpy as np

class Agent(BaseAgent):

    def __init__(self):
        self.last_action = None
        self.num_actions = None
        self.q_values = None
        self.step_size = None
        self.epsilon = None
        self.initial_value = 0.0
        self.arm_count = [0.0 for _ in range(10)]

    def agent_init(self, agent_info={}):
        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.q_values = np.ones(agent_info.get("num_actions", 2)) * self.initial_value
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.0)

        self.last_action = 0

    def agent_start(self, observation):
        self.last_action = np.random.choice(self.num_actions)  
        return self.last_action

    def agent_step(self, reward, observation):
        self.last_action = np.random.choice(self.num_actions)
        return self.last_action

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass