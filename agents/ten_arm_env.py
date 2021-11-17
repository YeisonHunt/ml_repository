#!/usr/bin/env python

from rlglue.environment import BaseEnvironment

import numpy as np

class Environment(BaseEnvironment):

    actions = [0]

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0
        self.arms = []
        self.seed = None

    def env_init(self, env_info={}):

        self.arms = np.random.randn(10)
        local_observation = 0  

        self.reward_obs_term = (0.0, local_observation, False)


    def env_start(self):
  
        return self.reward_obs_term[1]

    def env_step(self, action):


        reward = self.arms[action] + np.random.randn()

        obs = self.reward_obs_term[1]

        self.reward_obs_term = (reward, obs, False)

        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):

        if message == "provide current reward":
            return "{}".format(self.reward_obs_term[0])

        return "Error, cant respond to your inquiry"
