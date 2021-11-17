
"""
"""

from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseEnvironment:
    """
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)

    @abstractmethod
    def env_init(self, env_info={}):
        """
        """

    @abstractmethod
    def env_start(self):
        """
        """

    @abstractmethod
    def env_step(self, action):
        """
        """

    @abstractmethod
    def env_cleanup(self):
        """ """

    @abstractmethod
    def env_message(self, message):
        """
        """
