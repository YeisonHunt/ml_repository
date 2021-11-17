"""
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod


class BaseAgent:
    """
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info= {}):
        """ """

    @abstractmethod
    def agent_start(self, observation):
        """
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """
        """

    @abstractmethod
    def agent_end(self, reward):
        """
        """

    @abstractmethod
    def agent_cleanup(self):
        """ """

    @abstractmethod
    def agent_message(self, message):
        """
        """