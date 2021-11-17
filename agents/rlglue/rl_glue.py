
from __future__ import print_function


class RLGlue:


    def __init__(self, env_class, agent_class):
        self.environment = env_class()
        self.agent = agent_class()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def rl_init(self, agent_init_info={}, env_init_info={}):
        self.environment.env_init(env_init_info)
        self.agent.agent_init(agent_init_info)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self, agent_start_info={}, env_start_info={}):

        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation):

        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
 
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):

        self.agent.agent_end(reward)

    def rl_env_start(self):

        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self):
        (reward, last_state, term) = self.environment.env_step(self.last_action)

        self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat

    def rl_cleanup(self):
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def rl_return(self):
        return self.total_reward

    def rl_num_steps(self):
        return self.num_steps

    def rl_num_episodes(self):
        return self.num_episodes
