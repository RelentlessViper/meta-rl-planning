import gymnasium as gym
import numpy as np
from copy import deepcopy

class RL2DarkRoom(gym.Wrapper):
    """
    RL^2 wrapper for Dark Room environment
    """

    def __init__(
        self,
        env,
        trials_per_episode=3,
    ):
        super().__init__(env)
        self.main_env = env # Do not remove TimeLimit, etc.
        self.trials_per_episode = trials_per_episode
        self.trial_counter = 0
        self.cumulative_reward_per_trial = np.zeros((trials_per_episode,), dtype=np.float32)
    
    def reset(self, **kwargs):
        """
        Reset the environment (the agent position). If it was the last trila of an episode, then the new goal position will be sampled. Otherwise, the goal position remains the same.
        """
        if self.trial_counter == 0:
            self.main_env.unwrapped.goal_pos = self.main_env.unwrapped.generate_goal()
            self.cumulative_reward_per_trial = np.zeros((self.trials_per_episode,), dtype=np.float32)

        obs, info = self.main_env.reset(**kwargs)
        info["trial_done"] = False
            
        return obs, info
    
    def step(self, action):
        """
        Modified RL^2 step method. If an episode within the trial ends, it will return a new episode state with `False` done signal. Otherwise, the usual output is sent.
        """
        next_obs, reward, terminated, truncated, info = self.main_env.step(action)
        self.cumulative_reward_per_trial[self.trial_counter] += reward
        done = terminated or truncated


        if done:
            info["cumulative_reward_per_trial"] = self.cumulative_reward_per_trial
            info["current_trial"] = deepcopy(self.trial_counter)
            self.trial_counter += 1

            if self.trial_counter >= self.trials_per_episode:
                self.trial_counter = 0
                info["trial_done"] = True
                return next_obs, reward, terminated, truncated, info
            else:
                next_obs, _ = self.reset()
                info["trial_done"] = True
                return next_obs, reward, False, False, info
        
        info["trial_done"] = False
        return next_obs, reward, terminated, truncated, info
    
# import toymeta
# env = RL2DarkRoom(
#     gym.make("Dark-Room-3x3-v0"),
#     trials_per_episode=4,
# )
# obs, info = env.reset()

# for step in range(45):
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     print(step + 1, obs, reward, terminated, truncated, info)