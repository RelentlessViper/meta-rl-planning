import gymnasium as gym

class RL2DarkRoom(gym.Wrapper):
    """
    RL^2 wrapper for Dark Room environment
    """

    def __init__(
        self,
        env,
        episodes_per_trial=3
    ):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.episodes_per_trial = episodes_per_trial
        self.episode_counter = 0
    
    def reset(self, **kwargs):
        """
        Reset the environment (the agent position). If it was the last episode of a trial, then the new goal position will be sampled. Otherwise, the goal position remains the same.
        """
        if self.episode_counter == 0:
            obs, info = self.base_env.reset(**kwargs)
            self.base_env.goal_pos = self.base_env.generate_goal()
        else:
            obs, info = self.base_env.reset(**kwargs)
            
        return obs, info
    
    def step(self, action):
        """
        Modified RL^2 step method. If an episode within the trial ends, it will return a new episode state with `False` done signal. Otherwise, the usual output is sent.
        """
        next_obs, reward, terminated, truncated, info = self.base_env.step(action)
        done = terminated or truncated

        if done:
            self.episode_counter += 1

            if self.episode_counter >= self.episodes_per_trial:
                self.episode_counter = 0
                return next_obs, reward, terminated, truncated, info
            else:
                next_obs, info = self.reset()
                return next_obs, reward, False, False, info
        
        return next_obs, reward, terminated, truncated, info