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
        gym.utils.RecordConstructorArgs.__init__(
            self, 
            trials_per_episode=trials_per_episode
        )
        gym.Wrapper.__init__(self, env)
        self.trials_per_episode = trials_per_episode
        self.trial_counter = 0
        self.cumulative_reward_per_trial = np.zeros((trials_per_episode,), dtype=np.float32)

        # adding prev_action, prev_reward, prev_done (in one-hot)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(
                self.env.unwrapped.observation_space.n + self.env.unwrapped.action_space.n + 4,
            )
        )

    @staticmethod
    def __one_hot(value, num_classes):
        out = np.zeros((num_classes,), dtype=np.int64)
        out[value] = 1
        return out
        
    def reset(self, *, seed = None, options = None, hard_reset = False):
        """
        Reset the environment (the agent position). If it was the last trila of an episode, then the new goal position will be sampled. Otherwise, the goal position remains the same.
        """
        if self.trial_counter == 0 or hard_reset:
            self.env.unwrapped.goal_pos = self.env.unwrapped.generate_goal()
            # self.env.unwrapped.goal_pos = self.env.unwrapped.np_random.choice(np.array([[0, 0], [2, 2]]))
            self.cumulative_reward_per_trial = np.zeros((self.trials_per_episode,), dtype=np.float32)

        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.concatenate([
            self.__one_hot(obs, num_classes=self.env.unwrapped.observation_space.n),
            self.__one_hot(0, num_classes=self.env.unwrapped.action_space.n),
            self.__one_hot(0, num_classes=2),
            self.__one_hot(1, num_classes=2),
        ])
        info["trial_done"] = False
        info["trial_goal"] = self.env.unwrapped.goal_pos
        return obs, info
    
    def step(self, action):
        """
        Modified RL^2 step method. If an episode within the trial ends, it will return a new episode state with `False` done signal. Otherwise, the usual output is sent.
        """
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.cumulative_reward_per_trial[self.trial_counter] += reward

        if done:
            info["trial_done"] = True
            info["cumulative_reward_per_trial"] = self.cumulative_reward_per_trial
            info["current_trial"] = deepcopy(self.trial_counter)
            self.trial_counter += 1

            if self.trial_counter >= self.trials_per_episode:
                self.trial_counter = 0
            else:
                next_obs, _ = self.reset()
                return next_obs, reward, False, False, info
        else:
            info["trial_done"] = False
        
        # o_t+1, a_t, r_t, d_t
        next_obs = np.concatenate([
            self.__one_hot(next_obs, num_classes=self.env.unwrapped.observation_space.n),
            self.__one_hot(action, num_classes=self.env.unwrapped.action_space.n),
            self.__one_hot(int(reward), num_classes=2),
            self.__one_hot(int(done), num_classes=2),
        ])
        info["trial_goal"] = self.env.unwrapped.goal_pos
        return next_obs, reward, terminated, truncated, info


class RenderScalerDarkRoom(gym.Wrapper):
    """
    Wrapper that scales render height and width with a `scale_factor` multiplier. The final resolution shape is calculated as follows: (`base_env.size` * `scale_factor`, `base_env.size` * `scale_factor`).
    """
    def __init__(
        self,
        env,
        scale_factor=100,
        draw_lines=True,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, 
            scale_factor=scale_factor, 
            draw_lines=draw_lines,
        )
        gym.Wrapper.__init__(self, env)

        self.scale_factor = scale_factor
        self.draw_lines = draw_lines
    
    def render(self):
        if self.render_mode == "rgb_array":
            base_grid = np.full((self.env.unwrapped.size, self.env.unwrapped.size, 3), fill_value=(255, 255, 255), dtype=np.uint8)
            base_grid[self.env.unwrapped.goal_pos[0], self.env.unwrapped.goal_pos[1]] = (255, 0, 0)

            if self.env.unwrapped.agent_pos is None:
                raise AttributeError("The agent position is `None`. Execute `env.reset()` first")

            base_grid[int(self.env.unwrapped.agent_pos[0]), int(self.env.unwrapped.agent_pos[1])] = (0, 255, 0)
            scaled_grid = base_grid.repeat(self.scale_factor, axis=0).repeat(self.scale_factor, axis=1)

            if self.draw_lines:
                height, width = scaled_grid.shape[:2]

                for x in range(self.scale_factor, width, self.scale_factor):
                    scaled_grid[:, x - 1:x + 1] = [0, 0, 0]
                
                for y in range(self.scale_factor, height, self.scale_factor):
                    scaled_grid[y - 1:y + 1, :] = [0, 0, 0]
            
            return scaled_grid