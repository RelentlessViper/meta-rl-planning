import gymnasium as gym
import numpy as np
from copy import deepcopy

class RL2BoxWorld(gym.Wrapper):
    """
    RL^2 wrapper for Box World environment
    """
    def __init__(
        self,
        env,
        trials_per_episode=3,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            trials_per_episode=trials_per_episode,
        )
        self.scaled_env = gym.wrappers.TransformObservation(
            env,
            lambda obs: obs.astype(np.float32) / 255.0,
            env.observation_space,
        )
        gym.Wrapper.__init__(self, self.scaled_env)
        self.trials_per_episode = trials_per_episode
        self.trial_counter = 0
        self.cumulative_reward_per_trial = np.zeros((trials_per_episode,), dtype=np.float32)

        # Add prev_action, prev_reward, prev_done
        # Channels:
        # 0-2: image (normalized to [0.0, 1.0])
        # 3-6: action (one-hot)
        # 7: prev_reward (normalized to [-1.0, 1.0])
        # 8: prev_done (0.0 or 1.0)
        # Final shape: (9, env.field_size, env.field_size)
        self.action_c = self.scaled_env.action_space.n.__int__()
        self.image_c, self.h, self.w = self.scaled_env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(
                (self.image_c + self.action_c + 1 + 1, self.h, self.w)
            )
        )
    
    @staticmethod
    def __one_hot_spacial(value, num_classes, h, w):
        out = np.zeros((num_classes, h, w), dtype=np.float32)
        out[value, :, :] = 1.0
        return out
    
    def reset(self, *, seed=None, options=None, hard_reset=False):
        """
        Reset the environment. If it was the last trial of an episode, then the new grid will be generated. Otherwise, the grid remains the same.
        """
        if self.trial_counter == 0 or hard_reset:
            keep_prev_world = False
            self.cumulative_reward_per_trial = np.zeros((self.trials_per_episode,), dtype=np.float32)
        else:
            keep_prev_world = True

        if isinstance(options, dict):
            options["keep_prev_world"] = keep_prev_world
        else:
            options = {"keep_prev_world": keep_prev_world}
        
        obs, info = self.scaled_env.reset(seed=seed, options=options)
        obs = np.concatenate(
            [
                obs,
                np.zeros((self.action_c, self.h, self.w), dtype=np.float32),
                np.zeros((1, self.h, self.w), dtype=np.float32),
                np.ones((1, self.h, self.w), dtype=np.float32),
            ],
            axis=0,
        )
        info["tril_done"] = False
        
        return obs, info
    
    def step(self, action):
        """
        Modified RL^2 step method. If a trial within an episode ends, it will return a new episode state with `False` done signal. Otherwise, the usual output is sent.
        """
        next_obs, reward, terminated, truncated, info = self.scaled_env.step(action)
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
        
        # obs_t+1, prev_action_t, prev_reward_scaled_t, prev_done_t
        next_obs = np.concatenate(
            [
                next_obs,
                self.__one_hot_spacial(action, self.action_c, self.h, self.w),
                np.full((1, self.h, self.w), reward, dtype=np.float32),
                np.full((1, self.h, self.w), done, dtype=np.float32),
            ],
            axis=0,
        )
        return next_obs, reward, terminated, truncated, info