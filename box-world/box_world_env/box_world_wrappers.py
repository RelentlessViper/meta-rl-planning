import gymnasium as gym
from gymnasium import ObservationWrapper

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from .box_world_gen import goal_color

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
            low=-1.0,
            high=1.0,
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
        
        # obs_t+1, action_t, reward_scaled_t, done_t
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

class RevealChestContentsWrapper(gym.Wrapper):
    """
    Masks chest contents until the chest is opened.

    Assumptions:
      - collect_key must be False in the base env
      - each chest is encoded as:
            [ content cell | lock cell ]
        where content cell is at lock_pos + content_offset
      - opening a chest means the agent successfully moves onto the lock cell

    Behavior:
      - unopened chest contents are replaced by mask_color in observations/render
      - once a chest is opened, its content stays revealed for the rest of the episode
      - the gem chest content (goal_color) is never masked
      - loose keys are untouched, because they are not in world_dic
    """

    def __init__(self, env, mask_color=(35, 46, 52), content_offset=-1):
        super().__init__(env)

        if getattr(self.env.unwrapped, "collect_key", None) is True:
            raise ValueError(
                "RevealChestContentsWrapper only supports collect_key=False."
            )

        self.mask_color = np.asarray(mask_color, dtype=np.uint8)
        self.content_offset = content_offset

        # Track which chests have been opened so far
        self._opened_chests = set()

        # Keep same spaces as base env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._opened_chests.clear()
        obs = self._mask_observation(obs)
        return obs, info

    def step(self, action):
        prev_pos = None
        if getattr(self.env.unwrapped, "player_position", None) is not None:
            prev_pos = self.env.unwrapped.player_position.copy()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Detect whether this step opened a chest:
        # if the agent moved onto a lock cell that belongs to a chest,
        # then that chest is now open and should be revealed.
        if prev_pos is not None:
            new_pos = self.env.unwrapped.player_position
            if new_pos is not None and not np.array_equal(prev_pos, new_pos):
                lock_pos = (int(new_pos[0]), int(new_pos[1]))
                if lock_pos in self.env.unwrapped.world_dic:
                    self._opened_chests.add(lock_pos)

        obs = self._mask_observation(obs)
        return obs, reward, terminated, truncated, info

    def observation(self, observation):
        # Kept for compatibility if someone calls it manually
        return self._mask_observation(observation)

    def _mask_observation(self, observation):
        """
        observation is CHW: [3, H, W]
        """
        world_dic = getattr(self.env.unwrapped, "world_dic", None)
        world = getattr(self.env.unwrapped, "world", None)

        if world_dic is None or world is None:
            return observation

        obs = observation.copy()

        for lock_pos in world_dic.keys():
            r, c_lock = lock_pos
            c_content = c_lock + self.content_offset

            if c_content < 0 or c_content >= obs.shape[2]:
                continue

            # Do not mask the gem content
            content_rgb = world[r, c_content].astype(np.uint8)
            if np.array_equal(content_rgb, goal_color):
                continue

            # Reveal only after the chest is opened
            if lock_pos not in self._opened_chests:
                obs[:, r, c_content] = self.mask_color

        return obs

    def _get_masked_world_image(self):
        """
        Returns masked RGB image in HWC format, upscaled like the base env.
        """
        world = self.env.unwrapped.world.copy().astype(np.uint8)
        world_dic = getattr(self.env.unwrapped, "world_dic", None)

        if world_dic is not None:
            for lock_pos in world_dic.keys():
                r, c_lock = lock_pos
                c_content = c_lock + self.content_offset

                if c_content < 0 or c_content >= world.shape[1]:
                    continue

                content_rgb = world[r, c_content]
                if np.array_equal(content_rgb, goal_color):
                    continue

                if lock_pos not in self._opened_chests:
                    world[r, c_content] = self.mask_color

        scale = 32
        world = np.repeat(np.repeat(world, scale, axis=0), scale, axis=1)
        return world

    def _get_original_world_image(self):
        """
        Returns the unmasked RGB image in HWC format, upscaled like the base env.
        """
        world = self.env.unwrapped.world.astype(np.uint8)
        scale = 32
        world = np.repeat(np.repeat(world, scale, axis=0), scale, axis=1)
        return world

    def render(self):
        original = self._get_original_world_image()
        masked = self._get_masked_world_image()

        import matplotlib.pyplot as plt
        import numpy as np

        # Build the frame with titles baked in
        if not hasattr(self, "_fig"):
            self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

        ax_orig, ax_mask = self._axes

        ax_orig.clear()
        ax_orig.imshow(original, interpolation="nearest")
        ax_orig.set_title("Original")
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        ax_mask.clear()
        ax_mask.imshow(masked, interpolation="nearest")
        ax_mask.set_title("Masked")
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])

        self._fig.tight_layout()
        self._fig.canvas.draw()

        frame = np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3].copy()

        if self.env.render_mode == "human":
            plt.pause(1.0 / self.env.metadata["render_fps"])
            plt.draw()

        return frame