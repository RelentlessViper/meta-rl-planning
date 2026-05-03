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

    def __init__(self, env, mask_color=(35, 46, 52), content_offset=-1, render_original_env=True):
        super().__init__(env)

        if getattr(self.env.unwrapped, "collect_key", None) is True:
            raise ValueError(
                "RevealChestContentsWrapper only supports collect_key=False."
            )

        self.mask_color = np.asarray(mask_color, dtype=np.uint8)
        self.content_offset = content_offset
        self.render_original_env = render_original_env

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

        if not self.render_original_env:

            # Cache a single-axes figure
            if not hasattr(self, "_fig"):
                self._fig, self._ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)

            ax = self._ax
            ax.clear()
            ax.imshow(masked, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            self._fig.tight_layout(pad=0)
            self._fig.canvas.draw()

            frame = np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3].copy()

            if getattr(self.env, "render_mode", None) == "human":
                plt.pause(1.0 / self.env.metadata.get("render_fps", 30))
                plt.draw()

            return frame

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

class DifficultyRandomizerWrapper(gym.Wrapper):
    """
    Randomizes BoxWorld difficulty on every reset by sampling:
      - goal_length
      - num_distractor
      - distractor_length

    Optional:
      - max_steps can also be randomized or scaled with difficulty
    """

    def __init__(
        self,
        env,
        max_goal_length=3,
        max_num_distractor=3,
        max_distractor_length=3,
        seed=None,
    ):
        super().__init__(env)

        self.max_goal_length = max_goal_length
        self.max_num_distractor = max_num_distractor
        self.max_distractor_length = max_distractor_length

        self.rng = np.random.default_rng(seed)

        # Preserve spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _sample_int_range(self, low_high):
        low, high = low_high
        return int(self.rng.integers(low, high + 1))

    def _sample_task(self):
        base = self.env.unwrapped
        task = {
            "goal_length": self._sample_int_range((1, self.max_goal_length)) if self.max_goal_length is not None else base.goal_length,
            "num_distractor": self._sample_int_range((1, self.max_num_distractor)) if self.max_num_distractor is not None else base.num_distractor,
            "distractor_length": self._sample_int_range((1, self.max_distractor_length)) if self.max_distractor_length is not None else base.distractor_length,
        }
        if task["distractor_length"] == 0 and task["num_distractor"] > 0: # Safety measure
            task["distractor_length"] = 1

        task["max_steps"] = self._calculate_max_timesteps(
            self.env.unwrapped.n,
            task["goal_length"],
            task["num_distractor"],
            task["distractor_length"],
        )

        return task
    
    def _calculate_max_timesteps(self, n, goal_length, num_distractor, distractor_length):
        return int(
            0.35 * n * n
            + 4 * goal_length
            + 3 * num_distractor
            + 2 * num_distractor * distractor_length
        )

    def reset(self, seed=None, options=None):
        options = dict(options or {})

        if options.get("keep_prev_world", True):
            options["keep_prev_world"] = True
            obs, info = self.env.reset(seed=seed, options=options)
            return obs, info

        # If a specific world is injected manually, do not override it.
        if options.get("world", None) is None:
            task = self._sample_task()

            base = self.env.unwrapped
            base.goal_length = task["goal_length"]
            base.num_distractor = task["num_distractor"]
            base.distractor_length = task["distractor_length"]
            base.max_steps = task["max_steps"]

            obs, info = self.env.reset(seed=seed, options=options)
            info = dict(info)
            info["task_params"] = task
            return obs, info

        # Manual world override
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["task_params"] = {
            "goal_length": getattr(self.env.unwrapped, "goal_length", None),
            "num_distractor": getattr(self.env.unwrapped, "num_distractor", None),
            "distractor_length": getattr(self.env.unwrapped, "distractor_length", None),
            "max_steps": getattr(self.env.unwrapped, "max_steps", None),
        }
        return obs, info
    

class ProbeRenderWrapper(gym.Wrapper):
    """
    Overlay per-cell probe predictions on top of the base env render.

    Expected hidden state shape:
        (n_hidden, H, W)

    Expected probe API:
        probe.predict(X) -> action ids
    where X has shape:
        (H * W, n_hidden)
    """

    def __init__(
        self,
        env,
        probe,
        action_meanings=None,
        alpha=0.3,
        fps=8,   # lower = slower animation
    ):
        super().__init__(env)
        self.probe = probe
        self.hidden_state = None
        self.alpha = alpha
        self.fps = fps

        self.action_meanings = action_meanings or {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→",
            5: "x",
        }

        self._fig = None
        self._ax = None

        # Make render speed explicit for wrappers that consult metadata.
        self.metadata = dict(getattr(env, "metadata", {}))
        self.metadata["render_fps"] = fps

    def set_hidden_state(self, hidden_state):
        self.hidden_state = hidden_state

    def _predict_actions(self):
        if self.hidden_state is None:
            return None

        # (C, H, W) -> (H, W, C) -> (H*W, C)
        hs = np.transpose(self.hidden_state, (1, 2, 0))
        H, W, C = hs.shape
        features = hs.reshape(-1, C)

        pred_ids = self.probe.predict(features)
        return pred_ids.reshape(H, W)

    def render(self):
        base_frame = self.env.render()
        if base_frame is None:
            return None

        pred_map = self._predict_actions()

        if self._fig is None:
            self._fig, self._ax = plt.subplots(
                figsize=(base_frame.shape[1] / 100, base_frame.shape[0] / 100),
                dpi=100,
            )

        ax = self._ax
        ax.clear()
        ax.imshow(base_frame)
        ax.set_xticks([])
        ax.set_yticks([])

        if pred_map is not None:
            H, W = pred_map.shape
            img_h, img_w = base_frame.shape[:2]
            cell_h = img_h / H
            cell_w = img_w / W

            for r in range(1, H - 1):
                for c in range(1, W - 1):
                    action_id = int(pred_map[r, c])
                    label = self.action_meanings.get(action_id, str(action_id))

                    ax.text(
                        (c + 0.5) * cell_w,
                        (r + 0.5) * cell_h,
                        label,
                        color="white",
                        fontsize=12,
                        ha="center",
                        va="center",
                        weight="normal",
                        bbox=dict(
                            facecolor="black",
                            alpha=self.alpha,
                            edgecolor="none",
                            pad=1.0,
                        ),
                    )

        self._fig.tight_layout(pad=0)
        self._fig.canvas.draw()

        frame = np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3].copy()

        if getattr(self.env, "render_mode", None) == "human":
            plt.pause(1.0 / self.fps)
            plt.draw()

        return frame