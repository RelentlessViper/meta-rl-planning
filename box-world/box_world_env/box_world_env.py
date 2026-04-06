import numpy as np
import gymnasium as gym
from gymnasium import spaces

from collections import deque
import matplotlib.pyplot as plt

from .box_world_gen import world_gen, is_empty, update_color, goal_color, wall_color, grid_color

ACTION_LOOKUP = {
    0: "move up",
    1: "move down",
    2: "move left",
    3: "move right",
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


class BoxWorld(gym.Env):
    """Boxworld representation
    Args:
      n (int): Size of the field (n x n)
      goal_length (int): Number of keys to collect to solve the level
      num_distractor (int): Number of distractor trajectories
      distractor_length (int): Number of distractor keys in each distractor trajectory
      max_steps (int): Maximum number of env step for a given level
      collect_key (bool): If true, a key is collected immediately when its corresponding lock is opened
      world (np.ndarray): an existing level. If None, generates a new level by calling the world_gen() function 
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    def __init__(
        self,
        n=8,
        goal_length=3,
        num_distractor=1,
        distractor_length=1,
        max_steps=1000,
        collect_key=True,
        world=None,
        keep_prev_world=True,
        step_cost=0.0,
        reward_gem = 10.0,
        reward_key = 1.0,
        reward_distractor = -1.0,
        render_mode=None,
    ):
        super().__init__()

        self.n = n
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.max_steps = max_steps
        self.collect_key = collect_key
        self.render_mode = render_mode

        self.step_cost = step_cost
        self.reward_gem = reward_gem
        self.reward_key = reward_key
        self.reward_distractor = reward_distractor

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, n + 2, n + 2),
            dtype=np.float32,
        )

        self.world = world
        self.player_position = None
        self.world_dic = None
        self.owned_key = None

        self.keep_prev_world = keep_prev_world
        self.prev_seed = None

        self.num_steps = 0
        self.episode_reward = 0.0

        self.last_frames = deque(maxlen=3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        world_data = None

        if options is not None:
            world_data = options.get("world", None)

        if world_data is not None:
            self.world, self.player_position, self.world_dic = world_data
        elif self.keep_prev_world and self.prev_seed is not None:
            self.world, self.player_position, self.world_dic = world_gen(
                n=self.n,
                goal_length=self.goal_length,
                num_distractor=self.num_distractor,
                distractor_length=self.distractor_length,
                seed=self.prev_seed,
            )
        else:
            if seed is None:
                seed = np.random.randint(0, 2**31 - 1)

            self.prev_seed = seed

            self.world, self.player_position, self.world_dic = world_gen(
                n=self.n,
                goal_length=self.goal_length,
                num_distractor=self.num_distractor,
                distractor_length=self.distractor_length,
                seed=seed,
            )

        self.num_steps = 0
        self.episode_reward = 0.0
        self.owned_key = np.array([220, 220, 220], dtype=np.uint8)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        assert self.action_space.contains(action)

        move = np.array(CHANGE_COORDINATES[action])
        new_pos = self.player_position + move
        current_pos = self.player_position.copy()

        self.num_steps += 1

        reward = -self.step_cost
        terminated = False
        truncated = self.num_steps >= self.max_steps
        solved = False

        possible_move = False

        if np.any(new_pos < 1) or np.any(new_pos >= self.n + 1):
            possible_move = False

        elif is_empty(self.world[new_pos[0], new_pos[1]]):
            possible_move = True

        elif new_pos[1] == 1 or is_empty(self.world[new_pos[0], new_pos[1] - 1]):
            if is_empty(self.world[new_pos[0], new_pos[1] + 1]):
                possible_move = True

                self.owned_key = self.world[new_pos[0], new_pos[1]].copy()
                self.world[0, 0] = self.owned_key

                if np.array_equal(self.owned_key, goal_color):
                    reward += self.reward_gem
                    terminated = True
                    solved = True
                else:
                    reward += self.reward_key
            else:
                possible_move = False

        else:
            if np.array_equal(self.world[new_pos[0], new_pos[1]], self.owned_key):
                possible_move = True

                if self.collect_key:
                    next_key = self.world[new_pos[0], new_pos[1] - 1]

                    if np.array_equal(next_key, goal_color):
                        reward += self.reward_gem
                        terminated = True
                        solved = True
                    else:
                        self.owned_key = next_key.copy()
                        reward += self.reward_key

                        if self.world_dic.get(tuple(new_pos), 1) == 0:
                            reward += self.reward_distractor
                            terminated = True
                else:
                    self.owned_key = np.array([220, 220, 220], dtype=np.uint8)

                    if self.world_dic.get(tuple(new_pos), 1) == 0:
                        reward += self.reward_distractor
                        terminated = True
            else:
                possible_move = False

        if possible_move:
            self.player_position = new_pos
            update_color(self.world, current_pos, new_pos)

        self.episode_reward += reward

        info = {
            "action_name": ACTION_LOOKUP[action],
            "moved": possible_move,
        }

        if terminated or truncated:
            info["episode_internal"] = {
                "r": self.episode_reward,
                "length": self.num_steps,
                "solved": solved,
            }

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # [h, w, c] -> [c, h, w]
        return np.transpose(self.world.astype(np.float32), (2, 0, 1))

    def render(self):
        img = self._get_render_image()

        if self.render_mode == "rgb_array":
            return img

        elif self.render_mode == "human":
            self._fig, self._ax = plt.subplots()

            self._ax.clear()
            self._ax.imshow(img, interpolation="nearest")
            self._ax.set_xticks([])
            self._ax.set_yticks([])

            plt.pause(1.0 / self.metadata["render_fps"])
            plt.draw()

    def _get_render_image(self):
        """
        Returns a properly formatted RGB image (uint8, enlarged).
        """
        img = self.world.astype(np.uint8)

        # Upscale the image so it's visible
        scale = 32  # each cell becomes 32x32 pixels
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        return img
