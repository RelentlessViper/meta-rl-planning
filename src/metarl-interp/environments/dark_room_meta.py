import gymnasium as gym
import toymeta

import numpy as np

env = gym.make("Dark-Room-3x3-v0")
env.observation_space.sample()

class DarkRoomRL2Wrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
    ) -> None:
        """
        Initialize the distirbution of tasks.

        Parameters
        ----------
        env: gym.Env
            A [Dark Room environment](https://github.com/corl-team/toy-meta-gym/blob/main/src/toymeta/__init__.py)
        """
        super().__init__(env)
        base_env = self.env.unwrapped

        base_env.terminate_on_goal = True
        base_env.random_start = False

        self._grid_size = base_env.size

    def sample_task(self) -> dict[str:any]:
        """
        Sample a task with a fixed random goal position

        Returns
        ----------
        dict[str:any]
            Position of fixed random goal
        """
        goal_pos = np.random.randint(0, self._grid_size, size=2)
        return {"goal": goal_pos}
    
    def set_task(self, task: dict[str:any]) -> None:
        """
        Set a fixed goal position for a task

        Parameters
        ----------
        task: dict[str:any]
            A dictionary for task-specific parameters. In our case it is a goal position {"goal": goal_pos}, where goal_pos in [0;self._grid_size]
        """
        self._task = task
        self._fixed_goal = np.array(task["goal"], dtype=np.int64)

    # Override the reset method
    def reset(self, *, seed=None, options=None) -> None:
        """
        Reset the task without resetting the goal position (required for RL^2)

        Parameters
        ----------
        *args: list[any]
            List of arguments
        seed: int
            Random seed
        options: dict[str:any]
            Options dictionary

        Returns
        ----------
        tuple[int,any]
            The tuple containing the current observation and the logging info
        """
        state, info = self.env.reset(seed=seed, options=options)

        if self._fixed_goal is None:
            raise RuntimeError("Task settings are not set. Call env.set_task(task) before calling this method")
        
        base_env = self.env.unwrapped
        base_env.goal_pos = self._fixed_goal.copy()
        assert np.array_equal(base_env.goal_pos, self._fixed_goal)

        return state, info
