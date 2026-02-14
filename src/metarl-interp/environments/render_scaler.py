import gymnasium as gym

import numpy as np

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
        super().__init__(env)
        self.scale_factor = scale_factor
        self.draw_lines = draw_lines
        self.base_env = env.unwrapped
    
    def render(
        self,
        scale_factor = None,
        draw_lines = None,
    ):
        scale_factor = self.scale_factor if scale_factor is None else scale_factor
        draw_lines = self.draw_lines if draw_lines is None else draw_lines

        if self.render_mode == "rgb_array":
            base_grid = np.full((self.base_env.size, self.base_env.size, 3), fill_value=(255, 255, 255), dtype=np.uint8)
            base_grid[self.base_env.goal_pos[0], self.base_env.goal_pos[1]] = (255, 0, 0)

            if self.base_env.agent_pos is None:
                raise AttributeError("The agent position is `None`. Execute `env.reset()` first")

            base_grid[int(self.base_env.agent_pos[0]), int(self.base_env.agent_pos[1])] = (0, 255, 0)
            
            scaled_grid = base_grid.repeat(
                scale_factor,
                axis=0,
            ).repeat(
                scale_factor,
                axis=1,
            )

            if draw_lines:
                height, width = scaled_grid.shape[:2]

                for x in range(scale_factor, width, scale_factor):
                    scaled_grid[:, x - 1:x + 1] = [0, 0, 0]
                
                for y in range(scale_factor, height, scale_factor):
                    scaled_grid[y - 1:y + 1, :] = [0, 0, 0]
            
            return scaled_grid