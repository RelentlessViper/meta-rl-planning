import matplotlib
import matplotlib.pyplot as plt
from box_world_env import BoxWorld  # change this import

matplotlib.use("QtAgg")

KEY_TO_ACTION = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}


def main():
    env = BoxWorld(
        n=5,
        goal_length=2,
        num_distractor=0,
        distractor_length=1,
        max_steps=64,
        keep_prev_world=False,
    )
    obs, info = env.reset()

    fig, ax = plt.subplots()
    img = env._get_render_image()
    im = ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("BoxWorld — arrows/WASD to move, R reset, Q quit")

    running = {"value": True}

    def redraw():
        im.set_data(env._get_render_image())
        fig.canvas.draw_idle()

    def finish_episode(info):
        ep = info.get("episode_internal", {})
        print("\nEpisode finished")
        if ep:
            print(
                f"Return: {ep.get('r', 0.0):.2f}, "
                f"Length: {ep.get('length', 0)}, "
                f"Solved: {ep.get('solved', False)}"
            )
        print("Press R to reset, or Q to quit.")

    def on_key(event):
        key = event.key.lower() if event.key else ""

        if key in ("q", "escape"):
            running["value"] = False
            plt.close(fig)
            return

        if key == "r":
            env.reset()
            redraw()
            print("\nEnvironment reset.")
            return

        if key not in KEY_TO_ACTION:
            return

        action = KEY_TO_ACTION[key]
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Action: {info['action_name']}, "
            f"moved: {info['moved']}, "
            f"reward: {reward:.2f}"
        )

        redraw()

        if terminated or truncated:
            finish_episode(info)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    # optional cleanup
    plt.close("all")


if __name__ == "__main__":
    main()
