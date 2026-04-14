import os
import sys
import time

import matplotlib.pyplot as plt

from box_world_env import BoxWorld, RL2BoxWorld

if os.name == "nt":
    import msvcrt

    def get_key():
        ch = msvcrt.getch()
        try:
            return ch.decode("utf-8").lower()
        except UnicodeDecodeError:
            return ""
else:
    import tty
    import termios

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            return ch.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


ACTION_MAP = {
    "w": 0,  # up
    "s": 1,  # down
    "a": 2,  # left
    "d": 3,  # right
}


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    # Create your base env and wrap it
    base_env = BoxWorld(
        n=8,
        goal_length=3,
        num_distractor=2,
        distractor_length=2,
        max_steps=64,
        collect_key=False,
        render_mode="rgb_array",  # easier to display from this script
    )

    env = RL2BoxWorld(base_env, trials_per_episode=3)

    obs, info = env.reset()
    current_trial_return = 0.0
    last_step_reward = 0.0
    last_trial_done = False
    terminated, truncated = False, False

    # Matplotlib live viewer
    plt.ion()
    fig, ax = plt.subplots()
    frame = env.render()
    im = ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show(block=False)

    try:
        while True:
            clear_screen()

            current_trial_index = env.trial_counter

            print("RL^2 BoxWorld controller")
            print("-" * 30)
            print("Controls: W/A/S/D = move, r = reset episode, q = quit")
            print()
            print(f"Current trial index: {current_trial_index}")
            print(f"Last step reward:    {last_step_reward:.3f}")
            print(f"Trial return:        {current_trial_return:.3f}")
            print(f"Last trial done:     {last_trial_done}")
            print(f"Terminated:          {terminated}")
            print(f"Truncated:           {truncated}")
            print()

            print("Press a key...")

            key = get_key()

            if key == "q":
                break

            if key == "r":
                obs, info = env.reset(hard_reset=True)
                current_trial_return = 0.0
                last_step_reward = 0.0
                last_trial_done = False
            elif key in ACTION_MAP:
                action = ACTION_MAP[key]
                obs, reward, terminated, truncated, info = env.step(action)

                last_step_reward = float(reward)
                current_trial_return += float(reward)

                last_trial_done = bool(info.get("trial_done", False))

                # If a trial just ended, the wrapper may already have advanced
                # to the next trial or reset the episode.
                if last_trial_done:
                    ended_trial = info.get("current_trial", None)
                    print(f"\nTrial finished: {ended_trial}")
                    print(f"Cumulative trial rewards: {info.get('cumulative_reward_per_trial', None)}")
                    print(f"Terminated: {terminated}, truncated: {truncated}")
                    current_trial_return = 0.0

            # Update live image
            frame = env.render()
            im.set_data(frame)
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plt.close(fig)


if __name__ == "__main__":
    main()