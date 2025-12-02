import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import threading
import time

from dec_mcts_robot import DecMCTSRobot, GridEnvironment, DecentralizedCommunicationChannel
from dec_mcts_visualizer import TreeVisualizer

# --- Configuration ---
GRID_SIZE = 25
NUM_ROBOTS = 8
NUM_OBSTACLES = round(0.1 * GRID_SIZE * GRID_SIZE)
NUM_REWARDS = round(0.4 * GRID_SIZE * GRID_SIZE)  # 1250 rewards
ACTION_SET_SIZE = 10
SIMULATION_STEPS = 25
NUM_SAMPLES = 10
TAU_N = 10  # Refinement steps per physical step
BUDGET = 10  # Path depth
ROLLOUT_BUDGET = 200  # BFS expansions
GAMMA = 0.95
CP = 1.0
ALPHA = 0.1  # Gradient step size
BETA = 10.0  # Temperature

CONFIG = {
    'GRID_SIZE': GRID_SIZE,
    'NUM_ROBOTS': NUM_ROBOTS,
    'ACTION_SET_SIZE': ACTION_SET_SIZE,
    'SIMULATION_STEPS': SIMULATION_STEPS,
    'NUM_SAMPLES': NUM_SAMPLES,
    'TAU_N': TAU_N,
    'BUDGET': BUDGET,
    'ROLLOUT_BUDGET': ROLLOUT_BUDGET,
    'GAMMA': GAMMA,
    'CP': CP,
    'ALPHA': ALPHA,
    'BETA': BETA
}


class SimpleLogger:
    def __init__(self):
        # trajectories: {robot_id: [(step, x, y), ...]}
        self.trajectories = {i: [] for i in range(NUM_ROBOTS)}
        self.lock = threading.Lock()

    def log_step(self, step, rid, plan):
        pass

    def update_trajectory(self, rid, pos, step):
        with self.lock:
            self.trajectories[rid].append((step, pos[0], pos[1]))


def generate_environment():
    print(f"Generating {GRID_SIZE}x{GRID_SIZE} Grid...")
    all_coords = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]

    # Obstacles
    obs_indices = np.random.choice(len(all_coords), NUM_OBSTACLES, replace=False)
    obstacles = set(all_coords[i] for i in obs_indices)

    # Rewards
    remaining_indices = list(set(range(len(all_coords))) - set(obs_indices))
    rew_indices = np.random.choice(remaining_indices, NUM_REWARDS, replace=False)

    rewards = {}
    for idx in rew_indices:
        rewards[all_coords[idx]] = random.randint(1, 10)

    return obstacles, rewards


def visualize_results(env, trajectories, initial_rewards):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Dec-MCTS Final Trajectories")
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    # Obstacles
    if env.obstacles:
        ox, oy = zip(*env.obstacles)
        ax.scatter(ox, oy, c='black', marker='s', s=10, alpha=0.3, label='Obstacles')

    # Rewards (Remaining vs Taken)
    taken_x, taken_y, taken_s = [], [], []
    rem_x, rem_y, rem_s = [], [], []

    for pos, val in initial_rewards.items():
        size = val * 5  # Scale size by value
        if pos in env.rewards:
            rem_x.append(pos[0])
            rem_y.append(pos[1])
            rem_s.append(size)
        else:
            taken_x.append(pos[0])
            taken_y.append(pos[1])
            taken_s.append(size)

    if rem_x:
        ax.scatter(rem_x, rem_y, s=rem_s, c='green', alpha=0.6, label='Remaining Rewards')
    if taken_x:
        ax.scatter(taken_x, taken_y, s=taken_s, c='lightgray', alpha=0.4, label='Collected Rewards')

    # Trajectories (Plot just the lines)
    colors = cm.rainbow(np.linspace(0, 1, NUM_ROBOTS))
    for rid, path_data in trajectories.items():
        if not path_data: continue
        # Extract x, y ignoring step for the static plot
        px = [p[1] for p in path_data]
        py = [p[2] for p in path_data]

        ax.plot(px, py, c=colors[rid], linewidth=2, alpha=0.8, label=f'R{rid}')
        ax.scatter(px[0], py[0], c=colors[rid], marker='o', s=50, edgecolors='black')  # Start
        ax.scatter(px[-1], py[-1], c=colors[rid], marker='^', s=80, edgecolors='black')  # End

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_animation(env, trajectories, initial_rewards, filename="dec_mcts_simulation.mp4"):
    print(f"Generating Animation: {filename}...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.grid(True, alpha=0.3)

    # 1. Static Background (Obstacles)
    if env.obstacles:
        ox, oy = zip(*env.obstacles)
        ax.scatter(ox, oy, c='black', marker='s', s=10, alpha=0.3)

    # 2. Dynamic Elements Setup
    # Rewards scatter plot needs to update as they are collected
    # We'll split rewards into 'active' and 'collected'

    # Initial Rewards Data
    rew_x, rew_y, rew_s = [], [], []
    for pos, val in initial_rewards.items():
        rew_x.append(pos[0])
        rew_y.append(pos[1])
        rew_s.append(val * 5)

    # Scatter for remaining rewards (Green)
    scat_rewards = ax.scatter(rew_x, rew_y, s=rew_s, c='green', alpha=0.6)

    # Scatter for collected rewards (Gray - initially empty, but we can just overlay gray dots)
    # Actually simpler: replot rewards every frame or just overlay trajectories.
    # To be efficient, let's keep rewards static green for now, and maybe just show robots moving.
    # High fidelity: Remove green dot when robot hits it.
    # For simplicity here: Static rewards, dynamic robots.

    # Robot Markers
    colors = cm.rainbow(np.linspace(0, 1, NUM_ROBOTS))
    robot_dots = []
    robot_trails = []

    for i in range(NUM_ROBOTS):
        dot, = ax.plot([], [], 'o', c=colors[i], markeredgecolor='black', markersize=8)
        trail, = ax.plot([], [], '-', c=colors[i], alpha=0.5, linewidth=1)
        robot_dots.append(dot)
        robot_trails.append(trail)

    # Title
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")

    # Helper to get position at step T
    # Trajectories are list of (step, x, y). steps might not be perfectly contiguous if threads sleep weirdly,
    # but generally they are 0, 1, 2...
    # We will interpolate or just find the closest step <= T

    max_step = SIMULATION_STEPS

    def update(frame):
        title.set_text(f"Step {frame}/{max_step}")

        # Update each robot
        for rid in range(NUM_ROBOTS):
            path = trajectories[rid]
            # Find the state at this frame
            # Filter for steps <= frame
            history = [p for p in path if p[0] <= frame]

            if history:
                # Current pos is the last one in history
                curr = history[-1]
                robot_dots[rid].set_data([curr[1]], [curr[2]])

                # Trail is all history positions
                tx = [p[1] for p in history]
                ty = [p[2] for p in history]
                robot_trails[rid].set_data(tx, ty)

        return robot_dots + robot_trails + [title]

    # Create Animation
    # Frames = Simulation Steps. Interval = ms between frames.
    anim = animation.FuncAnimation(fig, update, frames=max_step + 1, interval=200, blit=True)

    # Save
    try:
        # Requires ffmpeg installed. If not, use writer='pillow' for gif
        anim.save(filename, writer='ffmpeg', fps=5)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Could not save video (is ffmpeg installed?): {e}")
        try:
            print("Attempting to save as GIF instead...")
            anim.save("dec_mcts_simulation.gif", writer='pillow', fps=5)
            print("GIF saved successfully.")
        except Exception as e2:
            print(f"Could not save GIF either: {e2}")


def run_simulation():
    obstacles, initial_rewards = generate_environment()
    # Deep copy rewards because environment consumes them
    env = GridEnvironment(GRID_SIZE, obstacles, initial_rewards.copy())
    comms = DecentralizedCommunicationChannel(NUM_ROBOTS)
    logger = SimpleLogger()

    # Create tree visualizer
    tree_visualizer = TreeVisualizer()

    robots = []
    print("Initializing Robots...")
    for i in range(NUM_ROBOTS):
        while True:
            sx, sy = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if env.is_valid((sx, sy)): break
        robots.append(DecMCTSRobot(i, (sx, sy), env, comms, CONFIG, logger, tree_visualizer))
        # Log initial pos at step 0
        logger.update_trajectory(i, (sx, sy), 0)

    print(f"Starting Simulation ({SIMULATION_STEPS} steps)...")
    start_time = time.time()

    threads = []
    for r in robots:
        r.start()
        threads.append(r)

    for t in threads:
        t.join()

    duration = time.time() - start_time
    print(f"\nSimulation Finished in {duration:.2f}s")

    # Capture final tree state with root positions for Robot 1
    robot_1 = robots[1]
    tree_visualizer.capture_final_tree_with_roots(
        robot_id=1,
        tree=robot_1.tree,
        root_states_by_step=robot_1.root_states_by_step
    )

    # Statistics
    total_val_start = sum(initial_rewards.values())
    total_val_left = sum(env.rewards.values())
    collected = total_val_start - total_val_left

    print("=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Total Available Value: {total_val_start}")
    print(f"Total Collected Value: {collected}")
    print(f"Percentage Collected:  {collected / total_val_start * 100:.2f}%")
    print(f"Remaining Value:       {total_val_left}")

    # Trajectory Stats
    for rid, path in logger.trajectories.items():
        print(f"Robot {rid}: Traveled {len(path)} steps")

    # Visualization (Static)
    visualize_results(env, logger.trajectories, initial_rewards)

    # Visualization (Video)
    create_animation(env, logger.trajectories, initial_rewards)

    # Visualize MCTS Trees for Robot 1 at steps 2, 4, 6
    print("\n" + "=" * 40)
    print("VISUALIZING MCTS TREES")
    print("=" * 40)

    # Individual tree visualizations (from each step's root)
    print("\nGenerating tree visualizations from each step's root...")
    tree_visualizer.visualize_tree(robot_id=1, step=1, max_depth=5,
                                   save_path="tree_robot1_step2.png", layout='radial')
    tree_visualizer.visualize_tree(robot_id=1, step=2, max_depth=5,
                                   save_path="tree_robot1_step4.png", layout='radial')
    tree_visualizer.visualize_tree(robot_id=1, step=3, max_depth=5,
                                   save_path="tree_robot1_step6.png", layout='radial')

    # Comparison visualization (same final tree, different root markers)
    print("\nGenerating comparison view (final tree with different roots marked)...")
    tree_visualizer.visualize_rollout_comparison(robot_id=1, steps=[1, 2, 3],
                                                 save_path="tree_robot1_comparison.png",
                                                 layout='radial')

    # Action distribution visualization (Algorithm 3)
    print("\n" + "=" * 40)
    print("VISUALIZING ACTION DISTRIBUTIONS")
    print("=" * 40)
    print("\nGenerating action distribution visualization (Algorithm 3)...")
    tree_visualizer.visualize_action_distributions(robot_id=1, steps=[15, 16],
                                                   save_path="action_distributions_robot1.png")


if __name__ == "__main__":
    run_simulation()