import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import threading
import time

from dec_mcts_robot import DecMCTSRobot, GridEnvironment, DecentralizedCommunicationChannel

# --- Configuration ---
GRID_SIZE = 40
NUM_ROBOTS = 8
NUM_OBSTACLES = round(0.1 * GRID_SIZE * GRID_SIZE)
NUM_REWARDS = round(0.5 * GRID_SIZE * GRID_SIZE)  # 1250 rewards
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
        self.trajectories = {i: [] for i in range(NUM_ROBOTS)}
        self.lock = threading.Lock()

    def log_step(self, step, rid, plan):
        pass

    def update_trajectory(self, rid, pos):
        with self.lock:
            self.trajectories[rid].append(pos)


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
    # We reconstruct taken from initial vs current env
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

    # Trajectories
    colors = cm.rainbow(np.linspace(0, 1, NUM_ROBOTS))
    for rid, path in trajectories.items():
        if not path: continue
        px, py = zip(*path)
        ax.plot(px, py, c=colors[rid], linewidth=2, alpha=0.8, label=f'R{rid}')
        ax.scatter(px[0], py[0], c=colors[rid], marker='o', s=50, edgecolors='black')  # Start
        ax.scatter(px[-1], py[-1], c=colors[rid], marker='^', s=80, edgecolors='black')  # End

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_simulation():
    obstacles, initial_rewards = generate_environment()
    # Deep copy rewards because environment consumes them
    env = GridEnvironment(GRID_SIZE, obstacles, initial_rewards.copy())
    comms = DecentralizedCommunicationChannel(NUM_ROBOTS)
    logger = SimpleLogger()

    robots = []
    print("Initializing Robots...")
    for i in range(NUM_ROBOTS):
        while True:
            sx, sy = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if env.is_valid((sx, sy)): break
        robots.append(DecMCTSRobot(i, (sx, sy), env, comms, CONFIG, logger))
        # Log initial pos
        logger.update_trajectory(i, (sx, sy))

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

    visualize_results(env, logger.trajectories, initial_rewards)


if __name__ == "__main__":
    run_simulation()