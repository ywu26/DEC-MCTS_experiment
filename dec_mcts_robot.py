import threading
import time
import copy
import random
import numpy as np
import math

try:
    from dec_mcts_tree import MCTSTree
except ImportError:
    print("Error: dec_mcts_tree.py not found.")


# --- Environment ---
class GridEnvironment:
    def __init__(self, size, obstacles, rewards):
        self.size = size
        self.obstacles = obstacles
        self.rewards = rewards  # dict {(x,y): value}
        self.initial_total_value = sum(rewards.values())
        self.lock = threading.Lock()

    def is_valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size: return False
        if pos in self.obstacles: return False
        return True

    def get_global_utility(self, joint_paths):
        visited = set()
        total_utility = 0
        with self.lock:
            current_rewards = self.rewards.copy()
        for path in joint_paths:
            for pos in path:
                if pos in current_rewards:
                    visited.add(pos)
        for pos in visited:
            total_utility += current_rewards[pos]
        return total_utility

    def consume_reward(self, pos):
        with self.lock:
            if pos in self.rewards:
                val = self.rewards.pop(pos)
                return val
            return 0


# --- Communication ---
class RobotCommSlot:
    def __init__(self):
        self.paths = []
        self.probs = []
        self.lock = threading.Lock()


class DecentralizedCommunicationChannel:
    def __init__(self, num_robots):
        self.slots = [RobotCommSlot() for _ in range(num_robots)]

    def publish(self, robot_id, paths, probs):
        if robot_id >= len(self.slots): return
        with self.slots[robot_id].lock:
            self.slots[robot_id].paths = paths
            self.slots[robot_id].probs = probs

    def read_other(self, target_robot_id):
        if target_robot_id >= len(self.slots): return [], []
        with self.slots[target_robot_id].lock:
            return copy.deepcopy(self.slots[target_robot_id].paths), copy.deepcopy(self.slots[target_robot_id].probs)

    def get_num_robots(self):
        return len(self.slots)


# --- Robot Agent ---
class DecMCTSRobot(threading.Thread):
    def __init__(self, robot_id, start_pos, env, comms, config, logger=None, tree_visualizer=None):
        super().__init__()
        self.id = robot_id
        self.pos = start_pos
        self.env = env
        self.comms = comms
        self.config = config
        self.logger = logger
        self.tree_visualizer = tree_visualizer
        self.daemon = True

        # Track root states for visualization
        self.root_states_by_step = {}

        self.tree = MCTSTree(
            start_state=start_pos,
            env=env,
            robot_id=robot_id,
            path_limit=config['BUDGET'],
            gamma=config['GAMMA'],
            cp=config['CP'],
            rollout_budget=config['ROLLOUT_BUDGET']
        )

        self.action_set = []  # X_hat
        self.action_probs = []  # q_n
        self.others_distributions = {}

        # Initialize distribution
        self.action_set = [[self.pos]] * config['ACTION_SET_SIZE']
        self.action_probs = [1.0 / config['ACTION_SET_SIZE']] * config['ACTION_SET_SIZE']

    def select_set_of_sequences(self):
        """Line 3: Update X_hat using Top-K from tree"""
        self.action_set = self.tree.get_top_k_paths(self.config['ACTION_SET_SIZE'])
        # Reset probs to uniform initially (or keep history if possible, but structure changes)
        self.action_probs = [1.0 / len(self.action_set)] * len(self.action_set)

    def update_distribution(self):
        """
        Algorithm 3: Update Distribution using Gradient Descent.
        """
        alpha = self.config.get('ALPHA', 0.1)
        beta = self.config.get('BETA', 1.0)

        new_probs = []

        # 1. Calculate Expectations via Sampling
        num_mc_samples = 10
        sampled_others = []
        for _ in range(num_mc_samples):
            sampled_others.append(self.tree._sample_others(self.others_distributions))

        # Calculate Expected Marginal Utilities for each path in action_set
        expected_utilities = []

        for my_path in self.action_set:
            utils = []
            for others_paths in sampled_others:
                u = self.env.get_global_utility([my_path] + others_paths)
                utils.append(u)
            expected_utilities.append(np.mean(utils))

        E_f_avg = np.dot(self.action_probs, expected_utilities)  # E[f]

        # 2. Gradient Step
        entropy = -np.sum([p * np.log(p + 1e-9) for p in self.action_probs])

        for i, p in enumerate(self.action_probs):
            E_f_cond = expected_utilities[i]
            # Gradient term
            term = (E_f_avg - E_f_cond) / beta + entropy + np.log(p + 1e-9)
            # Update
            new_p = p - alpha * p * term
            new_probs.append(max(0.01, new_p))  # Clamp to avoid 0

        # Normalize
        total = sum(new_probs)
        self.action_probs = [x / total for x in new_probs]

    def transmit(self):
        self.comms.publish(self.id, self.action_set, self.action_probs)

    def receive(self):
        num = self.comms.get_num_robots()
        for r_id in range(num):
            if r_id == self.id: continue
            paths, probs = self.comms.read_other(r_id)
            if paths:
                self.others_distributions[r_id] = (paths, probs)

    def run(self):
        for step in range(self.config['SIMULATION_STEPS']):

            # Track root state for this step
            self.root_states_by_step[step] = self.tree.current_root.state

            # --- Algorithm 1 ---
            # Line 3
            self.select_set_of_sequences()

            # Line 4: Inner Loop
            for _ in range(self.config['TAU_N']):

                # Line 5: Grow Tree (Algorithm 2) - NEEDS LOOP as per user instruction
                for _ in range(self.config['NUM_SAMPLES']):
                    self.tree.grow(self.others_distributions)

                # Capture BEFORE optimization (steps 3 and 4 only)
                if self.tree_visualizer and self.id == 1 and step in [15, 16]:
                    import copy
                    self.tree_visualizer.capture_action_distribution(
                        robot_id=self.id,
                        step=step,
                        action_set=copy.deepcopy(self.action_set),
                        action_probs=copy.deepcopy(self.action_probs),
                        others_distributions=copy.deepcopy(self.others_distributions),
                        current_pos=self.pos,
                        before_optimization=True
                    )

                # Line 6: Update Distribution (Algorithm 3)
                self.update_distribution()

                # Capture AFTER optimization (steps 3 and 4 only)
                if self.tree_visualizer and self.id == 1 and step in [15, 16]:
                    import copy
                    self.tree_visualizer.capture_action_distribution(
                        robot_id=self.id,
                        step=step,
                        action_set=copy.deepcopy(self.action_set),
                        action_probs=copy.deepcopy(self.action_probs),
                        others_distributions=copy.deepcopy(self.others_distributions),
                        current_pos=self.pos,
                        before_optimization=False
                    )

                # Line 7 & 8
                self.transmit()
                self.receive()

            # Capture tree snapshot for visualization (Robot 1 at steps 2, 4, 6)
            if self.tree_visualizer and self.id == 1 and step in [1, 2, 3, 4]:
                self.tree_visualizer.capture_tree_snapshot(self.id, step, self.tree)

            # --- Execution ---
            # Pick best path based on optimized probability
            best_idx = np.argmax(self.action_probs)
            best_plan = self.action_set[best_idx]

            # Logging
            if self.logger:
                self.logger.log_step(step, self.id, best_plan)

            if len(best_plan) > 1:
                next_pos = best_plan[1]
                self.pos = next_pos

                self.tree.advance_root(self.pos)

                if self.logger:
                    # Log with timestamp (step)
                    self.logger.update_trajectory(self.id, self.pos, step)

                val = self.env.consume_reward(self.pos)
                # (Optional statistics logging here)

            time.sleep(0.001)