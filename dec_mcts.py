import threading
import time
import copy
import random

# Import the specific tree implementation (Algorithm 2)
# This assumes dec_mcts_tree.py is in the same directory
try:
    from dec_mcts_tree import MCTSTree
except ImportError:
    print("CRITICAL ERROR: 'dec_mcts_tree.py' not found. This script requires the MCTS Tree implementation.")


class DecMCTSRobot(threading.Thread):
    """
    Implements Algorithm 1: Overview of Dec-MCTS for robot r[cite: 210].

    This agent:
    1. Maintains a persistent MCTS tree[cite: 212].
    2. Iteratively refines the tree and probability distributions[cite: 213].
    3. Communicates with other robots [cite: 225-227].
    4. Executes the best action in a receding horizon fashion[cite: 198].
    """

    def __init__(self, robot_id, start_pos, env, comms, logger, config):
        super().__init__()
        self.id = robot_id
        self.pos = start_pos
        self.env = env
        self.comms = comms
        self.logger = logger
        self.config = config
        self.daemon = True  # Daemon thread ensures it dies when main simulation ends

        # 1. Initialize MCTS Tree [cite: 212]
        # The tree persists throughout the robot's lifetime
        self.tree = MCTSTree(
            start_state=start_pos,
            env=env,
            robot_id=robot_id,
            path_limit=config['PATH_LIMIT'],  # B^r (Physical budget)
            gamma=config['GAMMA'],
            cp=config['CP'],
            rollout_budget=config['ROLLOUT_BUDGET']  # Computational budget
        )

        # Local view of other robots' plans (q_n^(r))
        self.others_distributions = {}

        # The robot's current optimized domain X_hat [cite: 185]
        self.action_set = []

        # Log initial state
        self.logger.update_trajectory(self.id, self.pos)

    def select_set_of_sequences(self):
        """
        Algorithm 1, Line 3: SelectSetOfSequences(T) [cite: 214]
        Updates the local domain X_hat based on the most promising nodes in the tree.
        """
        # In this implementation, we select the single best path found so far.
        # (Generalized: could select top-K diverse paths)
        best_path = self.tree.get_best_path()
        self.action_set = [best_path]

    def update_distribution(self):
        """
        Algorithm 1, Line 6: UpdateDistribution [cite: 222]
        Optimizes q_n based on the tree statistics.
        """
        # Placeholder for Algorithm 3 (Gradient Descent).
        # Currently implements a greedy deterministic assignment:
        # P(best_path) = 1.0, others = 0.0
        pass

    def transmit(self):
        """
        Algorithm 1, Line 7: CommunicationTransmit [cite: 225]
        Sends the compressed plan (X_hat, q_n) to the team.
        """
        # Create the message
        # We replicate the path to match ACTION_SET_SIZE for compatibility
        paths = self.action_set * self.config['ACTION_SET_SIZE']

        # Uniform probability if we only have one path, or specific probs otherwise
        probs = [1.0 / len(paths)] * len(paths)

        self.comms.publish(self.id, paths, probs)

    def receive(self):
        """
        Algorithm 1, Line 8: CommunicationReceive [cite: 227]
        Updates local cache of other robots' distributions.
        """
        num_robots = self.comms.get_num_robots()
        for r_id in range(num_robots):
            if r_id == self.id: continue

            # Read from the decentralized buffer
            paths, probs = self.comms.read_other(r_id)

            # Only update if we received valid data
            if paths:
                self.others_distributions[r_id] = (paths, probs)

    def run(self):
        """
        Main Execution Loop.
        Combines Algorithm 1 (Planning) with Online Execution (Replanning).
        """
        for step in range(self.config['SIMULATION_STEPS']):

            # --- Algorithm 1: Planning Block ---
            # Line 2: "While computation budget not met" [cite: 213]
            # We treat PLANNING_BUDGET as the number of refinement loops per physical step
            for _ in range(self.config['PLANNING_BUDGET']):

                # Line 3: Update X_hat
                self.select_set_of_sequences()

                # Line 4: For t_n iterations do [cite: 217]
                # (Inner MCTS sampling loop)
                for _ in range(self.config['NUM_SAMPLES']):
                    # Line 5: GrowTree (Algorithm 2) [cite: 219]
                    # This is the only point where we touch the Tree logic
                    self.tree.grow(self.others_distributions)

                    # Line 6: Update Probabilities
                    self.update_distribution()

                    # Line 7: Broadcast Plan
                    self.transmit()

                    # Line 8: Listen to Neighbors
                    self.receive()

                # Line 9: Cool beta (Skipped in this version)

            # --- Online Execution (Receding Horizon) ---
            # Line 10: Select best action sequence [cite: 231]
            best_plan = self.tree.get_best_path()

            # LOGGING: Save the tree state for visualization
            if self.logger:
                tree_viz = self.tree.extract_segments()
                self.logger.log_step(step, self.id, tree_viz, best_plan)

            # EXECUTE: Move physical robot one step
            if len(best_plan) > 1:
                next_pos = best_plan[1]
                self.pos = next_pos

                # CRITICAL: Prune/Update Tree Root [cite: 198-199]
                # "Online replanning... by adapting the previous search tree"
                self.tree.advance_root(self.pos)

                # Update global state (for visualization/simulation only)
                self.logger.update_trajectory(self.id, self.pos)

                # Attempt to collect reward
                # (Note: In a real decentralized system, the env handles this interaction)
                val = self.env.consume_reward(self.pos)
                if val > 0:
                    self.logger.log_collection(self.id, self.pos, val, step)

            # Simulate processing time
            time.sleep(0.01)