import math
import random
import numpy as np


# --- MCTS Node ---
class MCTSNode:
    """
    Represents a single state in the search tree.
    """

    def __init__(self, state, parent=None, action=None, valid_actions=None):
        self.state = state
        self.parent = parent
        self.action = action  # The action taken to reach this node
        self.children = {}  # Map: action -> MCTSNode

        # D-UCT Statistics
        self.t_gamma = 0.0  # Discounted visit count
        self.v_gamma = 0.0  # Discounted value sum

        # Expansion status
        # Default grid moves: Up, Down, Right, Left, Stay
        self.untried_actions = valid_actions if valid_actions else [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self, current_depth, max_depth):
        """
        Terminality is defined by the physical path limit (max_depth),
        NOT the computational budget.
        """
        return current_depth >= max_depth

    def best_child(self, exploration_weight):
        """
        [cite_start]Selection Policy (D-UCB). [cite: 303]
        """
        best_score = -float('inf')
        best_child_node = None
        t_parent = self.t_gamma

        for child in self.children.values():
            if child.t_gamma < 1e-9: return child

            mean_reward = child.v_gamma / child.t_gamma
            # Exploration term
            exploration = 2 * exploration_weight * math.sqrt(math.log(t_parent) / child.t_gamma)

            score = mean_reward + exploration
            if score > best_score:
                best_score = score
                best_child_node = child
        return best_child_node

    def update_stats(self, reward, gamma):
        """Recursive Discounting Update [cite: 318]"""
        self.t_gamma = (self.t_gamma * gamma) + 1
        self.v_gamma = (self.v_gamma * gamma) + reward


# --- MCTS Tree ---
class MCTSTree:
    """
    Handles the persistent tree structure and Algorithm 2 (GrowTree).
    """

    def __init__(self, start_state, env, robot_id, path_limit, gamma, cp, rollout_budget):
        self.env = env
        self.id = robot_id
        self.path_limit = path_limit  # Physical max path length (Depth limit) B^r [cite: 157]
        self.rollout_budget = rollout_budget  # Computational limit for BFS (Nodes explored)
        self.gamma = gamma
        self.cp = cp

        # PERSISTENCE:
        self.original_root = MCTSNode(start_state)
        self.current_root = self.original_root

    def advance_root(self, new_state):
        """Updates the current planning root after execution."""
        found_child = None
        for child in self.current_root.children.values():
            if child.state == new_state:
                found_child = child
                break

        if found_child:
            self.current_root = found_child
        else:
            new_node = MCTSNode(new_state, parent=self.current_root)
            self.current_root = new_node

    # --- ALGORITHM 2: GROW TREE ---

    def grow(self, others_distributions):
        """
        Executes one iteration of MCTS expansion.
        """
        # 1. SELECTION & 2. EXPANSION (Recursive)
        # [cite_start]Uses path_limit to stop recursion if tree is too deep [cite: 264]
        leaf_node, depth = self._select_recursive(self.current_root, 0)

        # 3. SIMULATION
        x_others = self._sample_others(others_distributions)

        # Rollout uses 'rollout_budget' to limit search effort, passed depth to enforce path limit
        rollout_path = self._rollout_policy_bfs(leaf_node, depth, x_others)

        path_from_root = self._reconstruct_path_from_current_root(leaf_node)
        x_self = path_from_root + rollout_path

        # [cite_start]4. EVALUATION (Local Utility F_t) [cite: 201]
        reward_joint = self.env.get_global_utility([x_self] + x_others)
        reward_baseline = self.env.get_global_utility([[]] + x_others)
        F_t = reward_joint - reward_baseline

        # [cite_start]5. BACKPROPAGATION [cite: 279]
        curr = leaf_node
        while curr is not None:
            curr.update_stats(F_t, self.gamma)
            if curr == self.current_root.parent: break
            curr = curr.parent

    # --- Recursive Selection Logic ---

    def _select_recursive(self, node, depth):
        # Stop if we hit the physical path limit
        if node.is_terminal(depth, self.path_limit):
            return node, depth

        # Recursive Step: Is Node fully expanded?
        if node.is_fully_expanded():
            best_child = node.best_child(self.cp)
            if best_child is None: return node, depth
            return self._select_recursive(best_child, depth + 1)

        else:
            # EXPANSION
            return self._expand_node(node), depth + 1

    def _expand_node(self, node):
        move = node.untried_actions.pop()
        nx, ny = node.state[0] + move[0], node.state[1] + move[1]

        if not self.env.is_valid((nx, ny)):
            nx, ny = node.state

        if move in node.children:
            return node.children[move]
        else:
            new_node = MCTSNode((nx, ny), parent=node, action=move)
            node.children[move] = new_node
            return new_node

    # --- Exhaustive BFS Rollout (Budget-Limited) ---

    def _rollout_policy_bfs(self, start_node, current_depth, x_others):
        """
        Exhaustive BFS Rollout limited by COMPUTATIONAL BUDGET.
        Expands 'self.rollout_budget' number of nodes to find best local path.
        """
        if self.rollout_budget <= 0:
            return []

        # 1. Pre-calculate blocked rewards
        taken_by_others = set()
        for path in x_others:
            for p in path:
                taken_by_others.add(p)

        # 2. BFS Initialization
        # Queue: (current_pos, path_history, current_score, visited_set)
        queue = [(start_node.state, [], 0.0, set())]

        best_path = []
        best_score = -1.0

        nodes_explored = 0  # Counter for budget

        while queue:
            # Check budget - This is the NEW NODE exploration limit
            if nodes_explored >= self.rollout_budget:
                break

            curr_pos, path, score, taken_set = queue.pop(0)  # Pop from front (BFS)
            nodes_explored += 1

            # Check all moves
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            for dx, dy in moves:
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy

                # Check Physical Path Limit
                if self.env.is_valid((nx, ny)) and (current_depth + len(path) + 1 <= self.path_limit):

                    # Calc Reward
                    step_reward = 0
                    is_new_reward = False
                    if (nx, ny) in self.env.rewards:
                        if (nx, ny) not in taken_by_others and (nx, ny) not in taken_set:
                            step_reward = self.env.rewards[(nx, ny)]
                            is_new_reward = True

                    new_score = score + step_reward

                    # Update Global Best
                    if new_score > best_score:
                        best_score = new_score
                        best_path = path + [(nx, ny)]

                    # Add to queue
                    new_taken = taken_set.copy()
                    if is_new_reward: new_taken.add((nx, ny))

                    queue.append(((nx, ny), path + [(nx, ny)], new_score, new_taken))

        return best_path

    # --- Helpers ---

    def _sample_others(self, others_distributions):
        """Samples one path per robot. [cite: 274]"""
        x_others = []
        for rid, (paths, probs) in others_distributions.items():
            if not paths: continue
            if len(probs) == len(paths) and sum(probs) > 0:
                p_sum = sum(probs)
                norm_probs = [p / p_sum for p in probs]
                idx = np.random.choice(len(paths), p=norm_probs)
                x_others.append(paths[idx])
            else:
                x_others.append(paths[0])
        return x_others

    def _reconstruct_path_from_current_root(self, node):
        path = []
        curr = node
        while curr is not None and curr != self.current_root.parent:
            path.append(curr.state)
            curr = curr.parent
        return list(reversed(path))

    def get_best_path(self):
        """Greedy selection for execution. [cite: 388]"""
        curr = self.current_root
        path = [curr.state]
        depth = 0
        while curr.children and depth < self.path_limit:
            best = max(curr.children.values(), key=lambda n: n.t_gamma)
            path.append(best.state)
            curr = best
            depth += 1
        return path

    def extract_segments(self):
        segments = []
        stack = [self.current_root]
        while stack:
            curr = stack.pop()
            for child in curr.children.values():
                segments.append((curr.state, child.state))
                stack.append(child)
        return segments