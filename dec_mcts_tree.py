import math
import random
import numpy as np


# --- MCTS Node ---
class MCTSNode:
    def __init__(self, state, parent=None, action=None, valid_actions=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.t_gamma = 0.0  # Discounted visits
        self.v_gamma = 0.0  # Discounted value
        # Default grid moves: Up, Down, Right, Left, Stay
        self.untried_actions = valid_actions if valid_actions else [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self, current_depth, max_depth):
        return current_depth >= max_depth

    def best_child(self, exploration_weight):
        best_score = -float('inf')
        best_child_node = None
        t_parent = self.t_gamma
        for child in self.children.values():
            if child.t_gamma < 1e-9: return child
            mean_reward = child.v_gamma / child.t_gamma
            exploration = 2 * exploration_weight * math.sqrt(math.log(t_parent) / child.t_gamma)
            score = mean_reward + exploration
            if score > best_score:
                best_score = score
                best_child_node = child
        return best_child_node

    def update_stats(self, reward, gamma):
        self.t_gamma = (self.t_gamma * gamma) + 1
        self.v_gamma = (self.v_gamma * gamma) + reward


# --- MCTS Tree ---
class MCTSTree:
    def __init__(self, start_state, env, robot_id, path_limit, gamma, cp, rollout_budget):
        self.env = env
        self.id = robot_id
        self.path_limit = path_limit
        self.rollout_budget = rollout_budget
        self.gamma = gamma
        self.cp = cp
        self.original_root = MCTSNode(start_state)
        self.current_root = self.original_root

    def advance_root(self, new_state):
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

    def grow(self, others_distributions):
        leaf_node, depth = self._select_recursive(self.current_root, 0)
        x_others = self._sample_others(others_distributions)
        rollout_path = self._rollout_policy_bfs(leaf_node, depth, x_others)
        path_from_root = self._reconstruct_path_from_current_root(leaf_node)
        x_self = path_from_root + rollout_path
        reward_joint = self.env.get_global_utility([x_self] + x_others)
        reward_baseline = self.env.get_global_utility([[]] + x_others)
        F_t = reward_joint - reward_baseline
        curr = leaf_node
        while curr is not None:
            curr.update_stats(F_t, self.gamma)
            if curr == self.current_root.parent: break
            curr = curr.parent

    def _select_recursive(self, node, depth):
        if node.is_terminal(depth, self.path_limit):
            return node, depth
        if node.is_fully_expanded():
            best_child = node.best_child(self.cp)
            if best_child is None: return node, depth
            return self._select_recursive(best_child, depth + 1)
        else:
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

    def _rollout_policy_bfs(self, start_node, current_depth, x_others):
        if self.rollout_budget <= 0: return []
        taken_by_others = set()
        for path in x_others:
            for p in path: taken_by_others.add(p)
        queue = [(start_node.state, [], 0.0, set())]
        best_path = []
        best_score = -1.0
        nodes_explored = 0
        while queue:
            if nodes_explored >= self.rollout_budget: break
            curr_pos, path, score, taken_set = queue.pop(0)
            nodes_explored += 1
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            random.shuffle(moves)
            for dx, dy in moves:
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy
                if self.env.is_valid((nx, ny)) and (current_depth + len(path) + 1 <= self.path_limit):
                    step_reward = 0
                    is_new_reward = False
                    if (nx, ny) in self.env.rewards:
                        if (nx, ny) not in taken_by_others and (nx, ny) not in taken_set:
                            step_reward = self.env.rewards[(nx, ny)]
                            is_new_reward = True
                    new_score = score + step_reward
                    if new_score > best_score:
                        best_score = new_score
                        best_path = path + [(nx, ny)]
                    new_taken = taken_set.copy()
                    if is_new_reward: new_taken.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)], new_score, new_taken))
        return best_path

    def _sample_others(self, others_distributions):
        x_others = []
        for rid, (paths, probs) in others_distributions.items():
            if not paths: continue
            if len(probs) == len(paths) and sum(probs) > 0:
                norm_probs = np.array(probs) / np.sum(probs)
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
        curr = self.current_root
        path = [curr.state]
        depth = 0
        while curr.children and depth < self.path_limit:
            best = max(curr.children.values(), key=lambda n: n.t_gamma)
            path.append(best.state)
            curr = best
            depth += 1
        return path

    def get_top_k_paths(self, k):
        """
        Identify top K paths from the tree based on value.
        Uses a BFS traversal to collect candidate nodes, sorts by value, and reconstructs paths.
        """
        # Collect all nodes in the subtree of current_root
        candidates = []
        queue = [(self.current_root, [])]  # Node, Path from root

        # Traverse tree to collect all valid full or partial paths
        # We limit collection to somewhat deep nodes to avoid just returning the root 10 times
        while queue:
            curr, path = queue.pop(0)
            full_path = path + [curr.state]

            # Use discounted value as score
            score = curr.v_gamma / curr.t_gamma if curr.t_gamma > 0 else 0
            candidates.append((score, full_path))

            for child in curr.children.values():
                queue.append((child, full_path))

        # Sort desc by score
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Extract unique paths (Top K)
        top_k = []
        seen_paths = set()

        for score, p in candidates:
            # Convert to tuple for set hashing
            path_tuple = tuple(p)
            if path_tuple not in seen_paths:
                # Ensure path length is reasonable (not just length 1)
                if len(p) > 1:
                    top_k.append(p)
                    seen_paths.add(path_tuple)
            if len(top_k) >= k:
                break

        # If we don't have enough, pad with best path or current state
        while len(top_k) < k:
            if top_k:
                top_k.append(top_k[0])
            else:
                top_k.append([self.current_root.state])

        return top_k