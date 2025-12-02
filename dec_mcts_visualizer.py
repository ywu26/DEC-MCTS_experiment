import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import copy


class TreeVisualizer:
    """Visualizes MCTS tree structure with node statistics and rollout values"""

    def __init__(self):
        self.snapshots = {}  # {(robot_id, step): tree_data}
        self.action_distributions = {}  # {(robot_id, step): distribution_data}
        self.final_tree = None

    def capture_tree_snapshot(self, robot_id, step, tree):
        """
        Capture a deep copy of the tree structure for later visualization

        Args:
            robot_id: Robot ID
            step: Current simulation step
            tree: MCTSTree instance
        """
        # Capture from CURRENT root to show tree from this step's perspective
        tree_data = self._extract_tree_data(tree.current_root, tree)

        # Store metadata
        tree_data['current_root_state'] = tree.current_root.state
        tree_data['step'] = step

        self.snapshots[(robot_id, step)] = tree_data
        print(
            f"[Snapshot] Captured tree for Robot {robot_id} at Step {step} (from current root: {tree.current_root.state})")

    def capture_final_tree_with_roots(self, robot_id, tree, root_states_by_step):
        """
        Capture final tree state with markers for where roots were at different steps
        Used for comparison visualization

        Args:
            robot_id: Robot ID
            tree: MCTSTree instance (final state)
            root_states_by_step: Dict mapping step -> root state position
        """
        # Capture from original root to show full tree
        tree_data = self._extract_tree_data(tree.original_root, tree)
        tree_data['root_states_by_step'] = root_states_by_step
        tree_data['is_final'] = True

        self.final_tree = (robot_id, tree_data)
        print(f"[Final Tree] Captured for Robot {robot_id} with {len(root_states_by_step)} root positions")

    def capture_action_distribution(self, robot_id, step, action_set, action_probs, others_distributions, current_pos,
                                    before_optimization=False):
        """
        Capture action sequence probability distributions for Algorithm 3 visualization

        Args:
            robot_id: Robot ID
            step: Current simulation step
            action_set: List of action sequences (paths)
            action_probs: List of probabilities for each action sequence
            others_distributions: Dict of {other_robot_id: (action_set, action_probs)}
            current_pos: Current position of the robot
            before_optimization: If True, this is before optimization; if False, after
        """
        key_suffix = '_before' if before_optimization else '_after'
        key = (robot_id, step, key_suffix)

        data = {
            'action_set': action_set,
            'action_probs': action_probs,
            'others_distributions': others_distributions,
            'current_pos': current_pos,
            'step': step,
            'before_optimization': before_optimization
        }

        self.action_distributions[key] = data
        stage = "BEFORE" if before_optimization else "AFTER"
        print(
            f"[Distribution] Captured for Robot {robot_id} at Step {step} ({stage} optimization): {len(action_set)} sequences, {len(others_distributions)} other robots")

    def visualize_action_distributions(self, robot_id, steps, save_path=None):
        """
        Visualize action sequence probability distributions showing before/after optimization
        Highlights the best path in red

        Args:
            robot_id: Robot ID to visualize
            steps: List of steps to show (should be [3, 4])
            save_path: Optional path to save figure
        """
        num_steps = len(steps)
        fig = plt.figure(figsize=(24, 6 * num_steps))

        for step_idx, step in enumerate(steps):
            key_before = (robot_id, step, '_before')
            key_after = (robot_id, step, '_after')

            if key_before not in self.action_distributions or key_after not in self.action_distributions:
                print(f"Missing before/after data for Robot {robot_id} at Step {step}")
                continue

            data_before = self.action_distributions[key_before]
            data_after = self.action_distributions[key_after]

            # Get data
            action_set = data_after['action_set']
            probs_before = data_before['action_probs']
            probs_after = data_after['action_probs']
            others_distributions = data_after['others_distributions']
            current_pos = data_after['current_pos']

            # Find best action index (after optimization)
            best_idx = probs_after.index(max(probs_after))

            # Create grid: [Paths Before | Prob Before | Paths After | Prob After | Others...]
            num_others = len(others_distributions)
            num_cols = 4 + num_others

            gs = fig.add_gridspec(1, num_cols,
                                  left=0.05, right=0.95,
                                  top=0.95 - step_idx / num_steps,
                                  bottom=0.95 - (step_idx + 1) / num_steps,
                                  wspace=0.25,
                                  width_ratios=[2, 1, 2, 1] + [1] * num_others)

            # Panel 1: Paths BEFORE optimization
            ax1 = fig.add_subplot(gs[0, 0])
            self._draw_action_paths_highlighted(ax1, action_set, probs_before, current_pos, best_idx,
                                                title=f"Step {step} - BEFORE Optimization\nAction Sequences")

            # Panel 2: Probabilities BEFORE
            ax2 = fig.add_subplot(gs[0, 1])
            self._draw_probability_bars_highlighted(ax2, probs_before, best_idx,
                                                    title=f"Probabilities\n(Before)",
                                                    color='blue')

            # Panel 3: Paths AFTER optimization
            ax3 = fig.add_subplot(gs[0, 2])
            self._draw_action_paths_highlighted(ax3, action_set, probs_after, current_pos, best_idx,
                                                title=f"Step {step} - AFTER Optimization\nBest Path in Red")

            # Panel 4: Probabilities AFTER
            ax4 = fig.add_subplot(gs[0, 3])
            self._draw_probability_bars_highlighted(ax4, probs_after, best_idx,
                                                    title=f"Probabilities\n(After)",
                                                    color='blue')

            # Panel 5+: Other robots
            for idx, (other_id, (other_actions, other_probs)) in enumerate(sorted(others_distributions.items())):
                ax_other = fig.add_subplot(gs[0, 4 + idx])
                self._draw_probability_bars(ax_other, other_probs,
                                            title=f"Robot {other_id}\n(Received)",
                                            color='orange')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Action distribution visualization saved to {save_path}")

        plt.show()

    def _draw_action_paths_highlighted(self, ax, action_set, action_probs, current_pos, best_idx, title):
        """Draw action sequences with best path highlighted in red and action labels"""
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Find grid bounds
        all_positions = [current_pos]
        for path in action_set:
            all_positions.extend(path)

        if not all_positions:
            return

        xs = [p[0] for p in all_positions]
        ys = [p[1] for p in all_positions]

        # Add padding to ensure all content fits
        min_x, max_x = min(xs) - 2, max(xs) + 2
        min_y, max_y = min(ys) - 2, max(ys) + 2

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # Draw non-best paths first (transparent blue)
        for idx, (path, prob) in enumerate(zip(action_set, action_probs)):
            if idx == best_idx:
                continue  # Skip best, draw later

            if len(path) < 2:
                continue

            xs = [p[0] for p in path]
            ys = [p[1] for p in path]

            # Transparent blue for non-best paths
            ax.plot(xs, ys, 'o-', color='blue', alpha=0.15,
                    linewidth=1.5, markersize=3, zorder=1)

        # Draw best path (red, opaque)
        if best_idx < len(action_set):
            best_path = action_set[best_idx]
            if len(best_path) >= 2:
                xs = [p[0] for p in best_path]
                ys = [p[1] for p in best_path]

                ax.plot(xs, ys, 'o-', color='red', alpha=1.0,
                        linewidth=3, markersize=6, zorder=3,
                        label=f'Best Path (p={action_probs[best_idx]:.3f})')

                # Add action sequence text
                actions = self._get_action_sequence(best_path)
                action_text = " → ".join(actions)

                # Place text above the path
                mid_idx = len(best_path) // 2
                text_x = xs[mid_idx]
                text_y = max(ys) - 3.5

                ax.text(text_x, text_y, action_text,
                        fontsize=9, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                        fontweight='bold', zorder=5)

        # Highlight current position
        ax.plot(current_pos[0], current_pos[1], '*', color='darkgreen',
                markersize=20, markeredgecolor='black', markeredgewidth=1,
                label='Current Pos', zorder=4)

        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)

    def _get_action_sequence(self, path):
        """Convert a path to action labels"""
        if len(path) < 2:
            return ["STAY"]

        actions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]

            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]

            if dx == 0 and dy == 0:
                actions.append("STAY")
            elif dx == 0 and dy == 1:
                actions.append("UP")
            elif dx == 0 and dy == -1:
                actions.append("DOWN")
            elif dx == 1 and dy == 0:
                actions.append("RIGHT")
            elif dx == -1 and dy == 0:
                actions.append("LEFT")
            else:
                actions.append(f"({dx},{dy})")

        return actions

    def _draw_probability_bars_highlighted(self, ax, probs, best_idx, title, color):
        """Draw probability distribution with best action highlighted"""
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

        indices = range(len(probs))
        colors = ['red' if i == best_idx else color for i in indices]
        alphas = [1.0 if i == best_idx else 0.6 for i in indices]

        for i, (prob, c, alpha) in enumerate(zip(probs, colors, alphas)):
            ax.barh(i, prob, color=c, alpha=alpha, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Probability', fontsize=9)
        ax.set_ylabel('Sequence #', fontsize=9)
        ax.set_ylim(-0.5, len(probs) - 0.5)

        # Set x-limit with extra space for text labels
        max_prob = max(probs) if probs else 1.0
        ax.set_xlim(0, max_prob * 1.25)  # Increased from 1.1 to 1.25 for text space

        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        # Add probability values on bars
        for i, prob in enumerate(probs):
            if prob > 0.01:  # Only show if significant
                # Position text slightly inside the bar if it fits, otherwise outside
                text_x = min(prob - 0.01, prob * 0.95) if prob > 0.1 else prob + 0.01
                text_ha = 'right' if prob > 0.1 else 'left'

                ax.text(text_x, i, f'{prob:.3f}',
                        va='center', ha=text_ha, fontsize=8,
                        fontweight='bold' if i == best_idx else 'normal',
                        color='white' if (prob > 0.1 and i == best_idx) else 'black')

    def _draw_probability_bars(self, ax, probs, title, color):
        """Draw probability distribution as bar chart"""
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

        indices = range(len(probs))
        ax.barh(indices, probs, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Probability', fontsize=9)
        ax.set_ylabel('Sequence #', fontsize=9)
        ax.set_ylim(-0.5, len(probs) - 0.5)

        # Set x-limit with extra space for text labels
        max_prob = max(probs) if probs else 1.0
        ax.set_xlim(0, max_prob * 1.25)  # Extra space for labels

        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        # Add probability values on bars
        for i, prob in enumerate(probs):
            if prob > 0.01:  # Only show if significant
                # Position text appropriately
                text_x = min(prob - 0.01, prob * 0.95) if prob > 0.1 else prob + 0.01
                text_ha = 'right' if prob > 0.1 else 'left'
                text_color = 'white' if prob > 0.1 else 'black'

                ax.text(text_x, i, f'{prob:.3f}',
                        va='center', ha=text_ha, fontsize=7,
                        color=text_color)

    def _extract_tree_data(self, root, tree):
        """
        Recursively extract tree structure into a serializable format

        Returns:
            dict with tree structure and statistics
        """

        def traverse(node, depth=0, parent_id=None, action=None):
            """Recursively build tree data structure"""
            node_id = id(node)

            # Calculate node statistics
            mean_value = node.v_gamma / node.t_gamma if node.t_gamma > 0 else 0

            # Calculate rollout statistics
            if hasattr(node, 'rollout_rewards') and node.rollout_rewards:
                avg_rollout = np.mean(node.rollout_rewards)
                std_rollout = np.std(node.rollout_rewards)
                min_rollout = np.min(node.rollout_rewards)
                max_rollout = np.max(node.rollout_rewards)
                rollout_count = node.rollout_count if hasattr(node, 'rollout_count') else len(node.rollout_rewards)
            else:
                avg_rollout = 0
                std_rollout = 0
                min_rollout = 0
                max_rollout = 0
                rollout_count = 0

            node_data = {
                'id': node_id,
                'state': node.state,
                'parent_id': parent_id,
                'action': action,
                'depth': depth,
                't_gamma': node.t_gamma,
                'v_gamma': node.v_gamma,
                'mean_value': mean_value,
                'num_children': len(node.children),
                'is_fully_expanded': node.is_fully_expanded(),
                'rollout_count': rollout_count,
                'avg_rollout': avg_rollout,
                'std_rollout': std_rollout,
                'min_rollout': min_rollout,
                'max_rollout': max_rollout,
                'children': []
            }

            # Recursively process children
            for child_action, child_node in node.children.items():
                child_data = traverse(child_node, depth + 1, node_id, child_action)
                node_data['children'].append(child_data)

            return node_data

        tree_structure = traverse(root)

        return {
            'tree': tree_structure,
            'gamma': tree.gamma,
            'cp': tree.cp,
            'path_limit': tree.path_limit,
            'root_state': root.state
        }

    def visualize_tree(self, robot_id, step, max_depth=5, save_path=None, layout='radial'):
        """
        Visualize the MCTS tree for a specific robot at a specific step
        Shows tree from the current root at that step

        Args:
            robot_id: Robot ID
            step: Simulation step
            max_depth: Maximum depth to visualize (to avoid clutter)
            save_path: Optional path to save figure
            layout: 'radial' (better for deep trees) or 'hierarchical'
        """
        key = (robot_id, step)
        if key not in self.snapshots:
            print(f"No snapshot found for Robot {robot_id} at Step {step}")
            return

        data = self.snapshots[key]
        tree = data['tree']
        current_root_state = data.get('current_root_state', tree['state'])

        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_title(
            f"MCTS Tree - Robot {robot_id} at Step {step}\nRoot: {current_root_state} ({layout.title()} Layout)",
            fontsize=16, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')

        # Position nodes using selected layout
        positions = {}
        if layout == 'radial':
            self._layout_tree_radial(tree, positions, 0, 2 * np.pi, 0, 0, max_depth)
        else:
            self._layout_tree(tree, positions, x=0, y=0, level=0, width=40, max_depth=max_depth)

        # Draw edges first (so they appear behind nodes)
        self._draw_edges(ax, tree, positions, max_depth)

        # Draw nodes (without current root marker since tree IS from current root)
        self._draw_nodes(ax, tree, positions, max_depth, current_root_id=None)

        # Add legend
        self._add_legend(ax, data)

        # Add statistics panel
        self._add_statistics_panel(ax, tree, data)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Tree visualization saved to {save_path}")

        plt.show()

    def _layout_tree_radial(self, node, positions, angle_start, angle_end, radius, level, max_depth):
        """
        Calculate positions for tree nodes using a radial layout
        Better for deep trees - children arranged in arcs around parent

        Args:
            node: Current node data
            positions: Dictionary to store positions
            angle_start: Start angle (radians) for this subtree
            angle_end: End angle (radians) for this subtree
            radius: Distance from center
            level: Current depth level
            max_depth: Maximum depth to layout
        """
        if level > max_depth:
            return

        node_id = node['id']

        # Calculate position
        if level == 0:
            # Root at center
            x, y = 0, 0
        else:
            # Position on arc
            angle = (angle_start + angle_end) / 2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

        positions[node_id] = (x, y)

        # Layout children
        num_children = len(node['children'])
        if num_children > 0 and level < max_depth:
            # Calculate angular space for each child
            angle_span = angle_end - angle_start

            # For root, use full circle; for others, use available arc
            if level == 0:
                child_angle_start = 0
                child_angle_end = 2 * np.pi
                angle_span = 2 * np.pi
            else:
                child_angle_start = angle_start
                child_angle_end = angle_end

            angle_per_child = angle_span / num_children
            next_radius = radius + 1.2  # COMPACT: smaller increment

            for i, child in enumerate(node['children']):
                child_start = child_angle_start + i * angle_per_child
                child_end = child_start + angle_per_child

                self._layout_tree_radial(
                    child, positions,
                    child_start, child_end,
                    next_radius, level + 1, max_depth
                )

    def _layout_tree(self, node, positions, x, y, level, width, max_depth):
        """
        Calculate positions for tree nodes using a hierarchical layout
        (Legacy - kept for compatibility, but radial is better)
        """
        if level > max_depth:
            return

        node_id = node['id']
        positions[node_id] = (x, -level * 1.5)  # Increased vertical spacing

        # Layout children
        num_children = len(node['children'])
        if num_children > 0 and level < max_depth:
            # Divide available width among children
            child_width = width / max(num_children, 1)
            start_x = x - (width / 2) + (child_width / 2)

            for i, child in enumerate(node['children']):
                child_x = start_x + (i * child_width)
                self._layout_tree(child, positions, child_x, y - 1, level + 1, child_width * 0.8, max_depth)

    def _draw_edges(self, ax, node, positions, max_depth, level=0):
        """Draw edges between parent and child nodes"""
        if level >= max_depth:
            return

        node_id = node['id']
        if node_id not in positions:
            return

        parent_pos = positions[node_id]

        for child in node['children']:
            child_id = child['id']
            if child_id not in positions:
                continue

            child_pos = positions[child_id]

            # Draw edge
            ax.plot([parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    'k-', alpha=0.3, linewidth=1.5, zorder=1)

            # Add action label on edge
            mid_x = (parent_pos[0] + child_pos[0]) / 2
            mid_y = (parent_pos[1] + child_pos[1]) / 2
            action = child['action']
            if action:
                action_str = self._action_to_string(action)
                ax.text(mid_x, mid_y, action_str, fontsize=7,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                        zorder=2)

            # Recursively draw child edges
            self._draw_edges(ax, child, positions, max_depth, level + 1)

    def _draw_nodes(self, ax, node, positions, max_depth, current_root_id=None, level=0):
        """Draw nodes with statistics"""
        if level > max_depth:
            return

        node_id = node['id']
        if node_id not in positions:
            return

        pos = positions[node_id]

        # Node color based on value
        mean_value = node['mean_value']
        if node['t_gamma'] > 0:
            # Normalize color (green = high value, red = low value)
            norm_value = np.clip(mean_value / 10, 0, 1)  # Assuming max reward ~10
            color = plt.cm.RdYlGn(norm_value)
        else:
            color = 'lightgray'  # Unexplored

        # SMALLER node size based on visit count
        size = min(400, 100 + node['t_gamma'] * 10)

        # Draw node circle - SMALLER RADIUS
        circle = plt.Circle(pos, 0.08, color=color, ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)

        # Node label with statistics (including rollout info)
        state_str = f"({node['state'][0]},{node['state'][1]})"
        stats_str = f"V={node['t_gamma']:.1f}\nQ={mean_value:.2f}"

        # Add rollout information if available
        if node['rollout_count'] > 0:
            rollout_str = f"\nR={node['avg_rollout']:.1f}±{node['std_rollout']:.1f}"
            stats_str += rollout_str

        # State position above node
        ax.text(pos[0], pos[1] + 0.15, state_str,
                fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=4)

        # Statistics inside/below node
        ax.text(pos[0], pos[1], stats_str,
                fontsize=5.5, ha='center', va='center', zorder=4)

        # Highlight root node (tree is from this root)
        if level == 0:
            highlight = plt.Circle(pos, 0.12, fill=False, ec='red', linewidth=2, zorder=4)
            ax.add_patch(highlight)
            ax.text(pos[0], pos[1] - 0.2, "ROOT",
                    fontsize=8, ha='center', va='top', color='red', fontweight='bold', zorder=4)

        # Mark specific root if comparison view
        if current_root_id and node_id == current_root_id:
            highlight = plt.Circle(pos, 0.11, fill=False, ec='red', linewidth=3, zorder=4)
            ax.add_patch(highlight)

        # Recursively draw children
        for child in node['children']:
            self._draw_nodes(ax, child, positions, max_depth, current_root_id, level + 1)

    def _action_to_string(self, action):
        """Convert action tuple to readable string"""
        dx, dy = action
        if dx == 0 and dy == 0:
            return "Stay"
        elif dx == 1 and dy == 0:
            return "R"
        elif dx == -1 and dy == 0:
            return "L"
        elif dx == 0 and dy == 1:
            return "U"
        elif dx == 0 and dy == -1:
            return "D"
        else:
            return f"({dx},{dy})"

    def _add_legend(self, ax, data):
        """Add legend explaining visualization elements"""
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label='High Value Node', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Low Value Node', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markersize=8, label='Unexplored Node', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markersize=12, label='Tree Root', markeredgecolor='red', markeredgewidth=2),
            Line2D([0], [0], color='none', label='────────────'),
            Line2D([0], [0], color='none', label='V = Discounted Visits'),
            Line2D([0], [0], color='none', label='Q = Mean Value'),
            Line2D([0], [0], color='none', label='R = Avg Rollout ± Std'),
        ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def _add_statistics_panel(self, ax, tree, data):
        """Add text panel with tree statistics"""
        stats_text = f"""Tree Statistics:
━━━━━━━━━━━━━━━━━━━━
Root State: {data['root_state']}
Gamma (γ): {data['gamma']}
Exploration (Cp): {data['cp']}
Path Limit: {data['path_limit']}
━━━━━━━━━━━━━━━━━━━━
Total Nodes: {self._count_nodes(tree)}
Max Depth: {self._get_max_depth(tree)}
Root Visits: {tree['t_gamma']:.1f}
Root Value: {tree['v_gamma']:.2f}
Root Mean Q: {tree['mean_value']:.3f}
━━━━━━━━━━━━━━━━━━━━
Rollout Statistics:
Root Rollouts: {tree['rollout_count']}
Avg Rollout: {tree['avg_rollout']:.2f}
Std Rollout: {tree['std_rollout']:.2f}
Min/Max: [{tree['min_rollout']:.1f}, {tree['max_rollout']:.1f}]
"""

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _count_nodes(self, node):
        """Count total nodes in tree"""
        count = 1
        for child in node['children']:
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node, current_depth=0):
        """Get maximum depth of tree"""
        if not node['children']:
            return current_depth
        return max(self._get_max_depth(child, current_depth + 1)
                   for child in node['children'])

    def visualize_all_snapshots(self, output_dir="./tree_visualizations"):
        """Visualize all captured snapshots"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for (robot_id, step) in sorted(self.snapshots.keys()):
            save_path = f"{output_dir}/robot_{robot_id}_step_{step}.png"
            self.visualize_tree(robot_id, step, save_path=save_path)

    def visualize_rollout_comparison(self, robot_id, steps, save_path=None, layout='radial'):
        """
        Create comparison visualization showing same final tree with different root positions marked

        Args:
            robot_id: Robot ID
            steps: List of steps to show root positions for
            save_path: Optional path to save figure
            layout: 'radial' or 'hierarchical'
        """
        if not hasattr(self, 'final_tree') or self.final_tree is None:
            print("No final tree captured. Use capture_final_tree_with_roots() first.")
            return

        final_robot_id, tree_data = self.final_tree
        if final_robot_id != robot_id:
            print(f"Final tree is for robot {final_robot_id}, not {robot_id}")
            return

        tree = tree_data['tree']
        root_states_by_step = tree_data.get('root_states_by_step', {})

        num_steps = len(steps)
        fig, axes = plt.subplots(1, num_steps, figsize=(7 * num_steps, 7))

        if num_steps == 1:
            axes = [axes]

        fig.suptitle(f"Tree Evolution - Robot {robot_id} (Same Final Tree, Different Roots)",
                     fontsize=18, fontweight='bold', y=0.98)

        for idx, step in enumerate(steps):
            ax = axes[idx]

            if step not in root_states_by_step:
                ax.text(0.5, 0.5, f"No root position for step {step}",
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue

            root_state = root_states_by_step[step]

            ax.set_title(f"Step {step}\nRoot: {root_state}", fontsize=12, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')

            # Position nodes with selected layout
            positions = {}
            if layout == 'radial':
                self._layout_tree_radial(tree, positions, 0, 2 * np.pi, 0, 0, max_depth=5)
            else:
                self._layout_tree(tree, positions, x=0, y=0, level=0, width=8, max_depth=5)

            # Find node ID with matching state
            root_node_id = None

            def find_node_by_state(node, target_state):
                if node['state'] == target_state:
                    return node['id']
                for child in node['children']:
                    result = find_node_by_state(child, target_state)
                    if result is not None:
                        return result
                return None

            root_node_id = find_node_by_state(tree, root_state)

            # Draw tree
            self._draw_edges_simple(ax, tree, positions, max_depth=5)
            self._draw_nodes_simple(ax, tree, positions, max_depth=5, current_root_id=root_node_id)

            # Add mini statistics
            stats_text = f"Nodes: {self._count_nodes(tree)}\nMarked Root: {root_state}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison visualization saved to {save_path}")

        plt.show()

    def _draw_edges_simple(self, ax, node, positions, max_depth, level=0):
        """Simplified edge drawing for comparison view"""
        if level >= max_depth:
            return

        node_id = node['id']
        if node_id not in positions:
            return

        parent_pos = positions[node_id]

        for child in node['children']:
            child_id = child['id']
            if child_id not in positions:
                continue

            child_pos = positions[child_id]
            ax.plot([parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    'k-', alpha=0.3, linewidth=1, zorder=1)

            self._draw_edges_simple(ax, child, positions, max_depth, level + 1)

    def _draw_nodes_simple(self, ax, node, positions, max_depth, current_root_id=None, level=0):
        """Simplified node drawing for comparison view"""
        if level > max_depth:
            return

        node_id = node['id']
        if node_id not in positions:
            return

        pos = positions[node_id]

        # Color based on value
        mean_value = node['mean_value']
        if node['t_gamma'] > 0:
            norm_value = np.clip(mean_value / 10, 0, 1)
            color = plt.cm.RdYlGn(norm_value)
        else:
            color = 'lightgray'

        # Draw node - SMALLER SIZE
        circle = plt.Circle(pos, 0.06, color=color, ec='black', linewidth=1, zorder=3)
        ax.add_patch(circle)

        # Mark root if this is level 0
        if level == 0:
            highlight = plt.Circle(pos, 0.09, fill=False, ec='blue', linewidth=1.5, linestyle='--', zorder=4)
            ax.add_patch(highlight)

        # Mark current root with red circle
        if current_root_id and node_id == current_root_id:
            highlight = plt.Circle(pos, 0.08, fill=False, ec='red', linewidth=2, zorder=4)
            ax.add_patch(highlight)

        # Label - only for important nodes
        if level == 0 or (current_root_id and node_id == current_root_id):
            ax.text(pos[0], pos[1], f"{node['t_gamma']:.0f}",
                    fontsize=6, ha='center', va='center', fontweight='bold', zorder=4)

        # Recursively draw children
        for child in node['children']:
            self._draw_nodes_simple(ax, child, positions, max_depth, current_root_id, level + 1)