# Decentralized Monte Carlo Tree Search (Dec-MCTS) Simulation

Implementation of decentralized MCTS for multi-robot coordination with comprehensive visualization tools.

## Overview

This implementation features:
- **Decentralized MCTS** with distributed action optimization
- **D-UCT tree policy** for action selection
- **Inter-robot communication** of action distributions
- **Comprehensive visualizations** for tree structure and probability distributions

## Modules

### 1. `dec_mcts_tree.py` - MCTS Tree Structure

**Purpose:** Implements the MCTS tree with nodes and search algorithms.

**Key Classes:**
- `MCTSNode`: Tree node storing state, visits, values, rollout statistics
- `MCTSTree`: Tree structure with growth and selection methods

**Key Methods:**
- `grow()`: Performs one iteration of MCTS (selection → expansion → rollout → backpropagation)
- `select()`: D-UCT selection with consideration of other robots
- `expand()`: Adds new child node to tree
- `rollout()`: Random simulation from node
- `backpropagate()`: Updates node statistics up the tree
- `advance_root()`: Moves tree root forward after execution

**Used By:** `dec_mcts_robot.py`

**Uses:** None (standalone tree implementation)

---

### 2. `dec_mcts_robot.py` - Robot Agent

**Purpose:** Robot agent that runs MCTS, optimizes actions, and communicates with neighbors.

**Key Classes:**
- `GridEnvironment`: Defines grid world with obstacles and rewards
- `DecMCTSRobot`: Robot agent implementing Algorithm 1

**Key Methods:**
- `run()`: Main loop executing simulation steps
- `select_set_of_sequences()`: Extracts action sequences from MCTS tree (Algorithm 1, Line 3)
- `update_distribution()`: Optimizes probability distribution (Algorithm 3)
- `transmit()`: Sends action distribution to neighbors
- `receive()`: Receives distributions from neighbors

**Communication:**
- Uses `comms` object for inter-robot message passing
- Shares `(action_set, action_probs)` tuples

**Used By:** `dec_mcts.py`

**Uses:**
- `MCTSTree` from `dec_mcts_tree.py` - Creates and grows tree
- `TreeVisualizer` from `dec_mcts_visualizer.py` - Captures distributions and tree snapshots

---

### 3. `dec_mcts.py` - Main Simulation

**Purpose:** Sets up and runs multi-robot simulation with visualization.

**Key Functions:**
- `main()`: Initializes environment, robots, communication, and runs simulation

**Workflow:**
1. Creates grid environment with obstacles and rewards
2. Initializes `TreeVisualizer`
3. Creates `DecMCTSRobot` instances with shared communication
4. Runs robots in parallel threads
5. Captures final tree state
6. Generates visualizations

**Used By:** User (entry point)

**Uses:**
- `GridEnvironment`, `DecMCTSRobot` from `dec_mcts_robot.py` - Creates robot agents
- `TreeVisualizer` from `dec_mcts_visualizer.py` - Captures and visualizes data

**Output Files:**
- `tree_robot1_step2.png`, `step4.png`, `step6.png` - Individual tree visualizations
- `tree_robot1_comparison.png` - Tree evolution comparison
- `action_distributions_robot1.png` - Action probability distributions

---

### 4. `dec_mcts_visualizer.py` - Visualization

**Purpose:** Visualizes MCTS trees and action probability distributions.

**Key Classes:**
- `TreeVisualizer`: Captures snapshots and generates visualizations

**Tree Visualization Methods:**
- `capture_tree_snapshot()`: Captures tree at specific step
  - **Called by:** `dec_mcts_robot.py` at steps 2, 4, 6
- `capture_final_tree_with_roots()`: Captures final tree with root positions
  - **Called by:** `dec_mcts.py` after simulation
- `visualize_tree()`: Renders tree from current root with radial layout
  - **Called by:** `dec_mcts.py` for steps 2, 4, 6
- `visualize_rollout_comparison()`: Shows final tree with different roots marked
  - **Called by:** `dec_mcts.py` for comparison view

**Action Distribution Visualization Methods:**
- `capture_action_distribution()`: Captures distributions before/after optimization
  - **Called by:** `dec_mcts_robot.py` at steps 3, 4 (before and after `update_distribution()`)
- `visualize_action_distributions()`: Renders before/after comparison with action labels
  - **Called by:** `dec_mcts.py` for steps 3, 4

**Helper Methods:**
- `_layout_tree_radial()`: Positions nodes in circular layout
- `_draw_action_paths_highlighted()`: Draws paths with best action in red
- `_draw_probability_bars_highlighted()`: Draws probability bars
- `_get_action_sequence()`: Converts path to action labels (UP, DOWN, etc.)

**Used By:** `dec_mcts.py`, `dec_mcts_robot.py`

**Uses:** matplotlib, numpy for plotting

---

## Module Dependencies

```
dec_mcts.py (main entry point)
    ├─> dec_mcts_robot.py
    │       ├─> dec_mcts_tree.py
    │       └─> dec_mcts_visualizer.py (captures data)
    └─> dec_mcts_visualizer.py (generates plots)
```

## Usage

### Run Simulation

```bash
python dec_mcts.py
```

**Generates:**
- `tree_robot1_step2.png` - Tree at step 2
- `tree_robot1_step4.png` - Tree at step 4
- `tree_robot1_step6.png` - Tree at step 6
- `tree_robot1_comparison.png` - Tree evolution
- `action_distributions_robot1.png` - Probability distributions (steps 3, 4)

### Run Tests

**Tree visualization test:**
```bash
python test_tree_visualization.py
```
Output: `test_tree.png`

**Action distribution test:**
```bash
python test_action_distributions.py
```
Output: `test_action_distributions.png`

---

## Configuration

**In `dec_mcts.py`:**

```python
CONFIG = {
    'GRID_SIZE': 50,              # Grid dimensions
    'NUM_ROBOTS': 8,              # Number of robots
    'SIMULATION_STEPS': 10,       # Total simulation steps
    'BUDGET': 5,                  # Path length limit
    'GAMMA': 0.95,                # Discount factor
    'CP': 1.0,                    # Exploration constant
    'ROLLOUT_BUDGET': 50,         # Rollout simulation budget
    'TAU_N': 1,                   # Inner loop iterations
    'NUM_SAMPLES': 50,            # MCTS samples per inner loop
    'ACTION_SET_SIZE': 10,        # Number of action sequences
}
```

---

## Algorithm Overview

### Algorithm 1: Main Loop (in `dec_mcts_robot.py::run()`)
1. Select set of action sequences from tree
2. **Inner loop (τ_n times):**
   - Grow MCTS tree with samples
   - Update probability distribution (Algorithm 3)
   - Transmit distribution to neighbors
   - Receive distributions from neighbors
3. Execute best action
4. Advance tree root

### Algorithm 2: Tree Growth (in `dec_mcts_tree.py::grow()`)
1. **Selection:** D-UCT traversal to leaf
2. **Expansion:** Add new child node
3. **Rollout:** Random simulation from new node
4. **Backpropagation:** Update statistics to root

### Algorithm 3: Update Distribution (in `dec_mcts_robot.py::update_distribution()`)
1. Extract action sequences from tree
2. Calculate probabilities based on Q-values
3. Normalize distribution

---

## Key Features

### Tree Visualization
- **Radial layout** for deep trees
- **Rollout statistics** on each node (V, Q, R)
- **Root tracking** across steps
- **Color coding** by Q-value

### Action Distribution Visualization
- **Before/After comparison** of optimization
- **Action sequence labels** (e.g., "UP → RIGHT")
- **Best path highlighting** in red
- **Neighbor distributions** in orange
- **All values within plot bounds**

---

## Visualization Details

### Tree Visualization (Steps 2, 4, 6)

**Individual Trees:**
- Shows tree from current root at each step
- Nodes display: position, visits (V), Q-value (Q), rollout (R)
- Root marked with red circle

**Comparison View:**
- Shows final tree with roots marked at different steps
- Red circles indicate where root was at each step
- Demonstrates tree evolution

### Action Distribution Visualization (Steps 3, 4)

**Layout per step:**
```
[Paths BEFORE | Prob BEFORE | Paths AFTER | Prob AFTER | Robot 2 | Robot 3 ...]
```

**Features:**
- Best path in RED with action label (yellow box)
- Other paths in transparent blue
- Probability bars with values
- Received distributions from neighbors (orange)

**Adjustment:** To move action labels, edit `text_y` in `_draw_action_paths_highlighted()` (line ~153)

---

## Output Files

| File | Description |
|------|-------------|
| `tree_robot1_step2.png` | Tree visualization at step 2 |
| `tree_robot1_step4.png` | Tree visualization at step 4 |
| `tree_robot1_step6.png` | Tree visualization at step 6 |
| `tree_robot1_comparison.png` | Tree evolution comparison |
| `action_distributions_robot1.png` | Action probability distributions |

---

## Customization

### Change Visualization Steps

**Tree visualization:**
Edit `dec_mcts_robot.py` line ~195:
```python
if self.tree_visualizer and self.id == 1 and step in [2, 4, 6]:
```

**Action distribution:**
Edit `dec_mcts_robot.py` lines ~169 and ~181:
```python
if self.tree_visualizer and self.id == 1 and step in [3, 4]:
```

### Change Robot to Visualize

Change `self.id == 1` to desired robot ID in both locations above.

### Adjust Action Label Position

Edit `dec_mcts_visualizer.py` line ~153:
```python
text_y = max(ys) + 0.5  # Decrease to move down, increase to move up
```

---

## Requirements

- Python 3.x
- numpy
- matplotlib

---

## Documentation

- `ACTION_DISTRIBUTION_VISUALIZATION.md` - Detailed guide for action distribution visualization
- `CHANGES.txt` - Summary of changes and features

---

## References

Algorithm based on decentralized MCTS for multi-robot coordination. See project documentation for algorithm details and theoretical background.