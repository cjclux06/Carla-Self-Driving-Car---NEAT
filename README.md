# Autonomous Driving in CARLA Through the Application of NEAT

This is a from scratch implemenation of **NEAT** (NeuroEvolution of Augmenting Topologies) running inside a [CARLA](https://carla.org/) driving simulator to train an autonomous vehicle to drive with some commonality to humans. 

The neural network starts with the most simplistic form with inputs directly connected to outputs. Over generations the topology of the network changes through purposely randomized evolution to force the car to stay in lanes and avoid collision. 

---

## Demo / Results

At cullmaination data s saved to "training_results.png showing things like fitness curves, species count, and network complexity over all the generations. 

---

## How It Works

### NEAT Overview

NEAT evolves both the weights *and* the structure of neural networks at the same time. Each genome is given a directed graph of nodes and weighted connections. Over the generations, mutations can:

- **Perturb weights** — small Gaussian nudges or random resets
- **Add a connection** — a new synapse between two previously unconnected nodes
- **Add a node** — an existing connection is split; a new hidden node is inserted in between

Genomes are "grouped" into **species** to allow closly built genomes to go through evolution together and allow new networks to grow with less competition. 
Genomes are grouped into **species** using a compatibility distance metric (excess genes, disjoint genes, and average weight difference), so structural innovations are protected from being immediately out-competed.

### Fitness Function

Each genome is evaluated for 60 seconds of real simulation time. Fitness is a weighted sum of:

| Signal | Weight | Notes |
|---|---|---|
| Forward progress | +2.0 | Dot product of movement with forward vector |
| Reverse movement | -3.0 | Penalises driving backwards |
| Heading alignment | -1.5 | Penalises facing away from the road direction |
| Lane offset | -1.0 | Penalises distance from center of closest lane |
| Speed shaping | +0.5 | Rewards driving close to 8 m/s |
| Idle penalty | -1.5 | Penalises not moving except when stopped with obstacle in front |
| Checkpoint reached | +10.0 | Bonus per route waypoint passed |
| Collision | -50.0 | Applied on collision, run ends as well |
| Stuck penalty | -20.0 | Eval ends early if good movement for 10 s |

### Neural Network Inputs (16)

| Index | Description |
|---|---|
| 0 | Speed that has been normalized |
| 1 | Current steering angle |
| 2 | Direction error to next waypoint |
| 3 | Distance to next waypoint |
| 4 | Signed lane offset |
| 5–11 | Raycasts at −90°, −45°, −20°, 0°, +20°, +45°, +90° |
| 12–13 | Nearest vehicle: signed lateral angle |
| 14–15 | 2nd nearest vehicle: signed lateral angle |

### Neural Network Outputs (3)

| Index | Description |
|---|---|
| 0 | Throttle `[0, 1]` |
| 1 | Brake `[0, 1]` (below 0.6 deadzone; suppressed when throttle > 0.5) |
| 2 | Steer — done with sigmoid output remapped to `[-1, 1]` |

---

## Project Structure

```
.
├── Main.py              # Entry point — CARLA setup, population init, training loop
├── Evaluator.py         # NEAT evaluation, speciation, and reproduction logic
├── Genome.py            # Genome class — nodes, connections, mutation, crossover, activation
├── Node.py              # Node class — ReLU (hidden) and sigmoid (output) activations
├── Connection.py        # Connection (gene) class with innovation tracking
├── Species.py           # Species class for NEAT speciation
├── Counter.py           # Global innovation number counter
├── Fitness_Genome.py    # Simple wrapper pairing a genome with its fitness score
└── config.py            # CARLA path configuration (not included — see Setup)
```

---

## Requirements

- Python 3.8+
- [CARLA Simulator](https://carla.org/) (tested on 0.9.x)
- CARLA Python API (bundled with the simulator)

**Python dependencies:**

```bash
pip install matplotlib
```

No ML frameworks (PyTorch, TensorFlow, etc.) are required — the neural network is implemented from scratch.

---

## Setup

1. **Install CARLA** — download from the [official releases page](https://github.com/carla-simulator/carla/releases).

2. **Create `config.py`** in the project root:

   ```python
   CARLA_PATH       = "/path/to/carla"          # e.g. /opt/carla-simulator
   CARLA_EXECUTABLE = "/path/to/CarlaUE4.sh"    # or CarlaUE4.exe on Windows
   ```

3. **Run training:**

   ```bash
   python Main.py
   ```

   CARLA will launch automatically. Training runs for 100 generations with a population of 100 genomes. Each genome is evaluated for up to 60 seconds.

---

## Configuration

Key hyperparameters can be adjusted at the top of `Evaluator.py`:

```python
C1 = 1          # Excess gene coefficient (compatibility distance)
C2 = 2          # Disjoint gene coefficient
C3 = 0.8        # Weight difference coefficient
DT = 10         # Compatibility threshold for speciation

MUTATION_RATE       = 0.75
ADD_CONNECTION_RATE = 0.15
ADD_NODE_RATE       = 0.15

EVAL_DURATION_SECONDS = 60
MAX_RAY_DISTANCE      = 50
NUM_NEARBY_VEHICLES   = 2
```

Population size and generation count are set in `Main.py`:

```python
inputs  = 16
outputs = 3
evals   = 100 generations

eval = Evaluator(100, ...)  # 100 = population size
```

---

## Key Design Decisions

**Synchronous CARLA mode** — the simulator only advances when `world.tick()` is called, giving deterministic, reproducible evaluations without timing drift.

**Shared route per generation** — all genomes in a generation are evaluated on the same randomly chosen origin so that fitness scores are dependent on driving not randomly chosen route.

**Traffic refresh** — before each generation, destroyed or culled traffic vehicles are automatically replaced so the traffic density stays consistent throughout a long training run so no uneveness is allowed in training.

**Brake deadzone** — sigmoid activations cluster near 0.5 with random initial weights. A 0.6 deadzone prevents the brake from constantly fighting the throttle due to carla prioritizing the brake when it is being applied. This allows early generations to learn to drive without getting softlocked into not moving.

**Stuck detection** — if a genome goes 10 seconds without moving more than 0.05 m per tick, evaluation ends early with a −20 penalty, so time can be saved during training on genomes that haven't actually learned to drive yet.

---

## References

- Stanley, K.O. & Miikkulainen, R. (2002). [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). *Evolutionary Computation*, 10(2), 99–127.
- [CARLA Simulator Documentation](https://carla.readthedocs.io/)
