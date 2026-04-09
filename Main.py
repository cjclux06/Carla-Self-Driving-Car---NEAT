import sys
import os
from config import CARLA_PATH, CARLA_EXECUTABLE

# Add CARLA's Python API to the path so 'agents' module can be found.
# This must happen before any other imports.
sys.path.append(os.path.join(CARLA_PATH, "carla"))
sys.path.append(os.path.join(CARLA_PATH, "carla", "agents"))

from Evaluator import Evaluator
from Counter import Counter
from Genome import Genome
from Node import Node, NodeType
from Connection import Connection
import random
import carla
import subprocess
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for long training runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

inputs = 16
outputs = 3
evals = 100

if __name__ == "__main__":

    carla_process = subprocess.Popen([CARLA_EXECUTABLE, "-quality-level=Epic"])

    time.sleep(20)

    print("Launching CARLA... waiting for server to be ready")

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)

    world = None
    while world is None:
        try:
            world = client.get_world()
            print("CARLA world loaded!")
        except RuntimeError as e:
            print("ERROR:", e)
            time.sleep(2)
    #world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Get a vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    #print("Vehicle spawned:", vehicle is not None)
    #print("Vehicle ID:", vehicle.id if vehicle else "None")
    #print("Vehicle transform:", vehicle.get_transform())

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=15), carla.Rotation(pitch=-90)))

    # --- Configure synchronous mode FIRST ---
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()

    num_vehicles = 50
    vehicles = []

    # --- Spawn vehicles ---
    for i in range(num_vehicles):
        blueprint = random.choice(blueprints)
        spawn_point = random.choice(spawn_points)

        vehicle_actor = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle_actor:
            vehicle_actor.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(vehicle_actor)

# --- IMPORTANT: Tick the world so everything actually spawns ---
    for _ in range(20):
        world.tick()

    print("Connected to CARLA!")

    connection_innovation = Counter()
    node_innovation = Counter()

    genome = Genome()

    for i in range(inputs):
        genome.add_node(Node(type=NodeType.INPUT, innovation=node_innovation.get_current_innovation()))
    
    for i in range(outputs):
        genome.add_node(Node(type=NodeType.OUTPUT, innovation=node_innovation.get_current_innovation()))
    
    for i in range(inputs):
        for j in range(outputs):
            genome.add_connection(Connection(in_node_id=genome.nodes.get(i).innovation, out_node_id=genome.nodes.get(inputs + j).innovation, weight=random.random() * 2 - 1, expressed=True, innovation=connection_innovation.get_current_innovation()))
    
    eval = Evaluator(50, genome, node_innovation, connection_innovation, world=world, vehicle=vehicle, vehicle_bp=vehicle_bp, spawn_point=spawn_point, spectator=spectator, vehicles=vehicles)

    # --- Training history ---
    history = {
        "generation":        [],
        "best_fitness":      [],
        "avg_fitness":       [],
        "worst_fitness":     [],
        "num_species":       [],
        "num_connections":   [],
    }

    for i in range(evals):
        print(f"\n=== GENERATION {i} ===")

        prev_genomes = set(id(g) for g in eval.genomes) if i > 0 else set()

        eval.evaluate()

        curr_genomes = set(id(g) for g in eval.genomes)
        new_genomes  = curr_genomes - prev_genomes

        # Collect per-genome scores for avg / worst
        all_scores   = list(eval.score_map.values())
        avg_fitness   = sum(all_scores) / len(all_scores) if all_scores else 0.0
        worst_fitness = min(all_scores) if all_scores else 0.0

        # Count expressed connections in best performer
        expressed_count = sum(
            1 for c in eval.fittest_genome.connections.values() if c.expressed
        )
        weight_sum = sum(
            c.weight for c in eval.fittest_genome.connections.values() if c.expressed
        )

        # Record history
        history["generation"].append(i)
        history["best_fitness"].append(eval.highest_score)
        history["avg_fitness"].append(avg_fitness)
        history["worst_fitness"].append(worst_fitness)
        history["num_species"].append(len(eval.species))
        history["num_connections"].append(expressed_count)

        # Console output
        print(f"New genomes in population : {len(new_genomes)}")
        print(f"Highest fitness           : {eval.highest_score:.2f}")
        print(f"Average fitness           : {avg_fitness:.2f}")
        print(f"Worst fitness             : {worst_fitness:.2f}")
        print(f"Species count             : {len(eval.species)}")
        print(f"Expressed connections (best): {expressed_count}")
        print(f"Weight sum (best)         : {weight_sum:.3f}")

        if i > 0 and len(new_genomes) == 0:
            print("❌ ALERT: No new genomes created! Evolution is stuck!")
            break

    # ── Save training graph ───────────────────────────────────────────────────
    gens = history["generation"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("NEAT Training Progress", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Fitness over generations ---
    ax1 = axes[0]
    ax1.plot(gens, history["best_fitness"],   color="#2ecc71", linewidth=2,   label="Best")
    ax1.plot(gens, history["avg_fitness"],    color="#3498db", linewidth=1.5, label="Average", linestyle="--")
    ax1.plot(gens, history["worst_fitness"],  color="#e74c3c", linewidth=1,   label="Worst",   linestyle=":")
    ax1.fill_between(gens, history["worst_fitness"], history["best_fitness"],
                     alpha=0.08, color="#3498db")
    ax1.set_title("Fitness over Generations", fontweight="bold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Species count ---
    ax2 = axes[1]
    ax2.bar(gens, history["num_species"], color="#9b59b6", alpha=0.75, width=0.8)
    ax2.set_title("Species Count over Generations", fontweight="bold")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Number of Species")
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Network complexity (expressed connections in best genome) ---
    ax3 = axes[2]
    ax3.plot(gens, history["num_connections"], color="#e67e22", linewidth=2, marker="o",
             markersize=3, label="Expressed connections")
    ax3.set_title("Network Complexity — Best Genome (Expressed Connections)",
                  fontweight="bold")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Connections")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    graph_path = "training_results.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Training complete. Graph saved to: {graph_path}")