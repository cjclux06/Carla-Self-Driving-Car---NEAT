from Genome import Genome
from Species import Species
from Fitness_Genome import Fitness_Genome
from agents.navigation.global_route_planner import GlobalRoutePlanner
import random
import time
import carla
import math


class Evaluator:

    # NEAT compatibility distance coefficients
    C1 = 1
    C2 = 2
    C3 = 0.4
    DT = 3  # compatibility threshold for speciation

    # Mutation rates
    MUTATION_RATE         = 0.75
    ADD_CONNECTION_RATE   = 0.1
    ADD_NODE_RATE         = 0.1

    # Evaluation settings
    EVAL_DURATION_SECONDS = 60
    RAY_ANGLES            = [-90, -45, -20, 0, 20, 45, 90]
    MAX_RAY_DISTANCE      = 50.0
    MAX_VEHICLE_SCAN_DIST = 50.0
    NUM_NEARBY_VEHICLES   = 2
    BRAKE_DEADZONE        = 0.6   # FIX: raised from 0.3 — sigmoid outputs ~0.5 with random weights,
                                  #      so the old threshold let brake through constantly and
                                  #      fight the throttle (CARLA gives brake priority → car stalls)
    MAX_SPAWN_ATTEMPTS    = 25
    SPAWN_SETTLE_TICKS    = 5     # FIX: ticks to wait after spawning before eval starts,
                                  #      so the vehicle is physically ready in synchronous mode

    def __init__(self, population_size, starting_genome, node_innovation,
                 connection_innovation, world=None, vehicle=None,
                 vehicle_bp=None, spawn_point=None, spectator=None,
                 vehicles=None, traffic_manager_port=8000):

        self.population_size       = population_size
        self.node_innovation       = node_innovation
        self.connection_innovation = connection_innovation
        self.world                 = world
        self.vehicle               = vehicle
        self.vehicle_bp            = vehicle_bp
        self.spawn_point           = spawn_point
        self.spectator             = spectator
        self.vehicles              = vehicles if vehicles is not None else []
        self.traffic_manager_port  = traffic_manager_port  # FIX: needed for traffic refresh

        self.genomes          = [Genome(starting_genome) for _ in range(population_size)]
        self.next_gen_genomes = []
        self.mapped_species   = {}
        self.score_map        = {}
        self.species          = []
        self.highest_score    = float("-inf")
        self.fittest_genome   = None

    # -------------------------------------------------------------------------
    # Traffic health check
    # -------------------------------------------------------------------------

    def _refresh_traffic(self):
        """
        FIX: Check every traffic vehicle is still alive and respawn any that
        have been destroyed (collision, spawn conflict, CARLA culling after
        a laptop sleep/wake cycle, etc.).  Without this the vehicles list
        goes stale and traffic gradually disappears over long runs.
        """
        blueprints   = self.world.get_blueprint_library().filter('vehicle.*')
        all_spawns   = self.world.get_map().get_spawn_points()

        alive      = [v for v in self.vehicles if v.is_alive]
        dead_count = len(self.vehicles) - len(alive)
        self.vehicles = alive

        for _ in range(dead_count):
            bp    = random.choice(blueprints)
            sp    = random.choice(all_spawns)
            actor = self.world.try_spawn_actor(bp, sp)
            if actor:
                actor.set_autopilot(True, self.traffic_manager_port)
                self.vehicles.append(actor)

        if dead_count > 0:
            print(f"[Traffic] Replaced {dead_count} dead vehicle(s) "
                  f"({len(self.vehicles)} total alive)")

    # -------------------------------------------------------------------------
    # Input helpers
    # -------------------------------------------------------------------------

    def _get_ego_inputs(self, vehicle):
        """Speed (normalized) and current steering angle."""
        vel   = vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
        steer = vehicle.get_control().steer
        return [speed / 30.0, steer]

    def _get_navigation_inputs(self, location, forward, right, waypoint):
        """
        Heading error, distance to next waypoint, and signed lane offset.
        - heading_error : cross product of forward vs direction-to-waypoint [-1, 1]
                          negative = waypoint is left, positive = waypoint is right
        - dist_to_wp    : normalized distance to the next road waypoint [0, 1]
        - signed_offset : signed distance from lane center (negative = left of center)
        """
        # Signed lane offset from center
        lane_center  = waypoint.transform.location
        offset_vec   = location - lane_center
        signed_offset = offset_vec.x * right.x + offset_vec.y * right.y
        signed_offset = max(-1.0, min(1.0, signed_offset / 4.0))

        # Heading error via next waypoint
        next_wps = waypoint.next(5.0)
        if next_wps:
            next_loc  = next_wps[0].transform.location
            to_wp     = next_loc - location
            to_wp_mag = (to_wp.x**2 + to_wp.y**2) ** 0.5 + 1e-6

            heading_error = forward.x * (to_wp.y / to_wp_mag) - forward.y * (to_wp.x / to_wp_mag)
            heading_error = max(-1.0, min(1.0, heading_error))
            dist_to_wp    = min(location.distance(next_loc) / 10.0, 1.0)
        else:
            heading_error = 0.0
            dist_to_wp    = 0.0

        return [heading_error, dist_to_wp, signed_offset]

    def _get_raycast_inputs(self, vehicle):
        """7 raycasts at predefined angles, each normalized to [0, 1]."""
        return [
            self.raycaster_distance(self.world, vehicle, angle)
            for angle in self.RAY_ANGLES
        ]

    def _get_nearby_vehicle_inputs(self, location, right):
        """
        Distance and signed lateral angle for the N nearest other vehicles.
        - dist         : normalized [0, 1], where 1.0 = at or beyond MAX_VEHICLE_SCAN_DIST
        - signed_angle : negative = vehicle is to the left, positive = to the right [-1, 1]
        """
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        others = sorted(
            [v for v in all_vehicles if v.id != self.vehicle.id],
            key=lambda v: v.get_location().distance(location)
        )

        inputs = []
        for v in others[:self.NUM_NEARBY_VEHICLES]:
            v_loc   = v.get_location()
            rel     = v_loc - location
            rel_mag = (rel.x**2 + rel.y**2) ** 0.5 + 1e-6

            dist         = min(v_loc.distance(location) / self.MAX_VEHICLE_SCAN_DIST, 1.0)
            signed_angle = (rel.x * right.x + rel.y * right.y) / rel_mag
            signed_angle = max(-1.0, min(1.0, signed_angle))
            inputs.extend([dist, signed_angle])

        # Pad with defaults if fewer than NUM_NEARBY_VEHICLES vehicles found
        while len(inputs) < self.NUM_NEARBY_VEHICLES * 2:
            inputs.extend([1.0, 0.0])

        return inputs

    # -------------------------------------------------------------------------
    # Main raycast helper
    # -------------------------------------------------------------------------

    def raycaster_distance(self, world, vehicle, angle_deg):
        """
        Cast a ray from the vehicle at angle_deg relative to its forward direction.
        Returns normalized hit distance [0, 1]. Returns 1.0 if nothing is hit.
        """
        transform = vehicle.get_transform()
        forward   = transform.get_forward_vector()
        right     = transform.get_right_vector()
        angle_rad = math.radians(angle_deg)

        direction = carla.Vector3D(
            forward.x * math.cos(angle_rad) - right.x * math.sin(angle_rad),
            forward.y * math.cos(angle_rad) - right.y * math.sin(angle_rad),
            0.0
        )

        start = transform.location + carla.Location(z=1.5)
        end   = start + direction * self.MAX_RAY_DISTANCE
        hits  = world.cast_ray(start, end)

        if hits:
            return start.distance(hits[0].location) / self.MAX_RAY_DISTANCE
        return 1.0

    # -------------------------------------------------------------------------
    # Genome evaluation
    # -------------------------------------------------------------------------

    def evaluate_genome(self, genome, route_locations):
        """
        Run the genome for EVAL_DURATION_SECONDS in CARLA synchronous mode.

        Inputs (16 total):
            [0]    Speed (normalized)
            [1]    Current steering angle
            [2]    Heading error to next waypoint
            [3]    Distance to next waypoint (normalized)
            [4]    Signed lane offset
            [5-11] Raycasts at -90°, -45°, -20°, 0°, 20°, 45°, 90°
            [12-13] Nearest vehicle: distance, signed lateral angle
            [14-15] 2nd nearest vehicle: distance, signed lateral angle

        Outputs (3):
            [0] Throttle  [0, 1]
            [1] Brake     [0, 1]  (zeroed below BRAKE_DEADZONE)
            [2] Steer     [0, 1] → remapped to [-1, 1]

        Returns:
            float: fitness score for this genome
        """

        # Fitness weights
        W_FORWARD    =  2.0
        W_REVERSE    = -3.0
        W_HEADING    = -1.5
        W_OFFSET     = -1.0
        W_SPEED      =  0.5
        W_IDLE       = -1.5
        W_COLLISION  = -50.0
        W_CHECKPOINT =  10.0
        TARGET_SPEED        = 8.0   # m/s (~28 km/h)
        IDLE_RAY_THRESHOLD  = 0.5   # forward ray: below this means something is ahead
        CHECKPOINT_RADIUS   = 3.0   # meters — how close counts as reaching a checkpoint
        STUCK_TIME_LIMIT    = 10.0  # FIX: seconds without meaningful movement before ending eval

        # Set up collision sensor
        collision_history = []
        collision_bp      = self.world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor  = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        collision_sensor.listen(lambda e: collision_history.append(e))

        fitness          = 0.0
        start_time       = time.time()
        prev_location    = self.vehicle.get_transform().location
        checkpoint_index = 0

        # FIX: track time without meaningful movement to detect a stuck car
        stuck_timer_start    = time.time()
        STUCK_MOVEMENT_THRESHOLD = 0.05  # meters per tick to count as "moving"

        while time.time() - start_time < self.EVAL_DURATION_SECONDS:
            self.world.tick()

            transform = self.vehicle.get_transform()
            location  = transform.location
            forward   = transform.get_forward_vector()
            right     = transform.get_right_vector()
            waypoint  = self.world.get_map().get_waypoint(
                location, project_to_road=True, lane_type=carla.LaneType.Driving
            )

            # Assemble inputs
            inputs = (
                self._get_ego_inputs(self.vehicle)                               +  # [0-1]
                self._get_navigation_inputs(location, forward, right, waypoint)  +  # [2-4]
                self._get_raycast_inputs(self.vehicle)                           +  # [5-11]
                self._get_nearby_vehicle_inputs(location, right)                    # [12-15]
            )

            outputs  = genome.activate_inputs(inputs)
            throttle = max(0.0, min(1.0, float(outputs[0])))
            brake    = max(0.0, min(1.0, float(outputs[1])))
            steer    = max(-1.0, min(1.0, float(outputs[2]) * 2 - 1))

            # FIX: zero brake below deadzone AND suppress it entirely when
            # throttle is dominant.  Sigmoid outputs ~0.5 with random weights,
            # so without this suppression the car brakes itself to a standstill
            # every generation during early training.
            if brake < self.BRAKE_DEADZONE:
                brake = 0.0
            if throttle > 0.5:
                brake = 0.0

            self.vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, brake=brake, steer=steer
            ))

            # RAY_ANGLES = [-90, -45, -20, 0, 20, 45, 90]
            # inputs[5] = -90°  →  inputs[8] = 0° (dead ahead)
            forward_ray = inputs[8]

            # ── Checkpoint progress ───────────────────────────────────────────
            if route_locations and checkpoint_index < len(route_locations):
                dist_to_next = location.distance(route_locations[checkpoint_index])
                if dist_to_next < CHECKPOINT_RADIUS:
                    fitness += W_CHECKPOINT
                    checkpoint_index += 1

            # ── Forward progress ──────────────────────────────────────────────
            current_location = self.vehicle.get_transform().location
            dx = current_location.x - prev_location.x
            dy = current_location.y - prev_location.y
            forward_progress = dx * forward.x + dy * forward.y

            if forward_progress > 0:
                fitness += forward_progress * W_FORWARD
            else:
                fitness += forward_progress * abs(W_REVERSE)

            # ── Stuck detection ───────────────────────────────────────────────
            # FIX: if the car hasn't moved meaningfully for STUCK_TIME_LIMIT
            # seconds, end this genome's eval early with a flat penalty.
            # This prevents wasting 60 seconds per genome on a car that will
            # never move, which is the dominant failure mode in early training.
            movement = (dx**2 + dy**2) ** 0.5
            if movement > STUCK_MOVEMENT_THRESHOLD:
                stuck_timer_start = time.time()  # reset timer whenever car moves
            elif time.time() - stuck_timer_start > STUCK_TIME_LIMIT:
                fitness -= 20.0  # penalty for being stuck
                print(f"  [Stuck] Ending eval early after {STUCK_TIME_LIMIT}s without movement")
                break

            # ── Idle penalty (only when road ahead is clear) ──────────────────
            if abs(forward_progress) < 0.001:
                if forward_ray > IDLE_RAY_THRESHOLD:
                    fitness += W_IDLE

            # ── Speed shaping (context-aware) ─────────────────────────────────
            vel   = self.vehicle.get_velocity()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            if forward_ray > IDLE_RAY_THRESHOLD:
                speed_reward = 1.0 - abs(speed - TARGET_SPEED) / TARGET_SPEED
            else:
                expected_speed = TARGET_SPEED * forward_ray
                speed_reward   = 1.0 - abs(speed - expected_speed) / TARGET_SPEED

            fitness += max(0.0, speed_reward) * W_SPEED

            # ── Lane keeping ──────────────────────────────────────────────────
            lane_center  = waypoint.transform.location
            offset_vec   = current_location - lane_center
            lateral_dist = abs(offset_vec.x * right.x + offset_vec.y * right.y)
            fitness += lateral_dist * W_OFFSET

            # ── Heading alignment ─────────────────────────────────────────────
            road_forward = waypoint.transform.get_forward_vector()
            heading_dot  = forward.x * road_forward.x + forward.y * road_forward.y
            fitness += (1.0 - heading_dot) * W_HEADING

            # ── Collision ─────────────────────────────────────────────────────
            if collision_history:
                fitness += W_COLLISION * len(collision_history)
                collision_history.clear()
                break

            prev_location = current_location

        collision_sensor.destroy()
        print(f"Fitness: {fitness:.2f} | Checkpoints reached: {checkpoint_index}/{len(route_locations)}")
        return fitness

    # -------------------------------------------------------------------------
    # Species selection helpers
    # -------------------------------------------------------------------------

    def _get_random_species_biased(self):
        """Pick a species with probability proportional to its adjusted fitness."""
        total      = sum(s.total_adjusted_fitness for s in self.species)
        target     = random.random() * total
        cumulative = 0.0
        for species in self.species:
            cumulative += species.total_adjusted_fitness
            if cumulative >= target:
                return species
        return random.choice(self.species)

    def _get_random_genome_biased(self, species):
        """Pick a genome from a species with probability proportional to its fitness."""
        total      = sum(fg.fitness for fg in species.fitness_pop)
        target     = random.random() * total
        cumulative = 0.0
        for fg in species.fitness_pop:
            cumulative += fg.fitness
            if cumulative >= target:
                return fg.genome
        return species.fitness_pop[-1].genome

    # -------------------------------------------------------------------------
    # Main evaluation loop
    # -------------------------------------------------------------------------

    def evaluate(self):
        """Run one full generation: evaluate all genomes, speciate, reproduce."""

        # FIX: refresh traffic before doing anything else this generation.
        # Vehicles lost to collisions, spawn conflicts, or a laptop sleep/wake
        # are silently removed by CARLA and never replaced unless we check.
        self._refresh_traffic()

        # Reset species for this generation
        for s in self.species:
            s.reset()
        self.score_map.clear()
        self.mapped_species.clear()
        self.next_gen_genomes.clear()
        self.highest_score  = float("-inf")
        self.fittest_genome = None

        # --- Speciation ---
        for genome in self.genomes:
            assigned = False
            for species in self.species:
                if genome.compatibility_distance(genome, species.mascot, self.C1, self.C2, self.C3) < self.DT:
                    species.members.append(genome)
                    self.mapped_species[genome] = species
                    assigned = True
                    break
            if not assigned:
                new_species = Species(genome)
                self.species.append(new_species)
                self.mapped_species[genome] = new_species

        self.species = [s for s in self.species if s.members]

        # --- Pick one spawn and one destination for the entire generation ---
        all_spawn_points = self.world.get_map().get_spawn_points()
        generation_spawn = random.choice(all_spawn_points)

        MIN_ROUTE_DISTANCE = 100.0
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2.0)
        destination     = None
        route_locations = []

        attempts = 0
        while attempts < 20:
            candidate = random.choice(all_spawn_points)
            if candidate.location.distance(generation_spawn.location) < MIN_ROUTE_DISTANCE:
                attempts += 1
                continue
            try:
                route = grp.trace_route(generation_spawn.location, candidate.location)
                if len(route) > 5:
                    destination     = candidate
                    route_locations = [wp.transform.location for wp, _ in route]
                    break
            except Exception:
                pass
            attempts += 1

        if not route_locations:
            print("Warning: could not find a valid route — falling back to no-destination mode")

        print(f"Route has {len(route_locations)} waypoints | "
              f"Destination: {destination.location if destination else 'None'}")

        # --- Evaluation ---
        for genome in self.genomes:
            species = self.mapped_species[genome]
            score   = self.evaluate_genome(genome, route_locations)

            # Respawn vehicle for next genome
            self.vehicle.destroy()
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, generation_spawn)
            if not self.vehicle:
                random.shuffle(all_spawn_points)
                for spawn in all_spawn_points:
                    self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn)
                    if self.vehicle:
                        break
            if not self.vehicle:
                raise RuntimeError("Failed to respawn vehicle — all spawn points are blocked.")

            # FIX: tick several times so the vehicle is fully initialised in
            # synchronous mode before the next evaluate_genome reads its state.
            # A single tick left the vehicle in an uninitialised physics state,
            # causing garbage velocity/position readings at the start of each eval.
            for _ in range(self.SPAWN_SETTLE_TICKS):
                self.world.tick()

            # Keep spectator camera above the vehicle
            t = self.vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(
                t.location + carla.Location(z=15),
                carla.Rotation(pitch=-90)
            ))

            species.add_adjusted_fitness(score)
            species.fitness_pop.append(Fitness_Genome(genome, score))
            self.score_map[genome] = score

            if score > self.highest_score:
                self.highest_score  = score
                self.fittest_genome = genome

        # --- Elitism: carry forward the best genome from each species ---
        for species in self.species:
            species.fitness_pop.sort(key=lambda fg: fg.fitness, reverse=True)
            self.next_gen_genomes.append(species.fitness_pop[0].genome)

        # --- Reproduction ---
        while len(self.next_gen_genomes) < self.population_size:
            species = self._get_random_species_biased()
            p1      = self._get_random_genome_biased(species)
            p2      = self._get_random_genome_biased(species)

            if self.score_map.get(p1, float("-inf")) >= self.score_map.get(p2, float("-inf")):
                child = Genome.crossover(p1, p2)
            else:
                child = Genome.crossover(p2, p1)

            if random.random() < self.MUTATION_RATE:
                child.mutation()
            if random.random() < self.ADD_CONNECTION_RATE:
                child.connection_mutation(self.connection_innovation, 100)
            if random.random() < self.ADD_NODE_RATE:
                child.node_mutation(self.connection_innovation, self.node_innovation)

            self.next_gen_genomes.append(child)

        self.genomes = self.next_gen_genomes
        self.next_gen_genomes = []
