import gymnasium as gym
from gymnasium import spaces
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

class ParallelParkingEnv(AbstractEnv):
    """
    A specific environment for Parallel Parking with HER compatibility.
    State Space: 8 Dimensions (3 Relative + 3 Absolute + 2 Physics)
    Action Space: 2 Dimensions (Steering, Acceleration)
    """
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": True,
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction",
                # Reduced acceleration for precise parking control
                "acceleration_range": [-1.0, 1.0], 
                # Increased steering range for tighter parking maneuvers
                "steering_range": [-np.pi / 4, np.pi / 4], # Steering range +/- 45 degrees
                "longitudinal": True,
                "lateral": True
            },
            # 30Hz simulation frequency makes the car move slower and smoother
            "simulation_frequency": 30, 
            "policy_frequency": 5,
            "duration": 200, 
            "screen_width": 800,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            "scaling": 10,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        })
        return config
    
    def __init__(self, config: dict = None, render_mode: str = None):
        self.spot_width = 7.5  
        self.spot_length = 3.5 
        self.parking_slots = {}
        self._define_parking_slots()
        
        # Initialize goal placeholder
        self.goal = np.array([0, 0, 0], dtype=np.float32)
        self.goal_slot_id = 1
        
        super().__init__(config, render_mode)

        # Define Observation Space Manually (Size 8)
        # This prevents the recursion error in spaces.Box logic
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
    
    def _define_parking_slots(self):
        """Defines the coordinates of the 8 parking spots."""
        spacing = 7.5
        # Row 1 (Top)
        for i in range(4):
            slot_num = i + 1
            x_pos = -12 + i * spacing
            self.parking_slots[slot_num] = {
                'position': np.array([x_pos, 10]),
                'heading': np.pi,
                'occupied': False,
                'row': 1
            }
        # Row 2 (Bottom)
        for i in range(4):
            slot_num = i + 5
            x_pos = -12 + i * spacing
            self.parking_slots[slot_num] = {
                'position': np.array([x_pos, -10]),
                'heading': 0,
                'occupied': False,
                'row': 2
            }

    def _reset(self) -> None:
        """
        Internal reset logic.
        1. Pick Goal FIRST.
        2. Draw Road (with highlighted goal lines).
        3. Spawn Vehicles (Avoiding goal).
        """
        # --- 1. Select a Random Goal ---
        # Pick any slot from 1 to 8
        self.goal_slot_id = self.np_random.integers(1, 9)
        
        target_data = self.parking_slots[self.goal_slot_id]
        
        # Goal Vector: [x, y, heading]
        self.goal = np.array([
            target_data['position'][0],
            target_data['position'][1],
            target_data['heading']
        ], dtype=np.float32)

        # --- 2. Create Road (Draws lines based on goal_slot_id) ---
        self._create_road()

        # --- 3. Create Vehicles (Spawn cars, skipping the goal slot) ---
        self._create_vehicles()

    def reset(self, seed=None, options=None):
        """
        The main public reset method.
        Overridden to return our custom observation vector.
        """
        # 1. Call Parent Reset (This calls our _reset() internally)
        super().reset(seed=seed, options=options)
        
        # 2. Compute OUR custom observation
        obs = self._compute_observation()
        info = self._info(obs, None)
        
        return obs, info

    def step(self, action):
        """Execute one time step."""
        super().step(action)
        max_speed = 2.0
        self.vehicle.speed = np.clip(self.vehicle.speed, -max_speed, max_speed)
        # 1. Get Observation
        obs = self._compute_observation()
        
        # 2. Check Termination Conditions
        is_crashed = self.vehicle.crashed
        
        # Check Out of Bounds (Map limits)
        x, y = self.vehicle.position
        is_out_of_bounds = (x < -40 or x > 40 or y < -20 or y > 20)
        
        # --- SUCCESS CHECK: "Inside the Box" ---
        gx, gy = self.goal[0], self.goal[1]
        
        dx = abs(x - gx)
        dy = abs(y - gy)
        angle_error = abs(obs[2]) # rel_theta
        speed = abs(self.vehicle.speed)
        
        # Spot dimensions: 7.5m long, 3.5m wide.
        # We check if car is within the central area of the spot.
        in_x_bounds = (dx < 4.0) # Car within 6m length (safe margin)
        in_y_bounds = (dy < 1.5) # Car within 2m width (centered)
        is_parallel = (angle_error < 0.8) # Roughly parallel (<28 degrees)
        is_stopped = (speed < 0.5) # Almost stopped
        
        is_success = in_x_bounds and in_y_bounds and is_parallel and is_stopped

        terminated = is_crashed or is_out_of_bounds or is_success
        truncated = self.time >= self.config["duration"]

        # 3. Compute Reward
        # +1 for parking successfully, 0 otherwise
        reward = 10.0 if is_success else -0.001
        
        # Optional: -1 penalty for crashing (helps learning speed)
        if is_crashed or is_out_of_bounds:
            reward = -5.0

        # 4. Info
        info = self._info(obs, action)
        info['is_success'] = is_success # Critical for HER
        info['is_crash'] = (is_crashed or is_out_of_bounds)
        info['is_timeout'] = truncated

        return obs, reward, terminated, truncated, info

    def _compute_observation(self):
        """
        Calculates the Hybrid State Vector (Relative + Absolute).
        Returns a numpy array of shape (8,).
        """
        v = self.vehicle
        
        # Raw State
        raw_state = np.array([
            v.position[0], v.position[1], v.heading,
            v.speed, v.action['steering']
        ], dtype=np.float32)

        # Relative State Calculation (Goal - Car)
        dx = self.goal[0] - raw_state[0]
        dy = self.goal[1] - raw_state[1]
        
        cos_h = np.cos(raw_state[2])
        sin_h = np.sin(raw_state[2])
        
        # Rotate difference into car's perspective (Ego-centric)
        rel_x = dx * cos_h + dy * sin_h
        rel_y = -dx * sin_h + dy * cos_h
        rel_theta = (self.goal[2] - raw_state[2] + np.pi) % (2 * np.pi) - np.pi

        # Combine [Rel(3) + Abs(3) + Phys(2)]
        obs = np.array([
            rel_x, rel_y, rel_theta,
            raw_state[0] / 30.0, # Norm X
            raw_state[1] / 20.0, # Norm Y
            raw_state[2] / np.pi, # Norm Angle
            raw_state[3] / 10.0,  # Norm Speed
            raw_state[4]          # Steering
        ], dtype=np.float32)
        
        return obs
        
    def _create_road(self) -> None:
        """Create a road network with parking spots."""
        net = RoadNetwork()
        width = 20.0
        lt = (LineType.NONE, LineType.NONE)
        
        # Main Road
        net.add_lane("a", "b", StraightLane([-30, 0], [30, 0], width=width, line_types=lt))
        
        # Parking Spots (Pass the goal ID to highlight it)
        self._add_parking_lanes(net, highlight_slot=self.goal_slot_id)
        
        road = Road(network=net, np_random=self.np_random, 
                   record_history=self.config["show_trajectories"])
        self.road = road
    
    def _add_parking_lanes(self, net: RoadNetwork, highlight_slot: int = None) -> None:
        """
        Draws the parking spots.
        If a slot matches 'highlight_slot', we draw it with STRIPED lines to visualize the goal.
        """
        lt_normal = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        lt_goal = (LineType.STRIPED, LineType.STRIPED) # VISUAL GOAL INDICATOR
        
        for slot_num, slot_info in self.parking_slots.items():
            pos = slot_info['position']
            
            # Select line type: Is this the goal?
            current_lt = lt_goal if slot_num == highlight_slot else lt_normal
            
            left_x = pos[0] - self.spot_width / 2
            right_x = pos[0] + self.spot_width / 2
            
            if slot_info['row'] == 1:
                # Top Row
                top_y = pos[1] + self.spot_length / 2
                bottom_y = pos[1] - self.spot_length / 2
                net.add_lane(f"s{slot_num}_l", f"s{slot_num}_le", 
                             StraightLane([left_x, bottom_y], [left_x, top_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_t", f"s{slot_num}_te", 
                             StraightLane([left_x, top_y], [right_x, top_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_r", f"s{slot_num}_re", 
                             StraightLane([right_x, top_y], [right_x, bottom_y], width=0.1, line_types=current_lt))
            else:
                # Bottom Row
                top_y = pos[1] + self.spot_length / 2
                bottom_y = pos[1] - self.spot_length / 2
                net.add_lane(f"s{slot_num}_l", f"s{slot_num}_le", 
                             StraightLane([left_x, bottom_y], [left_x, top_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_b", f"s{slot_num}_be", 
                             StraightLane([left_x, bottom_y], [right_x, bottom_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_r", f"s{slot_num}_re", 
                             StraightLane([right_x, top_y], [right_x, bottom_y], width=0.1, line_types=current_lt))
    
    def _create_vehicles(self) -> None:
        spawn_side = self.np_random.choice(['left', 'right'])
        x_pos = -25 if spawn_side == 'left' else 25
        heading = 0 if spawn_side == 'left' else np.pi
        
        ego_vehicle = self.action_type.vehicle_class(self.road, [x_pos, 0], heading=heading, speed=0)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        
        self.agent_spawn_side = spawn_side
        self._spawn_random_parked_cars()
    
    def _spawn_random_parked_cars(self):
        """
        Spawns cars in ALL slots except the goal.
        This makes the environment harder (tight squeeze) but PREDICTABLE.
        """
        # Get all slots
        all_slots = list(range(1, 9))
        
        # Remove the goal slot so we don't spawn a car on top of the target
        if self.goal_slot_id in all_slots:
            all_slots.remove(self.goal_slot_id) 
            
        # Spawn a car in EVERY remaining slot
        for slot_num in all_slots:
            slot_info = self.parking_slots[slot_num]
            parked_vehicle = self.action_type.vehicle_class(
                self.road, slot_info['position'], heading=slot_info['heading'], speed=0
            )
            # Make them visible/colored
            try:
                if hasattr(parked_vehicle, 'color'): 
                    parked_vehicle.color = (100, 200, 255)
            except: pass
            
            self.road.vehicles.append(parked_vehicle)
            self.parking_slots[slot_num]['occupied'] = True
            
        self.num_parked_cars = len(all_slots)
    
    def _reward(self, action) -> float:
        return 0.0 
    
    def _is_terminated(self) -> bool:
        return False 
    
    def _is_truncated(self) -> bool:
        return False 
    
    def _info(self, obs, action) -> dict:
        info = {
            "parking_slots": self.parking_slots,
            "agent_spawn_side": getattr(self, 'agent_spawn_side', None),
            "num_parked_cars": getattr(self, 'num_parked_cars', 0),
            "goal_slot": getattr(self, 'goal_slot_id', None)
        }
        return info


# --- Main Block for Visual Testing ---
if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL PARKING ENV - FINAL VERSION")
    print("=" * 60)
    
    try:
        import pygame
        # Initialize env
        env_visual = ParallelParkingEnv(config={}, render_mode="human")
        
        # Test Reset
        obs, info = env_visual.reset()
        
        print(f"\n GOAL SLOT: {info['goal_slot']} (Look for STRIPED lines)")
        print(f" Observation Size: {obs.shape} (Should be 8)")
        print("\n Drive! Press ESC to exit...")
        
        running = True
        clock = pygame.time.Clock()
        
        print("\n Drive! Controls:")
        print(" [UP]    = Gas")
        print(" [DOWN]  = Reverse / Gentle Brake")
        print(" [LEFT]  = Steer Left")
        print(" [RIGHT] = Steer Right")
        print(" [SPACE] = HANDBRAKE (Stops the car)")
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # 1. Get Keyboard Input
            keys = pygame.key.get_pressed()
            
            steering = 0.0
            acceleration = 0.0
            
            # Steering
            if keys[pygame.K_LEFT]:
                steering = -0.5
            if keys[pygame.K_RIGHT]:
                steering = 0.5
                
            # Acceleration / Reverse
            if keys[pygame.K_UP]:
                acceleration = 0.5  # Gas
            elif keys[pygame.K_DOWN]:
                acceleration = -0.5 # Reverse / Gentle Brake
            
            # --- NEW: SMART HANDBRAKE (SPACEBAR) ---
            if keys[pygame.K_SPACE]:
                # Get current speed directly from the vehicle
                current_speed = env_visual.vehicle.speed
                
                # If moving forward, apply MAX negative force
                if current_speed > 0.05:
                    acceleration = -1.0
                # If moving backward, apply MAX positive force
                elif current_speed < -0.05:
                    acceleration = 1.0
                # If basically stopped, force zero
                else:
                    acceleration = 0.0
                    env_visual.vehicle.speed = 0 # Physics hack to snap to 0
            
            # 2. Send Action
            action = [steering, acceleration]
            
            # 3. Handle Window Events (Exit)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 4. Step Environment
            obs, reward, terminated, truncated, info = env_visual.step(action)
            env_visual.render()
            
            if terminated:
                status = "SUCCESS! PARKED!" if info['is_success'] else "CRASHED/FAILED"
                print(f"Episode Ends. Result: {status} | Reward: {reward}")
                obs, info = env_visual.reset()
                print(f"New Goal: {info['goal_slot']}")
            
            clock.tick(30)
        
        env_visual.close()
        print("\n Closed!")
        
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()