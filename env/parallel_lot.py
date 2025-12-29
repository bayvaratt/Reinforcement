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
    State Space: 10 Dimensions (3 Relative + 3 Absolute + 2 Physics + 2 Goal)
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
                "acceleration_range": [-1.0, 1.0],
                "steering_range": [-np.pi / 4, np.pi / 4],
                "longitudinal": True,
                "lateral": True
            },
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
        
        self.goal = np.array([0, 0, 0], dtype=np.float32)
        self.goal_slot_id = 1
        
        super().__init__(config, render_mode)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
    
    def _define_parking_slots(self):
        spacing = 7.5
        for i in range(4):
            slot_num = i + 1
            x_pos = -12 + i * spacing
            self.parking_slots[slot_num] = {
                'position': np.array([x_pos, 10]),
                'heading': np.pi,
                'occupied': False,
                'row': 1
            }
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
        self.goal_slot_id = self.np_random.integers(1, 9)
        target_data = self.parking_slots[self.goal_slot_id]
        
        self.goal = np.array([
            target_data['position'][0],
            target_data['position'][1],
            target_data['heading']
        ], dtype=np.float32)

        self._create_road()
        self._create_vehicles()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs = self._compute_observation()
        info = self._info(obs, None)
        return obs, info

    def step(self, action):
        super().step(action)
        max_speed = 2.0
        self.vehicle.speed = np.clip(self.vehicle.speed, -max_speed, max_speed)
        
        obs = self._compute_observation()
        is_crashed = self.vehicle.crashed
        
        x, y = self.vehicle.position
        is_out_of_bounds = (x < -40 or x > 40 or y < -20 or y > 20)
        
        gx, gy = self.goal[0], self.goal[1]
        dx = abs(x - gx)
        dy = abs(y - gy)
        
        # --- SUCCESS CHECK: CAPTURE THE FLAG ---
        # Matches Trainer logic exactly (Distance < 3.0)
        dist_to_goal = np.linalg.norm([dx, dy])
        is_success = (dist_to_goal < 3.0) 
        
        # REMOVED: Orientation check. 
        # We first want the agent to learn to GO to the spot.
        # Alignment is a Level 2 problem.
        
        terminated = is_crashed or is_out_of_bounds or is_success
        truncated = self.time >= self.config["duration"]

        # A) Base Penalty (Leaky Bucket)
        reward = -0.15
        
        # B) Distance Reward
        normalised_distance = min(dist_to_goal / 50.0, 1.0)
        reward += (1.0 - normalised_distance) * 0.1
        
        # C) Success / Crash Rewards
        if is_success:
            reward += 50.0
        
        if is_crashed or is_out_of_bounds:
            reward = -40.0

        info = self._info(obs, action)
        info['is_success'] = is_success 
        info['is_crash'] = (is_crashed or is_out_of_bounds)
        info['is_timeout'] = truncated

        return obs, reward, terminated, truncated, info

    def _compute_observation(self):
        v = self.vehicle
        raw_state = np.array([
            v.position[0], v.position[1], v.heading,
            v.speed, v.action['steering']
        ], dtype=np.float32)

        dx = self.goal[0] - raw_state[0]
        dy = self.goal[1] - raw_state[1]
        
        cos_h = np.cos(raw_state[2])
        sin_h = np.sin(raw_state[2])
        
        rel_x = dx * cos_h + dy * sin_h
        rel_y = -dx * sin_h + dy * cos_h
        rel_theta = (self.goal[2] - raw_state[2] + np.pi) % (2 * np.pi) - np.pi

        # --- NORMALISE OBSERVATIONS ---
        obs = np.array([
            rel_x / 30.0,          # Normalised Rel X
            rel_y / 30.0,          # Normalised Rel Y
            rel_theta / np.pi,     
            raw_state[0] / 30.0,   
            raw_state[1] / 20.0,   
            raw_state[2] / np.pi,  
            raw_state[3] / 10.0,   
            raw_state[4],          
            self.goal[0] / 30.0,   # Normalised Goal X
            self.goal[1] / 20.0    # Normalised Goal Y
        ], dtype=np.float32)
        
        return obs
        
    def _create_road(self) -> None:
        net = RoadNetwork()
        width = 20.0
        lt = (LineType.NONE, LineType.NONE)
        net.add_lane("a", "b", StraightLane([-30, 0], [30, 0], width=width, line_types=lt))
        self._add_parking_lanes(net, highlight_slot=self.goal_slot_id)
        road = Road(network=net, np_random=self.np_random, 
                   record_history=self.config["show_trajectories"])
        self.road = road
    
    def _add_parking_lanes(self, net: RoadNetwork, highlight_slot: int = None) -> None:
        lt_normal = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        lt_goal = (LineType.STRIPED, LineType.STRIPED) 
        
        for slot_num, slot_info in self.parking_slots.items():
            pos = slot_info['position']
            current_lt = lt_goal if slot_num == highlight_slot else lt_normal
            left_x = pos[0] - self.spot_width / 2
            right_x = pos[0] + self.spot_width / 2
            
            if slot_info['row'] == 1:
                top_y = pos[1] + self.spot_length / 2
                bottom_y = pos[1] - self.spot_length / 2
                net.add_lane(f"s{slot_num}_l", f"s{slot_num}_le", 
                             StraightLane([left_x, bottom_y], [left_x, top_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_t", f"s{slot_num}_te", 
                             StraightLane([left_x, top_y], [right_x, top_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_r", f"s{slot_num}_re", 
                             StraightLane([right_x, top_y], [right_x, bottom_y], width=0.1, line_types=current_lt))
            else:
                top_y = pos[1] + self.spot_length / 2
                bottom_y = pos[1] - self.spot_length / 2
                net.add_lane(f"s{slot_num}_l", f"s{slot_num}_le", 
                             StraightLane([left_x, bottom_y], [left_x, top_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_b", f"s{slot_num}_be", 
                             StraightLane([left_x, bottom_y], [right_x, bottom_y], width=0.1, line_types=current_lt))
                net.add_lane(f"s{slot_num}_r", f"s{slot_num}_re", 
                             StraightLane([right_x, top_y], [right_x, bottom_y], width=0.1, line_types=current_lt))
    
    def _create_vehicles(self) -> None:
        """
        Modified to spawn the agent LATERALLY closer to the goal.
        """
        dist_to_goal = self.np_random.uniform(8.0, 15.0)
        spawn_side = self.np_random.choice([-1, 1]) 
        
        goal_x = self.goal[0]
        goal_y = self.goal[1] # 10 or -10
        
        spawn_x = goal_x + (dist_to_goal * spawn_side)
        spawn_x = np.clip(spawn_x, -28, 28)
        
        # --- LATERAL CURRICULUM ---
        # Spawn somewhere between the road (y=0) and the spot (y=10 or -10)
        # This ensures the car is partially "in the lane" to start.
        if goal_y > 0:
             spawn_y = self.np_random.uniform(0, goal_y)
        else:
             spawn_y = self.np_random.uniform(goal_y, 0)
        
        heading = 0 if spawn_side == -1 else np.pi
        
        # Spawn at [spawn_x, spawn_y] instead of [spawn_x, 0]
        ego_vehicle = self.action_type.vehicle_class(self.road, [spawn_x, spawn_y], heading=heading, speed=0)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        
        self.agent_spawn_side = 'left' if spawn_side == -1 else 'right'
        self._spawn_random_parked_cars()
    
    def _spawn_random_parked_cars(self):
        """
        Spawns cars but LEAVES NEIGHBORS EMPTY.
        """
        all_slots = list(range(1, 9))
        blocked_slots = {self.goal_slot_id, self.goal_slot_id - 1, self.goal_slot_id + 1}
        
        slots_to_spawn = [s for s in all_slots if s not in blocked_slots]
            
        for slot_num in slots_to_spawn:
            slot_info = self.parking_slots[slot_num]
            parked_vehicle = self.action_type.vehicle_class(
                self.road, slot_info['position'], heading=slot_info['heading'], speed=0
            )
            try:
                if hasattr(parked_vehicle, 'color'): 
                    parked_vehicle.color = (100, 200, 255)
            except: pass
            
            self.road.vehicles.append(parked_vehicle)
            self.parking_slots[slot_num]['occupied'] = True
            
        self.num_parked_cars = len(slots_to_spawn)
    
    def _reward(self, action) -> float: return 0.0 
    def _is_terminated(self) -> bool: return False 
    def _is_truncated(self) -> bool: return False 
    def _info(self, obs, action) -> dict:
        return {
            "parking_slots": self.parking_slots,
            "agent_spawn_side": getattr(self, 'agent_spawn_side', None),
            "num_parked_cars": getattr(self, 'num_parked_cars', 0),
            "goal_slot": getattr(self, 'goal_slot_id', None)
        }

if __name__ == "__main__":
    env_visual = ParallelParkingEnv(config={}, render_mode="human")
    obs, info = env_visual.reset()
    env_visual.close()