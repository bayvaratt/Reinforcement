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
    Layout:
    [1][2][3][4]
    
    [5][6][7][8]
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
                "acceleration_range": [-3, 3],
                "steering_range": [-np.pi / 4, np.pi / 4],
                "longitudinal": True,
                "lateral": True
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
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
        ''' ***Swap width and length due to rotation of the parking lot*** '''
        self.spot_width = 7.5  # Length of parking spot 
        self.spot_length = 3.5  # Width of parking spot
        self.parking_slots = {}
        self._define_parking_slots()
        super().__init__(config, render_mode)
    
    def _define_parking_slots(self):
        spacing = 7.5  # Space between slots
        
        # Row 1: Slots 1-4 (top row, facing LEFT)
        row1_y = 10
        start_x_row1 = -12
        for i in range(4):
            slot_num = i + 1
            x_pos = start_x_row1 + i * spacing
            self.parking_slots[slot_num] = {
                'position': np.array([x_pos, row1_y]),
                'heading': np.pi,
                'occupied': False,
                'vehicle': None,
                'row': 1
            }
        
        # Row 2: Slots 5-8 (bottom row, facing RIGHT)
        row2_y = -10
        start_x_row2 = -12
        for i in range(4):
            slot_num = i + 5
            x_pos = start_x_row2 + i * spacing
            self.parking_slots[slot_num] = {
                'position': np.array([x_pos, row2_y]),
                'heading': 0,  # Facing right (0 degrees)
                'occupied': False,
                'vehicle': None,
                'row': 2
            }
    
    ''' Do not remove '''
    def _create_road(self) -> None:
        """Create a road network with parking spots."""
        net = RoadNetwork()
        
        width = 20.0
        lt = (LineType.NONE, LineType.NONE)
        net.add_lane("a", "b", StraightLane([-30, 0], [30, 0], width=width, line_types=lt))
        
        self._add_parking_lanes(net)
        
        road = Road(network=net, np_random=self.np_random, 
                   record_history=self.config["show_trajectories"])
        self.road = road
    
    def _add_parking_lanes(self, net: RoadNetwork) -> None:
        lt_spots = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        
        for slot_num, slot_info in self.parking_slots.items():
            pos = slot_info['position']
            
            # For parallel parking: spots are horizontal rectangles
            # Left and right edges (vertical lines)
            left_x = pos[0] - self.spot_width / 2
            right_x = pos[0] + self.spot_width / 2
            
            if slot_info['row'] == 1:
                # Top row - opening faces down
                top_y = pos[1] + self.spot_length / 2
                bottom_y = pos[1] - self.spot_length / 2
                
                # Left vertical line
                net.add_lane(f"spot{slot_num}_left", f"spot{slot_num}_left_end",
                           StraightLane([left_x, bottom_y], [left_x, top_y],
                                      width=0.1, line_types=lt_spots, forbidden=False))
                # Top horizontal line
                net.add_lane(f"spot{slot_num}_top", f"spot{slot_num}_top_end",
                           StraightLane([left_x, top_y], [right_x, top_y],
                                      width=0.1, line_types=lt_spots, forbidden=False))
                # Right vertical line
                net.add_lane(f"spot{slot_num}_right", f"spot{slot_num}_right_end",
                           StraightLane([right_x, top_y], [right_x, bottom_y],
                                      width=0.1, line_types=lt_spots, forbidden=False))
            else:
                # Bottom row - opening faces up
                top_y = pos[1] + self.spot_length / 2
                bottom_y = pos[1] - self.spot_length / 2
                
                # Left vertical line
                net.add_lane(f"spot{slot_num}_left", f"spot{slot_num}_left_end",
                           StraightLane([left_x, bottom_y], [left_x, top_y],
                                      width=0.1, line_types=lt_spots, forbidden=False))
                # Bottom horizontal line
                net.add_lane(f"spot{slot_num}_bottom", f"spot{slot_num}_bottom_end",
                           StraightLane([left_x, bottom_y], [right_x, bottom_y],
                                      width=0.1, line_types=lt_spots, forbidden=False))
                # Right vertical line
                net.add_lane(f"spot{slot_num}_right", f"spot{slot_num}_right_end",
                           StraightLane([right_x, top_y], [right_x, bottom_y],
                                      width=0.1, line_types=lt_spots, forbidden=False))
    
    def _create_vehicles(self) -> None:
        ''' agent car btw, train this '''
        spawn_side = self.np_random.choice(['left', 'right'])
        
        if spawn_side == 'left':
            x_pos = -25
            heading = 0
        else:
            x_pos = 25
            heading = np.pi
        
        ego_vehicle = self.action_type.vehicle_class(
            self.road, [x_pos, 0], heading=heading, speed=0
        )
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        
        self.agent_spawn_side = spawn_side
        self.agent_spawn_position = [x_pos, 0]
        self.agent_spawn_heading = heading
        
        self._spawn_random_parked_cars()
    
    def _spawn_random_parked_cars(self):
        ''' static random car '''
        num_parked_cars = self.np_random.integers(1, 8)  # 1-7 cars (leave at least 1 free)
        all_slots = list(range(1, 9))
        occupied_slots = self.np_random.choice(all_slots, size=num_parked_cars, replace=False)
        
        for slot_num in occupied_slots:
            slot_info = self.parking_slots[slot_num]
            
            parked_vehicle = self.action_type.vehicle_class(
                self.road, slot_info['position'], 
                heading=slot_info['heading'], speed=0
            )
            
            try:
                if hasattr(parked_vehicle, 'color'):
                    parked_vehicle.color = (100, 200, 255)
                elif hasattr(parked_vehicle, 'COLOR'):
                    parked_vehicle.COLOR = (100, 200, 255)
            except:
                pass
            
            self.road.vehicles.append(parked_vehicle)
            self.parking_slots[slot_num]['occupied'] = True
            self.parking_slots[slot_num]['vehicle'] = parked_vehicle
        
        self.num_parked_cars = num_parked_cars
        self.occupied_slot_numbers = sorted(occupied_slots.tolist())
    
    def _reset(self) -> None:
        """Reset the environment."""
        self._create_road()
        self._create_vehicles()
    
    def _reward(self, action) -> float:
        """Calculate reward."""
        return 0.0
    
    def _is_terminated(self) -> bool:
        '''
        Placeholder but can be use for "Car successfully parked", "Car crashed" or "Car out of bound"
        '''
        return False
    
    def _is_truncated(self) -> bool:
        '''
        Time Out -- can set the time for agent. If taken too long, the map reset (reset the whole thing)
        ctrl+f, search duration to change the session time out.
        '''
        return self.time >= self.config["duration"]
    
    def _info(self, obs, action) -> dict:
        """Additional information about the environment state."""
        info = {
            "parking_slots": self.parking_slots,
            "available_slots": [k for k, v in self.parking_slots.items() if not v['occupied']],
            "agent_spawn_side": getattr(self, 'agent_spawn_side', None),
            "agent_spawn_position": getattr(self, 'agent_spawn_position', None),
            "agent_spawn_heading": getattr(self, 'agent_spawn_heading', None),
            "num_parked_cars": getattr(self, 'num_parked_cars', 0),
            "occupied_slots": getattr(self, 'occupied_slot_numbers', [])
        }
        return info
    
    def get_slot_info(self, slot_number: int) -> dict:
        """Get information about a specific parking slot."""
        if slot_number not in self.parking_slots:
            raise ValueError(f"Invalid slot number. Must be between 1 and 8.")
        return self.parking_slots[slot_number]
    
    def print_slot_layout(self):
        """Print the current parking slot layout."""
        print("\nParking Lot Layout - PARALLEL:")
        print("=" * 50)
        
        row1 = []
        for i in range(1, 5):
            status = "X" if self.parking_slots[i]['occupied'] else str(i)
            row1.append(f"[{status}]")
        print("  ".join(row1))
        
        print()
        
        row2 = []
        for i in range(5, 9):
            status = "X" if self.parking_slots[i]['occupied'] else str(i)
            row2.append(f"[{status}]")
        print("  ".join(row2))
        
        print("=" * 50)
        print("Legend: [1-8] = Available, [X] = Occupied\n")


if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL PARKING ENVIRONMENT")
    print("=" * 60)
    
    env = ParallelParkingEnv(config={})
    print(f"\nTotal parking slots: {len(env.parking_slots)}")
    env.unwrapped.print_slot_layout()
    
    obs, info = env.reset()
    
    print("\n Environment created!")
    print(" Yellow car = Agent")
    print("  White lines = Parking spots")
    
    if info.get('agent_spawn_side'):
        print(f" Agent spawned: {info['agent_spawn_side'].upper()}")
    
    if info.get('num_parked_cars'):
        print(f" Parked cars: {info['num_parked_cars']}")
        print(f"   Occupied: {info.get('occupied_slots')}")
        print(f"   Available: {info.get('available_slots')}")
    
    env.unwrapped.print_slot_layout()
    
    print("\n" + "=" * 60)
    print("DRIVE THE CAR")
    print("=" * 60)
    print("Controls: LEFT/RIGHT=Steer, UP/DOWN=Accelerate/Brake")
    print("Press ESC to exit")
    print("=" * 60)
    
    try:
        import pygame
        env_visual = ParallelParkingEnv(config={}, render_mode="human")
        obs, info = env_visual.reset()
        
        print("\n Drive! Press ESC to exit...")
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            keys = pygame.key.get_pressed()
            
            steering = 0.0
            acceleration = 0.0
            
            if keys[pygame.K_LEFT]:
                steering = -0.5
            if keys[pygame.K_RIGHT]:
                steering = 0.5
            if keys[pygame.K_UP]:
                acceleration = 0.3
            if keys[pygame.K_DOWN]:
                acceleration = -0.3
            
            action = [steering, acceleration]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            obs, reward, terminated, truncated, info = env_visual.step(action)
            env_visual.render()
            
            if terminated or truncated:
                obs, info = env_visual.reset()
            
            clock.tick(30)
        
        env_visual.close()
        print("\n Closed!")
        
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()