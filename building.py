from shapely.geometry import Point
from commonroad.geometry.shape import Rectangle

import helper as hf


class Building:
    def __init__(self, obstacle_id, obstacle):
        self.obstacle_id = obstacle_id
        self.obstacle = obstacle
        self.initial_state = obstacle.initial_state
        self.obstacle_shape = obstacle.obstacle_shape
    
    def draw(self, renderer):
        pass

    @property
    def current_pos(self):
        return self.obstacle.initial_state.position
    
    @property
    def current_pos_point(self):
        return Point(self.initial_state.position)
    
    @property
    def current_orientation(self):
        return self.obstacle.initial_state.orientation

    @property
    def current_polygon(self):
        return Rectangle(
            width=self.obstacle_shape.width,
            length=self.obstacle_shape.length,
            center=self.initial_state.position,
            orientation=self.initial_state.orientation,
        ).shapely_object
    
    @property
    def current_corner_points(self):
        return hf.calc_corner_points(
            self.initial_state.position,
            self.initial_state.orientation,
            self.obstacle_shape,
        )

def create_buildings(static_obstacles):
    buildings = []
    for i, obstacle in enumerate(static_obstacles):
        building = Building(2000 + i, obstacle)
        buildings.append(building)
    return buildings