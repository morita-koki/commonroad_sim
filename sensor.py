import numpy as np
from functools import reduce
import shapely
from shapely.geometry import Point
from commonroad.geometry.shape import ShapeParams
from commonroad.geometry.shape import Polygon

from agent import Agent
from building import Building

import helper as hf

class Sensor:
    def __init__(self, config_sensor, lanelet_network):
        self.config_sensor = config_sensor
        # self.scenario = scenario
        self.distance_range = config_sensor.distance_range
        self.angle_range = config_sensor.angle_range

        self.lanelet_network = lanelet_network
        self.road_polygon, self.lanelet_polygons = self._convert_lanelet_network()

        self.obstacle_occlusions = []

    def _create_sector(self, center, angle_start, angle_end):
        points = [
            (
                center[0] + self.distance_range * np.cos(angle),
                center[1] + self.distance_range * np.sin(angle),
            )
            for angle in np.linspace(angle_start, angle_end, 100)
        ]
        sector = shapely.geometry.Polygon([center] + points + [center])
        return sector
    
    def _convert_lanelet_network(self):
        lanelet_polygons = [polygon.shapely_object for polygon in self.lanelet_network.lanelet_polygons]
        road_polygon = reduce(lambda x, y: x.union(y), lanelet_polygons)

        return road_polygon, lanelet_polygons
    
    def calc_occluded_lanelets(self, ego_pos, ego_orientation, dynamic_obstacles, static_obstacles):
        visible_area, obstacle_polygon = self.calc_visible_area(
            ego_pos, ego_orientation, dynamic_obstacles, static_obstacles
        )
        occluded_lanelet = obstacle_polygon.intersection(self.road_polygon)
        return occluded_lanelet


    def calc_visible_area(
        self,
        ego_pos,
        ego_orientation,
        dynamic_obstacles: list[Agent] = [],
        static_obstacles: list[Building] = [],
    ):
        angle_start = ego_orientation - np.radians(self.angle_range / 2)
        angle_end = ego_orientation + np.radians(self.angle_range / 2)
        visible_area = self._create_sector(ego_pos, angle_start, angle_end)

        obstacles_polygon = shapely.geometry.Polygon([])
        for obst in dynamic_obstacles + static_obstacles:
            # obstacle position is not empty, this happens if dynamic obstacle is not available at timestep
            # if (
            #     obst.current_pos is not None
            #     and obst.cr_obstacle.obstacle_type.value != "bicycle"
            # ):
            # check if within sensor radius or if obstacle intersects with visible area
            if obst.current_pos_point.within(
                visible_area
            ) or obst.current_polygon.intersects(visible_area):
                # calculate occlusion polygon that is caused by the obstacle
                occlusion, c1, c2 = hf.get_polygon_from_obstacle_occlusion(
                    ego_pos, obst.current_corner_points
                )
                # self.obstacle_occlusions[obst.cr_obstacle.obstacle_id] = (
                #     occlusion.difference(obst.current_polygon)
                # )

                # Subtract obstacle shape from visible area
                visible_area = visible_area.difference(
                    obst.current_polygon.buffer(0.005, join_style=2)
                )
                obstacles_polygon = obstacles_polygon.union(occlusion)
                obstacles_polygon = obstacles_polygon.union(obst.current_polygon)

                # Subtract occlusion caused by obstacle (everything behind obstacle) from visible area
                if occlusion.is_valid:
                    visible_area = visible_area.difference(occlusion)
                
                

        return visible_area, obstacles_polygon

    def _get_occlusion_polygon(self, ego_pos, obstacle):
        obstacle_coords = list(obstacle.exterior.coords)
        occlusion_coords = []

        for coord in obstacle_coords:
            direction = np.arctan2(coord[1] - ego_pos[1], coord[0] - ego_pos[0])
            occlusion_coords.append(
                (
                    coord[0] + self.distance_range * np.cos(direction),
                    coord[1] + self.distance_range * np.sin(direction),
                )
            )

        occlusion_polygon = shapely.geometry.Polygon(
            obstacle_coords + occlusion_coords[::-1]
        )
        return occlusion_polygon, obstacle_coords, occlusion_coords

    def calc_occluded_area(
        self, ego_pos, ego_orientation, dynamic_obstacles, static_obstacles
    ):
        visible_area = self.calc_visible_area(
            ego_pos, ego_orientation, dynamic_obstacles, static_obstacles
        )
        full_area = Point(ego_pos).buffer(self.distance_range)
        occluded_area = full_area.difference(visible_area)

        return occluded_area

    def draw(
        self,
        renderer,
        ego_pos,
        ego_orientation,
        dynamic_obstacles=[],
        static_obstacles=[],
    ):
        visible_area, obstacle_polygon = self.calc_visible_area(
            ego_pos, ego_orientation, dynamic_obstacles, static_obstacles
        )
        # occluded_area = self.calc_occluded_area(
        #     ego_pos, ego_orientation, dynamic_obstacles, static_obstacles
        # )

        # draw visible sensor area
        # print(visible_area.geom_type)
        # print(visible_area.exterior.xy)



        params = ShapeParams()
        params.facecolor = "g"
        params.edgecolor = "g"
        params.opacity = 0.2

        if visible_area is not None:
            if visible_area.geom_type == "MultiPolygon":
                for geom in visible_area.geoms:
                    vertices = np.array(geom.exterior.xy).T
                    Polygon(vertices).draw(renderer, params)
                    # renderer.ax.fill(*geom.exterior.xy, "g", alpha=0.2, zorder=10)
            elif visible_area.geom_type == "Polygon":
                vertices = np.array(visible_area.exterior.xy).T
                Polygon(vertices).draw(renderer, params)
                # renderer.ax.fill(*visible_area.exterior.xy, "g", alpha=0.2, zorder=10)
            else:
                for obj in visible_area.geoms:
                    if obj.geom_type == "Polygon":
                        vertices = np.array(visible_area.exterior.xy).T
                        Polygon(vertices).draw(renderer, params)
                        # renderer.ax.fill(*obj.exterior.xy, "g", alpha=0.2, zorder=10)


        # draw obstacles
        # params = ShapeParams()
        # params.facecolor = "r"
        # params.edgecolor = "r"
        # params.opacity = 0.2
        # if obstacle_polygon is not None:
        #         if obstacle_polygon.geom_type == "MultiPolygon":
        #             for geom in obstacle_polygon.geoms:
        #                 vertices = np.array(geom.exterior.xy).T
        #                 Polygon(vertices).draw(renderer, params)
        #                 # renderer.ax.fill(*geom.exterior.xy, "g", alpha=0.2, zorder=10)
        #         elif obstacle_polygon.geom_type == "Polygon":
        #             vertices = np.array(obstacle_polygon.exterior.xy).T
        #             Polygon(vertices).draw(renderer, params)
        #             # renderer.ax.fill(*obstacle_polygon.exterior.xy, "g", alpha=0.2, zorder=10)
        #         else:
        #             for obj in obstacle_polygon.geoms:
        #                 if obj.geom_type == "Polygon":
        #                     vertices = np.array(obstacle_polygon.exterior.xy).T
        #                     Polygon(vertices).draw(renderer, params)
        #                     # renderer.ax.fill(*obj.exterior.xy, "g", alpha=0.2, zorder=10) 


        # draw occluded lanelets
        params = ShapeParams()
        params.facecolor = "r"
        params.edgecolor = "r"
        params.opacity = 0.2
        obstacle_polygon = obstacle_polygon.intersection(self.road_polygon)
        if obstacle_polygon is not None:
                if obstacle_polygon.geom_type == "MultiPolygon":
                    for geom in obstacle_polygon.geoms:
                        vertices = np.array(geom.exterior.xy).T
                        Polygon(vertices).draw(renderer, params)
                        # renderer.ax.fill(*geom.exterior.xy, "g", alpha=0.2, zorder=10)
                elif obstacle_polygon.geom_type == "Polygon":
                    vertices = np.array(obstacle_polygon.exterior.xy).T
                    Polygon(vertices).draw(renderer, params)
                    # renderer.ax.fill(*obstacle_polygon.exterior.xy, "g", alpha=0.2, zorder=10)
                else:
                    pass
                    # for obj in obstacle_polygon.geoms:
                    #     if obj.geom_type == "Polygon":
                    #         vertices = np.array(obstacle_polygon.exterior.xy).T
                    #         Polygon(vertices).draw(renderer, params)
                            # renderer.ax.fill(*obj.exterior.xy, "g", alpha=0.2, zorder=10) 



