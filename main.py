import os
import sys
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory, State, CustomState, InitialState
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.visualization.mp_renderer import MPRenderer, MPDrawParams
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle, Circle, Polygon, ShapeParams

import shapely
from shapely.geometry import Point
from shapely.geometry.multipolygon import MultiPolygon

import copy

from config import ConfigManager
import helper as hf


def set_non_blocking() -> None:
    """
    Ensures that interactive plotting is enabled for non-blocking plotting.

    :return: None
    """

    plt.ion()
    if not mpl.is_interactive():
        warnings.warn(
            "The current backend of matplotlib does not support "
            "interactive "
            "mode: " + str(mpl.get_backend()) + ". Select another backend with: "
            "\"matplotlib.use('TkAgg')\"",
            UserWarning,
            stacklevel=3,
        )


class StaticObstacle:
    pass


class Agent:
    def __init__(self, agent_id, scenario, config):
        self.agent_id = agent_id
        self.scenario = scenario
        self.config_agent = config

        self.state = CustomState(
            position=np.array(self.config_agent.initial_position),
            velocity=self.config_agent.initial_velocity,
            orientation=self.config_agent.initial_orientation / 180.0 * np.pi,
            acceleration=self.config_agent.initial_acceleration,
            yaw_rate=0.0,
            time_step=0,
        )

    @property
    def current_pos(self):
        return self.state.position

    @property
    def current_pos_point(self):
        return Point(self.state.position)

    @property
    def current_orientation(self):
        return self.state.orientation

    @property
    def current_polygon(self):
        return Rectangle(
            width=2.0,
            length=4.0,
            center=self.state.position,
            orientation=self.state.orientation,
        ).shapely_object

    @property
    def current_corner_points(self):
        return hf.calc_corner_points(
            self.state.position,
            self.state.orientation,
            Rectangle(
                width=self.config_agent.obstacle_shape[0],
                length=self.config_agent.obstacle_shape[1],
            ),
        )

    def draw(self, renderer):
        shape = Rectangle(
            width=2.0,
            length=4.0,
            center=self.state.position,
            orientation=self.state.orientation,
        )
        shape.draw(renderer)


def create_agents(scenario, config_agents: list):
    agents = []
    for i, agent_config in enumerate(config_agents):
        agent = Agent(1000 + i, scenario, agent_config)
        agents.append(agent)
    return agents


def update_agents(agents, timestep, dt):
    for agent in agents:
        agent.state.time_step = timestep
        agent.state.position = (
            agent.state.position
            + agent.state.velocity
            * np.array(
                [np.cos(agent.state.orientation), np.sin(agent.state.orientation)]
            )
            * dt
        )


class Sensor:
    def __init__(self, config_sensor):
        self.config_sensor = config_sensor
        # self.scenario = scenario
        self.distance_range = config_sensor.distance_range
        self.angle_range = config_sensor.angle_range

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

    def calc_visible_area(
        self,
        ego_pos,
        ego_orientation,
        dynamic_obstacles: list[Agent] = [],
        static_obstacles=[],
    ):
        angle_start = ego_orientation - np.radians(self.angle_range / 2)
        angle_end = ego_orientation + np.radians(self.angle_range / 2)
        visible_area = self._create_sector(ego_pos, angle_start, angle_end)

        # convert dynamic obstacles to shapely objects
        # dynamic_obstacles = [
        #     shapely.geometry.Polygon(
        #         [
        #             (obstacle.state.position[0] - 1, obstacle.state.position[1] - 1),
        #             (obstacle.state.position[0] + 1, obstacle.state.position[1] - 1),
        #             (obstacle.state.position[0] + 1, obstacle.state.position[1] + 1),
        #             (obstacle.state.position[0] - 1, obstacle.state.position[1] + 1),
        #         ]
        #     )
        #     for obstacle in dynamic_obstacles
        # ]

        # # convert static obstacles to shapely objects
        # static_obstacles = []

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
        visible_area, _ = self.calc_visible_area(
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

        # draw multipolygon objects
        # for polygon in visible_area:
        #     renderer.draw_polygon(polygon)
        # for polygon in occluded_area:
        #     renderer.draw_polygon(polygon)


class Visualizer:
    @staticmethod
    def draw_step(renderer, scenario, agents, timestep, config):
        mpparams = MPDrawParams()
        mpparams.dynamic_obstacle.time_begin = timestep
        mpparams.dynamic_obstacle.draw_icon = False
        # mpparams.dynamic_obstacle.draw_icon = config.visualization.draw_icons
        mpparams.dynamic_obstacle.show_label = False
        # mpparams.dynamic_obstacle.show_label = config.visualization.show_labels
        mpparams.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
        mpparams.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

        mpparams.static_obstacle.show_label = False
        # mpparams.static_obstacle.show_label = config.visualization.show_labels
        mpparams.static_obstacle.occupancy.shape.facecolor = "#A30000"
        mpparams.static_obstacle.occupancy.shape.edgecolor = "#756F61"

        scenario.draw(renderer, mpparams)
        for agent in agents:
            agent.draw(renderer)

        ego = agents[0]
        sensor = Sensor(config.sensor)
        sensor.draw(
            renderer,
            ego.state.position,
            ego.state.orientation,
            dynamic_obstacles=agents[1:],
            static_obstacles=[],
        )
        renderer.render(show=config.visualization.show)

        os.makedirs(f"{config.basic.result_dir}/images", exist_ok=True)
        plt.savefig(
            f"{config.basic.result_dir}/images/{scenario.scenario_id}_{timestep}.png"
        )

        if config.visualization.show:
            plt.pause(0.05)

    @staticmethod
    def make_gif(self, renderer, scenario, agents, time_steps):
        for timestep in time_steps:
            pass

        raise NotImplementedError


def main():
    config = ConfigManager.load_config("config.yaml")
    # print(config)
    if config.visualization.show:
        set_non_blocking()

    scenario, planning_problem_set = CommonRoadFileReader(
        "scenarios/ZAM_Occlusion-1_1_T-1.xml"
    ).open()

    renderer = MPRenderer()
    agents = create_agents(scenario, config.agents)

    config_basic = config.basic
    for timestep in tqdm(range(int(config_basic.duration / config_basic.dt))):
        update_agents(agents, timestep, dt=config_basic.dt)
        Visualizer.draw_step(renderer, scenario, agents, timestep, config)


if __name__ == "__main__":
    main()
