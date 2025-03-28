import os
import matplotlib as mpl
from tqdm import tqdm
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer, MPDrawParams
from commonroad.geometry.shape import ShapeParams
from commonroad.common.file_reader import CommonRoadFileReader

from config import ConfigManager
from agent import create_agents, update_agents
from building import create_buildings
from phantom import create_phantoms, update_phantoms
from sensor import Sensor
from motion_checker import MotionChecker


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


class Visualizer:
    @staticmethod
    def draw_step(renderer, scenario, agents, buildings, timestep, config):
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
        sensor = Sensor(config.sensor, scenario.lanelet_network)
        sensor.draw(
            renderer,
            ego.state.position,
            ego.state.orientation,
            dynamic_obstacles=agents[1:],
            static_obstacles=buildings,
        )
        renderer.render(show=config.visualization.show)

        os.makedirs(f"{config.basic.result_dir}/images", exist_ok=True)
        plt.savefig(
            f"{config.basic.result_dir}/images/{scenario.scenario_id}_{timestep}.png"
        )

        if config.visualization.show:
            plt.pause(0.05)

    @staticmethod
    def draw_step_with_phantoms(
        renderer, scenario, agents, buildings, phantoms, timestep, config
    ):
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
        ego_shape_params = ShapeParams()
        # edge is blue and face is sky blue
        ego_shape_params.edgecolor = "#003359"
        ego_shape_params.facecolor = "#87CEEB"
        agents[0].draw(renderer, ego_shape_params)
        for agent in agents[1:]:
            agent.draw(renderer)

        for phantom in phantoms:
            phantom.draw(renderer)

        ego = agents[0]
        sensor = Sensor(config.sensor, scenario.lanelet_network)
        sensor.draw(
            renderer,
            ego.state.position,
            ego.state.orientation,
            dynamic_obstacles=agents[1:],
            static_obstacles=buildings,
        )
        renderer.render(show=config.visualization.show)

        os.makedirs(f"{config.basic.result_dir}/images", exist_ok=True)
        plt.savefig(
            # f"{config.basic.result_dir}/images/{scenario.scenario_id}_{timestep}.png"
            f"{config.basic.result_dir}/images/{timestep}.png"
        )

        if config.visualization.show:
            plt.pause(0.05)

    @staticmethod
    def make_gif(config, time_steps):
        images = []
        for timestep in time_steps:
            filename = f"{config.basic.result_dir}/images/{timestep}.png"
            # load image with PIL
            img = Image.open(filename)
            # append image to list
            images.append(img)
            img.close()
        # save images as gif
        images[0].save(
            f"{config.basic.result_dir}/{config.basic.result_dir}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
            fps=3,
        )

        # raise NotImplementedError


def main():
    config = ConfigManager.load_config("config.yaml")
    # print(config)
    if config.visualization.show:
        set_non_blocking()

    scenario, planning_problem_set = CommonRoadFileReader(
        "scenarios/ZAM_Occlusion-1_1_T-1.xml"
    ).open()

    scenario.lanelet_network

    renderer = MPRenderer()
    agents = create_agents(scenario, config.agents)
    buildings = create_buildings(scenario.static_obstacles)

    phantoms = create_phantoms(scenario, config.phantom, agents[0].state, buildings)
    # sort by distance to ego
    # print(agents[0].state.position)
    phantoms = sorted(
        phantoms,
        key=lambda x: np.linalg.norm(x.state.position - agents[0].state.position),
    )
    # phantoms = phantoms[:2]
    phantoms = phantoms[2:3]
    motion_checker = MotionChecker(config)
    config_basic = config.basic
    for timestep in tqdm(range(int(config_basic.duration / config_basic.dt))):
        update_agents(agents, timestep, dt=config_basic.dt)
        update_phantoms(agents[0], agents[1:], phantoms, timestep, config)
        motion_checker.evaluate(agents[1], phantoms[0])
        # Visualizer.draw_step(renderer, scenario, agents, buildings, timestep, config)
        Visualizer.draw_step_with_phantoms(
            renderer, scenario, agents, buildings, phantoms, timestep, config
        )

    # plt.ion off
    plt.ioff()
    Visualizer.make_gif(config, range(int(config_basic.duration / config_basic.dt)))
    MotionChecker.make_gif(config, range(int(config_basic.duration / config_basic.dt)))


if __name__ == "__main__":
    main()
