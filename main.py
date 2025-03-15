import os
import matplotlib as mpl
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer, MPDrawParams
from commonroad.common.file_reader import CommonRoadFileReader

from config import ConfigManager
from agent import create_agents, update_agents
from building import create_buildings
from sensor import Sensor


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
        sensor = Sensor(config.sensor)
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
    buildings = create_buildings(scenario.static_obstacles)

    config_basic = config.basic
    for timestep in tqdm(range(int(config_basic.duration / config_basic.dt))):
        update_agents(agents, timestep, dt=config_basic.dt)
        Visualizer.draw_step(renderer, scenario, agents, buildings, timestep, config)


if __name__ == "__main__":
    main()
