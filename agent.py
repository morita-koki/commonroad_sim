
import numpy as np
from shapely.geometry import Point
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.trajectory import CustomState

import helper as hf


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