import numpy as np
from shapely.geometry import Point
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.trajectory import CustomState, Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

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

    def create_hypothesis(self):

        acceleration_range = np.array(self.config_agent.acceleration_range)
        prediction_time = self.config_agent.prediction_time


        num_split = 20
        accelerations = np.linspace(acceleration_range[0], acceleration_range[1], num_split)
        time_seq = np.arange(0, prediction_time, self.config_agent.basic.dt)

        agent_orient = np.array(
            [np.cos(self.state.orientation), np.sin(self.state.orientation)]
        )[:, np.newaxis]

        agent_hypothesis = [None for _ in range(len(accelerations))]
        for j, acceleration in enumerate(accelerations):
            hypothesis = (
                self.state.velocity * time_seq + 0.5 * acceleration * time_seq**2
            )
            hypothesis[hypothesis < 0] = 0
            agent_hypothesis[j] = (
                self.state.position[:, np.newaxis] + hypothesis * agent_orient
            )
        agent_predictions = [None for _ in range(len(accelerations))]
        for j, acceleration in enumerate(accelerations):
            agent_predictions[j] = TrajectoryPrediction(
                shape=Rectangle(width=2.0, length=4.0),
                trajectory=Trajectory(
                    initial_time_step=self.state.time_step,
                    state_list=[
                        CustomState(
                            position=agent_hypothesis[j][:, k],
                            velocity=self.state.velocity,
                            orientation=self.state.orientation,
                            time_step=self.state.time_step + k,
                        )
                        for k in range(len(time_seq))
                    ],
                ),
            )

        return accelerations, agent_predictions

    def draw(self, renderer, shapeparams=None):
        shape = Rectangle(
            width=2.0,
            length=4.0,
            center=self.state.position,
            orientation=self.state.orientation,
        )
        shape.draw(renderer, shapeparams)


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
