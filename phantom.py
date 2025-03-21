import numpy as np
from shapely.geometry import Point
from commonroad.scenario.scenario import Scenario
from commonroad.geometry.shape import Rectangle, ShapeParams
from commonroad.scenario.trajectory import CustomState, State, TraceState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction


from sensor import Sensor
import helper as hf
from motion_checker import MotionChecker


class PhantomAgent:
    def __init__(self, agent_id, scenario, config):
        self.agent_id = agent_id
        self.scenario = scenario
        self.config_agent = config

        self.state = CustomState()

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
        velocity_range = np.array(self.config_agent.velocity_range)
        acceleration_range = np.array(self.config_agent.acceleration_range)
        prediction_time = self.config_agent.prediction_time

        velocities = np.linspace(velocity_range[0], velocity_range[1], 10)
        accelerations = np.linspace(acceleration_range[0], acceleration_range[1], 10)

        time_seq = np.arange(0, prediction_time, self.config_agent.basic.dt)
        phantom_hypotheses = np.zeros(
            (len(velocities), len(accelerations), 2, len(time_seq))
        )
        orient = np.array(
            [np.cos(self.state.orientation), np.sin(self.state.orientation)]
        )[:, np.newaxis]

        for i, velocity in enumerate(velocities):
            for j, acceleration in enumerate(accelerations):
                hypothesis = velocity * time_seq + 0.5 * acceleration * time_seq**2
                hypothesis[hypothesis < 0] = 0
                phantom_hypotheses[i, j] = (
                    self.state.position[:, np.newaxis] + hypothesis * orient
                )

        phantom_predictions = [
            [None for _ in range(len(accelerations))] for _ in range(len(velocities))
        ]

        for i, velocity in enumerate(velocities):
            for j, acceleration in enumerate(accelerations):
                phantom_predictions[i][j] = TrajectoryPrediction(
                    id=self.agent_id,
                    shape=Rectangle(width=2.0, length=4.0),
                    trajectory=Trajectory(
                        initial_time_step=self.state.time_step,
                        state_list=[
                            CustomState(
                                position=phantom_hypotheses[i, j, :, k],
                                velocity=velocity,
                                orientation=self.state.orientation,
                                time_step=self.state.time_step + k,
                            )
                            for k in range(len(time_seq))
                        ],
                    ),
                )
        return (
            velocities,
            accelerations,
            phantom_predictions,
        )  # [acc_index, vel_index] -> TrajectoryPrediction

    def draw(self, renderer):
        shape = Rectangle(
            width=2.0,
            length=4.0,
            center=self.state.position,
            orientation=self.state.orientation,
        )
        params = ShapeParams()
        params.facecolor = "#E37222"
        params.edgecolor = "#003359"
        shape.occupancy = 0.5
        shape.draw(renderer, params)


def create_phantoms(scenario, config_phantom, ego_state, static_obstacles):
    occluded_lanelets_polygon = Sensor(
        config_phantom.sensor, scenario.lanelet_network
    ).calc_occluded_lanelets(
        ego_state.position,
        ego_state.orientation,
        dynamic_obstacles=[],
        static_obstacles=static_obstacles,
    )

    # phantomなlaneletを取得
    phantom_lanelets = []
    for lanelet in scenario.lanelet_network.lanelets:
        if (
            lanelet.polygon.shapely_object.intersection(occluded_lanelets_polygon).area
            > 0
        ):
            phantom_lanelets.append(lanelet)

    visible_area, _ = Sensor(
        config_phantom.sensor, scenario.lanelet_network
    ).calc_visible_area(
        ego_state.position,
        ego_state.orientation,
        dynamic_obstacles=[],
        static_obstacles=static_obstacles,
    )

    # 各phantom laneletに対して, 0.5m 間隔でPhantomAgentを生成
    phantom_agents = []
    for lanelet in phantom_lanelets:
        center_vertices = lanelet.center_vertices  # np.array([[x1, y1], [x2, y2], ...])
        # # 0.5 m 間隔となるようにverticesを線形補間
        # center_vertices = np.array([center_vertices[i] + (center_vertices[i+1] - center_vertices[i]) / 2 for i in range(len(center_vertices) - 1)])
        # lanelet_length = lanelet.center_line.length
        # num_phantom = int(lanelet_length / 0.5)
        for i, p in enumerate(center_vertices):
            # position = lanelet.center_line.interpolate(v).coords[0]
            phantom_agent = PhantomAgent(3000 + i, scenario, config_phantom)
            # print(p)
            phantom_agent.state = CustomState(
                position=np.array(p),
                velocity=0.0,
                orientation=lanelet.orientation_by_position(p),
                acceleration=0.0,
                yaw_rate=0.0,
                time_step=0,
            )

            # check if the phantom agent is not in the visible area (intersect and its area is not 0)
            if visible_area.intersection(phantom_agent.current_polygon).area > 0:
                continue

            phantom_agents.append(phantom_agent)

            # only pick phantom agents that are oriented to the the intersection
            # get intersection point in the lanelet

    return phantom_agents


def update_phantoms(ego, agents, phantoms, timestep, config):
    for phantom in phantoms:
        phantom.state.time_step = timestep

    return phantoms
