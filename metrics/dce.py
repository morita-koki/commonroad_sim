import numpy as np

from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle
import multiprocessing


class DCE:
    """
    Metric for Distance to Closest Encounter (DCE).
    """

    def __init__(self, config):
        self.config = config

    def evaluate(
        self,
        ego_predictions,
        agent_predictions,
    ):

        # initialize variables
        result = [
            [
                [{} for _ in range(len(agent_predictions))]
                for _ in range(len(agent_predictions[0]))
            ]
            for _ in range(len(ego_predictions))
        ]

        # ego_prediction: list [acc_index, 2, time_seq]
        # agent_prediction: list [acc_index, vel_index, 2, time_seq]
        # for i, ego_prediction in enumerate(ego_predictions):
        #     for acc_index in range(len(agent_predictions)):
        #         for vel_index in range(len(agent_predictions[acc_index])):
        #             dce, time_dce = self._calc_dce(
        #                 ego_prediction, agent_predictions[acc_index][vel_index]
        #             )
        #             result[i][acc_index][vel_index] = {
        #                 "dce": dce,
        #                 "time_dce": time_dce,
        #             }
        #             print(result[i][acc_index][vel_index], file=open("dce.txt", "a"))

        # def process_prediction(args):
        #     i, ego_prediction, acc_index, vel_index, agent_prediction = args
        #     dce, time_dce = _calc_dce(ego_prediction, agent_prediction)
        #     return i, acc_index, vel_index, {"dce": dce, "time_dce": time_dce}

        # Prepare arguments for multiprocessing
        args_list = []
        for i, ego_prediction in enumerate(ego_predictions):
            for acc_index in range(len(agent_predictions)):
                for vel_index in range(len(agent_predictions[acc_index])):
                    args_list.append(
                        (
                            i,
                            ego_prediction,
                            acc_index,
                            vel_index,
                            agent_predictions[acc_index][vel_index],
                        )
                    )

        # Use multiprocessing to process predictions
        with multiprocessing.Pool() as pool:
            results = pool.map(process_prediction, args_list)

        # Populate the result array
        for i, acc_index, vel_index, result_data in results:
            result[i][acc_index][vel_index] = result_data
            print(result_data, file=open("dce.txt", "a"))

        return result

    def _calc_dce(
        self,
        ego_prediction: TrajectoryPrediction,
        agent_prediction: TrajectoryPrediction,
    ) -> tuple[float, int]:

        return _calc_dce(ego_prediction, agent_prediction)


def process_prediction(args):
    i, ego_prediction, acc_index, vel_index, agent_prediction = args
    dce, time_dce = _calc_dce(ego_prediction, agent_prediction)
    return i, acc_index, vel_index, {"dce": dce, "time_dce": time_dce}


def _calc_dce(
    ego_prediction: TrajectoryPrediction,
    agent_prediction: TrajectoryPrediction,
) -> tuple[float, int]:

    # Cache attributes for quicker access

    # set dce to inf
    dce = np.inf
    time_dce = 0

    state_list = ego_prediction.trajectory.state_list

    # import pdb; pdb.set_trace()

    for i in range(len(state_list)):  # for each timestep
        if agent_prediction.trajectory.state_at_time_step(i) is None:
            continue
        ego_poly = Rectangle(
            center=ego_prediction.trajectory.state_at_time_step(i).position,
            width=2.0,
            length=4.0,
            orientation=ego_prediction.trajectory.state_at_time_step(i).orientation,
        ).shapely_object

        agent_poly = Rectangle(
            center=agent_prediction.trajectory.state_at_time_step(i).position,
            width=2.0,
            length=4.0,
            orientation=agent_prediction.trajectory.state_at_time_step(i).orientation,
        ).shapely_object

        distance = np.round(ego_poly.distance(agent_poly), 3)
        print(distance, file=open("dce.txt", "a"))

        if distance < dce:
            dce = distance
            time_dce = i
        # if dce == 0:
        #     break

    return dce, time_dce
