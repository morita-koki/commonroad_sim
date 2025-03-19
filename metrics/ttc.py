import numpy as np


class TTC:

    def __init__(self, config):
        self.config = config

    def evaluate(self, results):
        """
        Calculate the Time to Collision (TTC) metric."
        params:
            results: list: (ego_acc_index, agent_acc_index, agent_vel_index) -> {"dce": float, "time_dce": int}
        """

        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(len(results[i][j])):
                    result = results[i][j][k]

                    if np.isclose(result["dce"], 0.0):
                        result["ttc"] = np.round(
                            result["time_dce"] * self.config.basic.dt, 3
                        )
                    else:
                        result["ttc"] = np.inf

        # def process_result(result):
        #     if np.isclose(result["dce"], 0.0):
        #         result["ttc"] = np.round(result["time_dce"] * self.config.basic.dt, 3)
        #     else:
        #         result["ttc"] = np.inf
        #     return result

        # def process_sublist(sublist):
        #     return [process_result(result) for result in sublist]

        # with multiprocessing.Pool() as pool:
        #     for i in range(len(results)):
        #         results[i] = pool.map(process_sublist, results[i])

        return results
