import numpy as np


class TTCE:
    def __init__(self, config):
        self.config = config

    def evaluate(self, results) -> dict:

        # check if required dce is available
        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(len(results[i][j])):
                    result = results[i][j][k]

                    result["ttce"] = np.round(result["time_dce"] * self.config.basic.dt, 3)


        return results
        