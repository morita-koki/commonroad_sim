from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from commonroad.scenario.trajectory import State
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from metrics import DCE, TTC


class MotionChecker:

    def __init__(self, config):
        self.config = config
        if self.config.motion_checker.show:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax_ttc = self.fig.add_subplot(121)
            self.ax_dce = self.fig.add_subplot(122)
            self.ttc_cbar = None
            self.dce_cbar = None

    def evaluate(self, agent, phantom_agent, show=False):
        agent_accs, agent_predictions = agent.create_hypothesis()
        phantom_vels, phantom_accs, phantom_predictions = (
            phantom_agent.create_hypothesis()
        )

        dce = DCE(self.config)
        ttc = TTC(self.config)

        results = dce.evaluate(agent_predictions, phantom_predictions)
        results = ttc.evaluate(results)

        # save results to file
        save_dir = Path(self.config.basic.result_dir) / "metrics" / "tmp"
        save_dir.mkdir(parents=True, exist_ok=True)
        item = {
            "time_step": agent.state.time_step,
            "agent_accs": agent_accs,
            "agent_predictions": agent_predictions,
            "phantom_vels": phantom_vels,
            "phantom_accs": phantom_accs,
            "phantom_predictions": phantom_predictions,
            "results": results,
        }
        np.save(save_dir / f"results_{agent.state.time_step:03d}.npy", item)

        # test: pick one agent prediction and one phantom prediction
        if self.config.motion_checker.show:
            n = 5
            current_time = agent.state.time_step * self.config.basic.dt
            agent_prediction = agent_predictions[n]
            agent_acc = agent_accs[n]

            # plot ttc to one figure
            # x axis: phantom velocity
            # y axis: phantom acceleration
            # z axis: ttc
            ttc_values = np.zeros((len(phantom_vels), len(phantom_accs)))
            dce_values = np.zeros((len(phantom_vels), len(phantom_accs)))
            for i, vel in enumerate(phantom_vels):
                for j, acc in enumerate(phantom_accs):
                    ttc_values[i, j] = results[n][i][j]["ttc"]
                    dce_values[i, j] = results[n][i][j]["dce"]
                    # print(agent_acc)
                    # print(phantom_accs[j])
                    # print(phantom_vels[i])
                    # print(ttc_values[i, j])
            # print(ttc_values)

            # higher ttc is better, so we want to invert the blue color map
            cmap_ttc = plt.get_cmap("Blues")
            norm_ttc = mpl.colors.Normalize(
                vmin=0, vmax=self.config.phantom.prediction_time
            )
            # cmap_ttc = cmap_ttc.reversed()
            # detect the maximum ttc value
            # max_ttc = np.max(ttc_values)
            self.ax_ttc.clear()
            self.ax_ttc.imshow(ttc_values, cmap=cmap_ttc, norm=norm_ttc)
            self.ax_ttc.set_title("TTC")
            self.ax_ttc.set_xlabel("Phantom Acceleration [m/s^2]")
            self.ax_ttc.set_ylabel("Phantom Velocity [m/s]")
            self.ax_ttc.set_xticks(np.arange(len(phantom_accs)))
            self.ax_ttc.set_xticklabels(
                [f"{val:.2f}" for val in phantom_accs], rotation=45
            )
            self.ax_ttc.set_yticks(np.arange(len(phantom_vels)))
            self.ax_ttc.set_yticklabels([f"{val:.2f}" for val in phantom_vels])
            # if self.ttc_cbar is None:
            #     self.ttc_cbar = self.ax_ttc.figure.colorbar(ttc_im, ax=self.ax_dce)
            self.ax_ttc.set_title("Time to Collision")

            cmap_dce = plt.get_cmap("Blues")
            norm_dce = mpl.colors.Normalize(vmin=0, vmax=15.0)

            # cmap_dce = cmap_ttc.reversed()
            self.ax_dce.clear()
            self.ax_dce.imshow(dce_values, cmap=cmap_dce, norm=norm_dce)
            self.ax_dce.set_title("DCE")
            self.ax_dce.set_xlabel("Phantom Acceleration [m/s^2]")
            self.ax_dce.set_ylabel("Phantom Velocity [m/s]")
            self.ax_dce.set_xticks(np.arange(len(phantom_accs)))
            self.ax_dce.set_xticklabels(
                [f"{val:.2f}" for val in phantom_accs], rotation=45
            )
            self.ax_dce.set_yticks(np.arange(len(phantom_vels)))
            self.ax_dce.set_yticklabels([f"{val:.2f}" for val in phantom_vels])
            # if self.dce_cbar is None:
            #     self.dce_cbar = self.ax_dce.figure.colorbar(dce_im, ax=self.ax_dce)
            self.ax_dce.set_title("Distance to Closest Encounter")

            self.fig.suptitle(
                f"time {current_time: .1f} [s] \n Agent Velocity: {agent.state.velocity:.2f} [m/s] \n Agent Acceleration: {agent_acc:.2f} [m/s^2]"
            )
            self.fig.tight_layout()

            # show cmap colorbar
            # self.fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=self.ax_ttc)
            # self.fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=self.ax_dce)

        return results

    @staticmethod
    def make_gif(config, time_steps):
        dir = Path(config.basic.result_dir) / "metrics" / "tmp"
        items = []
        for timestep in time_steps:
            filename = dir / f"results_{timestep:03d}.npy"
            item = np.load(filename, allow_pickle=True).item()
            items.append(item)

        # create gif
        fig = plt.figure(figsize=(12, 8))
        ax_ttc = fig.add_subplot(121)
        ax_dce = fig.add_subplot(122)
        ttc_cbar = None
        dce_cbar = None
        converted = [[] for _ in range(len(items[0]["agent_accs"]))]

        for item in items:
            for i, acc in enumerate(item["agent_accs"]):
                converted[i].append(item["results"][i])
        # import pdb

        # pdb.set_trace()

        images = [[] for _ in range(len(items[0]["agent_accs"]))]
        time_steps = range(len(items))
        agent_accs = items[0]["agent_accs"]
        phantom_vels = items[0]["phantom_vels"]
        phantom_accs = items[0]["phantom_accs"]
        for i in range(len(converted)):  # agent_acc
            for j in range(len(converted[i])):  # time_step
                ttc_values = np.zeros((len(phantom_vels), len(phantom_accs)))
                dce_values = np.zeros((len(phantom_vels), len(phantom_accs)))
                for k, vel in enumerate(phantom_vels):
                    for l, acc in enumerate(phantom_accs):
                        ttc_values[k, l] = converted[i][j][k][l]["ttc"]
                        dce_values[k, l] = converted[i][j][k][l]["dce"]

                ax_ttc.clear()
                cmap_ttc = plt.get_cmap("Blues")
                norm_ttc = mpl.colors.Normalize(
                    vmin=0, vmax=config.phantom.prediction_time
                )
                ax_ttc.imshow(ttc_values, cmap=cmap_ttc, norm=norm_ttc)
                ax_ttc.set_title("TTC")
                ax_ttc.set_xlabel("Phantom Acceleration [m/s^2]")
                ax_ttc.set_ylabel("Phantom Velocity [m/s]")
                ax_ttc.set_xticks(np.arange(len(phantom_accs)))
                ax_ttc.set_xticklabels(
                    [f"{val:.2f}" for val in phantom_accs], rotation=45
                )
                ax_ttc.set_yticks(np.arange(len(phantom_vels)))
                ax_ttc.set_yticklabels([f"{val:.2f}" for val in phantom_vels])
                # if ttc_cbar is None:
                #     ttc_cbar = ax_ttc.figure.colorbar(ttc_im, ax=ax_dce)
                ax_ttc.set_title("Time to Collision")

                cmap_dce = plt.get_cmap("Blues")
                norm_dce = mpl.colors.Normalize(vmin=0, vmax=15.0)

                # cmap_dce = cmap_ttc.reversed()
                ax_dce.clear()
                ax_dce.imshow(dce_values, cmap=cmap_dce, norm=norm_dce)
                ax_dce.set_title("DCE")
                ax_dce.set_xlabel("Phantom Acceleration [m/s^2]")
                ax_dce.set_ylabel("Phantom Velocity [m/s]")
                ax_dce.set_xticks(np.arange(len(phantom_accs)))
                ax_dce.set_xticklabels(
                    [f"{val:.2f}" for val in phantom_accs], rotation=45
                )
                ax_dce.set_yticks(np.arange(len(phantom_vels)))
                ax_dce.set_yticklabels([f"{val:.2f}" for val in phantom_vels])
                # if dce_cbar is None:
                #     dce_cbar = ax_dce.figure.colorbar(dce_im, ax=ax_dce)
                ax_dce.set_title("Distance to Closest Encounter")

                fig.suptitle(
                    # f"time {j * config.basic.dt: .1f} [s] \n Agent Velocity: {agent_accs[i]:.2f} [m/s] \n Agent Acceleration: {agent_accs[i]:.2f} [m/s^2]"
                    f"time {j * config.basic.dt: .1f} [s] \n Agent Acceleration: {agent_accs[i]:.2f} [m/s^2]"
                )
                fig.tight_layout()
                fig.savefig(
                    dir / f"result_acc_{agent_accs[i]:.2f}_timestep_{j:03d}.png"
                )

                images[i].append(
                    Image.open(
                        dir / f"result_acc_{agent_accs[i]:.2f}_timestep_{j:03d}.png"
                    )
                )

        # save images as gif
        for i in range(len(images)):
            images[i][0].save(
                Path(config.basic.result_dir)
                / "metrics"
                / f"result_acc_{agent_accs[i]:.2f}.gif",
                save_all=True,
                append_images=images[i][1:],
                duration=100,
                loop=0,
                fps=3,
            )
