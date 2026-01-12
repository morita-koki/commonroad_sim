"""
Main script for CommonRoad simulation with multiple phantom vehicles
at specified distances from the collision region.

Phantom vehicles are placed at distances of 10, 20, 30, 40 meters
from the collision region (x=43.5).
"""

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
from commonroad.scenario.trajectory import CustomState

from config import ConfigManager
from agent import create_agents, update_agents
from building import create_buildings
from phantom import PhantomAgent, update_phantoms
from sensor import Sensor
from motion_checker import MotionChecker


# Collision region x-coordinate
COLLISION_REGION_X = 43.5

# Distances from collision region to place phantom vehicles [m]
PHANTOM_DISTANCES = [10, 20, 30, 40]


def set_non_blocking() -> None:
    """
    Ensures that interactive plotting is enabled for non-blocking plotting.
    """
    plt.ion()
    if not mpl.is_interactive():
        warnings.warn(
            "The current backend of matplotlib does not support "
            "interactive mode: " + str(mpl.get_backend()) + ". Select another backend with: "
            "\"matplotlib.use('TkAgg')\"",
            UserWarning,
            stacklevel=3,
        )


def create_phantom_at_distance(
    scenario,
    config_phantom,
    ego_state,
    static_obstacles,
    distance_to_collision: float,
    phantom_id: int = 3000,
) -> PhantomAgent:
    """
    Create a phantom vehicle at a specified distance from the collision region.

    Args:
        scenario: CommonRoad scenario object
        config_phantom: Phantom configuration
        ego_state: Ego vehicle state
        static_obstacles: List of static obstacles (buildings)
        distance_to_collision: Distance from collision region [m]
        phantom_id: ID for the phantom agent

    Returns:
        PhantomAgent at the specified position
    """
    # Calculate phantom position
    # Phantom is placed on the horizontal lane approaching x=43.5
    # Distance is measured from x=43.5, so phantom_x = 43.5 - distance
    phantom_x = COLLISION_REGION_X - distance_to_collision
    phantom_y = 0.0  # On the horizontal lane at y=0

    # Get the lanelet at this position to determine correct orientation
    lanelets = scenario.lanelet_network.lanelets
    target_lanelet = None
    for lanelet in lanelets:
        if lanelet.polygon.shapely_object.contains(
            __import__("shapely.geometry", fromlist=["Point"]).Point(phantom_x, phantom_y)
        ):
            target_lanelet = lanelet
            break

    # Determine orientation (moving towards collision region, i.e., +x direction)
    if target_lanelet is not None:
        orientation = target_lanelet.orientation_by_position(np.array([phantom_x, phantom_y]))
    else:
        # Default: moving in +x direction (0 radians)
        orientation = 0.0

    # Create phantom agent
    phantom_agent = PhantomAgent(phantom_id, scenario, config_phantom)
    phantom_agent.state = CustomState(
        position=np.array([phantom_x, phantom_y]),
        velocity=0.0,
        orientation=orientation,
        acceleration=0.0,
        yaw_rate=0.0,
        time_step=0,
    )

    return phantom_agent


def create_phantoms_at_distances(
    scenario,
    config_phantom,
    ego_state,
    static_obstacles,
    distances: list,
) -> list:
    """
    Create multiple phantom vehicles at specified distances from collision region.

    Args:
        scenario: CommonRoad scenario object
        config_phantom: Phantom configuration
        ego_state: Ego vehicle state
        static_obstacles: List of static obstacles
        distances: List of distances from collision region [m]

    Returns:
        List of PhantomAgent objects
    """
    phantoms = []
    for i, distance in enumerate(distances):
        phantom = create_phantom_at_distance(
            scenario=scenario,
            config_phantom=config_phantom,
            ego_state=ego_state,
            static_obstacles=static_obstacles,
            distance_to_collision=distance,
            phantom_id=3000 + i,
        )
        phantoms.append(phantom)
    return phantoms


class Visualizer:
    @staticmethod
    def draw_step_with_phantoms(
        renderer,
        scenario,
        agents,
        buildings,
        phantoms,
        timestep,
        config,
        phantom_distances=None,
        current_phantom_idx=None,
    ):
        """Draw a single step with phantom vehicles highlighted."""
        mpparams = MPDrawParams()
        mpparams.dynamic_obstacle.time_begin = timestep
        mpparams.dynamic_obstacle.draw_icon = False
        mpparams.dynamic_obstacle.show_label = False
        mpparams.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
        mpparams.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

        mpparams.static_obstacle.show_label = False
        mpparams.static_obstacle.occupancy.shape.facecolor = "#A30000"
        mpparams.static_obstacle.occupancy.shape.edgecolor = "#756F61"

        scenario.draw(renderer, mpparams)

        # Draw ego vehicle with distinct color
        ego_shape_params = ShapeParams()
        ego_shape_params.edgecolor = "#003359"
        ego_shape_params.facecolor = "#87CEEB"  # Sky blue
        agents[0].draw(renderer, ego_shape_params)

        # Draw other agents
        for agent in agents[1:]:
            agent.draw(renderer)

        # Draw all phantoms with different colors based on distance
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]  # Red, Teal, Blue, Green
        for i, phantom in enumerate(phantoms):
            phantom_params = ShapeParams()
            color_idx = i % len(colors)
            phantom_params.facecolor = colors[color_idx]
            phantom_params.edgecolor = "#333333"

            # Highlight current phantom being evaluated
            if current_phantom_idx is not None and i == current_phantom_idx:
                phantom_params.facecolor = "#FFD700"  # Gold for current
                phantom_params.edgecolor = "#FF4500"  # Orange-red edge

            from commonroad.geometry.shape import Rectangle
            shape = Rectangle(
                width=2.0,
                length=4.0,
                center=phantom.state.position,
                orientation=phantom.state.orientation,
            )
            shape.draw(renderer, phantom_params)

        # Draw sensor
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

        # Add title with phantom distance information
        if phantom_distances is not None and current_phantom_idx is not None:
            current_distance = phantom_distances[current_phantom_idx]
            plt.title(
                f"Timestep: {timestep}, Phantom Distance to Collision: {current_distance} m",
                fontsize=12
            )

        plt.xlim(15, 75)
        plt.ylim(-30, 30)

        if config.visualization.show:
            plt.pause(0.05)

    @staticmethod
    def save_frame(config, timestep, phantom_distance):
        """Save current frame with phantom distance in filename."""
        os.makedirs(f"{config.basic.result_dir}/images/dist_{phantom_distance}", exist_ok=True)
        plt.savefig(
            f"{config.basic.result_dir}/images/dist_{phantom_distance}/{timestep:03d}.png"
        )

    @staticmethod
    def make_gif(config, time_steps, phantom_distance):
        """Create GIF for a specific phantom distance."""
        images = []
        for timestep in time_steps:
            filename = f"{config.basic.result_dir}/images/dist_{phantom_distance}/{timestep:03d}.png"
            if os.path.exists(filename):
                img = Image.open(filename)
                images.append(img)

        if images:
            output_dir = f"{config.basic.result_dir}/gifs"
            os.makedirs(output_dir, exist_ok=True)
            images[0].save(
                f"{output_dir}/phantom_dist_{phantom_distance}.gif",
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )
            print(f"Saved GIF: {output_dir}/phantom_dist_{phantom_distance}.gif")


class MotionCheckerMultiPhantom(MotionChecker):
    """Extended MotionChecker that saves results with phantom distance info."""

    def __init__(self, config):
        super().__init__(config)

    def evaluate_with_distance(self, agent, phantom_agent, phantom_distance, show=False):
        """
        Evaluate motion metrics and save with phantom distance information.

        Args:
            agent: Observable agent
            phantom_agent: Phantom agent
            phantom_distance: Distance of phantom from collision region [m]
            show: Whether to show visualization
        """
        agent_accs, agent_predictions = agent.create_hypothesis()
        phantom_vels, phantom_accs, phantom_predictions = (
            phantom_agent.create_hypothesis()
        )

        from metrics import DCE, TTC, TTCE
        dce = DCE(self.config)
        ttc = TTC(self.config)
        ttce = TTCE(self.config)

        results = dce.evaluate(agent_predictions, phantom_predictions)
        results = ttc.evaluate(results)
        results = ttce.evaluate(results)

        # Save results to file with phantom distance info
        from pathlib import Path
        save_dir = Path(self.config.basic.result_dir) / "metrics" / f"dist_{phantom_distance}"
        save_dir.mkdir(parents=True, exist_ok=True)

        item = {
            "time_step": agent.state.time_step,
            "phantom_distance": phantom_distance,
            "phantom_position": phantom_agent.state.position.tolist(),
            "agent_accs": agent_accs,
            "agent_predictions": agent_predictions,
            "phantom_vels": phantom_vels,
            "phantom_accs": phantom_accs,
            "phantom_predictions": phantom_predictions,
            "results": results,
        }
        np.save(save_dir / f"results_{agent.state.time_step:03d}.npy", item)

        return results

    @staticmethod
    def make_gif_for_distance(config, time_steps, phantom_distance):
        """Create GIF for metrics visualization at specific phantom distance."""
        from pathlib import Path

        metrics_dir = Path(config.basic.result_dir) / "metrics" / f"dist_{phantom_distance}"
        items = []
        for timestep in time_steps:
            filename = metrics_dir / f"results_{timestep:03d}.npy"
            if filename.exists():
                item = np.load(filename, allow_pickle=True).item()
                items.append(item)

        if not items:
            print(f"No results found for distance {phantom_distance}")
            return

        fig = plt.figure(figsize=(12, 8))
        ax_ttc = fig.add_subplot(121)
        ax_dce = fig.add_subplot(122)

        converted = [[] for _ in range(len(items[0]["agent_accs"]))]
        for item in items:
            for i, acc in enumerate(item["agent_accs"]):
                converted[i].append(item["results"][i])

        images = [[] for _ in range(len(items[0]["agent_accs"]))]
        agent_accs = items[0]["agent_accs"]
        phantom_vels = items[0]["phantom_vels"]
        phantom_accs = items[0]["phantom_accs"]

        for i in range(len(converted)):  # agent_acc
            for j in range(len(converted[i])):  # time_step
                ttc_values = np.zeros((len(phantom_vels), len(phantom_accs)))
                dce_values = np.zeros((len(phantom_vels), len(phantom_accs)))
                for k, vel in enumerate(phantom_vels):
                    for m, acc in enumerate(phantom_accs):
                        ttc_values[k, m] = converted[i][j][k][m]["ttc"]
                        dce_values[k, m] = converted[i][j][k][m]["dce"]

                ax_ttc.clear()
                cmap_ttc = plt.get_cmap("Blues")
                norm_ttc = mpl.colors.Normalize(
                    vmin=0, vmax=config.phantom.prediction_time
                )
                ax_ttc.imshow(ttc_values, cmap=cmap_ttc, norm=norm_ttc)
                ax_ttc.set_xlabel("Phantom Acceleration [m/s^2]")
                ax_ttc.set_ylabel("Phantom Velocity [m/s]")
                ax_ttc.set_xticks(np.arange(len(phantom_accs)))
                ax_ttc.set_xticklabels(
                    [f"{val:.2f}" for val in phantom_accs], rotation=45
                )
                ax_ttc.set_yticks(np.arange(len(phantom_vels)))
                ax_ttc.set_yticklabels([f"{val:.2f}" for val in phantom_vels])
                ax_ttc.set_title("Time to Collision")

                cmap_dce = plt.get_cmap("Blues")
                norm_dce = mpl.colors.Normalize(vmin=0, vmax=15.0)

                ax_dce.clear()
                ax_dce.imshow(dce_values, cmap=cmap_dce, norm=norm_dce)
                ax_dce.set_xlabel("Phantom Acceleration [m/s^2]")
                ax_dce.set_ylabel("Phantom Velocity [m/s]")
                ax_dce.set_xticks(np.arange(len(phantom_accs)))
                ax_dce.set_xticklabels(
                    [f"{val:.2f}" for val in phantom_accs], rotation=45
                )
                ax_dce.set_yticks(np.arange(len(phantom_vels)))
                ax_dce.set_yticklabels([f"{val:.2f}" for val in phantom_vels])
                ax_dce.set_title("Distance to Closest Encounter")

                fig.suptitle(
                    f"Phantom Distance: {phantom_distance} m\n"
                    f"Time: {j * config.basic.dt:.1f} s, "
                    f"Agent Acc: {agent_accs[i]:.2f} m/s^2"
                )
                fig.tight_layout()

                save_path = metrics_dir / f"result_acc_{agent_accs[i]:.2f}_timestep_{j:03d}.png"
                fig.savefig(save_path)
                images[i].append(Image.open(save_path))

        # Save GIFs
        output_dir = Path(config.basic.result_dir) / "metrics" / "gifs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(images)):
            if images[i]:
                images[i][0].save(
                    output_dir / f"dist_{phantom_distance}_acc_{agent_accs[i]:.2f}.gif",
                    save_all=True,
                    append_images=images[i][1:],
                    duration=100,
                    loop=0,
                )

        plt.close(fig)
        print(f"Saved metric GIFs for distance {phantom_distance}")


def main():
    config = ConfigManager.load_config("config2.yaml")

    if config.visualization.show:
        set_non_blocking()

    # Load scenario
    scenario, planning_problem_set = CommonRoadFileReader(
        "scenarios/ZAM_Occlusion-1_1_T-1.xml"
    ).open()

    renderer = MPRenderer()
    agents = create_agents(scenario, config.agents)
    buildings = create_buildings(scenario.static_obstacles)

    # Create phantom vehicles at specified distances
    phantoms = create_phantoms_at_distances(
        scenario=scenario,
        config_phantom=config.phantom,
        ego_state=agents[0].state,
        static_obstacles=buildings,
        distances=PHANTOM_DISTANCES,
    )

    print(f"Created {len(phantoms)} phantom vehicles at distances: {PHANTOM_DISTANCES}")
    for i, (phantom, dist) in enumerate(zip(phantoms, PHANTOM_DISTANCES)):
        print(f"  Phantom {i}: distance={dist}m, position={phantom.state.position}")

    # Create motion checker
    motion_checker = MotionCheckerMultiPhantom(config)

    config_basic = config.basic
    num_timesteps = int(config_basic.duration / config_basic.dt)

    # Process each phantom separately
    for phantom_idx, (phantom, phantom_distance) in enumerate(zip(phantoms, PHANTOM_DISTANCES)):
        print(f"\n{'='*60}")
        print(f"Processing Phantom {phantom_idx}: Distance = {phantom_distance} m")
        print(f"{'='*60}")

        # Reset agents to initial state
        agents = create_agents(scenario, config.agents)

        # Reset phantom time step
        phantom.state.time_step = 0

        for timestep in tqdm(range(num_timesteps), desc=f"Phantom dist={phantom_distance}m"):
            # Update agents
            update_agents(agents, timestep, dt=config_basic.dt)

            # Update phantom time step
            phantom.state.time_step = timestep

            # Evaluate metrics
            motion_checker.evaluate_with_distance(
                agents[1],
                phantom,
                phantom_distance,
            )

            # Visualize
            Visualizer.draw_step_with_phantoms(
                renderer=renderer,
                scenario=scenario,
                agents=agents,
                buildings=buildings,
                phantoms=[phantom],  # Only show current phantom
                timestep=timestep,
                config=config,
                phantom_distances=[phantom_distance],
                current_phantom_idx=0,
            )
            Visualizer.save_frame(config, timestep, phantom_distance)

        # Create GIF for this phantom distance
        if config.visualization.save_gif:
            Visualizer.make_gif(config, range(num_timesteps), phantom_distance)

    # Create metric GIFs for all phantom distances
    plt.ioff()
    for phantom_distance in PHANTOM_DISTANCES:
        MotionCheckerMultiPhantom.make_gif_for_distance(
            config, range(num_timesteps), phantom_distance
        )

    print("\n" + "="*60)
    print("Simulation complete!")
    print(f"Results saved to: {config.basic.result_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
