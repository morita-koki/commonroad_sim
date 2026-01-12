# -*- coding: utf-8 -*-
"""
Publication-quality visualization script for CommonRoad simulation results.

This script generates figures suitable for academic papers with proper font sizes,
styling, and formatting. Supports multiple phantom vehicle distances.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ConfigManager

# Try to import japanize_matplotlib for Japanese support
try:
    import japanize_matplotlib  # noqa: F401
    JAPANESE_AVAILABLE = True
except ImportError:
    JAPANESE_AVAILABLE = False

# =============================================================================
# Publication-quality style settings
# =============================================================================

# Color palette (colorblind-friendly)
COLORS = {
    "red": "#FF4B00",
    "blue": "#005AFF",
    "green": "#03AF7A",
    "orange": "#F6AA00",
    "purple": "#990099",
    "gray": "#666666",
}

# Font sizes for publication
FONT_CONFIG = {
    "title": 14,
    "axis_label": 12,
    "tick_label": 10,
    "legend": 10,
    "annotation": 9,
}

# Figure sizes (in inches)
FIGURE_SIZES = {
    "single_column": (3.5, 2.8),
    "double_column": (7.0, 4.0),
    "heatmap": (5.5, 4.5),
    "line_plot": (5.0, 3.5),
    "comparison": (10.0, 4.0),
}

# Default phantom distances
DEFAULT_PHANTOM_DISTANCES = [10, 20, 30, 40]


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_CONFIG["tick_label"],
        "axes.titlesize": FONT_CONFIG["title"],
        "axes.labelsize": FONT_CONFIG["axis_label"],
        "xtick.labelsize": FONT_CONFIG["tick_label"],
        "ytick.labelsize": FONT_CONFIG["tick_label"],
        "legend.fontsize": FONT_CONFIG["legend"],
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
    })


# =============================================================================
# Data loading utilities
# =============================================================================

class MetricsData:
    """Container for loading and processing simulation metrics."""

    def __init__(
        self,
        data_dir: Path,
        num_timesteps: int = 100,
        phantom_distance: Optional[float] = None,
    ):
        self.data_dir = Path(data_dir)
        self.num_timesteps = num_timesteps
        self.phantom_distance = phantom_distance
        self.items = []
        self.result = None

        self._load_data()
        self._process_data()

    def _load_data(self):
        """Load all result files."""
        for timestep in range(self.num_timesteps):
            filename = self.data_dir / f"results_{timestep:03d}.npy"
            if filename.exists():
                item = np.load(filename, allow_pickle=True).item()
                self.items.append(item)
                # Extract phantom distance if available
                if self.phantom_distance is None and "phantom_distance" in item:
                    self.phantom_distance = item["phantom_distance"]
            else:
                break

        self.num_timesteps = len(self.items)

    def _process_data(self):
        """Process loaded data into numpy arrays."""
        if not self.items:
            raise ValueError("No data loaded")

        len_time_steps = len(self.items)
        len_agent_acc = len(self.items[0]["agent_accs"])
        len_phantom_vel = len(self.items[0]["phantom_vels"])
        len_phantom_acc = len(self.items[0]["phantom_accs"])
        len_metric = 3  # dce, ttc, ttce

        self.result = np.zeros(
            (len_time_steps, len_agent_acc, len_phantom_vel, len_phantom_acc, len_metric)
        )

        for i, item in enumerate(self.items):
            for j, _ in enumerate(item["agent_accs"]):
                for k, _ in enumerate(item["phantom_vels"]):
                    for m, _ in enumerate(item["phantom_accs"]):
                        for key, value in item["results"][j][k][m].items():
                            if key == "dce":
                                self.result[i, j, k, m, 0] = value
                            elif key == "ttc":
                                self.result[i, j, k, m, 1] = value
                            elif key == "ttce":
                                self.result[i, j, k, m, 2] = value

        self.dce = self.result[:, :, :, :, 0]
        self.ttc = self.result[:, :, :, :, 1]
        self.ttce = self.result[:, :, :, :, 2]

        self.agent_accs = self.items[0]["agent_accs"]
        self.phantom_vels = self.items[0]["phantom_vels"]
        self.phantom_accs = self.items[0]["phantom_accs"]

    def compute_risk_from_ttc(
        self,
        ttc_threshold: float = 2.0,
        alpha: float = 2.0
    ) -> np.ndarray:
        """Compute collision risk from TTC using sigmoid function."""
        risk = 1.0 / (1.0 + np.exp(alpha * (self.ttc - ttc_threshold)))
        return risk

    def compute_risk_mean_over_acc(self, risk: np.ndarray) -> np.ndarray:
        """Compute mean risk over agent accelerations."""
        return np.mean(risk, axis=1)

    def get_zero_acc_index(self, accs: np.ndarray) -> int:
        """Get the index of acceleration closest to zero.

        Args:
            accs: Array of acceleration values

        Returns:
            Index of the acceleration value closest to zero
        """
        return int(np.argmin(np.abs(accs)))

    def get_acc_index(self, accs: np.ndarray, target_acc: float) -> int:
        """Get the index of acceleration closest to the target value.

        Args:
            accs: Array of acceleration values
            target_acc: Target acceleration value

        Returns:
            Index of the acceleration value closest to target
        """
        return int(np.argmin(np.abs(accs - target_acc)))

    def extract_phantom_zero_acc(self, risk: np.ndarray) -> np.ndarray:
        """Extract risk slice where phantom acceleration is zero.

        Args:
            risk: Risk array with shape (timesteps, agent_acc, phantom_vel, phantom_acc)

        Returns:
            Risk array with shape (timesteps, agent_acc, phantom_vel)
        """
        zero_idx = self.get_zero_acc_index(self.phantom_accs)
        return risk[:, :, :, zero_idx]

    def extract_phantom_acc(self, risk: np.ndarray, target_acc: float) -> np.ndarray:
        """Extract risk slice for a specific phantom acceleration.

        Args:
            risk: Risk array with shape (timesteps, agent_acc, phantom_vel, phantom_acc)
            target_acc: Target phantom acceleration value

        Returns:
            Risk array with shape (timesteps, agent_acc, phantom_vel)
        """
        acc_idx = self.get_acc_index(self.phantom_accs, target_acc)
        return risk[:, :, :, acc_idx]

    def extract_agent_zero_acc(self, risk: np.ndarray) -> np.ndarray:
        """Extract risk slice where agent acceleration is zero.

        Args:
            risk: Risk array with shape (timesteps, agent_acc, phantom_vel, phantom_acc)

        Returns:
            Risk array with shape (timesteps, phantom_vel, phantom_acc)
        """
        zero_idx = self.get_zero_acc_index(self.agent_accs)
        return risk[:, zero_idx, :, :]

    def get_distance_to_collision(
        self,
        timestep: int,
        initial_distance: float = 20.0,
        dt: float = 0.05,
        velocity: float = 8.0
    ) -> float:
        """Calculate agent's distance to collision region."""
        return initial_distance - timestep * dt * velocity


class MultiPhantomMetricsData:
    """Container for loading metrics from multiple phantom distances."""

    def __init__(
        self,
        base_dir: Path,
        phantom_distances: List[float],
        num_timesteps: int = 100,
    ):
        self.base_dir = Path(base_dir)
        self.phantom_distances = phantom_distances
        self.num_timesteps = num_timesteps
        self.data = {}

        self._load_all_data()

    def _load_all_data(self):
        """Load data for all phantom distances."""
        for distance in self.phantom_distances:
            data_dir = self.base_dir / f"dist_{distance}"
            if data_dir.exists():
                try:
                    self.data[distance] = MetricsData(
                        data_dir, self.num_timesteps, distance
                    )
                    print(f"Loaded data for phantom distance {distance} m")
                except ValueError as e:
                    print(f"Warning: Could not load data for distance {distance}: {e}")
            else:
                print(f"Warning: Directory not found: {data_dir}")


# =============================================================================
# Visualization functions
# =============================================================================

def plot_risk_heatmap(
    data: MetricsData,
    timestep: int,
    risk: np.ndarray,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    phantom_distance: Optional[float] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "Reds",
) -> plt.Figure:
    """Plot heatmap of collision risk."""
    agent_distance = data.get_distance_to_collision(timestep, initial_distance)
    phantom_dist = phantom_distance if phantom_distance else data.phantom_distance

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["heatmap"], constrained_layout=True)

    im = ax.imshow(
        risk,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    cbar = plt.colorbar(im, ax=ax)

    if use_japanese and JAPANESE_AVAILABLE:
        ax.set_xlabel("仮想車両の加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("仮想車両の速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
        cbar.set_label("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
        title = f"タイムステップ {timestep}における衝突リスク\n"
        if phantom_dist:
            title += f"仮想車両の衝突領域までの距離: {phantom_dist:.1f} m\n"
        title += f"観測車両の衝突領域までの距離: {agent_distance:.2f} m"
    else:
        ax.set_xlabel("Phantom Vehicle Acceleration [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("Phantom Vehicle Velocity [m/s]", fontsize=FONT_CONFIG["axis_label"])
        cbar.set_label("Collision Risk", fontsize=FONT_CONFIG["axis_label"])
        title = f"Collision Risk at Timestep {timestep}\n"
        if phantom_dist:
            title += f"Phantom Distance to Collision: {phantom_dist:.1f} m\n"
        title += f"Agent Distance to Collision: {agent_distance:.2f} m"

    ax.set_title(title, fontsize=FONT_CONFIG["title"], pad=10)

    n_phantom_acc = len(data.phantom_accs)
    n_phantom_vel = len(data.phantom_vels)

    tick_step = max(1, n_phantom_acc // 10)
    xtick_indices = range(0, n_phantom_acc, tick_step)
    ax.set_xticks(list(xtick_indices))
    ax.set_xticklabels(
        [f"{data.phantom_accs[i]:.1f}" for i in xtick_indices],
        rotation=45,
        ha="right"
    )

    tick_step = max(1, n_phantom_vel // 10)
    ytick_indices = range(0, n_phantom_vel, tick_step)
    ax.set_yticks(list(ytick_indices))
    ax.set_yticklabels([f"{data.phantom_vels[i]:.1f}" for i in ytick_indices])


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_phantom_zero_acc(
    data: MetricsData,
    timestep: int,
    risk: np.ndarray,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    phantom_distance: Optional[float] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "Reds",
    dt: float = 0.1,
    velocity: float = 4.0,
) -> plt.Figure:
    """Plot heatmap of collision risk with phantom acceleration fixed at zero.

    This shows risk as a function of agent acceleration and phantom velocity,
    assuming the phantom vehicle maintains constant velocity (zero acceleration).

    Args:
        data: MetricsData object
        timestep: Timestep to visualize
        risk: Risk array with shape (agent_acc, phantom_vel) - already extracted for phantom_acc=0
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        initial_distance: Initial distance to collision region
        phantom_distance: Phantom distance to collision region
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
        dt: Time step [s]
        velocity: Agent velocity [m/s]

    Returns:
        matplotlib Figure object
    """
    agent_distance = data.get_distance_to_collision(timestep, initial_distance, dt, velocity)
    phantom_dist = phantom_distance if phantom_distance else data.phantom_distance
    zero_acc_idx = data.get_zero_acc_index(data.phantom_accs)
    zero_acc_value = data.phantom_accs[zero_acc_idx]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["heatmap"], constrained_layout=True)

    # Transpose so that X-axis is agent acceleration, Y-axis is phantom velocity
    # risk shape: (agent_acc, phantom_vel) -> (phantom_vel, agent_acc)
    im = ax.imshow(
        risk.T,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    cbar = plt.colorbar(im, ax=ax)

    if use_japanese and JAPANESE_AVAILABLE:
        ax.set_xlabel("観測車両の加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("仮想車両の速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
        cbar.set_label("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
        title = f"仮想車両加速度=0 ({zero_acc_value:.2f} m/s²) における衝突リスク\n"
        title += f"タイムステップ {timestep} (t={timestep*dt:.2f}s)\n"
        if phantom_dist:
            title += f"仮想車両距離: {phantom_dist:.1f} m, "
        title += f"観測車両距離: {agent_distance:.2f} m"
    else:
        ax.set_xlabel("Agent Acceleration [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("Phantom Velocity [m/s]", fontsize=FONT_CONFIG["axis_label"])
        cbar.set_label("Collision Risk", fontsize=FONT_CONFIG["axis_label"])
        title = f"Collision Risk at Phantom Acc=0 ({zero_acc_value:.2f} m/s²)\n"
        title += f"Timestep {timestep} (t={timestep*dt:.2f}s)\n"
        if phantom_dist:
            title += f"Phantom Distance: {phantom_dist:.1f} m, "
        title += f"Agent Distance: {agent_distance:.2f} m"

    ax.set_title(title, fontsize=FONT_CONFIG["title"], pad=10)

    # X-axis: agent accelerations
    n_agent_acc = len(data.agent_accs)
    tick_step = max(1, n_agent_acc // 10)
    xtick_indices = range(0, n_agent_acc, tick_step)
    ax.set_xticks(list(xtick_indices))
    ax.set_xticklabels(
        [f"{data.agent_accs[i]:.1f}" for i in xtick_indices],
        rotation=45,
        ha="right"
    )

    # Y-axis: phantom velocities
    n_phantom_vel = len(data.phantom_vels)
    tick_step = max(1, n_phantom_vel // 10)
    ytick_indices = range(0, n_phantom_vel, tick_step)
    ax.set_yticks(list(ytick_indices))
    ax.set_yticklabels([f"{data.phantom_vels[i]:.1f}" for i in ytick_indices])


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_agent_zero_acc(
    data: MetricsData,
    timestep: int,
    risk: np.ndarray,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    phantom_distance: Optional[float] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "Reds",
    dt: float = 0.1,
    velocity: float = 4.0,
) -> plt.Figure:
    """Plot heatmap of collision risk with agent acceleration fixed at zero.

    This shows risk as a function of phantom velocity and acceleration,
    assuming the agent (ego) vehicle maintains constant velocity (zero acceleration).

    Args:
        data: MetricsData object
        timestep: Timestep to visualize
        risk: Risk array with shape (phantom_vel, phantom_acc) - already extracted for agent_acc=0
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        initial_distance: Initial distance to collision region
        phantom_distance: Phantom distance to collision region
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
        dt: Time step [s]
        velocity: Agent velocity [m/s]

    Returns:
        matplotlib Figure object
    """
    agent_distance = data.get_distance_to_collision(timestep, initial_distance, dt, velocity)
    phantom_dist = phantom_distance if phantom_distance else data.phantom_distance
    zero_acc_idx = data.get_zero_acc_index(data.agent_accs)
    zero_acc_value = data.agent_accs[zero_acc_idx]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["heatmap"], constrained_layout=True)

    im = ax.imshow(
        risk,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    cbar = plt.colorbar(im, ax=ax)

    if use_japanese and JAPANESE_AVAILABLE:
        ax.set_xlabel("仮想車両の加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("仮想車両の速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
        cbar.set_label("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
        title = f"観測車両加速度=0 ({zero_acc_value:.2f} m/s²) における衝突リスク\n"
        title += f"タイムステップ {timestep} (t={timestep*dt:.2f}s)\n"
        if phantom_dist:
            title += f"仮想車両距離: {phantom_dist:.1f} m, "
        title += f"観測車両距離: {agent_distance:.2f} m"
    else:
        ax.set_xlabel("Phantom Acceleration [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("Phantom Velocity [m/s]", fontsize=FONT_CONFIG["axis_label"])
        cbar.set_label("Collision Risk", fontsize=FONT_CONFIG["axis_label"])
        title = f"Collision Risk at Agent Acc=0 ({zero_acc_value:.2f} m/s²)\n"
        title += f"Timestep {timestep} (t={timestep*dt:.2f}s)\n"
        if phantom_dist:
            title += f"Phantom Distance: {phantom_dist:.1f} m, "
        title += f"Agent Distance: {agent_distance:.2f} m"

    ax.set_title(title, fontsize=FONT_CONFIG["title"], pad=10)

    # X-axis: phantom accelerations
    n_phantom_acc = len(data.phantom_accs)
    tick_step = max(1, n_phantom_acc // 10)
    xtick_indices = range(0, n_phantom_acc, tick_step)
    ax.set_xticks(list(xtick_indices))
    ax.set_xticklabels(
        [f"{data.phantom_accs[i]:.1f}" for i in xtick_indices],
        rotation=45,
        ha="right"
    )

    # Y-axis: phantom velocities
    n_phantom_vel = len(data.phantom_vels)
    tick_step = max(1, n_phantom_vel // 10)
    ytick_indices = range(0, n_phantom_vel, tick_step)
    ax.set_yticks(list(ytick_indices))
    ax.set_yticklabels([f"{data.phantom_vels[i]:.1f}" for i in ytick_indices])


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_grid_phantom_acc(
    multi_data: "MultiPhantomMetricsData",
    agent_distances: List[float],
    phantom_acc: float = 0.0,
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    dt: float = 0.1,
    velocity: float = 4.0,
) -> plt.Figure:
    """
    Plot a grid of risk heatmaps with phantom acceleration fixed at a specific value.
    Rows = agent distances, Columns = phantom distances.
    Each cell shows: X-axis = Agent Acceleration, Y-axis = Phantom Velocity.

    Args:
        multi_data: MultiPhantomMetricsData object
        agent_distances: List of agent distances to collision region [m]
        phantom_acc: Target phantom acceleration value [m/s²]
        ttc_threshold: TTC threshold for risk calculation
        alpha: Steepness parameter for sigmoid
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        initial_distance: Initial distance at timestep 0 [m]
        dt: Time step [s]
        velocity: Agent velocity [m/s]

    Returns:
        matplotlib Figure object
    """
    phantom_distances = sorted(multi_data.data.keys())
    n_cols = len(phantom_distances)
    n_rows = len(agent_distances)

    if n_cols == 0:
        raise ValueError("No phantom data available")

    row_height = 2.8
    col_width = 3.0
    fig_width = col_width * n_cols + 1.5
    fig_height = row_height * n_rows + 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.92])  # Leave space for suptitle

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    im = None

    for row_idx, agent_dist in enumerate(agent_distances):
        timestep = get_timestep_from_agent_distance(agent_dist, initial_distance, dt, velocity)

        for col_idx, phantom_dist in enumerate(phantom_distances):
            data = multi_data.data[phantom_dist]

            if timestep >= data.num_timesteps:
                actual_timestep = data.num_timesteps - 1
            else:
                actual_timestep = timestep

            risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
            risk_phantom_slice = data.extract_phantom_acc(risk, phantom_acc)

            ax = axes[row_idx][col_idx]
            # Transpose: (agent_acc, phantom_vel) -> (phantom_vel, agent_acc)
            # So Y-axis = phantom_vel, X-axis = agent_acc
            im = ax.imshow(
                risk_phantom_slice[actual_timestep].T,
                interpolation="nearest",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                aspect="auto"
            )

            mean_risk_value = np.mean(risk_phantom_slice[actual_timestep])
            ax.set_title(f"$\\bar{{R}}$ = {mean_risk_value:.3f}", fontsize=FONT_CONFIG["tick_label"])

            # Column headers (phantom distance) - only for first row
            if row_idx == 0:
                if use_japanese and JAPANESE_AVAILABLE:
                    header = f"仮想車両\n{phantom_dist} m"
                else:
                    header = f"Phantom\n{phantom_dist} m"
                ax.annotate(header, xy=(0.5, 1.25), xycoords="axes fraction",
                           ha="center", va="bottom", fontsize=FONT_CONFIG["axis_label"],
                           fontweight="bold")

            # Row labels (agent distance) - only for first column
            if col_idx == 0:
                if use_japanese and JAPANESE_AVAILABLE:
                    row_label = f"観測車両\n{agent_dist:.0f} m"
                else:
                    row_label = f"Agent\n{agent_dist:.0f} m"
                ax.annotate(row_label, xy=(-0.35, 0.5), xycoords="axes fraction",
                           ha="center", va="center", fontsize=FONT_CONFIG["axis_label"],
                           fontweight="bold", rotation=90)

            # X-axis ticks (agent accelerations)
            n_agent_acc = len(data.agent_accs)
            tick_step = max(1, n_agent_acc // 4)
            xtick_indices = list(range(0, n_agent_acc, tick_step))
            ax.set_xticks(xtick_indices)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(
                    [f"{data.agent_accs[i]:.0f}" for i in xtick_indices],
                    fontsize=7,
                )
                if col_idx == n_cols // 2:
                    if use_japanese and JAPANESE_AVAILABLE:
                        ax.set_xlabel("観測車両加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
                    else:
                        ax.set_xlabel("Agent Acc. [m/s²]", fontsize=FONT_CONFIG["axis_label"])
            else:
                ax.set_xticklabels([])

            # Y-axis ticks (phantom velocities)
            n_phantom_vel = len(data.phantom_vels)
            tick_step = max(1, n_phantom_vel // 4)
            ytick_indices = list(range(0, n_phantom_vel, tick_step))
            ax.set_yticks(ytick_indices)
            if col_idx == 0:
                ax.set_yticklabels(
                    [f"{data.phantom_vels[i]:.0f}" for i in ytick_indices],
                    fontsize=7,
                )
                if row_idx == n_rows // 2:
                    if use_japanese and JAPANESE_AVAILABLE:
                        ax.set_ylabel("仮想車両速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
                    else:
                        ax.set_ylabel("Phantom Vel. [m/s]", fontsize=FONT_CONFIG["axis_label"])
            else:
                ax.set_yticklabels([])

    # Main title
    if use_japanese and JAPANESE_AVAILABLE:
        fig.suptitle(
            f"仮想車両加速度 = {phantom_acc:.1f} m/s² における衝突リスクマップ",
            fontsize=FONT_CONFIG["title"] + 2,
            y=0.98,
        )
    else:
        fig.suptitle(
            f"Collision Risk Map at Phantom Acc = {phantom_acc:.1f} m/s²",
            fontsize=FONT_CONFIG["title"] + 2,
            y=0.98,
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    if use_japanese and JAPANESE_AVAILABLE:
        cbar.set_label("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
    else:
        cbar.set_label("Risk", fontsize=FONT_CONFIG["axis_label"])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_grid_agent_zero_acc(
    multi_data: "MultiPhantomMetricsData",
    agent_distances: List[float],
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    dt: float = 0.1,
    velocity: float = 4.0,
) -> plt.Figure:
    """
    Plot a 4x4 grid of risk heatmaps with agent acceleration fixed at zero.
    Rows = agent distances, Columns = phantom distances.
    Each cell shows: X-axis = Phantom Acceleration, Y-axis = Phantom Velocity.

    Args:
        multi_data: MultiPhantomMetricsData object
        agent_distances: List of agent distances to collision region [m]
        ttc_threshold: TTC threshold for risk calculation
        alpha: Steepness parameter for sigmoid
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        initial_distance: Initial distance at timestep 0 [m]
        dt: Time step [s]
        velocity: Agent velocity [m/s]

    Returns:
        matplotlib Figure object
    """
    phantom_distances = sorted(multi_data.data.keys())
    n_cols = len(phantom_distances)
    n_rows = len(agent_distances)

    if n_cols == 0:
        raise ValueError("No phantom data available")

    row_height = 2.8
    col_width = 3.0
    fig_width = col_width * n_cols + 1.5
    fig_height = row_height * n_rows + 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.92])  # Leave space for suptitle

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    im = None

    for row_idx, agent_dist in enumerate(agent_distances):
        timestep = get_timestep_from_agent_distance(agent_dist, initial_distance, dt, velocity)

        for col_idx, phantom_dist in enumerate(phantom_distances):
            data = multi_data.data[phantom_dist]

            if timestep >= data.num_timesteps:
                actual_timestep = data.num_timesteps - 1
            else:
                actual_timestep = timestep

            risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
            risk_agent_zero = data.extract_agent_zero_acc(risk)

            ax = axes[row_idx][col_idx]
            # Shape is (phantom_vel, phantom_acc), so Y-axis = phantom_vel, X-axis = phantom_acc
            im = ax.imshow(
                risk_agent_zero[actual_timestep],
                interpolation="nearest",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                aspect="auto"
            )

            mean_risk_value = np.mean(risk_agent_zero[actual_timestep])
            ax.set_title(f"$\\bar{{R}}$ = {mean_risk_value:.3f}", fontsize=FONT_CONFIG["tick_label"])

            # Column headers (phantom distance) - only for first row
            if row_idx == 0:
                if use_japanese and JAPANESE_AVAILABLE:
                    header = f"仮想車両\n{phantom_dist} m"
                else:
                    header = f"Phantom\n{phantom_dist} m"
                ax.annotate(header, xy=(0.5, 1.25), xycoords="axes fraction",
                           ha="center", va="bottom", fontsize=FONT_CONFIG["axis_label"],
                           fontweight="bold")

            # Row labels (agent distance) - only for first column
            if col_idx == 0:
                if use_japanese and JAPANESE_AVAILABLE:
                    row_label = f"観測車両\n{agent_dist:.0f} m"
                else:
                    row_label = f"Agent\n{agent_dist:.0f} m"
                ax.annotate(row_label, xy=(-0.35, 0.5), xycoords="axes fraction",
                           ha="center", va="center", fontsize=FONT_CONFIG["axis_label"],
                           fontweight="bold", rotation=90)

            # X-axis ticks (phantom accelerations)
            n_phantom_acc = len(data.phantom_accs)
            tick_step = max(1, n_phantom_acc // 4)
            xtick_indices = list(range(0, n_phantom_acc, tick_step))
            ax.set_xticks(xtick_indices)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(
                    [f"{data.phantom_accs[i]:.0f}" for i in xtick_indices],
                    fontsize=7,
                )
                if col_idx == n_cols // 2:
                    if use_japanese and JAPANESE_AVAILABLE:
                        ax.set_xlabel("仮想車両加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
                    else:
                        ax.set_xlabel("Phantom Acc. [m/s²]", fontsize=FONT_CONFIG["axis_label"])
            else:
                ax.set_xticklabels([])

            # Y-axis ticks (phantom velocities)
            n_phantom_vel = len(data.phantom_vels)
            tick_step = max(1, n_phantom_vel // 4)
            ytick_indices = list(range(0, n_phantom_vel, tick_step))
            ax.set_yticks(ytick_indices)
            if col_idx == 0:
                ax.set_yticklabels(
                    [f"{data.phantom_vels[i]:.0f}" for i in ytick_indices],
                    fontsize=7,
                )
                if row_idx == n_rows // 2:
                    if use_japanese and JAPANESE_AVAILABLE:
                        ax.set_ylabel("仮想車両速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
                    else:
                        ax.set_ylabel("Phantom Vel. [m/s]", fontsize=FONT_CONFIG["axis_label"])
            else:
                ax.set_yticklabels([])

    # Main title
    if use_japanese and JAPANESE_AVAILABLE:
        fig.suptitle(
            "観測車両加速度 = 0 m/s² における衝突リスクマップ",
            fontsize=FONT_CONFIG["title"] + 2,
            y=0.98,
        )
    else:
        fig.suptitle(
            "Collision Risk Map at Agent Acc = 0 m/s²",
            fontsize=FONT_CONFIG["title"] + 2,
            y=0.98,
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    if use_japanese and JAPANESE_AVAILABLE:
        cbar.set_label("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
    else:
        cbar.set_label("Risk", fontsize=FONT_CONFIG["axis_label"])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_comparison_by_distance(
    multi_data: MultiPhantomMetricsData,
    timestep: int,
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
) -> plt.Figure:
    """
    Plot comparison of risk heatmaps across different phantom distances.
    """
    distances = sorted(multi_data.data.keys())
    n_distances = len(distances)

    if n_distances == 0:
        raise ValueError("No data available for comparison")

    fig, axes = plt.subplots(1, n_distances, figsize=(4 * n_distances, 4.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.92])  # Leave space for suptitle
    if n_distances == 1:
        axes = [axes]

    for idx, distance in enumerate(distances):
        data = multi_data.data[distance]
        risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
        risk_mean = data.compute_risk_mean_over_acc(risk)

        ax = axes[idx]
        im = ax.imshow(
            risk_mean[timestep],
            interpolation="nearest",
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            aspect="auto"
        )

        if use_japanese and JAPANESE_AVAILABLE:
            ax.set_title(f"仮想車両距離: {distance} m", fontsize=FONT_CONFIG["title"])
            if idx == 0:
                ax.set_ylabel("仮想車両速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
            ax.set_xlabel("仮想車両加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        else:
            ax.set_title(f"Phantom Distance: {distance} m", fontsize=FONT_CONFIG["title"])
            if idx == 0:
                ax.set_ylabel("Phantom Velocity [m/s]", fontsize=FONT_CONFIG["axis_label"])
            ax.set_xlabel("Phantom Acceleration [m/s²]", fontsize=FONT_CONFIG["axis_label"])

        n_phantom_acc = len(data.phantom_accs)
        n_phantom_vel = len(data.phantom_vels)

        tick_step = max(1, n_phantom_acc // 5)
        xtick_indices = range(0, n_phantom_acc, tick_step)
        ax.set_xticks(list(xtick_indices))
        ax.set_xticklabels(
            [f"{data.phantom_accs[i]:.1f}" for i in xtick_indices],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        tick_step = max(1, n_phantom_vel // 5)
        ytick_indices = range(0, n_phantom_vel, tick_step)
        ax.set_yticks(list(ytick_indices))
        ax.set_yticklabels(
            [f"{data.phantom_vels[i]:.1f}" for i in ytick_indices],
            fontsize=8,
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    if use_japanese and JAPANESE_AVAILABLE:
        cbar.set_label("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
    else:
        cbar.set_label("Collision Risk", fontsize=FONT_CONFIG["axis_label"])

    if use_japanese and JAPANESE_AVAILABLE:
        fig.suptitle(
            f"タイムステップ {timestep} における仮想車両距離別の衝突リスク比較",
            fontsize=FONT_CONFIG["title"] + 2,
        )
    else:
        fig.suptitle(
            f"Collision Risk Comparison by Phantom Distance (Timestep {timestep})",
            fontsize=FONT_CONFIG["title"] + 2,
        )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_mean_risk_by_distance(
    multi_data: MultiPhantomMetricsData,
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
) -> plt.Figure:
    """
    Plot mean risk over time for different phantom distances.
    """
    distances = sorted(multi_data.data.keys())

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["line_plot"], constrained_layout=True)

    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]]

    for idx, distance in enumerate(distances):
        data = multi_data.data[distance]
        risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
        mean_risk = np.mean(risk, axis=(1, 2, 3))

        timesteps = np.arange(len(mean_risk)) * 0.05

        color = colors[idx % len(colors)]
        ax.plot(
            timesteps,
            mean_risk,
            color=color,
            linewidth=2,
            label=f"{distance} m",
            marker="o",
            markersize=3,
            markevery=10,
        )

    if use_japanese and JAPANESE_AVAILABLE:
        ax.set_xlabel("時間 [s]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("平均衝突リスク", fontsize=FONT_CONFIG["axis_label"])
        ax.set_title("仮想車両距離別の平均衝突リスクの時間変化", fontsize=FONT_CONFIG["title"])
        ax.legend(title="仮想車両距離", loc="best")
    else:
        ax.set_xlabel("Time [s]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("Mean Collision Risk", fontsize=FONT_CONFIG["axis_label"])
        ax.set_title("Mean Collision Risk Over Time by Phantom Distance", fontsize=FONT_CONFIG["title"])
        ax.legend(title="Phantom Distance", loc="best")

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_of_ttc_mean_timeseries(
    data: MetricsData,
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    phantom_distance: Optional[float] = None,
    dt: float = 0.1,
) -> plt.Figure:
    """
    Plot the time series of mean risk (averaged over all parameters).

    Args:
        data: MetricsData object
        ttc_threshold: TTC threshold for risk calculation
        alpha: Steepness parameter for sigmoid
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        phantom_distance: Phantom distance to collision region
        dt: Time step [s]

    Returns:
        matplotlib Figure object
    """
    risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
    risk_mean = data.compute_risk_mean_over_acc(risk)

    # Global mean over phantom velocity and acceleration
    global_mean = np.mean(risk_mean, axis=(1, 2))

    # Also compute max and min for envelope
    risk_max = np.max(risk_mean, axis=(1, 2))
    risk_min = np.min(risk_mean, axis=(1, 2))

    timesteps = np.arange(len(global_mean)) * dt

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["line_plot"], constrained_layout=True)

    # Plot envelope
    ax.fill_between(
        timesteps,
        risk_min,
        risk_max,
        alpha=0.3,
        color=COLORS["blue"],
        label="Min-Max Range" if not use_japanese else "最小-最大範囲",
    )

    # Plot mean
    ax.plot(
        timesteps,
        global_mean,
        color=COLORS["blue"],
        linewidth=2,
        label="Mean Risk" if not use_japanese else "平均リスク",
    )

    phantom_dist = phantom_distance if phantom_distance else data.phantom_distance

    if use_japanese and JAPANESE_AVAILABLE:
        ax.set_xlabel("時間 [s]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("TTCから計算した平均リスク", fontsize=FONT_CONFIG["axis_label"])
        title = "平均衝突リスクの時間変化"
        if phantom_dist:
            title += f"\n仮想車両の衝突領域までの距離: {phantom_dist:.1f} m"
    else:
        ax.set_xlabel("Time [s]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("Mean Risk from TTC", fontsize=FONT_CONFIG["axis_label"])
        title = "Mean Collision Risk Over Time"
        if phantom_dist:
            title += f"\nPhantom Distance to Collision: {phantom_dist:.1f} m"

    ax.set_title(title, fontsize=FONT_CONFIG["title"], pad=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=FONT_CONFIG["legend"])
    ax.grid(True, alpha=0.3, linestyle="--")


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_of_ttc_mean_multi_timestep(
    data: MetricsData,
    timesteps: List[int],
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    phantom_distance: Optional[float] = None,
    dt: float = 0.1,
) -> plt.Figure:
    """
    Plot risk_of_ttc_mean heatmaps for multiple timesteps side by side.

    Args:
        data: MetricsData object
        timesteps: List of timesteps to plot
        ttc_threshold: TTC threshold for risk calculation
        alpha: Steepness parameter for sigmoid
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        initial_distance: Initial distance to collision region
        phantom_distance: Phantom distance to collision region
        dt: Time step [s]

    Returns:
        matplotlib Figure object
    """
    risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
    risk_mean = data.compute_risk_mean_over_acc(risk)

    n_plots = len(timesteps)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.90])  # Leave space for suptitle
    if n_plots == 1:
        axes = [axes]

    phantom_dist = phantom_distance if phantom_distance else data.phantom_distance

    for idx, ts in enumerate(timesteps):
        ax = axes[idx]
        agent_distance = data.get_distance_to_collision(ts, initial_distance)
        time_sec = ts * dt

        im = ax.imshow(
            risk_mean[ts],
            interpolation="nearest",
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            aspect="auto"
        )

        if use_japanese and JAPANESE_AVAILABLE:
            ax.set_title(
                f"t = {time_sec:.2f} s\n観測車両距離: {agent_distance:.1f} m",
                fontsize=FONT_CONFIG["title"]
            )
            if idx == 0:
                ax.set_ylabel("仮想車両速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
            ax.set_xlabel("仮想車両加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        else:
            ax.set_title(
                f"t = {time_sec:.2f} s\nAgent Distance: {agent_distance:.1f} m",
                fontsize=FONT_CONFIG["title"]
            )
            if idx == 0:
                ax.set_ylabel("Phantom Velocity [m/s]", fontsize=FONT_CONFIG["axis_label"])
            ax.set_xlabel("Phantom Acceleration [m/s²]", fontsize=FONT_CONFIG["axis_label"])

        # Set tick labels
        n_phantom_acc = len(data.phantom_accs)
        n_phantom_vel = len(data.phantom_vels)

        tick_step = max(1, n_phantom_acc // 5)
        xtick_indices = range(0, n_phantom_acc, tick_step)
        ax.set_xticks(list(xtick_indices))
        ax.set_xticklabels(
            [f"{data.phantom_accs[i]:.1f}" for i in xtick_indices],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        tick_step = max(1, n_phantom_vel // 5)
        ytick_indices = range(0, n_phantom_vel, tick_step)
        ax.set_yticks(list(ytick_indices))
        ax.set_yticklabels(
            [f"{data.phantom_vels[i]:.1f}" for i in ytick_indices],
            fontsize=8,
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    if use_japanese and JAPANESE_AVAILABLE:
        cbar.set_label("平均衝突リスク", fontsize=FONT_CONFIG["axis_label"])
    else:
        cbar.set_label("Mean Collision Risk", fontsize=FONT_CONFIG["axis_label"])

    if use_japanese and JAPANESE_AVAILABLE:
        suptitle = "時間経過に伴う平均衝突リスクの変化"
        if phantom_dist:
            suptitle += f" (仮想車両距離: {phantom_dist:.1f} m)"
    else:
        suptitle = "Mean Collision Risk Evolution Over Time"
        if phantom_dist:
            suptitle += f" (Phantom Distance: {phantom_dist:.1f} m)"

    fig.suptitle(suptitle, fontsize=FONT_CONFIG["title"] + 2)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_of_ttc_mean_by_phantom_distance(
    multi_data: MultiPhantomMetricsData,
    timestep: int,
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    dt: float = 0.1,
) -> plt.Figure:
    """
    Plot risk_of_ttc_mean comparison across different phantom distances at a specific timestep.

    Args:
        multi_data: MultiPhantomMetricsData object
        timestep: Timestep to visualize
        ttc_threshold: TTC threshold for risk calculation
        alpha: Steepness parameter for sigmoid
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        dt: Time step [s]

    Returns:
        matplotlib Figure object
    """
    distances = sorted(multi_data.data.keys())
    n_distances = len(distances)

    if n_distances == 0:
        raise ValueError("No data available for comparison")

    fig, axes = plt.subplots(1, n_distances, figsize=(4 * n_distances, 4.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.90])  # Leave space for suptitle
    if n_distances == 1:
        axes = [axes]

    time_sec = timestep * dt

    for idx, distance in enumerate(distances):
        data = multi_data.data[distance]
        risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
        risk_mean = data.compute_risk_mean_over_acc(risk)

        ax = axes[idx]
        im = ax.imshow(
            risk_mean[timestep],
            interpolation="nearest",
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            aspect="auto"
        )

        if use_japanese and JAPANESE_AVAILABLE:
            ax.set_title(
                f"仮想車両距離: {distance} m\n平均リスク: {np.mean(risk_mean[timestep]):.3f}",
                fontsize=FONT_CONFIG["title"]
            )
            if idx == 0:
                ax.set_ylabel("仮想車両速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
            ax.set_xlabel("仮想車両加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        else:
            ax.set_title(
                f"Phantom Distance: {distance} m\nMean Risk: {np.mean(risk_mean[timestep]):.3f}",
                fontsize=FONT_CONFIG["title"]
            )
            if idx == 0:
                ax.set_ylabel("Phantom Velocity [m/s]", fontsize=FONT_CONFIG["axis_label"])
            ax.set_xlabel("Phantom Acceleration [m/s²]", fontsize=FONT_CONFIG["axis_label"])

        n_phantom_acc = len(data.phantom_accs)
        n_phantom_vel = len(data.phantom_vels)

        tick_step = max(1, n_phantom_acc // 5)
        xtick_indices = range(0, n_phantom_acc, tick_step)
        ax.set_xticks(list(xtick_indices))
        ax.set_xticklabels(
            [f"{data.phantom_accs[i]:.1f}" for i in xtick_indices],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        tick_step = max(1, n_phantom_vel // 5)
        ytick_indices = range(0, n_phantom_vel, tick_step)
        ax.set_yticks(list(ytick_indices))
        ax.set_yticklabels(
            [f"{data.phantom_vels[i]:.1f}" for i in ytick_indices],
            fontsize=8,
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    if use_japanese and JAPANESE_AVAILABLE:
        cbar.set_label("平均衝突リスク (risk_of_ttc_mean)", fontsize=FONT_CONFIG["axis_label"])
    else:
        cbar.set_label("Mean Collision Risk (risk_of_ttc_mean)", fontsize=FONT_CONFIG["axis_label"])

    if use_japanese and JAPANESE_AVAILABLE:
        fig.suptitle(
            f"仮想車両距離別の平均衝突リスク比較 (t = {time_sec:.2f} s)",
            fontsize=FONT_CONFIG["title"] + 2,
        )
    else:
        fig.suptitle(
            f"Mean Collision Risk (risk_of_ttc_mean) by Phantom Distance (t = {time_sec:.2f} s)",
            fontsize=FONT_CONFIG["title"] + 2,
        )


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def get_timestep_from_agent_distance(
    target_distance: float,
    initial_distance: float = 20.0,
    dt: float = 0.1,
    velocity: float = 4.0,
) -> int:
    """
    Calculate timestep corresponding to a given agent distance from collision region.

    Args:
        target_distance: Target distance to collision region [m]
        initial_distance: Initial distance at timestep 0 [m]
        dt: Time step [s]
        velocity: Agent velocity [m/s]

    Returns:
        Timestep index
    """
    # distance = initial_distance - timestep * dt * velocity
    # timestep = (initial_distance - distance) / (dt * velocity)
    timestep = int((initial_distance - target_distance) / (dt * velocity))
    return max(0, timestep)


def plot_risk_by_agent_distance(
    multi_data: MultiPhantomMetricsData,
    agent_distances: List[float],
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    dt: float = 0.1,
    velocity: float = 4.0,
) -> plt.Figure:
    """
    Plot risk_of_ttc_mean comparison at specific agent distances from collision region.

    Args:
        multi_data: MultiPhantomMetricsData object
        agent_distances: List of agent distances to collision region [m]
        ttc_threshold: TTC threshold for risk calculation
        alpha: Steepness parameter for sigmoid
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        initial_distance: Initial distance at timestep 0 [m]
        dt: Time step [s]
        velocity: Agent velocity [m/s]

    Returns:
        matplotlib Figure object
    """
    phantom_distances = sorted(multi_data.data.keys())
    n_phantoms = len(phantom_distances)
    n_agent_dists = len(agent_distances)

    if n_phantoms == 0:
        raise ValueError("No phantom data available")

    fig, axes = plt.subplots(n_agent_dists, n_phantoms, figsize=(4 * n_phantoms, 3.5 * n_agent_dists), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    # Handle single row/column case
    if n_agent_dists == 1:
        axes = [axes]
    if n_phantoms == 1:
        axes = [[ax] for ax in axes]

    for row_idx, agent_dist in enumerate(agent_distances):
        timestep = get_timestep_from_agent_distance(agent_dist, initial_distance, dt, velocity)

        for col_idx, phantom_dist in enumerate(phantom_distances):
            data = multi_data.data[phantom_dist]

            # Check if timestep is valid
            if timestep >= data.num_timesteps:
                timestep = data.num_timesteps - 1

            risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
            risk_mean = data.compute_risk_mean_over_acc(risk)

            ax = axes[row_idx][col_idx]
            im = ax.imshow(
                risk_mean[timestep],
                interpolation="nearest",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                aspect="auto"
            )

            mean_risk_value = np.mean(risk_mean[timestep])

            # Title for each subplot
            if use_japanese and JAPANESE_AVAILABLE:
                title = f"平均: {mean_risk_value:.3f}"
            else:
                title = f"Mean: {mean_risk_value:.3f}"
            ax.set_title(title, fontsize=FONT_CONFIG["tick_label"])

            # Column headers (phantom distance)
            if row_idx == 0:
                if use_japanese and JAPANESE_AVAILABLE:
                    ax.text(0.5, 1.15, f"仮想車両距離: {phantom_dist} m",
                            transform=ax.transAxes, ha="center", fontsize=FONT_CONFIG["axis_label"],
                            fontweight="bold")
                else:
                    ax.text(0.5, 1.15, f"Phantom: {phantom_dist} m",
                            transform=ax.transAxes, ha="center", fontsize=FONT_CONFIG["axis_label"],
                            fontweight="bold")

            # Row labels (agent distance)
            if col_idx == 0:
                if use_japanese and JAPANESE_AVAILABLE:
                    ax.set_ylabel(f"観測車両距離\n{agent_dist:.0f} m", fontsize=FONT_CONFIG["axis_label"])
                else:
                    ax.set_ylabel(f"Agent Dist.\n{agent_dist:.0f} m", fontsize=FONT_CONFIG["axis_label"])

            # Tick labels
            n_phantom_acc = len(data.phantom_accs)
            n_phantom_vel = len(data.phantom_vels)

            tick_step = max(1, n_phantom_acc // 4)
            xtick_indices = range(0, n_phantom_acc, tick_step)
            ax.set_xticks(list(xtick_indices))
            if row_idx == n_agent_dists - 1:
                ax.set_xticklabels(
                    [f"{data.phantom_accs[i]:.1f}" for i in xtick_indices],
                    rotation=45, ha="right", fontsize=7,
                )
            else:
                ax.set_xticklabels([])

            tick_step = max(1, n_phantom_vel // 4)
            ytick_indices = range(0, n_phantom_vel, tick_step)
            ax.set_yticks(list(ytick_indices))
            if col_idx == 0:
                ax.set_yticklabels(
                    [f"{data.phantom_vels[i]:.1f}" for i in ytick_indices],
                    fontsize=7,
                )
            else:
                ax.set_yticklabels([])

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    if use_japanese and JAPANESE_AVAILABLE:
        cbar.set_label("平均衝突リスク", fontsize=FONT_CONFIG["axis_label"])
    else:
        cbar.set_label("Mean Collision Risk", fontsize=FONT_CONFIG["axis_label"])

    # Main title
    if use_japanese and JAPANESE_AVAILABLE:
        fig.suptitle(
            "観測車両位置別・仮想車両距離別の衝突リスク比較",
            fontsize=FONT_CONFIG["title"] + 2,
        )
    else:
        fig.suptitle(
            "Collision Risk by Agent Position and Phantom Distance",
            fontsize=FONT_CONFIG["title"] + 2,
        )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_grid_phantom_vs_agent(
    multi_data: MultiPhantomMetricsData,
    agent_distances: List[float],
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    dt: float = 0.1,
    velocity: float = 4.0,
) -> plt.Figure:
    """
    Plot a grid of risk heatmaps: rows = agent distances, columns = phantom distances.
    This creates a comprehensive comparison figure.

    Args:
        multi_data: MultiPhantomMetricsData object
        agent_distances: List of agent distances to collision region [m]
        ttc_threshold: TTC threshold for risk calculation
        alpha: Steepness parameter for sigmoid
        output_path: Path to save the figure
        use_japanese: Whether to use Japanese labels
        initial_distance: Initial distance at timestep 0 [m]
        dt: Time step [s]
        velocity: Agent velocity [m/s]

    Returns:
        matplotlib Figure object
    """
    phantom_distances = sorted(multi_data.data.keys())
    n_cols = len(phantom_distances)
    n_rows = len(agent_distances)

    if n_cols == 0:
        raise ValueError("No phantom data available")

    # Create figure with adaptive size
    # For many rows (>10), use compact layout
    if n_rows > 10:
        row_height = 1.2  # Compact height per row
        col_width = 2.0   # Compact width per column
    else:
        row_height = 2.8  # Standard height per row
        col_width = 3.0   # Standard width per column

    fig_width = col_width * n_cols + 1.5  # Extra space for colorbar
    fig_height = row_height * n_rows + 1.0  # Extra space for titles
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 0.92, 0.95])  # Leave space for colorbar and suptitle

    # Handle single row/column case
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    im = None  # To store the last imshow for colorbar

    for row_idx, agent_dist in enumerate(agent_distances):
        timestep = get_timestep_from_agent_distance(agent_dist, initial_distance, dt, velocity)

        for col_idx, phantom_dist in enumerate(phantom_distances):
            data = multi_data.data[phantom_dist]

            # Check if timestep is valid
            if timestep >= data.num_timesteps:
                actual_timestep = data.num_timesteps - 1
            else:
                actual_timestep = timestep

            risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
            risk_mean = data.compute_risk_mean_over_acc(risk)

            ax = axes[row_idx][col_idx]
            im = ax.imshow(
                risk_mean[actual_timestep],
                interpolation="nearest",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                aspect="auto"
            )

            mean_risk_value = np.mean(risk_mean[actual_timestep])

            # Compact mode adjustments
            is_compact = n_rows > 10

            # Subplot title showing mean risk (hide in compact mode)
            if not is_compact:
                ax.set_title(f"$\\bar{{R}}$ = {mean_risk_value:.3f}", fontsize=FONT_CONFIG["tick_label"])

            # Column headers (phantom distance) - only for first row
            if row_idx == 0:
                if use_japanese and JAPANESE_AVAILABLE:
                    header = f"仮想車両 {phantom_dist}m" if is_compact else f"仮想車両\n{phantom_dist} m"
                else:
                    header = f"Phantom {phantom_dist}m" if is_compact else f"Phantom\n{phantom_dist} m"
                header_y = 1.1 if is_compact else 1.25
                header_fontsize = 9 if is_compact else FONT_CONFIG["axis_label"]
                ax.annotate(header, xy=(0.5, header_y), xycoords="axes fraction",
                           ha="center", va="bottom", fontsize=header_fontsize,
                           fontweight="bold")

            # Row labels (agent distance) - only for first column
            if col_idx == 0:
                if is_compact:
                    row_label = f"{agent_dist:.0f}m"
                    label_x = -0.2
                    label_fontsize = 7
                else:
                    if use_japanese and JAPANESE_AVAILABLE:
                        row_label = f"観測車両\n{agent_dist:.0f} m"
                    else:
                        row_label = f"Agent\n{agent_dist:.0f} m"
                    label_x = -0.35
                    label_fontsize = FONT_CONFIG["axis_label"]
                ax.annotate(row_label, xy=(label_x, 0.5), xycoords="axes fraction",
                           ha="center", va="center", fontsize=label_fontsize,
                           fontweight="bold", rotation=90)

            # Tick labels - minimal for cleaner look
            n_phantom_acc = len(data.phantom_accs)
            n_phantom_vel = len(data.phantom_vels)

            # In compact mode, hide all tick labels except edges
            if is_compact:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # X-axis ticks (only for bottom row)
                tick_step = max(1, n_phantom_acc // 3)
                xtick_indices = list(range(0, n_phantom_acc, tick_step))
                ax.set_xticks(xtick_indices)
                if row_idx == n_rows - 1:
                    ax.set_xticklabels(
                        [f"{data.phantom_accs[i]:.0f}" for i in xtick_indices],
                        fontsize=7,
                    )
                    if col_idx == n_cols // 2:
                        if use_japanese and JAPANESE_AVAILABLE:
                            ax.set_xlabel("加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
                        else:
                            ax.set_xlabel("Acc. [m/s²]", fontsize=FONT_CONFIG["axis_label"])
                else:
                    ax.set_xticklabels([])

                # Y-axis ticks (only for first column)
                tick_step = max(1, n_phantom_vel // 3)
                ytick_indices = list(range(0, n_phantom_vel, tick_step))
                ax.set_yticks(ytick_indices)
                if col_idx == 0:
                    ax.set_yticklabels(
                        [f"{data.phantom_vels[i]:.0f}" for i in ytick_indices],
                        fontsize=7,
                    )
                    if row_idx == n_rows // 2:
                        if use_japanese and JAPANESE_AVAILABLE:
                            ax.set_ylabel("速度 [m/s]", fontsize=FONT_CONFIG["axis_label"])
                        else:
                            ax.set_ylabel("Vel. [m/s]", fontsize=FONT_CONFIG["axis_label"])
                else:
                    ax.set_yticklabels([])

    # Add colorbar
    is_compact = n_rows > 10
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    if use_japanese and JAPANESE_AVAILABLE:
        cbar.set_label("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
    else:
        cbar.set_label("Risk", fontsize=FONT_CONFIG["axis_label"])

    # Main title
    if use_japanese and JAPANESE_AVAILABLE:
        fig.suptitle(
            "観測車両位置 × 仮想車両距離 の衝突リスクマップ",
            fontsize=FONT_CONFIG["title"] + 2,
        )
    else:
        fig.suptitle(
            "Collision Risk Map: Agent Position × Phantom Distance",
            fontsize=FONT_CONFIG["title"] + 2,
        )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_risk_vs_acceleration(
    data: MetricsData,
    timestep: int,
    risk: np.ndarray,
    phantom_vel_index: int,
    phantom_acc_index: int,
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    phantom_distance: Optional[float] = None,
    show_mean: bool = True,
) -> plt.Figure:
    """Plot collision risk as a function of agent acceleration."""
    agent_distance = data.get_distance_to_collision(timestep, initial_distance)
    phantom_dist = phantom_distance if phantom_distance else data.phantom_distance
    risk_values = risk[timestep, :, phantom_vel_index, phantom_acc_index]
    mean_risk = np.mean(risk_values)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["line_plot"], constrained_layout=True)

    if use_japanese and JAPANESE_AVAILABLE:
        label_risk = "衝突リスク"
        label_mean = f"平均 ({mean_risk:.3f})"
    else:
        label_risk = "Collision Risk"
        label_mean = f"Mean ({mean_risk:.3f})"

    ax.plot(
        data.agent_accs,
        risk_values,
        color=COLORS["blue"],
        linewidth=2,
        label=label_risk,
        marker="o",
        markersize=4,
    )

    if show_mean:
        ax.axhline(
            y=mean_risk,
            color=COLORS["red"],
            linewidth=2,
            linestyle="--",
            label=label_mean,
        )

    if use_japanese and JAPANESE_AVAILABLE:
        ax.set_xlabel("観測車両の加速度 [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("TTCから計算したリスク", fontsize=FONT_CONFIG["axis_label"])
        title = f"加速度に対する衝突リスク (t={timestep})\n"
        if phantom_dist:
            title += f"仮想車両の衝突領域までの距離: {phantom_dist:.1f} m\n"
        title += f"観測車両の衝突領域までの距離: {agent_distance:.2f} m"
    else:
        ax.set_xlabel("Observable Vehicle Acceleration [m/s²]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("Risk from TTC", fontsize=FONT_CONFIG["axis_label"])
        title = f"Collision Risk vs. Acceleration (t={timestep})\n"
        if phantom_dist:
            title += f"Phantom Distance to Collision: {phantom_dist:.1f} m\n"
        title += f"Agent Distance to Collision: {agent_distance:.2f} m"

    ax.set_title(title, fontsize=FONT_CONFIG["title"], pad=10)

    ax.set_xlim(min(data.agent_accs), max(data.agent_accs))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.legend(loc="best", fontsize=FONT_CONFIG["legend"])
    ax.grid(True, alpha=0.3, linestyle="--")


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_sigmoid_risk_function(
    alpha: float = 3.0,
    tau: float = 2.0,
    x_range: Tuple[float, float] = (0, 5),
    output_path: Optional[Path] = None,
    use_japanese: bool = False,
) -> plt.Figure:
    """Plot the sigmoid function used to convert TTC to risk."""
    x = np.linspace(x_range[0], x_range[1], 200)
    y = 1.0 / (1.0 + np.exp(alpha * (x - tau)))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_column"], constrained_layout=True)

    ax.plot(x, y, color=COLORS["blue"], linewidth=2)

    ax.axvline(x=tau, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.axhline(y=0.5, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.scatter([tau], [0.5], color=COLORS["red"], s=50, zorder=5)

    param_text = f"alpha = {alpha:.1f}, tau = {tau:.1f} s"
    ax.annotate(
        param_text,
        xy=(tau, 0.5),
        xytext=(tau + 0.5, 0.7),
        fontsize=FONT_CONFIG["annotation"],
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"]),
    )

    if use_japanese and JAPANESE_AVAILABLE:
        ax.set_xlabel("TTC [秒]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("衝突リスク", fontsize=FONT_CONFIG["axis_label"])
        ax.set_title("TTCからリスクへの変換関数", fontsize=FONT_CONFIG["title"])
    else:
        ax.set_xlabel("TTC [s]", fontsize=FONT_CONFIG["axis_label"])
        ax.set_ylabel("Collision Risk", fontsize=FONT_CONFIG["axis_label"])
        ax.set_title("TTC to Risk Conversion Function", fontsize=FONT_CONFIG["title"])

    ax.set_xlim(x_range)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")


    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def generate_risk_heatmap_sequence(
    data: MetricsData,
    output_dir: Path,
    ttc_threshold: float = 2.0,
    alpha: float = 2.0,
    use_japanese: bool = False,
    initial_distance: float = 20.0,
    phantom_distance: Optional[float] = None,
) -> List[Path]:
    """Generate a sequence of risk heatmap images for all timesteps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    risk = data.compute_risk_from_ttc(ttc_threshold, alpha)
    risk_mean = data.compute_risk_mean_over_acc(risk)

    saved_paths = []

    for timestep in range(data.num_timesteps):
        output_path = output_dir / f"risk_heatmap_{timestep:03d}.png"

        fig = plot_risk_heatmap(
            data=data,
            timestep=timestep,
            risk=risk_mean[timestep],
            output_path=output_path,
            use_japanese=use_japanese,
            initial_distance=initial_distance,
            phantom_distance=phantom_distance,
        )
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def create_video_from_images(
    image_paths: List[Path],
    output_path: Path,
    fps: int = 20,
) -> None:
    """Create a video from a sequence of images."""
    if not image_paths:
        raise ValueError("No images provided")

    first_img = cv2.imread(str(image_paths[0]))
    height, width, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved: {output_path}")


def create_gif_from_images(
    image_paths: List[Path],
    output_path: Path,
    duration: int = 100,
) -> None:
    """Create a GIF from a sequence of images."""
    if not image_paths:
        raise ValueError("No images provided")

    images = [Image.open(str(p)) for p in image_paths]
    images[0].save(
        str(output_path),
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=duration,
    )
    print(f"GIF saved: {output_path}")


# =============================================================================
# Main execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality visualizations for CommonRoad simulation results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/metrics/tmp",
        help="Directory containing result .npy files (or base dir for multi-phantom)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualization_output",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=100,
        help="Number of timesteps to process",
    )
    parser.add_argument(
        "--japanese",
        action="store_true",
        help="Use Japanese labels",
    )
    parser.add_argument(
        "--ttc-threshold",
        type=float,
        default=2.0,
        help="TTC threshold for risk calculation [s]",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Steepness parameter for sigmoid function",
    )
    parser.add_argument(
        "--initial-distance",
        type=float,
        default=20.0,
        help="Initial distance to collision region [m]",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=15,
        help="Specific timestep for single-frame plots",
    )
    parser.add_argument(
        "--phantom-vel-idx",
        type=int,
        default=10,
        help="Phantom velocity index for line plot",
    )
    parser.add_argument(
        "--phantom-acc-idx",
        type=int,
        default=12,
        help="Phantom acceleration index for line plot",
    )
    parser.add_argument(
        "--phantom-distances",
        type=float,
        nargs="+",
        default=DEFAULT_PHANTOM_DISTANCES,
        help="List of phantom distances to process",
    )
    parser.add_argument(
        "--multi-phantom",
        action="store_true",
        help="Process multiple phantom distances",
    )
    parser.add_argument(
        "--generate-sequence",
        action="store_true",
        help="Generate full heatmap sequence and video/GIF",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (e.g., config2.yaml). If provided, dt, velocity, and initial_distance will be read from config.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step [s]. If not specified, read from config or use default 0.1",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=None,
        help="Observable vehicle velocity [m/s]. If not specified, read from config or use default 4.0",
    )

    args = parser.parse_args()

    setup_publication_style()

    script_dir = Path(__file__).parent.parent

    # Load config and extract values if config file is specified
    if args.config:
        config_path = script_dir / args.config
        print(f"Loading config from: {config_path}")
        config = ConfigManager.load_config(str(config_path))

        # Extract values from config
        config_dt = config.basic.dt
        config_velocity = config.agents[1].initial_velocity
        config_initial_distance = config.agents[1].initial_position[1]

        print(f"  dt: {config_dt} s")
        print(f"  velocity: {config_velocity} m/s")
        print(f"  initial_distance: {config_initial_distance} m")

        # Use config values if not explicitly specified via command line
        if args.dt is None:
            args.dt = config_dt
        if args.velocity is None:
            args.velocity = config_velocity
        if args.initial_distance == 20.0:  # Default value, override with config
            args.initial_distance = config_initial_distance
    else:
        # Use defaults if no config
        if args.dt is None:
            args.dt = 0.1
        if args.velocity is None:
            args.velocity = 4.0

    print("\nUsing parameters:")
    print(f"  dt: {args.dt} s")
    print(f"  velocity: {args.velocity} m/s")
    print(f"  initial_distance: {args.initial_distance} m")

    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Generate sigmoid function plot
    print("\nGenerating sigmoid function plot...")
    plot_sigmoid_risk_function(
        alpha=args.alpha,
        tau=args.ttc_threshold,
        output_path=output_dir / "sigmoid_risk_function.pdf",
        use_japanese=args.japanese,
    )
    plt.close()

    if args.multi_phantom:
        # Multi-phantom mode
        print(f"\nLoading data for phantom distances: {args.phantom_distances}")
        multi_data = MultiPhantomMetricsData(
            base_dir=data_dir,
            phantom_distances=args.phantom_distances,
            num_timesteps=args.num_timesteps,
        )

        if multi_data.data:
            # Generate comparison plot
            print(f"\nGenerating risk comparison plot for timestep {args.timestep}...")
            plot_risk_comparison_by_distance(
                multi_data=multi_data,
                timestep=args.timestep,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / f"risk_comparison_t{args.timestep:03d}.pdf",
                use_japanese=args.japanese,
            )
            plt.close()

            # Generate mean risk over time plot
            print("\nGenerating mean risk over time plot...")
            plot_mean_risk_by_distance(
                multi_data=multi_data,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "mean_risk_by_distance.pdf",
                use_japanese=args.japanese,
            )
            plt.close()

            # Generate risk_of_ttc_mean comparison by phantom distance
            print("\nGenerating risk_of_ttc_mean comparison by phantom distance...")
            plot_risk_of_ttc_mean_by_phantom_distance(
                multi_data=multi_data,
                timestep=args.timestep,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / f"risk_of_ttc_mean_comparison_t{args.timestep:03d}.pdf",
                use_japanese=args.japanese,
                dt=args.dt,
            )
            plt.close()

            # Generate comparison by agent distance
            agent_distances = [15, 10, 5, 0, -5]  # Distance to collision region [m]
            print(f"\nGenerating risk comparison by agent distance: {agent_distances} m...")
            plot_risk_by_agent_distance(
                multi_data=multi_data,
                agent_distances=agent_distances,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "risk_by_agent_distance.pdf",
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
                dt=args.dt,
                velocity=args.velocity,
            )
            plt.close()

            # Generate grid plot: phantom distance (columns) x agent distance (rows)
            grid_agent_distances = [10, 5, -5]  # Selected agent distances for 3x4 grid
            print(f"\nGenerating risk grid (phantom x agent): {grid_agent_distances} m...")
            plot_risk_grid_phantom_vs_agent(
                multi_data=multi_data,
                agent_distances=grid_agent_distances,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "risk_grid_phantom_vs_agent.pdf",
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
                dt=args.dt,
                velocity=args.velocity,
            )
            plt.close()

            # Generate fine-grained comparison: 1m intervals from 15m to -15m
            fine_agent_distances = list(range(15, -16, -1))  # [15, 14, 13, ..., 0, -1, ..., -15]
            print(f"\nGenerating fine-grained risk grid (1m intervals): {fine_agent_distances[0]}m to {fine_agent_distances[-1]}m...")
            plot_risk_grid_phantom_vs_agent(
                multi_data=multi_data,
                agent_distances=fine_agent_distances,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "risk_grid_fine_1m_interval.pdf",
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
                dt=args.dt,
                velocity=args.velocity,
            )
            plt.close()

            # Generate 4x4 grid with phantom acceleration slices
            zero_acc_agent_distances = [15, 10, 5, 0]

            # Phantom acc = 0 (constant velocity)
            print(f"\nGenerating 4x4 grid with phantom acc=0: {zero_acc_agent_distances} m...")
            plot_risk_grid_phantom_acc(
                multi_data=multi_data,
                agent_distances=zero_acc_agent_distances,
                phantom_acc=0.0,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "risk_grid_phantom_acc_0_4x4.pdf",
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
                dt=args.dt,
                velocity=args.velocity,
            )
            plt.close()

            # Phantom acc = -1.0 (decelerating)
            print(f"\nGenerating 4x4 grid with phantom acc=-1.0: {zero_acc_agent_distances} m...")
            plot_risk_grid_phantom_acc(
                multi_data=multi_data,
                agent_distances=zero_acc_agent_distances,
                phantom_acc=-1.0,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "risk_grid_phantom_acc_-1_4x4.pdf",
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
                dt=args.dt,
                velocity=args.velocity,
            )
            plt.close()

            # Phantom acc = -3.0 (hard braking)
            print(f"\nGenerating 4x4 grid with phantom acc=-3.0: {zero_acc_agent_distances} m...")
            plot_risk_grid_phantom_acc(
                multi_data=multi_data,
                agent_distances=zero_acc_agent_distances,
                phantom_acc=-3.0,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "risk_grid_phantom_acc_-3_4x4.pdf",
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
                dt=args.dt,
                velocity=args.velocity,
            )
            plt.close()

            print(f"\nGenerating 4x4 grid with agent acc=0: {zero_acc_agent_distances} m...")
            plot_risk_grid_agent_zero_acc(
                multi_data=multi_data,
                agent_distances=zero_acc_agent_distances,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                output_path=output_dir / "risk_grid_agent_zero_acc_4x4.pdf",
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
                dt=args.dt,
                velocity=args.velocity,
            )
            plt.close()

            # Generate individual heatmaps for each distance
            for distance, data in multi_data.data.items():
                print(f"\nProcessing phantom distance {distance} m...")
                dist_output_dir = output_dir / f"dist_{distance}"
                dist_output_dir.mkdir(parents=True, exist_ok=True)

                risk = data.compute_risk_from_ttc(args.ttc_threshold, args.alpha)
                risk_mean = data.compute_risk_mean_over_acc(risk)

                # Single heatmap
                plot_risk_heatmap(
                    data=data,
                    timestep=args.timestep,
                    risk=risk_mean[args.timestep],
                    output_path=dist_output_dir / f"risk_heatmap_t{args.timestep:03d}.pdf",
                    use_japanese=args.japanese,
                    initial_distance=args.initial_distance,
                    phantom_distance=distance,
                )
                plt.close()

                # Risk vs acceleration
                plot_risk_vs_acceleration(
                    data=data,
                    timestep=args.timestep,
                    risk=risk,
                    phantom_vel_index=args.phantom_vel_idx,
                    phantom_acc_index=args.phantom_acc_idx,
                    output_path=dist_output_dir / f"risk_vs_acceleration_t{args.timestep:03d}.pdf",
                    use_japanese=args.japanese,
                    initial_distance=args.initial_distance,
                    phantom_distance=distance,
                )
                plt.close()

                # risk_of_ttc_mean timeseries
                print("  Generating risk_of_ttc_mean timeseries...")
                plot_risk_of_ttc_mean_timeseries(
                    data=data,
                    ttc_threshold=args.ttc_threshold,
                    alpha=args.alpha,
                    output_path=dist_output_dir / "risk_of_ttc_mean_timeseries.pdf",
                    use_japanese=args.japanese,
                    phantom_distance=distance,
                    dt=args.dt,
                )
                plt.close()

                # risk_of_ttc_mean multi-timestep comparison
                print("  Generating risk_of_ttc_mean multi-timestep comparison...")
                comparison_timesteps = [0, args.timestep, min(args.timestep * 2, data.num_timesteps - 1)]
                plot_risk_of_ttc_mean_multi_timestep(
                    data=data,
                    timesteps=comparison_timesteps,
                    ttc_threshold=args.ttc_threshold,
                    alpha=args.alpha,
                    output_path=dist_output_dir / "risk_of_ttc_mean_evolution.pdf",
                    use_japanese=args.japanese,
                    initial_distance=args.initial_distance,
                    phantom_distance=distance,
                    dt=args.dt,
                )
                plt.close()

                # Risk with phantom acceleration = 0 (constant velocity phantom)
                print("  Generating risk heatmap with phantom acc=0...")
                risk_phantom_zero = data.extract_phantom_zero_acc(risk)
                plot_risk_phantom_zero_acc(
                    data=data,
                    timestep=args.timestep,
                    risk=risk_phantom_zero[args.timestep],
                    output_path=dist_output_dir / f"risk_phantom_zero_acc_t{args.timestep:03d}.pdf",
                    use_japanese=args.japanese,
                    initial_distance=args.initial_distance,
                    phantom_distance=distance,
                    dt=args.dt,
                    velocity=args.velocity,
                )
                plt.close()

                # Risk with agent acceleration = 0 (constant velocity agent)
                print("  Generating risk heatmap with agent acc=0...")
                risk_agent_zero = data.extract_agent_zero_acc(risk)
                plot_risk_agent_zero_acc(
                    data=data,
                    timestep=args.timestep,
                    risk=risk_agent_zero[args.timestep],
                    output_path=dist_output_dir / f"risk_agent_zero_acc_t{args.timestep:03d}.pdf",
                    use_japanese=args.japanese,
                    initial_distance=args.initial_distance,
                    phantom_distance=distance,
                    dt=args.dt,
                    velocity=args.velocity,
                )
                plt.close()

                if args.generate_sequence:
                    heatmap_dir = dist_output_dir / "heatmaps"
                    image_paths = generate_risk_heatmap_sequence(
                        data=data,
                        output_dir=heatmap_dir,
                        ttc_threshold=args.ttc_threshold,
                        alpha=args.alpha,
                        use_japanese=args.japanese,
                        initial_distance=args.initial_distance,
                        phantom_distance=distance,
                    )

                    create_video_from_images(
                        image_paths=image_paths,
                        output_path=dist_output_dir / "risk_heatmap.mp4",
                        fps=20,
                    )

                    create_gif_from_images(
                        image_paths=image_paths,
                        output_path=dist_output_dir / "risk_heatmap.gif",
                        duration=100,
                    )

    else:
        # Single phantom mode (original behavior)
        print(f"\nLoading data from: {data_dir}")
        data = MetricsData(data_dir, args.num_timesteps)
        print(f"Loaded {data.num_timesteps} timesteps")

        risk = data.compute_risk_from_ttc(args.ttc_threshold, args.alpha)
        risk_mean = data.compute_risk_mean_over_acc(risk)

        print(f"\nGenerating heatmap for timestep {args.timestep}...")
        plot_risk_heatmap(
            data=data,
            timestep=args.timestep,
            risk=risk_mean[args.timestep],
            output_path=output_dir / f"risk_heatmap_t{args.timestep:03d}.pdf",
            use_japanese=args.japanese,
            initial_distance=args.initial_distance,
        )
        plt.close()

        print(f"\nGenerating risk vs acceleration plot for timestep {args.timestep}...")
        plot_risk_vs_acceleration(
            data=data,
            timestep=args.timestep,
            risk=risk,
            phantom_vel_index=args.phantom_vel_idx,
            phantom_acc_index=args.phantom_acc_idx,
            output_path=output_dir / f"risk_vs_acceleration_t{args.timestep:03d}.pdf",
            use_japanese=args.japanese,
            initial_distance=args.initial_distance,
        )
        plt.close()

        # risk_of_ttc_mean timeseries
        print("\nGenerating risk_of_ttc_mean timeseries...")
        plot_risk_of_ttc_mean_timeseries(
            data=data,
            ttc_threshold=args.ttc_threshold,
            alpha=args.alpha,
            output_path=output_dir / "risk_of_ttc_mean_timeseries.pdf",
            use_japanese=args.japanese,
            dt=args.dt,
        )
        plt.close()

        # risk_of_ttc_mean multi-timestep comparison
        print("\nGenerating risk_of_ttc_mean multi-timestep comparison...")
        comparison_timesteps = [0, args.timestep, min(args.timestep * 2, data.num_timesteps - 1)]
        plot_risk_of_ttc_mean_multi_timestep(
            data=data,
            timesteps=comparison_timesteps,
            ttc_threshold=args.ttc_threshold,
            alpha=args.alpha,
            output_path=output_dir / "risk_of_ttc_mean_evolution.pdf",
            use_japanese=args.japanese,
            initial_distance=args.initial_distance,
            dt=args.dt,
        )
        plt.close()

        # Risk with phantom acceleration = 0 (constant velocity phantom)
        print("\nGenerating risk heatmap with phantom acc=0...")
        risk_phantom_zero = data.extract_phantom_zero_acc(risk)
        plot_risk_phantom_zero_acc(
            data=data,
            timestep=args.timestep,
            risk=risk_phantom_zero[args.timestep],
            output_path=output_dir / f"risk_phantom_zero_acc_t{args.timestep:03d}.pdf",
            use_japanese=args.japanese,
            initial_distance=args.initial_distance,
            dt=args.dt,
            velocity=args.velocity,
        )
        plt.close()

        # Risk with agent acceleration = 0 (constant velocity agent)
        print("\nGenerating risk heatmap with agent acc=0...")
        risk_agent_zero = data.extract_agent_zero_acc(risk)
        plot_risk_agent_zero_acc(
            data=data,
            timestep=args.timestep,
            risk=risk_agent_zero[args.timestep],
            output_path=output_dir / f"risk_agent_zero_acc_t{args.timestep:03d}.pdf",
            use_japanese=args.japanese,
            initial_distance=args.initial_distance,
            dt=args.dt,
            velocity=args.velocity,
        )
        plt.close()

        if args.generate_sequence:
            print("\nGenerating heatmap sequence...")
            heatmap_dir = output_dir / "heatmaps"
            image_paths = generate_risk_heatmap_sequence(
                data=data,
                output_dir=heatmap_dir,
                ttc_threshold=args.ttc_threshold,
                alpha=args.alpha,
                use_japanese=args.japanese,
                initial_distance=args.initial_distance,
            )

            print("\nCreating video...")
            create_video_from_images(
                image_paths=image_paths,
                output_path=output_dir / "risk_heatmap.mp4",
                fps=20,
            )

            print("\nCreating GIF...")
            create_gif_from_images(
                image_paths=image_paths,
                output_path=output_dir / "risk_heatmap.gif",
                duration=100,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
