import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LANDMARK_COLORS = {
    "Mesial": "#1f77b4",
    "Distal": "#ff7f0e",
    "Cusp": "#2ca02c",
    "InnerPoint": "#9467bd",
    "OuterPoint": "#d62728",
    "FacialPoint": "#8c564b",
}


def sample_indices(n_points, max_points=30000, seed=0):
    rng = np.random.default_rng(seed)
    if n_points <= max_points:
        return np.arange(n_points)
    return rng.choice(n_points, size=max_points, replace=False)


def set_axes_equal_3d(ax, points):
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def finish_3d_axis(ax, points, title, elev=22, azim=-65):
    set_axes_equal_3d(ax, points)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)


def plot_landmarks_3d(
    vertices,
    landmark_coords,
    landmark_labels,
    max_points=30000,
    seed=0,
    title="Scan brut avec landmarks",
    colors=None,
    figsize=(9, 7),
    elev=28,
    azim=-58,
    scan_color="#cfcfcf",
    scan_alpha=0.4,
    scan_size=0.7,
    landmark_size=52,
):
    """Plot a raw scan and its landmarks without depending on notebook globals."""
    vertices = np.asarray(vertices, dtype=np.float32)
    landmark_coords = np.asarray(landmark_coords, dtype=np.float32)
    landmark_labels = np.asarray(landmark_labels)
    colors = {} if colors is None else colors

    idx = sample_indices(len(vertices), max_points=max_points, seed=seed)
    scan_points = vertices[idx]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        scan_points[:, 0],
        scan_points[:, 1],
        scan_points[:, 2],
        s=scan_size,
        color=scan_color,
        alpha=scan_alpha,
        linewidths=0,
    )

    legend_handles = []
    for landmark_class in sorted(set(landmark_labels.tolist())):
        mask = landmark_labels == landmark_class
        pts = landmark_coords[mask]
        color = colors.get(str(landmark_class), "black")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=landmark_size,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            depthshade=False,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{landmark_class} ({len(pts)})",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=7,
            )
        )

    finish_3d_axis(ax, scan_points, title, elev=elev, azim=azim)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    return fig, ax


def plot_landmark_projections(
    vertices,
    landmark_coords,
    landmark_labels,
    colors=None,
    max_points=30000,
    seed=0,
    projection_specs=None,
    figsize=(18, 5),
    scan_color="#d4d4d4",
    scan_alpha=0.4,
    scan_size=0.45,
    landmark_size=34,
):
    """Plot XY, XZ and YZ landmark projections."""
    vertices = np.asarray(vertices, dtype=np.float32)
    landmark_coords = np.asarray(landmark_coords, dtype=np.float32)
    landmark_labels = np.asarray(landmark_labels)
    colors = {} if colors is None else colors

    if projection_specs is None:
        projection_specs = [
            (0, 1, "XY", "x", "y"),
            (0, 2, "XZ", "x", "z"),
            (1, 2, "YZ", "y", "z"),
        ]

    idx = sample_indices(len(vertices), max_points=max_points, seed=seed)
    scan_points = vertices[idx]

    fig, axes = plt.subplots(1, len(projection_specs), figsize=figsize, constrained_layout=True)
    if len(projection_specs) == 1:
        axes = [axes]

    classes = sorted(set(landmark_labels.tolist()))
    for ax, (a, b, title, xlabel, ylabel) in zip(axes, projection_specs):
        ax.scatter(
            scan_points[:, a],
            scan_points[:, b],
            s=scan_size,
            color=scan_color,
            alpha=scan_alpha,
            linewidths=0,
        )

        for landmark_class in classes:
            mask = landmark_labels == landmark_class
            pts = landmark_coords[mask]
            ax.scatter(
                pts[:, a],
                pts[:, b],
                s=landmark_size,
                color=colors.get(str(landmark_class), "black"),
                edgecolor="black",
                linewidth=0.35,
                label=landmark_class,
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cls,
            markerfacecolor=colors.get(str(cls), "black"),
            markeredgecolor="black",
            markersize=7,
        )
        for cls in classes
    ]
    if handles:
        fig.legend(handles=handles, loc="center right", frameon=False)

    return fig, axes


def plot_landmark_classes(
    vertices,
    landmark_coords,
    landmark_labels,
    colors=None,
    max_points=30000,
    seed=0,
    axes_pair=(0, 1),
    axis_names=("x", "y", "z"),
    n_cols=3,
    figsize_per_plot=(5, 4),
    scan_color="#d6d6d6",
    scan_alpha=0.4,
    scan_size=0.35,
    landmark_size=42,
):
    """Plot one 2D projection per landmark class."""
    vertices = np.asarray(vertices, dtype=np.float32)
    landmark_coords = np.asarray(landmark_coords, dtype=np.float32)
    landmark_labels = np.asarray(landmark_labels)
    colors = {} if colors is None else colors

    a, b = axes_pair
    projection_name = f"{axis_names[a].upper()}{axis_names[b].upper()}"

    idx = sample_indices(len(vertices), max_points=max_points, seed=seed)
    scan_points = vertices[idx]

    classes = sorted(set(landmark_labels.tolist()))
    n_rows = int(np.ceil(len(classes) / n_cols)) if classes else 1
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)

    for ax, landmark_class in zip(axes.ravel(), classes):
        ax.scatter(
            scan_points[:, a],
            scan_points[:, b],
            s=scan_size,
            color=scan_color,
            alpha=scan_alpha,
            linewidths=0,
        )

        mask = landmark_labels == landmark_class
        pts = landmark_coords[mask]
        color = colors.get(str(landmark_class), "black")
        ax.scatter(
            pts[:, a],
            pts[:, b],
            s=landmark_size,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )

        ax.set_title(f"{landmark_class} ({len(pts)}) - projection {projection_name}")
        ax.set_xlabel(axis_names[a])
        ax.set_ylabel(axis_names[b])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)

    for ax in axes.ravel()[len(classes) :]:
        ax.axis("off")

    return fig, axes
