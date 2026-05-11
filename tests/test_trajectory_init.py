import numpy as np

from src.trajectory_init import (
    initialize_constant_velocity_rollout,
    initialize_straight_line_trajectory,
)


def test_straight_line_trajectory_shape_and_endpoints():
    traj = initialize_straight_line_trajectory([0.1, 0.2], [0.9, 0.2], 10)
    assert traj.shape == (11, 2)
    assert np.allclose(traj[0], [0.1, 0.2])
    assert np.allclose(traj[-1], [0.9, 0.2])


def test_constant_velocity_rollout():
    traj = initialize_constant_velocity_rollout([0.0, 0.0], [1.0, 0.0], 4, 0.5)
    assert np.allclose(traj[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
