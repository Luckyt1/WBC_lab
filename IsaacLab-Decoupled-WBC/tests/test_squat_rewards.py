import torch

from legged_lab.utils.squat_rewards import height_coupled_knee_cost, standing_posture_cost


def test_knee_cost_encourages_flexion_from_elf3_stance_and_fades_at_target():
    limits = torch.tensor([[[-0.087266, 2.618], [-0.087266, 2.618]]])
    knees = torch.tensor([[0.6, 0.6], [1.0, 1.0], [1.265367, 1.265367]])
    costs = height_coupled_knee_cost(knees, limits, torch.full((3,), 0.25))
    assert costs[0] > costs[1] > costs[2]
    torch.testing.assert_close(costs[2], torch.tensor(0.), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        height_coupled_knee_cost(knees, limits, torch.zeros(3)), torch.zeros(3)
    )


def test_knee_cost_is_normalized_and_symmetric_not_a_signed_height_controller():
    limits = torch.tensor([[[-1., 3.], [0., 2.]]])
    knees = torch.tensor([[-1., 2.], [3., 0.], [1., 1.]])
    torch.testing.assert_close(
        height_coupled_knee_cost(knees, limits, torch.tensor([0.2, -0.2, 0.2])),
        torch.tensor([0.2, 0.2, 0.]),
    )
    # Equivalent physical coordinates must not change the normalized cost.
    torch.testing.assert_close(
        height_coupled_knee_cost(knees * 2 + 4, limits * 2 + 4, torch.ones(3)),
        height_coupled_knee_cost(knees, limits, torch.ones(3)),
    )


def test_posture_gate_uses_command_and_preserves_standing_l1():
    positions = torch.tensor([[0.2, -0.3]]).repeat(4, 1)
    defaults = torch.tensor([[0.1, 0.1]])
    torch.testing.assert_close(
        standing_posture_cost(positions, defaults, torch.tensor([0.8, 0.999, 1.0, 1.05]), 1.0),
        torch.tensor([0., 0., 0.5, 0.5]),
    )
