import torch

from legged_lab.utils.action_targets import raw_target_limit_excess


def test_in_range_targets_and_both_boundaries_have_no_cost():
    limits = torch.tensor([[[0., 2.], [-1., 1.]]])
    actions = torch.tensor([[-2., -1.], [0., 0.], [2., 1.]])
    excess = raw_target_limit_excess(actions, torch.tensor([[1., 0.]]), torch.tensor([.5, 1.]), limits)
    torch.testing.assert_close(excess, torch.zeros_like(actions))


def test_improvement_is_rewarded_even_when_motor_targets_are_identical():
    # Historical clipped-controller failure; the cost remains useful without clipping.
    actions = torch.tensor([[-30.], [-29.]])
    limits = torch.tensor([[[.048, 2.4827]]])
    executed = (.6 + actions * .213).clamp(min=.048, max=2.4827)
    torch.testing.assert_close(executed[0], executed[1])
    costs = raw_target_limit_excess(actions, torch.tensor([[.6]]), .213, limits)
    torch.testing.assert_close(costs[0] - costs[1], torch.tensor([.213]))
    assert (-.2 * costs[1] > -.2 * costs[0]).all()


def test_excess_is_joint_scaled_and_not_capped_at_global_action_clip():
    actions = torch.tensor([[-120., 120.], [-110., 110.]])
    limits = torch.tensor([[[-1., 1.], [-2., 2.]]])
    costs = raw_target_limit_excess(actions, torch.zeros(1, 2), torch.tensor([.2, .3]), limits)
    torch.testing.assert_close(costs, torch.tensor([[23., 34.], [21., 31.]]))
    assert (costs[0] > costs[1]).all()
