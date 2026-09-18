import unittest

from legged_lab.utils.joint_layout import resolve_action_scales, resolve_joint_layout


class JointLayoutTests(unittest.TestCase):
    def test_interleaved_usd_order_preserves_motor_targets_and_held_joints(self):
        usd_names = ["arm_r", "hip_l", "head", "waist", "arm_l", "hip_r"]
        policy = ["hip_l", "hip_r", "waist"]
        indices, inverse = resolve_joint_layout(usd_names, policy, ["arm_l", "arm_r"])
        defaults = [10, 20, 30, 40, 50, 60]
        targets = [defaults[i] for i in indices]
        for i, delta in enumerate([1, 2, 3]):
            targets[i] += delta
        self.assertEqual([targets[i] for i in inverse], [10, 21, 30, 43, 50, 62])

    def test_wrong_robot_or_overlapping_joints_fail_early(self):
        for action, arms in [(["missing"], []), (["hip"], ["hip"]), ([], [])]:
            with self.subTest(action=action, arms=arms), self.assertRaises(ValueError):
                resolve_joint_layout(["hip", "arm"], action, arms)

    def test_scales_follow_policy_order_and_reject_ambiguous_patterns(self):
        scales = {".*_hip.*": 0.2, "waist.*": 0.1}
        self.assertEqual(resolve_action_scales(["waist_y", "l_hip_y"], scales), [0.1, 0.2])
        with self.assertRaises(ValueError):
            resolve_action_scales(["ankle"], scales)
        with self.assertRaises(ValueError):
            resolve_action_scales(["waist_y"], {".*": 0.2, "waist.*": 0.1})


if __name__ == "__main__":
    unittest.main()
