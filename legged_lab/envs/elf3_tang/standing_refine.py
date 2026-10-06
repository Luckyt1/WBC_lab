"""Stationary refinement settings, isolated from the live 50k training task."""


def configure_standing_refinement(cfg):
    # Keep 65% locomotion samples, including side steps and both turn directions.
    cfg.commands.rel_standing_envs = .35
    cfg.commands.rel_turning_envs = .15
    cfg.commands.rel_lateral_envs = .15
    cfg.commands.initial_mode_fractions = None
    cfg.reward.standing_quiet_base.weight = 4.
    cfg.reward.standing_quiet_base.params = {'velocity_std': .03, 'yaw_rate_std': .10}
    cfg.reward.standing_base_motion.weight = -2.
    cfg.reward.standing_base_motion.params = {'velocity_scale': .08, 'yaw_rate_scale': .20}
    # Do not increase foot-motion/contact costs: recovery may require stepping.
    # Neutral arms match the first deployment Mod, while retaining moving-arm trials.
    cfg.arm_motion.amass_neutral_fraction = .50


def configure_standing_refinement_agent(cfg):
    cfg.policy.noise_std_bounds = (.05, .30)
    cfg.policy.init_noise_std = .30
    cfg.algorithm.entropy_coef = .001
    cfg.algorithm.learning_rate = 3.e-5
    cfg.algorithm.learning_rate_bounds = (1.e-5, 5.e-5)
