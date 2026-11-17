"""Shared setup for controlled GL-165 previews, recordings and evaluations."""

from pathlib import Path


def configure_controlled_evaluation(cfg):
    """Disable observation noise and random disturbances for repeatable playback."""
    cfg.noise.add_noise = False
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.physics_material = None
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.reset_base.params.update(pose_range={}, velocity_range={})
    cfg.domain_rand.events.reset_robot_joints.params.update(
        position_range=(1., 1.), velocity_range=(0., 0.))


def load_policy_runner(env, checkpoint):
    """Load a policy using its saved agent config after Isaac Sim has started."""
    from isaaclab.utils.io import load_yaml
    from rsl_rl.runners import OnPolicyRunner

    checkpoint = Path(checkpoint)
    config = load_yaml(str(checkpoint.parent / 'params/agent.yaml'))
    config['device'] = env.device
    runner = OnPolicyRunner(env, config, log_dir=None, device=env.device)
    runner.load(str(checkpoint), load_optimizer=False)
    return runner
