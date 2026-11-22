"""Inspect TorchRL API compatibility without changing trained checkpoints.

Passing this probe means the environment adapter works, NOT that the current
training loops use TorchRL objective modules. See STAGE7_GUIDE.md.
"""
from importlib.metadata import version


def main():
    from torchrl.envs import GymWrapper
    from torchrl.objectives import DQNLoss, ClipPPOLoss, A2CLoss, ReinforceLoss
    from torchrl.envs.utils import check_env_specs
    from environment import PatientNavigationEnv
    print("TorchRL version:", version("torchrl"))
    print("Official loss classes are importable:", ", ".join(
        item.__name__ for item in (DQNLoss, ClipPPOLoss, A2CLoss, ReinforceLoss)))
    env = GymWrapper(PatientNavigationEnv(), categorical_action_encoding=True)
    try:
        check_env_specs(env)
        env.set_seed(42)
        td = env.reset()
        print("Environment reset keys:", list(td.keys()))
        print("Action spec:", env.action_spec)
        print("TorchRL env specs: PASS")
    finally:
        env.close()
    print("Training implementation status: PyTorch losses with TorchRL GymWrapper.")
    print("A genuine TorchRL-objectives rewrite would change training and require rerunning the 40 experiments.")

if __name__ == "__main__":
    main()
