"""
#07 - Prediction
Launch a running session.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig

# Tutorial related imports
from T3_configuration import DummyEnvironment, net_config, dataset_config


def launch_prediction():
    # Adapt the Environment config to avoid using Client / Server Architecture
    env_config = BaseEnvironmentConfig(environment_class=DummyEnvironment,
                                       visualizer=None,
                                       simulations_per_step=1,
                                       use_dataset_in_environment=False,
                                       param_dict={'increment': 1},
                                       as_tcp_ip_client=False)
    # Create the Pipeline
    pipeline = BaseRunner(session_dir='sessions',
                          session_name='tutorial_online_training',
                          environment_config=env_config,
                          dataset_config=dataset_config,
                          network_config=net_config,
                          nb_steps=20)
    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_prediction()
