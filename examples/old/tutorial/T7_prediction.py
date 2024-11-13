"""
#07 - Prediction
Launch a running session.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig

# Tutorial related imports
from T1_environment import DummyEnvironment
from T2_network import DummyNetwork


def launch_prediction():

    # Create the Environment config
    env_config = BaseEnvironmentConfig(environment_class=DummyEnvironment)

    # Create the networks config
    net_config = BaseNetworkConfig(network_class=DummyNetwork)

    # Create the Pipeline
    pipeline = BasePrediction(network_config=net_config,
                              environment_config=env_config,
                              session_dir='sessions',
                              session_name='tutorial_online_training',
                              step_nb=20)

    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_prediction()
