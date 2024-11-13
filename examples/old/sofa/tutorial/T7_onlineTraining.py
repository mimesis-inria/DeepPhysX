"""
#07 - Online Training
Launch a training session and Dataset production simultaneously.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Tutorial related imports
from T1_environment import DummyEnvironment
from T2_network import DummyNetwork, DummyOptimization, DummyTransformation


def launch_training():

    # Create the Environment config
    env_config = SofaEnvironmentConfig(environment_class=DummyEnvironment,
                                       as_tcp_ip_client=True,
                                       number_of_thread=3)

    # Create the Network config
    net_config = BaseNetworkConfig(network_class=DummyNetwork,
                                   optimization_class=DummyOptimization,
                                   data_transformation_class=DummyTransformation,
                                   network_name='DummyNetwork',
                                   network_type='Dummy',
                                   save_each_epoch=False,
                                   require_training_stuff=False)

    # Create the Dataset config
    database_config = BaseDatabaseConfig(max_file_size=1,
                                         normalize=False)

    # Create the Pipeline
    pipeline = BaseTraining(network_config=net_config,
                            database_config=database_config,
                            environment_config=env_config,
                            session_dir='sessions',
                            session_name='tutorial_online_training',
                            new_session=True,
                            epoch_nb=2,
                            batch_nb=20,
                            batch_size=10,
                            debug=True)

    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_training()
