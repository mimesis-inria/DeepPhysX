"""
#06 - Offline Training
Launch a training session with an existing Dataset.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTraining import BaseTraining
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig

# Tutorial related imports
from T2_network import DummyNetwork, DummyOptimization, DummyTransformation


def launch_training():

    # Create the Network config
    net_config = BaseNetworkConfig(network_class=DummyNetwork,
                                   optimization_class=DummyOptimization,
                                   data_transformation_class=DummyTransformation,
                                   network_name='DummyNetwork',
                                   network_type='Dummy',
                                   save_each_epoch=False,
                                   require_training_stuff=False)

    # Create the Dataset config
    database_config = BaseDatabaseConfig(existing_dir='sessions/tutorial_data_generation',
                                         shuffle=False)
    # Create the Pipeline
    pipeline = BaseTraining(network_config=net_config,
                            database_config=database_config,
                            session_dir='sessions',
                            session_name='tutorial_offline_training',
                            new_session=True,
                            epoch_nb=2,
                            batch_nb=20,
                            batch_size=10,
                            debug=True)

    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_training()
