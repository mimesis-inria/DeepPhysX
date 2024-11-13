"""
#03 - Configurations
Define configurations for Environment, networks and Dataset
"""

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig

# Tutorial related imports
from T1_environment import DummyEnvironment
from T2_network import DummyNetwork, DummyOptimization, DummyTransformation

# Create the Environment config
env_config = BaseEnvironmentConfig(environment_class=DummyEnvironment,  # The Environment class to create
                                   as_tcp_ip_client=True,               # Create a Client / Server architecture
                                   number_of_thread=3,                  # Number of Clients connected to Server
                                   ip_address='localhost',              # IP address to use for communication
                                   port=10001,                          # Port number to use for communication
                                   simulations_per_step=1,              # The number of bus-steps to run
                                   load_samples=False,                  # Load samples from Database to Environment
                                   only_first_epoch=True,               # Use the Environment on the first epoch only
                                   always_produce=False)                # Environment is always producing data

# Create the networks config
net_config = BaseNetworkConfig(network_class=DummyNetwork,                     # The networks class to create
                               optimization_class=DummyOptimization,           # The Optimization class to create
                               data_transformation_class=DummyTransformation,  # The DataTransformation class to create
                               network_dir=None,                               # Path to an existing networks repository
                               network_name='DummyNetwork',                    # Nickname of the networks
                               network_type='Dummy',                           # Type of the networks
                               which_network=-1,                               # The index of networks to load
                               save_each_epoch=False,                          # Do not save the networks at each epoch
                               data_type='float32',                            # Training data type
                               require_training_stuff=False,                   # loss and optimizer can remain at None
                               lr=None,                                        # Learning rate
                               loss=None,                                      # Loss class
                               optimizer=None)                                 # Optimizer class

# Create the Dataset config
database_config = BaseDatabaseConfig(existing_dir=None,              # Path to an existing Database
                                     mode='training',                # Database mode
                                     max_file_size=1,                # Max size of the Dataset (Gb)
                                     shuffle=False,                  # Dataset should be shuffled
                                     normalize=False,                # Database should be normalized
                                     recompute_normalization=False)  # Normalization should be recomputed at loading
