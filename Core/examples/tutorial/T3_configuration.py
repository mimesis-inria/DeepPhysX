"""
#03 - Configurations
Define configurations for Environment, Network and Dataset
"""

# DeepPhysX related imports
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

# Tutorial related imports
from T1_environment import DummyEnvironment
from T2_network import DummyNetwork, DummyOptimization


# Create the Environment config
env_config = BaseEnvironmentConfig(environment_class=DummyEnvironment,      # The Environment class to create
                                   visualizer=None,                         # The Visualizer to use
                                   simulations_per_step=1,                  # The number of bus-steps to run
                                   use_dataset_in_environment=False,        # Dataset will not be sent to Environment
                                   param_dict={'increment': 1},             # Parameters to send at init
                                   as_tcp_ip_client=True,                   # Create a Client / Server architecture
                                   number_of_thread=3,                      # Number of Clients connected to Server
                                   ip_address='localhost',                  # IP address to use for communication
                                   port=10001)                              # Port number to use for communication

# Create the Network config
net_config = BaseNetworkConfig(network_class=DummyNetwork,                  # The Network class to create
                               optimization_class=DummyOptimization,        # The Optimization class to create
                               network_name='DummyNetwork',                 # Nickname of the Network
                               network_type='Dummy',                        # Type of the Network
                               save_each_epoch=False,                       # Do not save the network at each epoch
                               require_training_stuff=False,                # loss and optimizer can remain at None
                               lr=None,                                     # Learning rate
                               loss=None,                                   # Loss class
                               optimizer=None)                              # Optimizer class

# Create the Dataset config
dataset_config = BaseDatasetConfig(partition_size=1,                        # Max size of the Dataset
                                   shuffle_dataset=False)                   # Dataset should be shuffled
