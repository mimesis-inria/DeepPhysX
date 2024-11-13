# Python related imports
from os.path import exists
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.pipelines import TrainingPipeline
from DeepPhysX.simulation.simulation_config import SimulationConfig
from DeepPhysX.database.database_config import DatabaseConfig
from DeepPhysX.networks.architectures.mlp.mlp_config import MLPConfig

# Session related imports
from simulation import SpringEnvironment


if __name__ == '__main__':

    simulation_config = SimulationConfig(environment_class=SpringEnvironment,
                                          use_viewer=False,
                                          nb_parallel_env=1,
                                          simulations_per_step=5)

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_config = MLPConfig(lr=1e-4,
                               loss=MSELoss,
                               optimizer=Adam,
                               dim_layers=[5, 5, 2],
                               dim_output=2)

    # TODO: remove this line
    if exists('sessions/online_training'):
        from shutil import rmtree
        rmtree('sessions/online_training')

    # Dataset configuration with the path to the existing Dataset
    database_config = DatabaseConfig()

    # Create DataGenerator
    trainer = TrainingPipeline(network_config=network_config,
                               database_config=database_config,
                               simulation_config=simulation_config,
                               session_dir='sessions',
                               session_name='online_training',
                               epoch_nb=30,
                               batch_nb=1000,
                               batch_size=32)

    # Launch the training session
    trainer.execute()
