# Python related imports
from os.path import exists
from shutil import rmtree
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.pipelines.data_pipeline import DataPipeline
from DeepPhysX.pipelines.training_pipeline import TrainingPipeline
from DeepPhysX.simulation.simulation_manager import SimulationManager
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.networks.architecture.mlp import MLP
from DeepPhysX.networks.network_manager import NetworkManager

# Session related imports
from simulation import SpringEnvironment

if __name__ == '__main__':

    if not exists('sessions/data_generation'):

        # Environment configuration
        simulation_manager = SimulationManager(simulation_class=SpringEnvironment,
                                               use_viewer=False,
                                               nb_parallel_env=1,
                                               simulations_per_step=5)

        # Create DataGenerator
        data_generator = DataPipeline(simulation_manager=simulation_manager,
                                      database_manager=DatabaseManager(),
                                      batch_nb=1000,
                                      batch_size=16)

        # Launch the training session
        data_generator.execute()

    # TODO: remove this line
    if exists('sessions/offline_training'):
        rmtree('sessions/offline_training')

    # Fully Connected configuration (the number of neurones on the first and last layer is defined by the total amount
    # of parameters in the input and the output vectors respectively)
    network_manager = NetworkManager(network_architecture=MLP,
                                     network_kwargs={'dim_layers': [5, 5, 5, 1],
                                                     'out_shape': (1,)},
                                     data_forward_fields='state',
                                     data_backward_fields='displacement')

    # Dataset configuration with the path to the existing Dataset
    database_manager = DatabaseManager(existing_dir='sessions/data_generation/',
                                       shuffle=True,
                                       normalize=True)

    # Create DataGenerator
    trainer = TrainingPipeline(network_manager=network_manager,
                               loss_fnc=MSELoss,
                               optimizer=Adam,
                               optimizer_kwargs={'lr': 1e-4},
                               database_manager=database_manager,
                               session_dir='sessions',
                               session_name='offline_training',
                               epoch_nb=20,
                               batch_nb=1000,
                               batch_size=16)

    # # Launch the training session
    trainer.execute()
