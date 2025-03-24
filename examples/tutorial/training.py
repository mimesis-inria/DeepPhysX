# Python related imports
from os.path import exists, join
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.pipelines import DataPipeline, TrainingPipeline
from DeepPhysX.simulation import SimulationManager
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import MLP

# Session related imports
from simulation import SpringEnvironment

if __name__ == '__main__':

    # Create the simulation manager
    simulation_manager = SimulationManager(simulation_class=SpringEnvironment,
                                           use_viewer=True,
                                           nb_parallel_env=2,
                                           simulations_per_step=5)

    # Launch the data generation pipeline
    data_session = join('sessions', 'data_generation')
    if not exists(data_session):
        data_pipeline = DataPipeline(simulation_manager=simulation_manager,
                                     database_manager=DatabaseManager(),
                                     batch_nb=1000,
                                     batch_size=16)
        data_pipeline.execute()

    # Create the database manager
    database_manager = DatabaseManager(existing_dir=data_session,
                                       shuffle_data=True,
                                       normalize=True)

    # Create the neural network manager
    network_manager = NetworkManager(network_architecture=MLP,
                                     network_kwargs={'dim_layers': [5, 5, 5, 1],
                                                     'out_shape': (1,)},
                                     data_forward_fields='state',
                                     data_backward_fields='displacement')

    # Launch the training pipeline
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
    trainer.execute()
