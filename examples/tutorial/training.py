# Python related imports
from os.path import exists, join
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.pipelines import TrainingPipeline
from DeepPhysX.simulation import SimulationManager
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import MLP

# Session related imports
from simulation import SpringEnvironment

if __name__ == '__main__':

    if not exists(join('sessions', 'data_generation')):
        raise FileNotFoundError('You must run the "data_generation.py" pipeline first.')

    # Create the simulation manager
    simulation_manager = SimulationManager(simulation_class=SpringEnvironment,
                                           use_viewer=True,
                                           nb_parallel_env=2,
                                           simulations_per_step=5)

    # Create the database manager
    database_manager = DatabaseManager(existing_dir=join('sessions', 'data_generation'),
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
                               session_name='training',
                               epoch_nb=20,
                               batch_nb=1000,
                               batch_size=16)
    trainer.execute()
