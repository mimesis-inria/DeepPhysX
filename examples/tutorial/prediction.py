# Python related imports
from os.path import exists, join

# DeepPhysX related imports
from DeepPhysX.pipelines.prediction_pipeline import PredictionPipeline
from DeepPhysX.simulation.simulation_manager import SimulationManager
from DeepPhysX.database.database_manager import DatabaseManager
from DeepPhysX.networks.architectures.mlp import MLP
from DeepPhysX.networks.network_manager import NetworkManager

# Session imports
from simulation import SpringEnvironmentPrediction


if __name__ == '__main__':

    if not exists(join('sessions', 'training')):
        raise FileNotFoundError('You must run the "training.py" pipeline first.')

    # Create the simulation manager
    simulation_manager = SimulationManager(simulation_class=SpringEnvironmentPrediction,
                                           use_viewer=True)

    # Create the database manager
    database_manager = DatabaseManager(normalize=True)

    # Create the neural network manager
    network_manager = NetworkManager(network_architecture=MLP,
                                     network_kwargs={'dim_layers': [5, 5, 5, 1],
                                                     'out_shape': (1,)},
                                     data_forward_fields='state',
                                     data_backward_fields='displacement')

    # Launch the prediction pipeline
    pipeline = PredictionPipeline(simulation_manager=simulation_manager,
                                  database_manager=database_manager,
                                  network_manager=network_manager,
                                  session_dir='sessions',
                                  session_name='training',
                                  step_nb=-1)
    pipeline.execute()
