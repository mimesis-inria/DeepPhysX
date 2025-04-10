from os import listdir, remove

from DeepPhysX.pipelines import SofaPredictionPipeline
from DeepPhysX.simulation import SimulationManager
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import MLP

from simulations.beam_prediction import BeamPrediction, grid


# Create the simulation manager, the data manager and the network manager
simu_manager = SimulationManager(simulation_class=BeamPrediction,
                                 simulation_kwargs={'compute_fem_solution': True},
                                 use_viewer=True)
data_manager = DatabaseManager(normalize=True)
net_manager = NetworkManager(network_architecture=MLP,
                             network_kwargs={'dim_layers': [grid['n'] * 3 for _ in range(5)],
                                             'out_shape': (grid['n'], 3),
                                             'use_biases': True},
                             data_forward_fields='forces',
                             data_backward_fields='displacement')

# Launch the prediction pipeline
SofaPredictionPipeline(simulation_manager=simu_manager,
                       database_manager=data_manager,
                       network_manager=net_manager,
                       session_name='training').execute()

# Delete log files
for file in listdir():
    if file.endswith('.ini') or file.endswith('.log'):
        remove(file)
