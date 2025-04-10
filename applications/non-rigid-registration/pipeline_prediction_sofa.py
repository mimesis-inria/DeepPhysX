from os import listdir, remove

from DeepPhysX.pipelines import SofaPredictionPipeline
from DeepPhysX.simulation import SimulationManager
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import UNet
from DeepPhysX.utils.grid.sparse_grid import get_scaled_regular_grid_resolution

from simulations.liver_prediction import LiverPrediction
from simulations.liver_simulation import p_grid


# Compute the UNet grid resolution
unet_resolution = get_scaled_regular_grid_resolution(grid_resolution=p_grid['res'],
                                                     grid_size=p_grid['max'] - p_grid['min'],
                                                     margin=p_grid['margin'])

# Create the simulation manager, the data manager and the network manager
simu_manager = SimulationManager(simulation_class=LiverPrediction,
                                 simulation_kwargs={'predict_at_each_step': True},
                                 use_viewer=True)
data_manager = DatabaseManager(normalize=True)
net_manager = NetworkManager(network_architecture=UNet,
                             network_kwargs={'input_size': unet_resolution,
                                             'nb_dims': 3,
                                             'nb_input_channels': 1,
                                             'nb_first_layer_channels': 128,
                                             'nb_output_channels': 3,
                                             'flatten_output': True,
                                             'nb_steps': 3,
                                             'two_sublayers': True,
                                             'border_mode': 'same',
                                             'skip_merge': False},
                             data_forward_fields='pcd',
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
