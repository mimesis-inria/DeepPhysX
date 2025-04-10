from os.path import join
from torch.nn import MSELoss
from torch.optim import Adam

from DeepPhysX.pipelines import TrainingPipeline
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import UNet
from DeepPhysX.utils.grid.sparse_grid import get_scaled_regular_grid_resolution

from simulations.liver_simulation import p_grid


# Compute the UNet grid resolution
unet_resolution = get_scaled_regular_grid_resolution(grid_resolution=p_grid['res'],
                                                     grid_size=p_grid['max'] - p_grid['min'],
                                                     margin=p_grid['margin'])


# Create the data manager and the network manager
data_manager = DatabaseManager(existing_dir=join('sessions', 'data_generation'),
                               shuffle_data=True, normalize=True)
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

# Launch the training pipeline
TrainingPipeline(database_manager=data_manager,
                 network_manager=net_manager,
                 loss_fnc=MSELoss,
                 optimizer=Adam,
                 optimizer_kwargs={'lr': 1e-5},
                 epoch_nb=100,
                 batch_nb=500,
                 batch_size=32).execute()
