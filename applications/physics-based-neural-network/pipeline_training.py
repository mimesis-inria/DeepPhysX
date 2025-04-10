from os.path import join
from torch.nn import MSELoss
from torch.optim import Adam

from DeepPhysX.pipelines import TrainingPipeline
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import MLP

from simulations.beam_simulation import grid


# Create the data manager and the network manager
data_manager = DatabaseManager(existing_dir=join('sessions', 'data_generation'),
                               shuffle_data=True, normalize=True)
net_manager = NetworkManager(network_architecture=MLP,
                             network_kwargs={'dim_layers': [grid['n'] * 3 for _ in range(5)],
                                             'out_shape': (grid['n'], 3),
                                             'use_biases': True},
                             data_forward_fields='forces',
                             data_backward_fields='displacement')

# Launch the training pipeline
TrainingPipeline(database_manager=data_manager,
                 network_manager=net_manager,
                 loss_fnc=MSELoss,
                 optimizer=Adam,
                 optimizer_kwargs={'lr': 1e-5},
                 epoch_nb=400,
                 batch_nb=500,
                 batch_size=32).execute()
