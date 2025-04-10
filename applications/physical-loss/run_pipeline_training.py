from os.path import join

import torch
from torch.optim import Adam

from DeepPhysX.pipelines import TrainingPipeline
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import UNet

from simulations.beam_training import grid
from phy_loss import PhysicalLoss


# Create the data manager and the network manager
data_manager = DatabaseManager(existing_dir=join('sessions', 'data_generation'),
                               shuffle_data=True, normalize=True)
net_manager = NetworkManager(network_architecture=UNet,
                             network_kwargs={'input_size': grid['res'],
                                             'nb_dims': 3,
                                             'nb_input_channels': 3,
                                             'nb_first_layer_channels': 128,
                                             'nb_output_channels': 3,
                                             'nb_steps': 3,
                                             'two_sublayers': True,
                                             'border_mode': 'same',
                                             'skip_merge': False},
                             data_forward_fields='forces',
                             data_backward_fields='displacement')

# Launch the training pipeline
trainer = TrainingPipeline(database_manager=data_manager,
                 network_manager=net_manager,
                 loss_fnc=PhysicalLoss,
                 loss_kwargs={'mu': 500, 'lmbd': 0.45},
                 optimizer=Adam,
                 optimizer_kwargs={'lr': 1e-5},
                 epoch_nb=50,
                 batch_nb=100,
                 batch_size=16)

# Get 100 predictions to initialize PhyLoss coefficients
preds, targets = [], []
data_lines = data_manager.get_data(batch_size=100)
data_fwd, data_bwd = net_manager.get_data(data_lines)
predict = net_manager.get_predict(batch_fwd=data_fwd)
net_manager.loss_fnc.init_loss_coeff(predict, data_bwd['displacement'])


trainer.execute()
