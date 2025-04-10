from DeepPhysX.pipelines import PredictionPipeline
from DeepPhysX.simulation import SimulationManager
from DeepPhysX.database import DatabaseManager
from DeepPhysX.networks import NetworkManager
from DeepPhysX.networks.architectures import UNet

from simulations.beam_prediction import BeamPrediction, grid


# Create the simulation manager, the data manager and the network manager
simu_manager = SimulationManager(simulation_class=BeamPrediction,
                                 use_viewer=True)
data_manager = DatabaseManager(normalize=True)
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

# Launch the prediction pipeline
PredictionPipeline(simulation_manager=simu_manager,
                   database_manager=data_manager,
                   network_manager=net_manager,
                   session_name='training').execute()
