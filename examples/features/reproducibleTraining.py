"""
offlineTraining.py
Run the pipeline BaseTrainer to create a training session with an existing Dataset.
"""

# Python related imports
import os

from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX.Core.Utils.yamlUtils import unpack_pipeline_config, BaseYamlExporter, BaseYamlLoader
from DeepPhysX.Core.Visualizer.VedoVisualizer import VedoVisualizer
from torch.nn import MSELoss
from torch.optim import Adam

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig
from Environment import MeanEnvironment


def set_pipepline_config():
    # Some int parameters
    nb_points = 30
    dimension = 3
    # The config object and its keyword arguments are put in a tuple
    environment_config = (BaseEnvironmentConfig, dict(environment_class=MeanEnvironment,
                                                      visualizer=VedoVisualizer,
                                                      param_dict={'constant': False,
                                                                  'data_size': [nb_points, dimension],
                                                                  'sleep': False,
                                                                  'allow_requests': True},
                                                      as_tcp_ip_client=True,
                                                      number_of_thread=10))
    network_config = (FCConfig, dict(loss=MSELoss,
                                     lr=1e-3,
                                     optimizer=Adam,
                                     dim_layers=[nb_points * dimension, nb_points * dimension, dimension],
                                     dim_output=dimension))

    dataset_config = (BaseDatasetConfig, dict(shuffle_dataset=True,
                                              normalize=False))

    # This is the config that will be passed to the Trainer
    # The config will be instantiated using unpack_pipeline_config
    pipeline_config = dict(session_dir='sessions',
                           session_name='offline_training',
                           environment_config=environment_config,
                           dataset_config=dataset_config,
                           network_config=network_config,
                           nb_epochs=1,
                           nb_batches=500,
                           batch_size=10)
    return pipeline_config


if __name__ == '__main__':
    pipeline_config = set_pipepline_config()
    # Initialize the network config and dataset config then initialize the Trainer
    pipeline = BaseTrainer(**unpack_pipeline_config(pipeline_config))
    # Save the Pipeline config in the current session dir
    config_export_path = os.path.join(pipeline.manager.session_dir, 'conf.yml')
    BaseYamlExporter(config_export_path, pipeline_config)
    # Loads the pipeline config from the file
    loaded_pipeline_config = BaseYamlLoader(config_export_path)
    if loaded_pipeline_config == pipeline_config:
        print("The loaded config is equal to the original config.")