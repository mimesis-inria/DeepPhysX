"""
#06 - Online Training
Launch a training session and Dataset production simultaneously.
"""

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseTrainer import BaseTrainer

# Tutorial related imports
from T3_configuration import env_config, net_config, dataset_config


def launch_training():
    # Create the Pipeline
    pipeline = BaseTrainer(session_dir='sessions',
                           session_name='tutorial_online_training',
                           environment_config=env_config,
                           dataset_config=dataset_config,
                           network_config=net_config,
                           nb_epochs=2,
                           nb_batches=100,
                           batch_size=10)
    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_training()
