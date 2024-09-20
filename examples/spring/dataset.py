from DeepPhysX.pipelines.core.data_generation import DataGeneration
from DeepPhysX.simulation.core.base_environment_config import BaseEnvironmentConfig
from DeepPhysX.database.database_config import DatabaseConfig

# Session related imports
from simulation import SpringEnvironment

if __name__ == '__main__':

    # Environment configuration
    environment_config = BaseEnvironmentConfig(environment_class=SpringEnvironment,
                                               visualizer=None,
                                               nb_parallel_env=2)
    # Database configuration
    database_config = DatabaseConfig()

    # Create DataGenerator
    data_generator = DataGeneration(environment_config=environment_config,
                                    database_config=database_config,
                                    batch_nb=100,
                                    batch_size=32)

    # Launch the training session
    data_generator.execute()
