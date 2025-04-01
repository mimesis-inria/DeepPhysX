# Python related imports
from os.path import join

# DeepPhysX related imports
from DeepPhysX.pipelines import DataPipeline
from DeepPhysX.simulation import SimulationManager
from DeepPhysX.database import DatabaseManager

# Session related imports
from simulation import SpringEnvironment

if __name__ == '__main__':

    # Create the simulation manager
    simulation_manager = SimulationManager(simulation_class=SpringEnvironment,
                                           use_viewer=True,
                                           nb_parallel_env=1,
                                           simulations_per_step=5)

    # Launch the data generation pipeline
    data_session = join('sessions', 'data_generation')
    data_pipeline = DataPipeline(simulation_manager=simulation_manager,
                                 database_manager=DatabaseManager(),
                                 batch_nb=1000,
                                 batch_size=16)
    data_pipeline.execute()
