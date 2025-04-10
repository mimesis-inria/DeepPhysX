from DeepPhysX.pipelines import DataPipeline
from DeepPhysX.simulation import SimulationManager
from DeepPhysX.database import DatabaseManager

from simulations.beam_training import BeamTraining


# Create the simulation manager and the data manager
simu_manager = SimulationManager(simulation_class=BeamTraining,
                                 simulations_per_step=50,
                                 nb_parallel_env=5,
                                 use_viewer=True)
data_manager = DatabaseManager()

# Launch the data generation pipeline
DataPipeline(simulation_manager=simu_manager,
             database_manager=data_manager,
             batch_nb=100,
             batch_size=16).execute()
