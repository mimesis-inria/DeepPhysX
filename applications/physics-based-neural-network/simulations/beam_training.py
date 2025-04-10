import os, sys
from numpy import zeros, ndarray
import Sofa

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from beam_simulation import BeamSimulation, grid


class BeamTraining(BeamSimulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def init_database(self):

        for field in ('forces', 'displacement'):
            self.add_data_field(field_name=field, field_type=ndarray)

    def init_visualization(self):

        ogl_model = self.root.fem.visual.getObject('OGL')
        self.viewer.objects.add_sofa_mesh(positions_data=ogl_model.position,
                                          cells_data=ogl_model.quads)

    def onAnimateEndEvent(self, event):

        if self.compute_training_data:

            if self.viewer:
                self.viewer.render()

            # Compute force vector to set as network input
            force_field = self.root.fem.getObject('CFF')
            forces = zeros((grid['n'], 3))
            forces[force_field.indices.value.copy()] = force_field.forces.value.copy()

            # Compute displacement vector to set as network ground truth
            m_state = self.root.fem.getObject('GridMO')
            displacement = m_state.position.value.copy() - m_state.rest_position.value.copy()

            self.set_data(forces=forces, displacement=displacement)

    def check_sample(self):

        # Check if the solver diverged
        if self.fem_node and not self.root.fem.getObject('ODESolver').converged.value:
            Sofa.Simulation.reset(self.root)
            return False
        return True
