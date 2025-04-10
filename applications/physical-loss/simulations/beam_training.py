import os, sys
from numpy import zeros, ndarray
import Sofa

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from beam_sofa import BeamSofa, grid, forces


class BeamTraining(BeamSofa):

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

        if self.viewer:
            self.viewer.render()

        if self.compute_training_data:

            # Compute force vector to set as network input
            f = zeros((grid['n'], 3))
            for i in range(forces['nb']):
                cff = self.root.fem.getObject(f'CFF_{i}')
                f[cff.indices.value.copy()] = cff.forces.value.copy()
            f = f.reshape(*grid['res'], 3)

            # Compute displacement vector to set as network ground truth
            m_state = self.root.fem.getObject('GridMO')
            d = m_state.position.value.copy() - m_state.rest_position.value.copy()
            d = d.reshape(*grid['res'], 3)

            self.set_data(forces=f, displacement=d)
