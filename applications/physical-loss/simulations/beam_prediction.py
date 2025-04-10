from typing import Dict
import os, sys
from numpy import ndarray, zeros

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .beam_training import BeamTraining, grid, forces


class BeamPrediction(BeamTraining):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def init_visualization(self):

        # Add OGL model for fem node
        ogl_model = self.root.fem.visual.getObject('OGL')
        self.viewer.objects.add_sofa_mesh(positions_data=ogl_model.position,
                                          cells_data=ogl_model.quads,
                                          alpha=0.9)

        # Add OGL model for network node
        ogl_model = self.root.net.visual.getObject('OGL')
        self.viewer.objects.add_sofa_mesh(positions_data=ogl_model.position,
                                          cells_data=ogl_model.quads,
                                          color='orange', wireframe=True, line_width=2)

    def onAnimateEndEvent(self, event):

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

    def apply_prediction(self, prediction: Dict[str, ndarray]):

        # Get the computed displacement
        displacement = prediction['displacement']

        # Update the MO and the OGL model (otherwise the update with SimRender will be displayed at next time step)
        m_state = self.root.net.getObject('GridMO')
        m_state.position.value = m_state.rest_position.value + displacement.reshape(m_state.rest_position.value.shape)
        ogl_model = self.root.net.visual.getObject('OGL')
        ogl_model.position.value = m_state.position.value

        if self.viewer:
            self.viewer.render()
