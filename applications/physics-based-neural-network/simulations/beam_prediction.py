from typing import Dict
import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .beam_training import BeamTraining, grid


class BeamPrediction(BeamTraining):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.fem_node = kwargs.pop('compute_fem_solution', False)
        self.net_node = True
        self.metrics = {'l2_err': [], 'l2_rel': [], 'mse_err': [], 'mse_rel': []}

    def init_visualization(self):

        # Add OGL model for fem node
        if self.fem_node:
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

            node = self.root.fem if self.fem_node else self.root.net

            # Compute force vector to set as network input
            force_field = node.getObject('CFF')
            forces = np.zeros((grid['n'], 3))
            forces[force_field.indices.value.copy()] = force_field.forces.value.copy()

            # Compute displacement vector to set as network ground truth
            m_state = node.getObject('GridMO')
            displacement = m_state.position.value.copy() - m_state.rest_position.value.copy()

            self.set_data(forces=forces, displacement=displacement)

    def apply_prediction(self, prediction: Dict[str, np.ndarray]):

        # Get the computed displacement
        displacement = prediction['displacement']

        # Update the MO and the OGL model (otherwise the update with SimRender will be displayed at next time step)
        m_state = self.root.net.getObject('GridMO')
        m_state.position.value = m_state.rest_position.value + displacement
        ogl_model = self.root.net.visual.getObject('OGL')
        ogl_model.position.value = m_state.position.value

        if self.viewer:
            self.viewer.render()

        # if self.fem_node and self.net_node:
        #     self.compute_metrics(displacement)

    def compute_metrics(self, prediction):

        prediction = prediction
        ground_truth = self.data['displacement']

        if np.linalg.norm(ground_truth) > 1 and self.root.fem.getObject('ODESolver').converged.value:
            error = (ground_truth - prediction).reshape(-1)
            self.metrics['l2_err'].append(np.linalg.norm(error))            
            self.metrics['l2_rel'].append(np.linalg.norm(error) / np.linalg.norm(ground_truth))
            mse_err = (error.T @ error) / error.shape[0]
            mse_def = (ground_truth.reshape(-1).T @ ground_truth.reshape(-1)) / ground_truth.shape[0]
            self.metrics['mse_err'].append(mse_err)
            self.metrics['mse_rel'].append(mse_err / mse_def)

    def close(self):

        if len(self.metrics['l2_err']) > 0:

            print("\nL2 ERROR METRICS")
            print(f"\t- Distribution : {np.mean(self.metrics['l2_err'])} ± {np.std(self.metrics['l2_err'])} mm")
            print(f"\t- Extrema : {np.min(self.metrics['l2_err'])} -> {np.max(self.metrics['l2_err'])} mm")
            print(f"\t- Relative Distribution : {1e2 * np.mean(self.metrics['l2_rel'])} ± {1e2 * np.std(self.metrics['l2_rel'])} %")
            print(f"\t- Relative Extrema : {1e2 * np.min(self.metrics['l2_rel'])} -> {1e2 * np.max(self.metrics['l2_rel'])} %")

            print("\nMSE METRICS")
            print(f"\t- Distribution : {np.mean(self.metrics['mse_err'])} ± {np.std(self.metrics['mse_err'])} mm")
            print(f"\t- Extrema : {np.min(self.metrics['mse_err'])} -> {np.max(self.metrics['mse_err'])} mm")
            print(f"\t- Relative Distribution : {1e2 * np.mean(self.metrics['mse_rel'])} ± {1e2 * np.std(self.metrics['mse_rel'])} %")
            print(f"\t- Relative Extrema : {1e2 * np.min(self.metrics['mse_rel'])} -> {1e2 * np.max(self.metrics['mse_rel'])} %")
