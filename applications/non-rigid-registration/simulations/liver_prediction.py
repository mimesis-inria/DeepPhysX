from typing import Dict
import os, sys
import numpy as np

from DeepPhysX.utils.grid.sparse_grid import regular_grid_get_cell
from DeepPhysX.utils.grid.mapping import GridMapping

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .liver_training import LiverTraining


class LiverPrediction(LiverTraining):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.predict_at_each_step = kwargs.pop('predict_at_each_step', False)
        self.metrics = {'l2_err': [], 'l2_rel': [], 'mse_err': [], 'mse_rel': []}

    def init_visualization(self):

        # Add OGL model for fem node
        ogl_model = self.root.fem.visual.getObject('OGL')
        self.viewer.objects.add_sofa_mesh(positions_data=ogl_model.position,
                                          cells_data=ogl_model.triangles,
                                          alpha=0.9)
        self.viewer.objects.add_points(positions=self.root.pcd.getObject('MO').position.value, color='orange')

        # Add OGL model for network node
        ogl_model = self.root.net.visual.getObject('OGL')
        self.viewer.objects.add_sofa_mesh(positions_data=ogl_model.position,
                                          cells_data=ogl_model.triangles,
                                          color='orange', alpha=0.9, wireframe=True, line_width=2)

    def onSimulationInitDoneEvent(self, event):

        LiverTraining.onSimulationInitDoneEvent(self, event)

        self.grid_mapping = GridMapping(master_positions=self.root.net.getObject('SparseGridMO').position.value,
                                        master_cells=self.root.net.getObject('SparseGridTopo').hexahedra.value,
                                        slave_positions=self.root.net.visual.getObject('OGL').position.value)

    def onAnimateEndEvent(self, event):

        self.idx_step += 1

        # Update the PDC
        pcd_mo = self.root.pcd.getObject('MO')
        surface_mo = self.root.fem.surface.getObject('SurfaceMO')
        pcd_mo.position.value = surface_mo.position.value[self.pcd_indices]

        # Encode the PCD in the unet grid
        pcd_unet = np.zeros((self.unet_grid.position.value.shape[0], 1))
        for pos in self.root.pcd.getObject('MO').position.value:
            # Get the cell containing the pos
            idx = regular_grid_get_cell(regular_grid=self.unet_grid, pos=pos)
            if idx < self.unet_grid.hexahedra.value.shape[0]:
                # For each node of the cell, compute the distance to the considered point
                for node in self.unet_grid.hexahedra.value[idx]:
                    d = np.linalg.norm(self.unet_grid.position.value[node] - pos)
                    # Encode in the node the distance to the closest point
                    if pcd_unet[node] == 0 or d < pcd_unet[node]:
                        pcd_unet[node] = d

        # Compute displacement vector to set as unet ground truth
        sparse_grid = self.root.fem.getObject('SparseGridMO')
        regular_grid_rest_positions = np.zeros(self.liver_grid.position.value.shape, dtype=float)
        regular_grid_rest_positions[self.idx_sparse_to_regular] = sparse_grid.rest_position.value
        regular_grid_positions = np.zeros(self.liver_grid.position.value.shape, dtype=float)
        regular_grid_positions[self.idx_sparse_to_regular] = sparse_grid.position.value
        unet_displacement = np.zeros(self.unet_grid.position.value.shape)
        unet_displacement[self.idx_liver_to_unet] = regular_grid_positions - regular_grid_rest_positions

        self.set_data(pcd=pcd_unet, displacement=unet_displacement,
                      surface=self.root.fem.surface.getObject('SurfaceMO').position.value)

        if not self.predict_at_each_step:
            self.viewer.render()

    def apply_prediction(self, prediction: Dict[str, np.ndarray]):

        if self.idx_step % self.nb_intermediate_step == 0 or self.predict_at_each_step:

            disp = prediction['displacement']
            grid_disp = disp[self.idx_liver_to_unet]
            liver_disp = grid_disp[self.idx_sparse_to_regular]
            m_state = self.root.net.getObject('SparseGridMO')
            m_state.position.value = m_state.rest_position.value + liver_disp

            ogl_model = self.root.net.visual.getObject('OGL')
            ogl_model.position.value = self.grid_mapping.apply(master_positions=m_state.position.value)

            if self.viewer:
                self.viewer.objects.update_points(object_id=1, positions=self.root.pcd.getObject('MO').position.value)
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
