import os, sys
from numpy import zeros, ndarray
from numpy.linalg import norm
import Sofa

from DeepPhysX.utils.grid.sparse_grid import regular_grid_get_cell

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from liver_simulation import LiverSimulation


class LiverTraining(LiverSimulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def init_database(self):

        for field in ('pcd', 'displacement', 'surface'):
            self.add_data_field(field_name=field, field_type=ndarray)

    def init_visualization(self):

        ogl_model = self.root.fem.visual.getObject('OGL')
        self.viewer.objects.add_sofa_mesh(positions_data=ogl_model.position, cells_data=ogl_model.triangles)
        self.viewer.objects.add_points(positions=self.root.pcd.getObject('MO').position.value, color='orange')

    def onAnimateEndEvent(self, event):

        LiverSimulation.onAnimateEndEvent(self, event)

        if self.compute_training_data:

            if self.viewer:
                self.viewer.objects.update_points(object_id=1, positions=self.root.pcd.getObject('MO').position.value)
                self.viewer.render()

            # Encode the PCD in the unet grid
            pcd_unet = zeros((self.unet_grid.position.value.shape[0], 1))
            for pos in self.root.pcd.getObject('MO').position.value:
                # Get the cell containing the pos
                idx = regular_grid_get_cell(regular_grid=self.unet_grid, pos=pos)
                if idx < self.unet_grid.hexahedra.value.shape[0]:
                    # For each node of the cell, compute the distance to the considered point
                    for node in self.unet_grid.hexahedra.value[idx]:
                        d = norm(self.unet_grid.position.value[node] - pos)
                        # Encode in the node the distance to the closest point
                        if pcd_unet[node] == 0 or d < pcd_unet[node]:
                            pcd_unet[node] = d

            # Compute displacement vector to set as unet ground truth
            sparse_grid = self.root.fem.getObject('SparseGridMO')
            regular_grid_rest_positions = zeros(self.liver_grid.position.value.shape, dtype=float)
            regular_grid_rest_positions[self.idx_sparse_to_regular] = sparse_grid.rest_position.value
            regular_grid_positions = zeros(self.liver_grid.position.value.shape, dtype=float)
            regular_grid_positions[self.idx_sparse_to_regular] = sparse_grid.position.value
            unet_displacement = zeros(self.unet_grid.position.value.shape)
            unet_displacement[self.idx_liver_to_unet] = regular_grid_positions - regular_grid_rest_positions

            self.set_data(pcd=pcd_unet, displacement=unet_displacement,
                          surface=self.root.fem.surface.getObject('SurfaceMO').position.value)
