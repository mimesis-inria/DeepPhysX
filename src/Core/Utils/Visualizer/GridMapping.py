import numpy as np
from vedo import Mesh


class GridMapping:

    def __init__(self, master: Mesh, slave: Mesh):
        """
        Python implementation of a trilinear interpolation between a master grid and a slave mesh.
        """

        # Clone meshes
        self.master_points = master.points().copy()
        self.master_cells = np.array(master.cells().copy())
        self.slave_points = np.array(slave.points().copy())

        # Compute trilinear interpolation coordinates
        self.b, self.cells = self.__init_mapping()

    def __init_mapping(self):

        # 1. CLOSEST CELL OF MASTER FOR EACH POINT OF SLAVE
        # 1.1 Get each cell center
        cells_centers = self.master_points[self.master_cells].mean(axis=1)
        # 1.2 Compute distances to all centers for each slave point
        diff_to_grd = np.subtract(np.tile(cells_centers,
                                          (1, 1, self.slave_points.shape[0])).reshape((cells_centers.shape[0],
                                                                                       self.slave_points.shape[0],
                                                                                       cells_centers.shape[1])),
                                  self.slave_points)
        D = np.linalg.norm(diff_to_grd, axis=2).T
        # 1.3 Get the closest cell of master for each slave node
        cells = self.master_cells[np.argmin(D, axis=1)]

        # 2. BARYCENTRIC COORDINATES
        # 2.1 Get closest cell positions
        hexas = self.master_points[cells]
        # 2.2 Get the min and max corners of the cell
        anchors, corners = np.min(hexas, axis=1), np.max(hexas, axis=1)
        sizes = corners - anchors
        # 2.3 Compute thr relative position of the point in the cell
        bars = (self.slave_points - anchors) / sizes

        return bars, cells

    def apply(self, source_positions) -> np.ndarray:
        """
        Apply mapping between new master positions and slave with barycentric coordinates.
        """

        # 1. Permute cells columns, define other vectors
        H_T = source_positions[self.cells].T
        ones = np.ones_like(self.b) - self.b

        # 2. Interpolate slave points
        P = ((H_T[:, 6] * self.b[:, 0] + H_T[:, 7] * ones[:, 0]) * self.b[:, 1] +
             (H_T[:, 5] * self.b[:, 0] + H_T[:, 4] * ones[:, 0]) * ones[:, 1]) * self.b[:, 2] + \
            ((H_T[:, 2] * self.b[:, 0] + H_T[:, 3] * ones[:, 0]) * self.b[:, 1] +
             (H_T[:, 1] * self.b[:, 0] + H_T[:, 0] * ones[:, 0]) * ones[:, 1]) * ones[:, 2]

        return P.T
