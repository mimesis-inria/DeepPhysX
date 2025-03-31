import numpy as np

class GridMapping:

    def __init__(self,
                 master_positions: np.ndarray,
                 master_cells: np.ndarray,
                 slave_positions: np.ndarray):
        """
        Python implementation of a trilinear interpolation between a master haxahedron grid and a slave mesh.

        :param master_positions: Array of positions of the master grid.
        :param master_cells: Array of cells of the master grid.
        :param slave_positions: Array of positions of the slave mesh.
        """

        # Clone meshes
        master_points = np.array(master_positions)
        master_cells = np.array(master_cells)
        slave_points = np.array(slave_positions)

        # Compute trilinear interpolation coordinates
        # 1. CLOSEST CELL OF MASTER FOR EACH POINT OF SLAVE
        # 1.1 Get each cell center
        cells_centers = master_points[master_cells].mean(axis=1)
        # 1.2 Compute distances to all centers for each slave point
        diff_to_grd = np.subtract(np.tile(cells_centers,
                                          (1, 1, slave_points.shape[0])).reshape((cells_centers.shape[0],
                                                                                       slave_points.shape[0],
                                                                                       cells_centers.shape[1])),
                                  slave_points)
        d = np.linalg.norm(diff_to_grd, axis=2).T
        # 1.3 Get the closest cell of master for each slave node
        self.cells = master_cells[np.argmin(d, axis=1)]

        # 2. BARYCENTRIC COORDINATES
        # 2.1 Get closest cell positions
        hexas = master_points[self.cells]
        # 2.2 Get the min and max corners of the cell
        anchors, corners = np.min(hexas, axis=1), np.max(hexas, axis=1)
        sizes = corners - anchors
        # 2.3 Compute thr relative position of the point in the cell
        self.b = (slave_points - anchors) / sizes

    def apply(self, master_positions: np.ndarray) -> np.ndarray:
        """
        Apply the mapping to the slave mesh with barycentric coordinates regarding new master grid positions.

        :param master_positions: New array of positions of the master grid.
        """

        # 1. Permute cells columns, define other vectors
        H_T = master_positions[self.cells].T
        ones = np.ones_like(self.b) - self.b

        # 2. Interpolate slave points
        P = ((H_T[:, 6] * self.b[:, 0] + H_T[:, 7] * ones[:, 0]) * self.b[:, 1] +
             (H_T[:, 5] * self.b[:, 0] + H_T[:, 4] * ones[:, 0]) * ones[:, 1]) * self.b[:, 2] + \
            ((H_T[:, 2] * self.b[:, 0] + H_T[:, 3] * ones[:, 0]) * self.b[:, 1] +
             (H_T[:, 1] * self.b[:, 0] + H_T[:, 0] * ones[:, 0]) * ones[:, 1]) * ones[:, 2]

        return P.T
