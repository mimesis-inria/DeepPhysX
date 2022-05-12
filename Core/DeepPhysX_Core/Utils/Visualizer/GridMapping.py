import numpy as np
from vedo import Mesh


class GridMapping:

    def __init__(self, master: Mesh, slave: Mesh):
        """
        Python implementation of a trilinear interpolation between a master grid and a slave mesh.
        """

        # Clone meshes
        self.master = master.clone()
        self.slave = slave.clone()

        # Compute trilinear interpolation coordinates
        self.bar_coord, self.cells = self.init_mapping()

    def init_mapping(self):
        """
        Compute barycentric coordinates of slave to master.
        """

        # 1. CLOSEST CELL OF MASTER FOR EACH POINT OF SLAVE
        # 1.1 Get each cell center
        cells_centers = self.master.points()[self.master.cells()].mean(axis=1)
        # 1.2 Compute distances to all centers for each slave point
        D = np.array([np.linalg.norm(cells_centers - p, axis=1) for p in self.slave.points()])
        # 1.3 Get the closest cell of master for each slave node
        cells = np.array(self.master.cells())[np.argmin(D, axis=1)]

        # 2. BARYCENTRIC COORDINATES
        bar = []
        for p, cell in zip(self.slave.points(), cells):
            # 2.1 Get closest cell positions
            hexa = self.master.points()[np.array(cell)]
            # 2.2 Get the min and max corners of the cell
            anchor, corner = np.min(hexa, axis=0), np.max(hexa, axis=0)
            size = corner - anchor
            # 2.3 Compute relative position of the point in the cell
            bar.append((p - anchor) / size)

        return bar, cells

    def apply(self, source_positions):
        """
        Apply mapping between new master positions and slave with barycentric coordinates.
        """

        # 1. Permute cells columns, define other vectors
        H = source_positions[self.cells][:, [0, 4, 7, 3, 1, 5, 6, 2]]
        b = np.array(self.bar_coord)
        one = np.ones_like(b[:, 0])
        # 2. Interpolate slave points
        P = ((H[:, 6].T * b[:, 0] + H[:, 2].T * (one - b[:, 0])) * b[:, 1] +
             (H[:, 5].T * b[:, 0] + H[:, 1].T * (one - b[:, 0])) * (one - b[:, 1])) * b[:, 2] + \
            ((H[:, 7].T * b[:, 0] + H[:, 3].T * (one - b[:, 0])) * b[:, 1] +
             (H[:, 4].T * b[:, 0] + H[:, 0].T * (one - b[:, 0])) * (one - b[:, 1])) * (one - b[:, 2])

        return P.T
