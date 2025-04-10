import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import numpy as np


class BarycentricMapping:

    def __init__(self, 
                 source_positions: np.ndarray,
                 source_cells: np.ndarray,
                 target_positions: np.ndarray):
        """
        Python implementation of a barycentric mapping between a source mesh and a target mesh.

        :param source_positions: Array of positions of the source mesh.
        :param source_cells: Array of cells of the source mesh.
        :param target_positions: Array of positions of the target mesh.
        """

        source_positions = np.array(source_positions)
        source_cells = np.array(source_cells)
        target_positions = np.array(target_positions)

        # Compute barycentric coordinates
        # 1. CONVERT SOURCE TO VTK
        # 1.1. Create a VTK mesh
        vtk_mesh = vtk.vtkPolyData()
        # 1.2. Set points
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(source_positions, deep=1))
        vtk_mesh.SetPoints(points)
        # 1.3. Set cells
        cells = vtk.vtkCellArray()
        dtype = np.int32 if vtk.vtkIdTypeArray().GetDataTypeSize() == 4 else np.int64
        cells_data = np.hstack((np.ones(len(source_cells))[:, None] * 3, source_cells)).astype(dtype).ravel()
        cells.SetCells(len(source_cells), numpy_to_vtkIdTypeArray(cells_data, deep=1))
        vtk_mesh.SetPolys(cells)

        # 2. CLOSEST POINT FROM SOURCE FOR EACH POINT OF TARGET
        # 2.1. Use a VTL locator
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(vtk_mesh)
        cell_locator.BuildLocator()
        # 2.2 Properties filled by vtk
        point = [0., 0., 0.]
        cell_id, sub_id, distance = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.)
        # 2.3 Find the closest points
        points, index = [], []
        for p in target_positions:
            cell_locator.FindClosestPoint(p, point, cell_id, sub_id, distance)
            points.append(point[:])
            index.append(cell_id.get())

        # 3. BARYCENTRIC COORDINATES
        # 3.1 Define triangles
        ABC = source_positions[source_cells[index]]
        A, B, C = ABC[:, 0, :], ABC[:, 1, :], ABC[:, 2, :]
        P = np.array(points)
        # 3.2 Compute barycentric coordinates
        v0, v1, v2 = B - A, C - A, P - A
        d00 = (v0 * v0).sum(axis=1)
        d01 = (v0 * v1).sum(axis=1)
        d11 = (v1 * v1).sum(axis=1)
        d20 = (v2 * v0).sum(axis=1)
        d21 = (v2 * v1).sum(axis=1)
        v = (d11 * d20 - d01 * d21) / (d00 * d11 - d01 * d01)
        w = (d00 * d21 - d01 * d20) / (d00 * d11 - d01 * d01)
        u = 1 - v - w

        self.bar_coord = np.vstack([u, v, w]).T
        self.index = index
        self.cells = source_cells

    def apply(self, source_positions: np.ndarray) -> np.ndarray:
        """
        Apply mapping between new source positions and target with barycentric coordinates.
        """

        ABC = source_positions[self.cells[self.index]]
        points = []
        for abc, p in zip(ABC, self.bar_coord):
            points.append(np.dot(abc.T, p))
        return np.array(points)
