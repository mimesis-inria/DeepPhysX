import math
import vedo
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import Sofa.SofaBaseTopology


def compute_grid_resolution(max_bbox, min_bbox, cell_size, print_log=False):
    """
    Compute the grid resolution from the desired cell size and the grid dimensions.

    :param list max_bbox: Max upper corner of the grid
    :param list min_bbox: Min lower corner of the grid
    :param float cell_size: Desired cell size
    :param bool print_log: Print info
    :return: Number of nodes for each direction of the Grid
    """

    # Absolute size values along 3 dimensions
    sx = math.fabs(max_bbox[0] - min_bbox[0])
    sy = math.fabs(max_bbox[1] - min_bbox[1])
    sz = math.fabs(max_bbox[2] - min_bbox[2])

    # Compute number of nodes in the grid
    cell_size = cell_size * min(sx, sy, sz)  # Cells need to be hexahedron
    nx = int(sx / cell_size)
    ny = int(sy / cell_size)
    nz = int(sz / cell_size)

    if print_log:
        print(f"Cell size = {cell_size}x{cell_size}x{cell_size}")
        print(f"Nx = {nx}, Ny = {ny}, Nz = {nz}")
        number_of_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        print(f"Number of nodes in regular grid = {number_of_nodes}")

    return [nx + 1, ny + 1, nz + 1]


def find_boundaries(source_file, objects_files_list, scale):
    """

    :param scale:
    :param source_file: Filename of the source object
    :param objects_files_list: List of filenames of objects which intersect the source object
    :return: List of boxes defined by xmin,ymin,zmin, xmax,ymax,zmax
    """
    # Source mesh object, list of points defining the boundary conditions
    source_mesh = vedo.Mesh(source_file)
    boundaries = []
    # Find intersections for each object
    for object_file in objects_files_list:
        object_mesh = vedo.Mesh(object_file)
        intersect = source_mesh.intersectWith(object_mesh)
        boundaries += intersect.points().tolist()
    # Find min and max corners of the bounding box
    bbox_min = np.array(boundaries).min(0) * scale
    bbox_max = np.array(boundaries).max(0) * scale
    return bbox_min.tolist() + bbox_max.tolist()


def define_bbox(source_file, margin_scale, scale):
    """

    :param scale:
    :param source_file: Filename of the source object
    :param margin_scale: Margin in percents
    :return: List of boxes defined by xmin,ymin,zmin, xmax,ymax,zmax
    """
    # Source object mesh
    source_mesh = vedo.Mesh(source_file)
    # Find min and max corners of the bounding box
    bbox_min = source_mesh.points().min(0)
    bbox_max = source_mesh.points().max(0)
    # Apply a margin scale to the bounding box
    bbox_min -= margin_scale * (bbox_max - bbox_min)
    bbox_max += margin_scale * (bbox_max - bbox_min)
    # Scale the bounding box
    bbox_min = bbox_min * scale
    bbox_max = bbox_max * scale
    return bbox_min, bbox_max, bbox_min.tolist() + bbox_max.tolist()


def get_nb_nodes(source_file):
    mesh = vedo.Mesh(source_file)
    return len(mesh.points())


def find_center(source_file, scale):
    return vedo.Mesh(source_file).scale(scale).centerOfMass()


def from_sparse_to_regular_grid(nb_nodes_regular_grid, sparse_grid, sparse_grid_mo):
    """
    Map the indices of nodes in the sparse grid with the indices of nodes in the regular grid.

    :param int nb_nodes_regular_grid: Total number of nodes in the regular grid
    :param sparse_grid: SparseGridTopology containing the sparse grid topology
    :param sparse_grid_mo: MechanicalObject containing the positions of the nodes in the sparse grid
    :return: Mapped indices from sparse to regular grids, Mapped indices from regular to sparse regular grid,
    Rest shape positions of the regular grid
    """

    # Initialize mapping between sparse grid and regular grid
    positions_sparse_grid = sparse_grid_mo.position.array()
    indices_sparse_to_regular = np.zeros(positions_sparse_grid.shape[0], dtype=np.int32)
    indices_regular_to_sparse = np.full(nb_nodes_regular_grid, -1, dtype=np.int32)

    # Map the indices of each node iteratively
    for i in range(positions_sparse_grid.shape[0]):
        # In Sofa, a SparseGrid in computed from a RegularGrid, just use the dedicated method to retrieve their link
        idx = sparse_grid.getRegularGridNodeIndex(i)
        indices_sparse_to_regular[i] = idx  # Node i in SparseGrid corresponds to node idx in RegularGrid
        indices_regular_to_sparse[idx] = i  # Node idx in RegularGrid corresponds to node i in SparseGrid

    # Recover rest shape positions of sparse grid nodes in the regular grid
    regular_grid_rest_shape_positions = np.zeros((nb_nodes_regular_grid, 3), dtype=np.double)
    regular_grid_rest_shape_positions[indices_sparse_to_regular] = sparse_grid_mo.rest_position.array()

    return indices_sparse_to_regular, indices_regular_to_sparse, regular_grid_rest_shape_positions
