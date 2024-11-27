from vedo import Mesh
from math import fabs


def define_bbox(source_file, margin_scale, scale):
    """
    Find the bounding box of the model.

    :param str source_file: Mesh file
    :param float scale: Scale to apply
    :param float margin_scale: Margin in percents of the bounding box
    :return: List of coordinates defined by xmin, ymin, zmin, xmax, ymax, zmax
    """

    # Find min and max corners of the bounding box
    mesh = Mesh(source_file).scale(scale)
    bbox_min = mesh.points().min(0)
    bbox_max = mesh.points().max(0)

    # Apply a margin scale to the bounding box
    bbox_min -= margin_scale * (bbox_max - bbox_min)
    bbox_max += margin_scale * (bbox_max - bbox_min)

    return bbox_min, bbox_max, bbox_min.tolist() + bbox_max.tolist()


def find_center(source_file, scale):
    """
    Find the center of mass of the object.

    :param source_file: Filename of the src object
    :param scale: Scaling to apply to the objects
    :return: Center of mass of the object
    """

    return Mesh(source_file).scale(scale).centerOfMass()


def compute_grid_resolution(max_bbox, min_bbox, cell_size, print_log=False):
    """
    Compute the grid resolution from the desired cell size and the grid dimensions.

    :param list max_bbox: Max upper corner of the grid
    :param list min_bbox: Min lower corner of the grid
    :param float cell_size: Desired cell size
    :param bool print_log: Print info
    :return: List of grid resolution for each dimension
    """

    # Absolute size values along 3 dimensions
    sx = fabs(max_bbox[0] - min_bbox[0])
    sy = fabs(max_bbox[1] - min_bbox[1])
    sz = fabs(max_bbox[2] - min_bbox[2])

    # Compute number of nodes in the grid
    cell_size = cell_size * min(sx, sy, sz)  # Cells need to be hexahedron
    nx = int(sx / cell_size)
    ny = int(sy / cell_size)
    nz = int(sz / cell_size)

    # Print grid infos
    if print_log:
        print(f"Cell size = {cell_size}x{cell_size}x{cell_size}")
        print(f"Nx = {nx}, Ny = {ny}, Nz = {nz}")
        number_of_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        print(f"Number of nodes in regular grid = {number_of_nodes}")

    return [nx + 1, ny + 1, nz + 1]
