import math
import vedo
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np


def compute_grid_resolution(max_bbox, min_bbox, cell_size):
    """
    Compute the grid resolution from the desired cell size and the grid dimensions.

    :param max_bbox: Max upper corner of the grid
    :param min_bbox: Min lower corner of the grid
    :param cell_size: Desired cell size
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

    return [nx + 1, ny + 1, nz + 1]


def find_boundaries(source_file, objects_files_list, scale):
    """
    Find the boundary conditions of the liver.

    :param source_file: Filename of the source object
    :param objects_files_list: List of filenames of objects which intersect the source object
    :param scale: Scaling to apply to the objects
    :return: Boundary box defined by [xmin, ymin, zmin, xmax, ymax, zmax]
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
    Define the global bounding box of the object.

    :param source_file: Filename of the source object
    :param margin_scale: Margin in percents
    :param scale: Scaling to apply to the objects
    :return: Boundary box defined by [xmin, ymin, zmin, xmax, ymax, zmax]
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
    """
    Get the number of node of the object.

    :param source_file: Filename of the source object
    :return: Number of node
    """

    return vedo.Mesh(source_file).N()


def find_center(source_file, scale):
    """
    Find the center of mass of the object.

    :param source_file: Filename of the source object
    :param scale: Scaling to apply to the objects
    :return: Center of mass of the object
    """

    return vedo.Mesh(source_file).scale(scale).centerOfMass()
