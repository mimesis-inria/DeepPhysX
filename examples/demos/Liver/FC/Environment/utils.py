import vedo
import numpy as np
from math import pow


def get_nb_nodes(source_file):
    """
    Get the number of node of the object.

    :param source_file: Filename of the src object
    :return: Number of node
    """

    return vedo.Mesh(source_file).N()


def find_center(source_file, scale):
    """
    Find the center of mass of the object.

    :param source_file: Filename of the src object
    :param scale: Scaling to apply to the objects
    :return: Center of mass of the object
    """

    return vedo.Mesh(source_file).scale(scale).centerOfMass()


def find_boundaries(source_file, objects_files_list, scale):
    """
    Find the boundary conditions of the liver.

    :param source_file: Filename of the src object
    :param objects_files_list: List of filenames of objects which intersect the src object
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


def filter_by_distance(pcd, distance):
    """
    Filter the points of a point cloud by distance.

    :param pcd: Array of points
    :param distance: Threshold distance
    :return: Filtered list of indices
    """

    # Init random seed and list of indices to analyze
    remaining_points = list(np.arange(0, len(pcd)))
    current = 0
    selected_points = [current]
    remaining_points.remove(current)

    # Remove nodes too close to seed, then process to the closet valid node
    while len(remaining_points) != 0:
        in_sphere = np.argwhere(np.sum(np.power(pcd[remaining_points] - pcd[current], 2), axis=1) <= np.power(distance, 2))
        remaining_points = list(set(remaining_points) - set(np.array(remaining_points)[in_sphere.reshape(-1)]))
        if len(remaining_points) > 0:
            current = remaining_points[np.argmin(np.linalg.norm(pcd[remaining_points] - pcd[current], axis=1))]
            selected_points.append(current)
            remaining_points.remove(current)

    # Return list of remaining nodes
    return np.array(selected_points)
