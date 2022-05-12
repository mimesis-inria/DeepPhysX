import vedo


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