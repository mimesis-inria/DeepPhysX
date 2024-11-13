from vedo import Mesh


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


def find_extremities(source_file, scale):
    """
    Find the different extremities of the model.

    :param str source_file: Mesh file
    :param float scale: Scale to apply
    :return: Key points of the mesh.
    """

    # Get the coordinates of the mesh
    mesh = Mesh(source_file).scale(scale)
    coords = mesh.points().copy()

    # Get the size of the bounding box
    b_min, b_max, _ = define_bbox(source_file, 0, scale)
    sizes = b_max - b_min

    # Find the tail
    tail = coords[coords[:, 2].argmax()].tolist()

    # Find the hands
    right = coords[coords[:, 0] >= sizes[0] / 3]
    left = coords[coords[:, 0] <= -sizes[0] / 3]
    r_hand = right[right[:, 2].argmin()].tolist()
    l_hand = left[left[:, 2].argmin()].tolist()

    # Find the ears
    right = coords[coords[:, 0] >= 0]
    left = coords[coords[:, 0] <= 0]
    r_ear = right[right[:, 1].argmax()].tolist()
    l_ear = left[left[:, 1].argmax()].tolist()

    # Find the muzzle
    middle = coords[coords[:, 0] >= -sizes[0] / 3]
    middle = middle[middle[:, 0] <= sizes[0] / 3]
    muzzle = middle[middle[:, 2].argmin()].tolist()

    return [tail, r_hand, l_hand, r_ear, l_ear, muzzle]


def get_nb_nodes(source_file):
    """
    Get the number of nodes of a mesh.

    :param str source_file: Mesh file
    :return: Number of nodes in the mesh
    """

    return Mesh(source_file).npoints
