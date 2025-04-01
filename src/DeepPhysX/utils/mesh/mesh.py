from vedo import Mesh


def find_center(mesh_file: str, scale: float):
    """
    Get the geometrical center of a mesh.

    :param mesh_file: Mesh file.
    :param scale: Scale to apply to the mesh.
    """

    return Mesh(mesh_file).scale(scale).center_of_mass()

def get_bbox(mesh_file: str, scale: float):
    """
    Get the bounding box of a mesh.
    """

    bounds = Mesh(mesh_file).scale(scale).bounds()
    return bounds[[0, 2, 4]], bounds[[1, 3, 5]]
