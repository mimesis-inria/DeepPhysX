from typing import Tuple, List
from numpy import array, ndarray, abs
from vedo import Mesh

try:
    import Sofa
    import Sofa.SofaBaseTopology
except ImportError:
    pass


def get_grid_resolution(grid_min: ndarray,
                        grid_max: ndarray,
                        cell_size: float):
    """
    Compute the grid resolution with hexahedron cells.

    :param grid_min: Min corner of the grid.
    :param grid_max: Max corner of the grid.
    :param cell_size: Individual cell size in % of the total grid size.
    """

    size = abs(grid_max - grid_min)
    return [int(s / (cell_size * size.min())) + 1 for s in size]


def create_sparse_grid(mesh_file: str,
                       scale: float,
                       cell_size: float):
    """
    Create the sparse grid of a mesh.

    :param mesh_file: Path to the mesh file.
    :param scale: Scale to apply to the mesh.
    :param cell_size: Individual cell size in % of the total grid size.
    :return: Tuple of 3 arrays containing ([n_x, n_y, n_z], [x_min, y_min, z_min], [x_max, y_max, z_max])
    """

    # Get the bounding box size along each axis
    bbox = array(Mesh(inputobj=mesh_file).scale(s=scale).bounds()).reshape((3, 2)).T
    size = abs(bbox[1] - bbox[0])

    # Compute the number of hexa cells along each axis
    n = [int(s / (cell_size * size.min())) + 1 for s in size]

    # Create the sparse grid with SOFA
    loader = 'MeshOBJLoader' if mesh_file.endswith('.obj') else 'MeshSTLLoader'
    plugins = ['Sofa.Component.IO.Mesh', 'Sofa.Component.Topology.Container.Grid']
    node = Sofa.Core.Node('root')
    node.addObject('RequiredPlugin', pluginName=plugins)
    node.addObject(loader, name='Mesh', filename=mesh_file, scale3d=scale)
    sparse_grid = node.addObject('SparseGridTopology', src='@Mesh', n=n)
    node.init()

    return sparse_grid.n.value, sparse_grid.min.value, sparse_grid.max.value

def sparse_to_regular_grid(sparse_grid: Sofa.SofaBaseTopology.SparseGridTopology) -> Tuple[
    Sofa.SofaBaseTopology.RegularGridTopology, ndarray]:
    """
    Create a regular grid from a sparse grid and compute their correspondences.
    Optionally define a margin around the sparse grid.

    :param sparse_grid: SOFA sparse grid.
    :return: The regular grid object and the correspondences from sparse to regular.
    """

    # Create a RegularGrid topology
    node = Sofa.Core.Node('root')
    regular_grid = node.addObject('RegularGridTopology', n=sparse_grid.n.value, min=sparse_grid.min.value,
                                  max=sparse_grid.max.value)
    node.init()

    # Get the correspondences from sparse to regular grids
    nb_node = sparse_grid.position.value.shape[0]
    idx_sparse_to_regular = array([sparse_grid.getRegularGridNodeIndex(i) for i in range(nb_node)], dtype=int)

    return regular_grid, idx_sparse_to_regular


def regular_grid_margin(regular_grid: Sofa.SofaBaseTopology.RegularGridTopology,
                        margin: float = 0.) -> Tuple[Sofa.SofaBaseTopology.RegularGridTopology, ndarray]:
    """
    Create a new regular grid with the same cell_size with a margin around it.

    :param regular_grid: Regular grid.
    :param margin: Margin to apply in % of the regular grid size.
    :return: The scaled regular grid object and the correspondences from original to scaled grids.
    """

    # Compute the new grid resolution
    grid_size = regular_grid.max.value - regular_grid.min.value
    grid_n = regular_grid.n.value
    cell_size = grid_size / grid_n
    nb_additional_cells = array([(0.5 * margin * g_s // c_s + 1) * 2 for g_s, c_s in zip(grid_size, cell_size)], dtype=int)
    margin_n = grid_n + nb_additional_cells
    margin_min = regular_grid.min.value - 0.5 * nb_additional_cells * cell_size

    # Create the new regular grid
    node = Sofa.Core.Node('root')
    margin_grid = node.addObject('RegularGridTopology',
                                 n=margin_n,
                                 min=margin_min,
                                 max=margin_min + margin_n * cell_size)
    node.init()

    # Correspondences from original to margin grids
    cell_margin = array(0.5 * nb_additional_cells, dtype=int)
    regular_to_margin = array([[[z * (margin_n[0] + 1) * (margin_n[1] + 1) + y * (margin_n[0] + 1) + x
                                 for x in range(cell_margin[0], cell_margin[0] + grid_n[0])]
                                for y in range(cell_margin[1], cell_margin[1] + grid_n[1])]
                               for z in range(cell_margin[2], cell_margin[2] + grid_n[2])], dtype=int).reshape(-1)

    return margin_grid, regular_to_margin


def get_scaled_regular_grid_resolution(grid_resolution: ndarray,
                                       grid_size: ndarray,
                                       margin: float):
    """

    """

    cell_size = grid_size / grid_resolution
    nb_additional_cells = array([(0.5 * margin * g_s // c_s + 1) * 2 for g_s, c_s in zip(grid_size, cell_size)])
    return grid_resolution + nb_additional_cells


def regular_grid_get_cell(regular_grid: Sofa.SofaBaseTopology.RegularGridTopology,
                          pos: ndarray) -> int:
    """
    Get the index of the cell in the regular grid containing the position.
    """

    # Get the cell coordinates
    coord = ((pos - regular_grid.min.value) / (regular_grid.max.value - regular_grid.min.value)) * regular_grid.n.value
    coord = coord.astype(int)

    # Get the cell index
    n = regular_grid.n.value
    idx = coord[2] * n[1] * n[0] + coord[1] * n[0] + coord[0]

    return idx
