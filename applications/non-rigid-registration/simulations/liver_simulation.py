from os.path import join, dirname
from functools import reduce
from numpy import array, load, zeros, unique, empty, concatenate, absolute
from numpy.random import randint, uniform
from numpy.linalg import norm

import Sofa
import SofaRuntime
import SofaCaribou

from DeepPhysX.simulation import SofaSimulation
from DeepPhysX.utils.mesh.mesh import find_center
from DeepPhysX.utils.grid.sparse_grid import create_sparse_grid, sparse_to_regular_grid, regular_grid_margin


p_liver = {'file': join(dirname(__file__), 'data', 'liver.obj'),
           'collision': join(dirname(__file__), 'data', 'liver_col.obj'),
           'scale': [1] * 3,
           'fixed': join(dirname(__file__), 'data', 'fixed.npy'),
           'pcd': join(dirname(__file__), 'data', 'pcd.npy')
           }
p_liver['center'] = find_center(mesh_file=p_liver['file'], scale=p_liver['scale'][0])
p_grid = {key: value for key, value in zip(['res', 'min', 'max'], create_sparse_grid(mesh_file=p_liver['file'],
                                                                                     scale=p_liver['scale'][0],
                                                                                     cell_size=0.12))}
p_grid['margin'] = 0.4
p_forces = {'nb': 10, 'amplitude': 1e5, 'radius': 15 * p_liver['scale'][0], 'distance': 2}


class LiverSimulation(SofaSimulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.cff, self.cff_sphere = [], []
        self.pcd_indices = load(p_liver['pcd'])
        self.liver_grid, self.idx_sparse_to_regular = None, None
        self.unet_grid, self.idx_liver_to_unet = None, None

        self.idx_step = 0
        self.nb_intermediate_step = 25

    def create(self):

        # Add required plugins
        with open(join(dirname(__file__), 'data', 'plugins.txt'), 'r') as f:
            plugins = [plugin[:-1] if plugin.endswith('\n') else plugin for plugin in f.readlines() if plugin != '\n']
        SofaRuntime.PluginRepository.addFirstPath(reduce(lambda d, _: dirname(d), range(4), SofaCaribou.__file__))
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        self.root.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels hideForceFields')
        self.root.addObject('DefaultAnimationLoop')

        # Add fem node and / or nn node
        self.create_pcd_node()
        self.create_rest_node()
        self.create_fem_node()
        self.create_net_node()

    def create_pcd_node(self):

        self.root.addChild('pcd')
        self.root.pcd.addObject('MechanicalObject', name='MO', position=zeros((len(self.pcd_indices), 3)),
                                showObject=True, showColor=[0.7, 0.5, 0], showObjectScale=5)

    def create_rest_node(self):

        # Rest shape node
        self.root.addChild('rest')
        self.root.rest.addObject('MeshOBJLoader', name='Mesh', filename=p_liver['file'], scale3d=p_liver['scale'], rotation=[0, 0, 180])
        self.root.rest.addObject('OglModel', src='@Mesh', color=[0.5, 0.5, 0.5, 0.5])

    def create_fem_node(self):

        # FEM root node
        self.root.addChild('fem')
        self.root.fem.addObject('EulerImplicitSolver', name='ODESolver', rayleighMass=0.1, rayleighStiffness=0.1,
                                vdamping=5)
        self.root.fem.addObject('CGLinearSolver', name='LinearSolver', iterations=2000, threshold=1e-10,
                                tolerance=1e-10)

        # Grid topology of the model
        self.root.fem.addObject(f'MeshOBJLoader', name='Mesh', filename=p_liver['file'], scale3d=p_liver['scale'], rotation=[0, 0, 180])
        self.root.fem.addObject('SparseGridTopology', name='SparseGridTopo', src='@Mesh', n=p_grid['res'])
        self.root.fem.addObject('MechanicalObject', name='SparseGridMO', src='@SparseGridTopo')

        # Material
        self.root.fem.addObject('SaintVenantKirchhoffMaterial', name='StVK', young_modulus=2000, poisson_ratio=0.45)
        self.root.fem.addObject('HyperelasticForcefield', name='Hexa', template="Hexahedron", material="@StVK",
                                topology='@SparseGridTopo')
        self.root.fem.addObject('UniformMass', totalMass=1)

        # Constraints
        self.root.fem.addObject('FixedConstraint', name='Boundaries', indices=[])
        self.root.fem.addObject('PlaneForceField', name='Plane', normal=[0., 1., 0.], stiffness=1e4,
                                showPlane=True, showPlaneSize=2e2 * p_liver['scale'][0])

        # Surface
        self.root.fem.addChild('surface')
        self.root.fem.surface.addObject('TriangleSetTopologyContainer', name='SurfaceTopo', src='@../Mesh')
        self.root.fem.surface.addObject('MechanicalObject', name='SurfaceMO', src='@SurfaceTopo', showObject=False)
        self.root.fem.surface.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

        # Forces
        for i in range(p_forces['nb']):
            self.cff_sphere.append(self.root.fem.surface.addObject('SphereROI', name=f'Sphere_{i}', radii=p_forces['radius'],
                                                                    centers=p_liver['center']))
            self.cff.append(self.root.fem.surface.addObject('ConstantForceField', name=f'CFF_{i}', indices='0',
                                                            forces=[0., 0., 0.], showArrowSize=2e-4))

        # Visual
        self.root.fem.addChild('visual')
        self.root.fem.visual.addObject('OglModel', name='OGL', src='@../Mesh', color=[0, 0.7, 0, 1])
        self.root.fem.visual.addObject('BarycentricMapping', input='@../SparseGridMO', output='@OGL')


    def create_net_node(self):

        # FEM root node
        self.root.addChild('net')

        # Grid topology of the model
        self.root.net.addObject(f'MeshOBJLoader', name='Mesh', filename=p_liver['file'], scale3d=p_liver['scale'], rotation=[0, 0, 180])
        self.root.net.addObject('SparseGridTopology', name='SparseGridTopo', src='@Mesh', n=p_grid['res'])
        self.root.net.addObject('MechanicalObject', name='SparseGridMO', src='@SparseGridTopo')

        # Visual
        self.root.net.addChild('visual')
        self.root.net.visual.addObject('OglModel', name='OGL', src='@../Mesh', color='orange')
        self.root.net.visual.addObject('BarycentricMapping', input='@../SparseGridMO', output='@OGL')

    def onSimulationInitDoneEvent(self, event):

        # Init the visible PCD
        pcd_mo = self.root.pcd.getObject('MO')
        surface_mo = self.root.fem.surface.getObject('SurfaceMO')
        pcd_mo.position.value = surface_mo.position.value[self.pcd_indices]

        # Create the regular grids
        self.root.addChild('unet')
        sparse_grid = self.root.fem.getObject('SparseGridTopo')
        self.liver_grid, self.idx_sparse_to_regular = sparse_to_regular_grid(sparse_grid=sparse_grid)
        self.unet_grid, self.idx_liver_to_unet = regular_grid_margin(regular_grid=self.liver_grid, margin=p_grid['margin'])
        self.root.unet.addObject('OglModel', position=self.unet_grid.position.value,
                                 edges=self.unet_grid.edges.value, color=[1., 1., 1., 0.2])

        # Init constraints
        fixed_idx = []
        for pos in surface_mo.position.value[load(p_liver['fixed'])]:
            cell = sparse_grid.findCube(pos.tolist())
            fixed_idx += sparse_grid.getNodeIndicesOfCube(cell[0])
        self.root.fem.getObject('Boundaries').indices.value = unique(fixed_idx).tolist()
        liver_grid_cell_size = (self.liver_grid.max.value - self.liver_grid.min.value) / self.liver_grid.n.value
        self.root.fem.getObject('Plane').d.value = self.liver_grid.min.value[1] - 0.5 * liver_grid_cell_size[1]

    def onAnimateBeginEvent(self, event):

        # New batch of time steps
        if self.idx_step % self.nb_intermediate_step == 0:

            # Reset fem positions
            sparse_grid_mo = self.root.fem.getObject('SparseGridMO')
            sparse_grid_mo.position.value = sparse_grid_mo.rest_position.value
            sparse_grid_mo.velocity.value = zeros(sparse_grid_mo.position.value.shape)

            # Build and set forces vectors
            selected_centers = empty([0, 3])
            surface = self.root.fem.surface.getObject('SurfaceMO').position.value
            ogl = self.root.fem.visual.getObject('OGL')
            nb_empty_cff = 0
            for i in range(p_forces['nb']):
                # Pick up a random visible surface point, select the points in a centered sphere
                id_center = randint(len(surface))
                current_center = surface[id_center]
                # Check distance to other points
                distance_check = True
                for p in selected_centers.reshape((-1, 3)):
                    if norm(current_center - p) < p_forces['radius'] * p_forces['distance']:
                        distance_check = False
                        break
                empty_cff = False
                if distance_check:
                    # Set sphere center
                    selected_centers = concatenate((selected_centers, array([current_center])))
                    self.cff_sphere[i].centers.value = [current_center]
                    # Build force vector
                    if len(self.cff_sphere[i].indices.value) > 0:
                        f = ogl.normal.value[id_center] * uniform(0., 1., (3,))
                        f = (f / norm(f)) * p_forces['amplitude']
                        self.cff[i].indices.value = self.cff_sphere[i].indices.array()
                        self.cff[i].forces.value = [f for _ in range(len(self.cff[i].indices.value))]
                    else:
                        empty_cff = True
                if not distance_check or empty_cff:
                    # Reset sphere position
                    self.cff_sphere[i].centers.value = [p_liver['center']]
                    nb_empty_cff += 1
                    # Reset force field
                    self.cff[i].indices.value = [0]
                    self.cff[i].forces.value = [[0.0, 0.0, 0.0]]
            # Avoid empty force field
            if nb_empty_cff == p_forces['nb']:
                Sofa.Simulation.animate(self.root, self.root.dt.value)

    def onAnimateEndEvent(self, event):

        self.idx_step += 1

        # End of batch of time step
        if self.idx_step % self.nb_intermediate_step == 0:
            self.check_sample()

            # Update the PDC
            pcd_mo = self.root.pcd.getObject('MO')
            surface_mo = self.root.fem.surface.getObject('SurfaceMO')
            pcd_mo.position.value = surface_mo.position.value[self.pcd_indices]

    def check_sample(self):

        is_solver_ok = True

        # Check to high displacements
        sparse_grid = self.root.fem.getObject('SparseGridMO')
        max_u = absolute(sparse_grid.position.value - sparse_grid.rest_position.value).max()
        if max_u > 100 * p_liver['scale'][0]:
            is_solver_ok = False

        if not is_solver_ok:
            self.idx_step = 0
            Sofa.Simulation.reset(self.root)

        return is_solver_ok