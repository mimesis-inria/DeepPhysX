"""
LiverSofa
Simulation of a Liver with FEM computed deformations.
The SOFA simulation contains two models of a Liver:
    * one to apply forces and compute deformations
    * one to apply the networks predictions
"""

# Python related imports
import os
import sys
from numpy import array, empty, concatenate
from numpy.random import randint, uniform
from numpy.linalg import norm

# Sofa & Caribou related imports
import Sofa.Simulation
import SofaRuntime
from Caribou.Topology import Grid3D

# DeepPhysX related imports
from DeepPhysX.Sofa.Environment.SofaEnvironment import SofaEnvironment

# Working session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_liver, p_grid, p_forces
from utils import from_sparse_to_regular_grid


class LiverSofa(SofaEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1):

        SofaEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb)

        # With flag set to True, the model is created
        self.create_model = {'fem': True, 'nn': False}

        # UNet regular grid
        self.regular_grid = None
        self.nb_nodes_regular_grid = None
        self.nb_nodes_sparse_grid = None
        self.idx_sparse_to_regular = None
        self.idx_regular_to_sparse = None
        self.regular_grid_rest_shape = None
        self.data_size = None

        # FEM objects
        self.solver = None
        self.f_sparse_grid_topo = None
        self.f_sparse_grid_mo = None
        self.f_surface_topo = None
        self.f_surface_mo = None
        self.f_visu = None

        # Network objects
        self.n_sparse_grid_topo = None
        self.n_sparse_grid_mo = None
        self.n_surface_topo = None
        self.n_surface_mo = None
        self.n_visu = None

        # Forces
        self.sphere = []
        self.force_field = []

    def create(self):
        """
        Create the Sofa scene graph. Automatically called by SofaEnvironmentConfig.
        """

        # Add required plugins
        SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_INSTALL'])
        plugins = ['Sofa.Component.IO.Mesh', 'Sofa.Component.Engine.Select', 'Sofa.Component.MechanicalLoad',
                   'Sofa.GL.Component.Rendering3D', 'SofaCaribou']
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        # Scene visual style
        self.root.addObject('VisualStyle', displayFlags="showVisualModels showWireframe")

        # Meshes
        self.root.addObject('MeshObjLoader', name='Mesh', filename=p_liver.mesh, scale3d=p_liver.scale3d)
        self.root.addObject('MeshObjLoader', name='MeshCoarse', filename=p_liver.mesh_coarse, scale3d=p_liver.scale3d)

        # UNet regular grid
        self.regular_grid = Grid3D(anchor_position=p_grid.bbox_anchor, n=p_grid.nb_cells, size=p_grid.bbox_size)
        self.root.addObject('MechanicalObject')
        self.root.addObject('BoxROI', name='RegularGridBox', box=p_grid.b_box, drawBoxes=True, drawSize=1.)

        # Create FEM and / or NN models
        if self.create_model['fem']:
            self.createFEM()
        if self.create_model['nn']:
            self.createNN()

    def createFEM(self):
        """
        FEM model of Liver. Used to apply forces and compute deformations.
        """

        # Create child node
        self.root.addChild('fem')

        # ODE solver + Static solver
        self.solver = self.root.fem.addObject('StaticODESolver', name='ODESolver', newton_iterations=20,
                                              printLog=False, correction_tolerance_threshold=1e-8,
                                              residual_tolerance_threshold=1e-8)
        self.root.fem.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-8, printLog=False)

        # Grid topology of the model
        self.f_sparse_grid_topo = self.root.fem.addObject('SparseGridTopology', name='SparseGridTopo',
                                                          src='@../MeshCoarse', n=p_grid.grid_resolution)
        self.f_sparse_grid_mo = self.root.fem.addObject('MechanicalObject', name='SparseGridMO', src='@SparseGridTopo',
                                                        showObject=False)

        # Material
        self.root.fem.addObject('NeoHookeanMaterial', name='NH', young_modulus=4000, poisson_ratio=0.4)
        self.root.fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@NH",
                                topology='@SparseGridTopo', printLog=False)

        # Fixed section
        self.root.fem.addObject('BoxROI', name='FixedBox', box=p_liver.fixed_box, drawBoxes=True, drawSize=1.)
        self.root.fem.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Surface
        self.root.fem.addChild('surface')
        self.f_surface_topo = self.root.fem.surface.addObject('TriangleSetTopologyContainer', name='SurfaceTopo',
                                                              src='@../../MeshCoarse')
        self.f_surface_mo = self.root.fem.surface.addObject('MechanicalObject', name='SurfaceMO',
                                                            src='@SurfaceTopo', showObject=False)
        self.root.fem.surface.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

        # Forces
        self.create_forces(self.root.fem.surface)

        # Visual
        self.root.fem.addChild('visual')
        self.f_visu = self.root.fem.visual.addObject('OglModel', name='OGL', src='@../../Mesh', color='green')
        self.root.fem.visual.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

    def createNN(self):
        """
        Network model of Liver. Used to apply predictions.
        """

        # Create child node
        self.root.addChild('nn')

        # Fake solvers (required to compute forces on the grid)
        self.root.nn.addObject('StaticODESolver', name='ODESolver', newton_iterations=0)
        self.root.nn.addObject('ConjugateGradientSolver', name='StaticSolver', maximum_number_of_iterations=0)

        # Grid topology
        self.n_sparse_grid_topo = self.root.nn.addObject('SparseGridTopology', name='SparseGrid', src='@../MeshCoarse',
                                                         n=p_grid.grid_resolution)
        self.n_sparse_grid_mo = self.root.nn.addObject('MechanicalObject', name='SparseGridMO', showObject=False,
                                                       src='@SparseGrid')

        # Fixed section
        if not self.create_model['fem']:
            self.root.nn.addObject('BoxROI', name='fixed_box', box=p_liver.fixed_box, drawBoxes=True)
            self.root.nn.addObject('FixedConstraint', indices='@fixed_box.indices')

        # Surface
        self.root.nn.addChild('surface')
        self.n_surface_topo = self.root.nn.surface.addObject('TriangleSetTopologyContainer', name='SurfaceTopo',
                                                             src='@../../MeshCoarse')
        self.n_surface_mo = self.root.nn.surface.addObject('MechanicalObject', name='SurfaceMO', src='@SurfaceTopo')
        self.root.nn.surface.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

        # Forces
        if not self.create_model['fem']:
            self.create_forces(self.root.nn.surface)

        # Visual
        self.root.nn.addChild('visual')
        self.n_visu = self.root.nn.visual.addObject('OglModel', name='OGL', src='@../../Mesh', color=(1, 0.6, 0.1, 1))
        self.root.nn.visual.addObject('BarycentricMapping', input='@../SparseGridMO', output='@./')

    def create_forces(self, node):
        """
        Generate the force fields on surface.
        """

        for i in range(p_forces.nb_simultaneous_forces):
            self.sphere.append(node.addObject('SphereROI', name=f'Sphere_{i}', radii=25 * p_liver.scale,
                                              drawSphere=True, drawSize=1., centers=p_liver.fixed_point.tolist()))
            self.force_field.append(node.addObject('ConstantForceField', name=f'CFF_{i}', indices='0',
                                                   force=[0., 0., 0.], showArrowSize=0.4))

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """

        # Correspondences between sparse grid and regular grid
        sparse_grid_mo = self.f_sparse_grid_mo if self.create_model['fem'] else self.n_sparse_grid_mo
        sparse_grid_topo = self.f_sparse_grid_topo if self.create_model['fem'] else self.n_sparse_grid_topo
        self.nb_nodes_regular_grid = self.regular_grid.number_of_nodes()
        self.nb_nodes_sparse_grid = len(sparse_grid_mo.rest_position.value)
        correspondence = from_sparse_to_regular_grid(self.nb_nodes_regular_grid, sparse_grid_topo, sparse_grid_mo)
        self.idx_sparse_to_regular = correspondence[0]
        self.idx_regular_to_sparse = correspondence[1]
        self.regular_grid_rest_shape = correspondence[2]

        # Get the data sizes
        self.data_size = (self.nb_nodes_regular_grid, 3)

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        # Reset positions
        if self.create_model['fem']:
            self.f_sparse_grid_mo.position.value = self.f_sparse_grid_mo.rest_position.value
        if self.create_model['nn']:
            self.n_sparse_grid_mo.position.value = self.n_sparse_grid_mo.rest_position.value

        # Build and set forces vectors
        selected_centers = empty([0, 3])
        surface_mo = self.f_surface_mo if self.create_model['fem'] else self.n_surface_mo
        for i in range(p_forces.nb_simultaneous_forces):
            # Pick up a random visible surface point, select the points in a centered sphere
            current_point = surface_mo.position.value[randint(len(surface_mo.position.value))]
            # Check distance to other points
            distance_check = True
            for p in selected_centers:
                distance = norm(current_point - p)
                if distance < p_forces.inter_distance_thresh:
                    distance_check = False
                    break
            empty_indices = False
            if distance_check:
                # Add center to the selection
                selected_centers = concatenate((selected_centers, array([current_point])))
                # Set sphere center
                self.sphere[i].centers.value = [current_point]
                # Build force vector
                if len(self.sphere[i].indices.value) > 0:
                    f = uniform(low=-1, high=1, size=(3,))
                    # from random import choice
                    # f = np.array([choice([-1, 1]), choice([-1, 1]), choice([-1, 1])])
                    f = (f / norm(f)) * p_forces.amplitude
                    self.force_field[i].indices.value = self.sphere[i].indices.array()
                    self.force_field[i].force.value = f
                else:
                    empty_indices = True
            if not distance_check or empty_indices:
                # Reset sphere position
                self.sphere[i].centers.value = [p_liver.fixed_point.tolist()]
                # Reset force field
                self.force_field[i].indices.value = [0]
                self.force_field[i].force.value = [0.0, 0.0, 0.0]

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step.
        """

        # Check whether if the solver diverged or not
        if not self.check_sample():
            print("Solver diverged.")

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # Check if the solver converged while computing FEM
        if self.create_model['fem']:
            if not self.solver.converged.value:
                # Reset simulation if solver diverged to avoid unwanted behaviour in following samples
                Sofa.Simulation.reset(self.root)
            return self.solver.converged.value
        return True

    def close(self):
        """
        Shutdown procedure.
        """

        print("Bye!")
