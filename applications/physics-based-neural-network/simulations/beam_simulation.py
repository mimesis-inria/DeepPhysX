from os.path import join, dirname
from functools import reduce
from operator import mul
from numpy.random import randint, uniform
from numpy.linalg import norm

import Sofa
import SofaRuntime
import SofaCaribou

from DeepPhysX.simulation import SofaSimulation


grid = {'min': [0., 0., 0.],
        'max': [100., 15., 15.],
        'res': [25, 5, 5],
        'fix': [0., 0., 0., 0., 15., 15.]}
grid['n'] = reduce(mul, grid['res'])


class BeamSimulation(SofaSimulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.idx_step = 0
        self.sub_steps = 5
        self.F = None

        self.fem_node = True
        self.net_node = False

    def create(self):

        # Add required plugins
        data_dir = join(dirname(__file__), 'data')
        with open(join(data_dir, 'plugins.txt'), 'r') as f:
            plugins = [plugin[:-1] if plugin.endswith('\n') else plugin for plugin in f.readlines() if plugin != '\n']
        SofaRuntime.PluginRepository.addFirstPath(reduce(lambda d, _: dirname(d), range(4), SofaCaribou.__file__))
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        self.root.addObject('VisualStyle', displayFlags='showVisualModels showWireframe showBehaviorModels showForceFields')
        self.root.addObject('DefaultAnimationLoop')

        # Add fem node and / or nn node
        if self.fem_node:
            self.create_fem_node()
        if self.net_node:
            self.create_net_node()

    def create_fem_node(self):

        # FEM root node
        self.root.addChild('fem')
        self.root.fem.addObject('StaticODESolver', name='ODESolver', printLog=False, newton_iterations=50,
                                correction_tolerance_threshold=1e-8, residual_tolerance_threshold=1e-8)
        self.root.fem.addObject('ConjugateGradientSolver', name='StaticSolver', preconditioning_method='Diagonal',
                                maximum_number_of_iterations=2000, residual_tolerance_threshold=1e-8, printLog=False)

        # Grid topology of the model
        self.root.fem.addObject('RegularGridTopology', name='GridTopo', min=grid['min'], max=grid['max'],
                                nx=grid['res'][0], ny=grid['res'][1], nz=grid['res'][2])
        self.root.fem.addObject('MechanicalObject', name='GridMO', src='@GridTopo', showObject=False)
        self.root.fem.addObject('HexahedronSetTopologyContainer', name='HexaTopo', src='@GridTopo')
        self.root.fem.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.fem.addObject('HexahedronSetTopologyModifier')

        # Material
        self.root.fem.addObject('NeoHookeanMaterial', name='NH', young_modulus=10000, poisson_ratio=0.45)
        self.root.fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@NH", topology='@HexaTopo')

        # Surface
        self.root.fem.addObject('QuadSetTopologyContainer', name='QuadTopo', src='@HexaTopo')
        self.root.fem.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.fem.addObject('QuadSetTopologyModifier')
        self.root.fem.addObject('Hexa2QuadTopologicalMapping', input='@HexaTopo', output='@QuadTopo')

        # Fixed section
        self.root.fem.addObject('BoxROI', name='FixedBox', box=grid['fix'])
        self.root.fem.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Forces
        self.root.fem.addObject('BoxROI', name='ForceBox', box=grid['min'] + grid['max'])
        self.root.fem.addObject('ConstantForceField', name='CFF', showArrowSize=0.1, forces=[0., 0., 0.],
                                indices=list(range(grid['n'])))

        # Visual
        self.root.fem.addChild('visual')
        self.root.fem.visual.addObject('OglModel', name='OGL', src='@../GridTopo', color='green')
        self.root.fem.visual.addObject('BarycentricMapping', input='@../GridMO', output='@./')

    def create_net_node(self):

        # Network root node
        self.root.addChild('net')
        self.root.net.addObject('StaticODESolver', name='ODESolver', newton_iterations=0)
        self.root.net.addObject('ConjugateGradientSolver', name='StaticSolver', maximum_number_of_iterations=0)

        # Grid topology of the model
        self.root.net.addObject('RegularGridTopology', name='GridTopo', min=grid['min'], max=grid['max'],
                                nx=grid['res'][0], ny=grid['res'][1], nz=grid['res'][2])
        self.root.net.addObject('MechanicalObject', name='GridMO', src='@GridTopo', showObject=False)
        self.root.net.addObject('HexahedronSetTopologyContainer', name='HexaTopo', src='@GridTopo')
        self.root.net.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.net.addObject('HexahedronSetTopologyModifier')

        # Surface
        self.root.net.addObject('QuadSetTopologyContainer', name='QuadTopo', src='@HexaTopo')
        self.root.net.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.net.addObject('QuadSetTopologyModifier')
        self.root.net.addObject('Hexa2QuadTopologicalMapping', input='@HexaTopo', output='@QuadTopo')

        # Fixed section
        self.root.net.addObject('BoxROI', name='FixedBox', box=grid['fix'])
        self.root.net.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Forces
        self.root.net.addObject('BoxROI', name='ForceBox', box=grid['min'] + grid['max'])
        self.root.net.addObject('ConstantForceField', name='CFF', showArrowSize=0.1, forces=[0., 0., 0.],
                                indices=list(range(grid['n'])))

        # Visual
        self.root.net.addChild('visual')
        self.root.net.visual.addObject('OglModel', name='OGL', src='@../GridTopo', color='orange')
        self.root.net.visual.addObject('BarycentricMapping', input='@../GridMO', output='@./')

    def onAnimateBeginEvent(self, event):

        # Select the node to apply forces
        node = self.root.fem if self.fem_node else self.root.net

        if self.idx_step == 0:

            # Reset positions
            if self.fem_node:
                m_state = self.root.fem.getObject('GridMO')
                m_state.position.value = m_state.rest_position.value
            if self.net_node:
                m_state = self.root.net.getObject('GridMO')
                m_state.position.value = m_state.rest_position.value

            # Define a new random box roi
            side = randint(0, 6)
            x_min = grid['min'][0] if side == 0 else randint(grid['min'][0], grid['max'][0] - 10)
            x_max = grid['max'][0] if side == 1 else randint(x_min + 10, grid['max'][0] + 1)
            y_min = grid['min'][1] if side == 2 else randint(grid['min'][1], grid['max'][1] - 10)
            y_max = grid['max'][1] if side == 3 else randint(y_min + 10, grid['max'][1] + 1)
            z_min = grid['min'][2] if side == 4 else randint(grid['min'][2], grid['max'][2] - 10)
            z_max = grid['max'][2] if side == 5 else randint(z_min + 10, grid['max'][2] + 1)
            node.getObject('ForceBox').box.value = [[x_min, y_min, z_min, x_max, y_max, z_max]]

            # Get the intersection with the surface DOFs
            surface_indices = node.getObject('QuadTopo').quads.value.reshape(-1)
            indices = list(set(node.getObject('ForceBox').indices.value).intersection(set(surface_indices)))

            # Define a new random force vector
            F = uniform(low=-1, high=1, size=(3,))
            self.F = 20 * (F / norm(F))

            # Update the force field
            node.getObject('CFF').indices.value = indices

        # Apply the force value incrementally
        n = len(node.getObject('CFF').indices.value)
        node.getObject('CFF').forces.value = [self.F * (self.idx_step + 1) for _ in range(n)]
        self.idx_step = (self.idx_step + 1) % self.sub_steps
