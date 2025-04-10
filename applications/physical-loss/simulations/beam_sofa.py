from os.path import join, dirname
from functools import reduce
from math import prod
from numpy import zeros
from numpy.random import randint, uniform
from numpy.linalg import norm

import SofaRuntime
import SofaCaribou

from DeepPhysX.simulation import SofaSimulation


grid = {'min': [0., 0., 0.],
        'max': [180., 15., 15.],
        'res': [60, 5, 5],
        'fix': [0., 0., 0., 0., 15., 15.]}
grid['n'] = prod(grid['res'])
forces = {'radius': 7, 'nb': 2, 'amplitude': 50}


class BeamSofa(SofaSimulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.idx_step = 0
        self.sub_steps = 50

    def create(self):

        # Add required plugins
        with open(join(dirname(__file__), 'plugins.txt'), 'r') as f:
            plugins = [plugin[:-1] if plugin.endswith('\n') else plugin for plugin in f.readlines() if plugin != '\n']
        SofaRuntime.PluginRepository.addFirstPath(reduce(lambda d, _: dirname(d), range(4), SofaCaribou.__file__))
        self.root.addObject('RequiredPlugin', pluginName=plugins)

        self.root.addObject('VisualStyle', displayFlags='showVisualModels showWireframe showBehaviorModels showForceFields')
        self.root.addObject('DefaultAnimationLoop')

        # Add fem node and / or nn node
        self.create_fem_node()
        self.create_net_node()

    def create_fem_node(self):

        # FEM root node
        self.root.addChild('fem')
        self.root.fem.addObject('EulerImplicitSolver', name='ODESolver', rayleighMass=0.01, rayleighStiffness=0.01,
                                vdamping=5)
        self.root.fem.addObject('CGLinearSolver', name='LinearSolver', iterations=1000, threshold=1e-8,
                                tolerance=1e-8)

        # Grid topology
        self.root.fem.addObject('RegularGridTopology', name='GridTopo', min=grid['min'], max=grid['max'],
                                nx=grid['res'][0], ny=grid['res'][1], nz=grid['res'][2])
        self.root.fem.addObject('MechanicalObject', name='GridMO', src='@GridTopo')
        self.root.fem.addObject('HexahedronSetTopologyContainer', name='HexaTopo', src='@GridTopo')
        self.root.fem.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.fem.addObject('HexahedronSetTopologyModifier')

        # Surface topology
        self.root.fem.addObject('QuadSetTopologyContainer', name='QuadTopo', src='@HexaTopo')
        self.root.fem.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.fem.addObject('QuadSetTopologyModifier')
        self.root.fem.addObject('Hexa2QuadTopologicalMapping', input='@HexaTopo', output='@QuadTopo')

        # Material
        self.root.fem.addObject('NeoHookeanMaterial', name='NH', young_modulus=500, poisson_ratio=0.45)
        self.root.fem.addObject('HyperelasticForcefield', template="Hexahedron", material="@NH", topology='@HexaTopo')
        self.root.fem.addObject('UniformMass', totalMass=10.)

        # Constraints
        self.root.fem.addObject('BoxROI', name='FixedBox', box=grid['fix'])
        self.root.fem.addObject('FixedConstraint', indices='@FixedBox.indices')

        # Forces
        for i in range(forces['nb']):
            self.root.fem.addObject('SphereROI', name=f'Sphere_{i}', radii=forces['radius'], drawSphere=False)
            self.root.fem.addObject('ConstantForceField', name=f'CFF_{i}', forces=zeros(3), indices=0,
                                    showArrowSize=0.5)

        # Visual
        self.root.fem.addChild('visual')
        self.root.fem.visual.addObject('OglModel', name='OGL', src='@../GridTopo', color='green')
        self.root.fem.visual.addObject('BarycentricMapping', input='@../GridMO', output='@./')

    def create_net_node(self):

        # Network root node
        self.root.addChild('net')
        self.root.net.addObject('EulerImplicitSolver', name='ODESolver')
        self.root.net.addObject('CGLinearSolver', name='LinearSolver', iterations=0)

        # Grid topology
        self.root.net.addObject('RegularGridTopology', name='GridTopo', min=grid['min'], max=grid['max'],
                                nx=grid['res'][0], ny=grid['res'][1], nz=grid['res'][2])
        self.root.net.addObject('MechanicalObject', name='GridMO', src='@GridTopo')
        self.root.net.addObject('HexahedronSetTopologyContainer', name='HexaTopo', src='@GridTopo')
        self.root.net.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
        self.root.net.addObject('HexahedronSetTopologyModifier')

        # Surface topology
        self.root.net.addObject('QuadSetTopologyContainer', name='QuadTopo', src='@HexaTopo')
        self.root.net.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
        self.root.net.addObject('QuadSetTopologyModifier')
        self.root.net.addObject('Hexa2QuadTopologicalMapping', input='@HexaTopo', output='@QuadTopo')

        # Visual
        self.root.net.addChild('visual')
        self.root.net.visual.addObject('OglModel', name='OGL', src='@../GridTopo', color='green')
        self.root.net.visual.addObject('BarycentricMapping', input='@../GridMO', output='@./')

    def onAnimateBeginEvent(self, event):

        if self.idx_step == 0:

            # Reset positions and velocities
            m_state = self.root.fem.getObject('GridMO')
            m_state.position.value = m_state.rest_position.value
            m_state.velocity.value = zeros(m_state.velocity.value.shape)

            # Select new SphereROI centers
            centers = []
            surface_pos = self.root.fem.getObject('QuadTopo').position.value
            for i in range(forces['nb']):
                # Select a random surface point
                center = surface_pos[randint(len(surface_pos))]
                # Check that the sphere does not intersect others
                intersect_spheres = False
                for c in centers:
                    if norm(center - c) <= forces['radius'] * 2:
                        intersect_spheres = True
                        break
                # Update ROI and force
                roi = self.root.fem.getObject(f'Sphere_{i}')
                cff = self.root.fem.getObject(f'CFF_{i}')
                if not intersect_spheres:
                    centers.append(center)
                    roi.centers.value = [center]
                    roi.init()
                    cff.indices.value = roi.indices.value
                    f = uniform(low=-1, high=1, size=(3,))
                    cff.forces.value = [forces['amplitude'] * f / norm(f)] * len(roi.indices.value)
                else:
                    roi.centers.value = [grid['min']]
                    cff.indices.value = [0]
                    cff.forces.value = zeros((1, 3))

        self.idx_step = (self.idx_step + 1) % self.sub_steps
