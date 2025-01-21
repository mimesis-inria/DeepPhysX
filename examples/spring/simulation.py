from numpy import ndarray, array, zeros
from numpy.random import uniform
from vedo import Spring, Cube, Box
from time import sleep

from DeepPhysX.simulation.dpx_simulation import DPXSimulation


class SpringEnvironment(DPXSimulation):

    def __init__(self, **kwargs):

        DPXSimulation.__init__(self, **kwargs)

        # Spring parameters
        self.spring_length: float = 1.
        self.spring_stiffness: float = 30.

        # Cube parameters
        self.cube_size: float = 0.2
        self.cube_mass: float = 10.5
        self.cube_friction: float = 5.

        # Simulation parameters
        self.t: float = 0.
        self.dt: float = 0.1
        self.T: float = 20.

        # State vectors of the tip of the spring
        self.A: ndarray = zeros(3)
        self.V: ndarray = zeros(3)
        self.X: ndarray = array([self.spring_length, 0.5 * self.cube_size, 0.])
        self.X_rest: ndarray = self.X.copy()

        # Meshes of the cube, spring, floor and wall
        self.mesh_cube = Cube(pos=self.X_rest, side=self.cube_size).clean()
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X_rest,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01).triangulate()
        self.mesh_floor = Box(pos=[-0.02, 1.75 * self.spring_length + 0.5 * self.cube_size,
                                   -0.01, 0., -self.cube_size, self.cube_size])
        self.mesh_wall = Box(pos=[-0.02, 0.02, 0., 2 * self.cube_size, -self.cube_size, self.cube_size])

    def create(self):

        # Define random parameters
        self.spring_stiffness = uniform(10., 50.)
        self.cube_mass = uniform(10., 20.)
        self.cube_friction = uniform(0., 1.)

        # Define a random initial position
        self.X[0] = uniform(0.25 * self.spring_length, 1.75 * self.spring_length)
        self.V = zeros(3)
        self.t = 0.

    def init_database(self):

        self.define_fields(fields=[('state', ndarray), ('displacement', ndarray)])
        # self.define_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    def init_visualization(self):

        # Add the meshes to the rendering window
        self.viewer.objects.add_mesh(positions=self.mesh_cube.vertices, cells=self.mesh_cube.cells, color='green7')
        self.viewer.objects.add_mesh(positions=self.mesh_spring.vertices, cells=self.mesh_spring.cells, color='grey')
        self.viewer.objects.add_mesh(positions=self.mesh_floor.vertices, cells=self.mesh_floor.cells, color='grey')
        self.viewer.objects.add_mesh(positions=self.mesh_wall.vertices, cells=self.mesh_wall.cells, color='grey')

    async def step(self):

        # Reset the simulation when the max time step is reached
        if self.t >= self.T:
            self.create()
        self.t += self.dt

        # Compute the ground truth of the networks
        net_input = array([self.X[0], self.V[0], self.spring_stiffness * 0.1, self.cube_mass * 0.1, self.cube_friction])
        F = -self.spring_stiffness * (self.X - self.X_rest) - self.cube_friction * self.V
        self.A = F / self.cube_mass
        self.V = self.V + self.A * self.dt
        self.X = self.X + self.V * self.dt
        net_output = array([self.X[0] - self.X_rest[0]])

        # Set the training data
        if self.compute_training_data:
            # self.set_data(input=net_input, ground_truth=net_output)
            self.set_data(state=net_input, displacement=net_output)


        self.mesh_cube.pos(self.X)
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        if self.viewer is not None:
            self.viewer.objects.update_mesh(object_id=0, positions=self.mesh_cube.vertices)
            self.viewer.objects.update_mesh(object_id=1, positions=self.mesh_spring.vertices)
            # self.update_visualisation()


class SpringEnvironmentPrediction(SpringEnvironment):

    def __init__(self, **kwargs):

        SpringEnvironment.__init__(self, **kwargs)

        self.mesh_pred = Cube(pos=self.X_rest, side=self.cube_size).clean()

    def init_visualization(self):

        SpringEnvironment.init_visualization(self)
        self.viewer.objects.add_mesh(positions=self.mesh_pred.vertices, cells=self.mesh_pred.cells, color='blue',
                                     wireframe=True, line_width=5)

    def apply_prediction(self, prediction):

        U = prediction['displacement'][0][0]
        self.mesh_pred.pos([self.X_rest[0] + U, 0.5 * self.cube_size, 0])
        if self.viewer is not None:
            self.viewer.objects.update_mesh(object_id=4, positions=self.mesh_pred.vertices)
            self.update_visualisation()
            sleep(0.01)
