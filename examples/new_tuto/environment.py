from numpy import ndarray, array, zeros
from numpy.random import uniform
from vedo import Spring, Cube, Box

from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


class SpringEnvironment(BaseEnvironment):

    def __init__(self, **kwargs):

        BaseEnvironment.__init__(self, **kwargs)

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
        self.T: float = 50.

        # State vectors of the tip of the spring
        self.A: ndarray = zeros(3)
        self.V: ndarray = zeros(3)
        self.X: ndarray = array([self.spring_length, 0.5 * self.cube_size, 0.])
        self.X_rest: ndarray = self.X.copy()

        # Meshes of the cube, spring, floor and wall
        self.mesh_cube = Cube(pos=self.X_rest, side=self.cube_size, c='tomato')
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X_rest,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        self.mesh_floor = Box(pos=[-0.02, 1.75 * self.spring_length + 0.5 * self.cube_size,
                                   -0.01, 0., -self.cube_size, self.cube_size], c='grey')
        self.mesh_wall = Box(pos=[-0.02, 0.02, 0., 2 * self.cube_size, -self.cube_size, self.cube_size], c='green')

    def create(self):

        # Define random parameters
        self.spring_stiffness = uniform(10., 50.)
        self.cube_mass = uniform(1., 20.)
        self.cube_friction = uniform(0., 10.)

        # Define a random initial position
        self.X[0] = uniform(0.25 * self.spring_length, 1.75 * self.spring_length)
        self.V = zeros(3)
        self.t = 0.

    def init_database(self):

        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    def init_visualization(self):

        # Add the meshes to the rendering window
        self.visual.add_mesh(positions=self.mesh_cube.vertices, cells=self.mesh_cube.cells,
                             at=self.environment_id, c='tomato')
        self.visual.add_mesh(positions=self.mesh_spring.vertices, cells=self.mesh_spring.cells,
                             at=self.environment_id, c='grey5')
        self.visual.add_mesh(positions=self.mesh_floor.vertices, cells=self.mesh_floor.cells,
                             at=self.environment_id, c='grey')
        self.visual.add_mesh(positions=self.mesh_wall.vertices, cells=self.mesh_wall.cells,
                             at=self.environment_id, c='green')

    async def step(self):

        # Reset the simulation when the max time step is reached
        if self.t >= self.T:
            self.create()
        self.t += self.dt

        # Define the input of the network
        net_input = array([self.X[0], self.V[0], self.spring_stiffness, self.cube_mass, self.cube_friction])

        # Compute the ground truth of the network
        F = -self.spring_stiffness * (self.X - self.X_rest) - self.cube_friction * self.V
        self.A = F / self.cube_mass
        self.V = self.V + self.A * self.dt
        self.X = self.X + self.V * self.dt
        net_output = array([self.X[0], self.V[0]])

        # Set the training data
        if self.compute_training_data:
            self.set_training_data(input=net_input, ground_truth=net_output)

        if self.visual is not None:
            self.visual.update_mesh(object_id=0, positions=self.mesh_cube.vertices)
            self.visual.update_mesh(object_id=1, positions=self.mesh_spring.vertices)
            self.update_visualisation()

    def apply_prediction(self, prediction):

        X = prediction['prediction'][0][0]
        self.mesh_cube.pos([X, 0.5 * self.cube_size, 0])
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=[X, 0.5 * self.cube_size, 0.],
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        if self.visual is not None:
            self.visual.update_mesh(object_id=0, positions=self.mesh_spring.vertices)
            self.visual.update_mesh(object_id=1, positions=self.mesh_cube.vertices)
            self.update_visualisation()
