from time import sleep
from numpy import ndarray, array, zeros
from numpy.random import uniform
from vedo import Spring, Cube, Box

from DeepPhysX.simulation import Simulation


class SpringEnvironment(Simulation):

    def __init__(self, **kwargs):
        """
        The numerical simulation of the scenario in a DeepPhysX compatible implementation.
        This class is used for the data generation pipeline.
        """

        Simulation.__init__(self, **kwargs)

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
        self.mesh_cube_init = self.mesh_cube.vertices
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X_rest,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01).triangulate()
        self.mesh_floor = Box(pos=[-0.02, 1.75 * self.spring_length + 0.5 * self.cube_size,
                                   -0.01, 0., -self.cube_size, self.cube_size])
        self.mesh_wall = Box(pos=[-0.02, 0.02, 0., 2 * self.cube_size, -self.cube_size, self.cube_size])

    def create(self):
        """
        Create the Simulation. Automatically called when the simulation is launched.
        """

        # Define random parameters
        self.spring_stiffness = uniform(10., 50.)
        self.cube_mass = uniform(10., 20.)
        self.cube_friction = uniform(0., 1.)

        # Define a random initial position
        self.X[0] = uniform(0.25 * self.spring_length, 1.75 * self.spring_length)
        self.V = zeros(3)
        self.t = 0.

    def init_database(self):
        """
        Define the fields of the training database. Automatically called when Simulation is launched.
        """

        # Define the training data fields 'state' and 'displacement'
        for field_name in ('state', 'displacement'):
            self.add_data_field(field_name=field_name, field_type=ndarray)

    def init_visualization(self):
        """
        Define the 3D objects to render in the viewer. Automatically called when Simulation is launched.
        """

        # Add the meshes to the rendering window
        self.viewer.objects.add_mesh(positions=self.mesh_cube.vertices, cells=self.mesh_cube.cells, color='green7')
        self.viewer.objects.add_mesh(positions=self.mesh_spring.vertices, cells=self.mesh_spring.cells, color='grey')
        self.viewer.objects.add_mesh(positions=self.mesh_floor.vertices, cells=self.mesh_floor.cells, color='grey')
        self.viewer.objects.add_mesh(positions=self.mesh_wall.vertices, cells=self.mesh_wall.cells, color='grey')

    def step(self):
        """
        Compute a time step of the numerical simulation. Automatically called when a data sample is requested.
        """

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
            self.set_data(state=net_input, displacement=net_output)

        self.mesh_cube.vertices = self.mesh_cube_init + self.X - self.X_rest
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        if self.viewer is not None:
            self.viewer.objects.update_mesh(object_id=0, positions=self.mesh_cube.vertices)
            self.viewer.objects.update_mesh(object_id=1, positions=self.mesh_spring.vertices)
            self.viewer.render()


class SpringEnvironmentPrediction(SpringEnvironment):

    def __init__(self, **kwargs):
        """
        The numerical simulation of the scenario in a DeepPhysX compatible implementation.
        This class is used for the prediction pipeline.
        """

        SpringEnvironment.__init__(self, **kwargs)

        # Add a mesh to apply the predictions of the network and compare with the real solution
        self.mesh_pred = Cube(pos=self.X_rest, side=self.cube_size).clean()

    def init_visualization(self):
        """
        Define the 3D objects to render in the viewer. Automatically called when Simulation is launched.
        """

        SpringEnvironment.init_visualization(self)
        self.viewer.objects.add_mesh(positions=self.mesh_pred.vertices, cells=self.mesh_pred.cells, color='blue',
                                     wireframe=True, line_width=5)

    def apply_prediction(self, prediction):
        """
        Apply networks prediction in the numerical simulation. Automatically called when the prediction is sent to the
        simulation.
        """

        # Get the data field 'displacement' produced by the network
        U = prediction['displacement'][0]

        # Update the cube position
        self.mesh_pred.pos([self.X_rest[0] + U, 0.5 * self.cube_size, 0])
        if self.viewer is not None:
            self.viewer.objects.update_mesh(object_id=4, positions=self.mesh_pred.vertices)
            self.viewer.render()
            sleep(0.01)
