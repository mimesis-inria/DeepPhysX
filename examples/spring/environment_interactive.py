from os.path import exists
from numpy import ndarray, array, zeros
from numpy.random import uniform
from vedo import Spring, Cube, Box, Text2D, Plotter

from DeepPhysX.Core.Pipelines.BasePrediction import BasePrediction
from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig, BaseEnvironment
from DeepPhysX.Core.Database.BaseDatabaseConfig import BaseDatabaseConfig
from DeepPhysX.Torch.FC.FCConfig import FCConfig


class SpringEnvironment(BaseEnvironment):

    def __init__(self,
                 spring_length: float = 1.,
                 spring_stiffness: float = 30.,
                 cube_mass: float = 20.,
                 cube_friction: float = 2.,
                 **kwargs):

        BaseEnvironment.__init__(self, **kwargs)

        # Spring parameters
        self.spring_length: float = spring_length
        self.spring_stiffness: float = spring_stiffness

        # Cube parameters
        self.cube_size: float = 0.2
        self.cube_mass: float = cube_mass
        self.cube_friction: float = cube_friction

        # Simulation parameters
        self.dt: float = 0.1
        self.T: float = 50.
        self.t: float = 50.
        self.animate: bool = False

        # State vectors of the tip of the spring
        self.A: ndarray = zeros(3)
        self.V: ndarray = zeros(3)
        self.X: ndarray = array([self.spring_length, 0.5 * self.cube_size, 0.])
        self.X_rest: ndarray = self.X.copy()
        self.X_init: ndarray = self.X.copy()

        # State vectors predicted by the networks
        self.V_net: ndarray = self.V.copy()
        self.X_net: ndarray = self.X.copy()

        # Create the Plotter to render the spring, the cube, the floor and the wall
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X_rest,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        self.mesh_cube = Cube(pos=self.X_rest, side=self.cube_size, c='blue5').wireframe(True).linewidth(5)
        self.mesh_cube_net = Cube(pos=self.X_rest, side=self.cube_size, c='tomato')
        mesh_floor = Box(pos=[-0.02, 1.75 * self.spring_length, -0.01, 0., -self.cube_size, self.cube_size], c='grey')
        mesh_wall = Box(pos=[-0.02, 0.02, 0., 2 * self.cube_size, -self.cube_size, self.cube_size], c='green')
        self.plt = Plotter(size=(1200, 800))
        self.plt.add(self.mesh_spring, self.mesh_cube, self.mesh_cube_net, mesh_floor, mesh_wall)

        # Add the parameters description
        txt = Text2D(txt=f"Spring parameters:"
                         f"\n  spring length = {spring_length}"
                         f"\n  spring stiffness = {spring_stiffness}"
                         f"\n  cube mass = {cube_mass}"
                         f"\n  cube friction = {cube_friction}", s=0.75)
        self.plt.add(txt)

        # Add widgets to the Plotter to control the simulation
        self.button_play = self.plt.add_button(fnc=self.play_pause, states=['Start', 'Pause'], pos=(0.9, 0.1))
        self.plt.add_button(fnc=self.reset, states=['Reset'], pos=(0.9, 0.15))
        self.slider = self.plt.add_slider(sliderfunc=self.slider, title='x_position', value=self.spring_length,
                                          xmin=0.25 * self.spring_length, xmax=1.75 * self.spring_length,
                                          pos=[(0.1, 0.1), (0.8, 0.1)])

        # Add a timer callback to loop on time steps
        self.plt.add_callback('timer', self.time_step, enable_picking=False)
        self.timer_id = self.plt.timer_callback('start')

    def create(self):
        pass

    def init_database(self):

        self.define_training_fields(fields=[('input', ndarray), ('ground_truth', ndarray)])

    async def step(self):

        # Launch the Plotter
        self.plt.show(axes=1).close()
        self.set_training_data(input=zeros(6), ground_truth=zeros(2))

    def time_step(self, _):

        if self.animate:
            # Compute the prediction of the networks
            net_input = array([self.X[0], self.V[0], self.spring_length, self.spring_stiffness, self.cube_mass,
                               self.cube_friction])
            prediction = self.get_prediction(input=net_input)
            self.X_net[0], self.V_net[0] = prediction['prediction'][0]

            # Compute the ground truth
            F = -self.spring_stiffness * (self.X - self.X_rest) - self.cube_friction * self.V
            self.A = F / self.cube_mass
            self.V = self.V + self.A * self.dt
            self.X = self.X + self.V * self.dt

            # Update the rendering view
            self.render()

    def render(self):

        # Update meshes with the current state of the simulation
        self.mesh_cube.pos(self.X)
        self.mesh_cube_net.pos(self.X_net)
        self.plt.remove(self.mesh_spring)
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=[self.X_net[0], 0.5 * self.cube_size, 0.],
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        self.plt.add(self.mesh_spring)
        self.slider.GetRepresentation().SetValue(self.X_net[0])

        # Update Vedo Plotter
        self.plt.render()

    def play_pause(self, *args):

        # Update the switch button and the animate flag
        self.button_play.switch()
        self.animate = not self.animate

    def reset(self, *args):

        # Stop the simulation if currently running
        if self.animate:
            self.button_play.switch()
            self.animate = False

        # Reset the state vectors
        self.A = zeros(3)
        self.V = zeros(3)
        self.X = self.X_rest.copy()
        self.V_net = zeros(3)
        self.X_net = self.X_rest.copy()

        # Reset the initial position of the tip of the spring
        self.X_init = self.X_rest.copy()
        self.slider.GetRepresentation().SetValue(self.X_init[0])
        self.render()

    def slider(self, *args):

        # Define the new initial x_position
        if not self.animate:
            self.X[0] = self.slider.value
            self.X_net[0] = self.slider.value
            self.X_init[0] = self.slider.value
            self.render()

        # The slider is disabled when the simulation is running
        else:
            self.slider.GetRepresentation().SetValue(self.X_init[0])


if __name__ == '__main__':

    if not exists('sessions/training'):
        quit(print("Trained networks required, 'sessions/training' not found. Run training.py script first."))

    environment_config = BaseEnvironmentConfig(environment_class=SpringEnvironment,
                                               env_kwargs={'spring_length': round(uniform(0.5, 2.), 2),
                                                           'spring_stiffness': round(uniform(10., 50.), 2),
                                                           'cube_mass': round(uniform(10., 50.), 2),
                                                           'cube_friction': round(uniform(0.5, 2.), 2)})
    database_config = BaseDatabaseConfig(normalize=False)
    network_config = FCConfig(dim_layers=[6, 6, 2],
                              dim_output=2)
    trainer = BasePrediction(environment_config=environment_config,
                             database_config=database_config,
                             network_config=network_config,
                             session_dir='sessions',
                             session_name='training',
                             step_nb=1)
    trainer.execute()
