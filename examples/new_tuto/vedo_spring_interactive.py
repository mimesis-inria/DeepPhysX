from numpy import ndarray, array, zeros
from vedo import Plotter, Spring, Cube, Box


class SpringScenario:

    def __init__(self,
                 spring_length: float = 1.,
                 spring_stiffness: float = 30.,
                 cube_mass: float = 20.,
                 cube_friction: float = 2.):
        """
        Render a Mass-Spring simulation with Vedo.
        (src: https://github.com/marcomusy/vedo/blob/master/examples/simulations/aspring.py)

        :param spring_length: Rest length of the spring.
        :param spring_stiffness: Stiffness value of the spring.
        :param cube_mass: Mass value of the attached cube.
        :param cube_friction: Friction coefficient between the cube and the floor.
        """

        # Spring parameters
        self.spring_length: float = spring_length
        self.spring_stiffness: float = spring_stiffness

        # Cube parameters
        self.cube_size: float = 0.2
        self.cube_mass: float = cube_mass
        self.cube_friction: float = cube_friction

        # Simulation parameters
        self.dt: float = 0.1
        self.animate: bool = False

        # State vectors of the tip of the spring
        self.A: ndarray = zeros(3,)
        self.V: ndarray = zeros(3,)
        self.X: ndarray = array([self.spring_length, 0.5 * self.cube_size, 0.])
        self.X_rest: ndarray = self.X.copy()
        self.X_init: ndarray = self.X.copy()

        # Create the Plotter to render the spring, the cube, the floor and the wall
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X_rest,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        self.mesh_cube = Cube(pos=self.X_rest, side=self.cube_size, c='tomato')
        mesh_floor = Box(size=[-0.02, 1.75 * self.spring_length, -0.01, 0., -self.cube_size, self.cube_size], c='grey')
        mesh_wall = Box(size=[-0.02, 0.02, 0., 2 * self.cube_size, -self.cube_size, self.cube_size], c='green')
        self.plt = Plotter(size=(1200, 800))
        self.plt.add(self.mesh_spring, self.mesh_cube, mesh_floor, mesh_wall)

        # Add widgets to the Plotter to control the simulation
        self.button_play = self.plt.add_button(fnc=self.play_pause, states=['Start', 'Pause'], pos=(0.9, 0.1))
        self.plt.add_button(fnc=self.reset, states=['Reset'], pos=(0.9, 0.15))
        self.slider_x = self.plt.add_slider(sliderfunc=self.slider_x, title='x_position', value=self.spring_length,
                                            xmin=0.25 * self.spring_length, xmax=1.75 * self.spring_length,
                                            pos=[(0.1, 0.1), (0.8, 0.1)])
        self.slider_z = self.plt.add_slider(sliderfunc=self.slider_z, title='z_position', value=0.,
                                            xmin=-0.5 * self.cube_size, xmax=0.5 * self.cube_size,
                                            pos=[(0.9, 0.2), (0.9, 0.9)])

        # Add a timer callback to loop on time steps
        self.plt.add_callback('timer', self.step)
        self.plt.timer_callback('start')
        self.plt.show(axes=1).close()

    def step(self, *args):
        """
        Time callback to compute a time step during the animation loop.
        """

        if self.animate:

            # Compute the next position of the tip of the spring
            F = -self.spring_stiffness * (self.X - self.X_rest) - self.cube_friction * self.V
            self.A = F / self.cube_mass
            self.V = self.V + self.A * self.dt
            self.X = self.X + self.V * self.dt + 0.5 * self.A * self.dt ** 2

            # Update the rendering view
            self.render()

    def render(self):
        """
        Render the current state of the simulation.
        """

        # Update meshes with the current state of the simulation
        self.mesh_cube.pos(self.X)
        self.mesh_spring.stretch(q1=[0., 0.5 * self.cube_size, 0.], q2=self.X)

        # Update Vedo Plotter
        self.plt.render()

    def play_pause(self):
        """
        Switch button to play / pause the simulation.
        """

        # Update the switch button and the animate flag
        self.button_play.switch()
        self.animate = not self.animate

    def reset(self):
        """
        Button to reset the simulation to the rest state.
        """

        # Stop the simulation if currently running
        if self.animate:
            self.button_play.switch()
            self.animate = False

        # Reset the state vectors
        self.A = zeros(3,)
        self.V = zeros(3,)
        self.X = self.X_rest.copy()

        # Reset the initial position of the tip of the spring
        self.X_init = self.X_rest.copy()
        self.slider_x.GetRepresentation().SetValue(self.X_init[0])
        self.slider_z.GetRepresentation().SetValue(self.X_init[2])
        self.render()

    def slider_x(self, *args):
        """
        Slider to define the initial x_position.
        """

        # Define the new initial x_position
        if not self.animate:
            self.X[0] = self.slider_x.value
            self.X_init[0] = self.slider_x.value
            self.render()

        # The slider is disabled when the simulation is running
        else:
            self.slider_x.GetRepresentation().SetValue(self.X_init[0])

    def slider_z(self, *args):
        """
        Slider to define the initial z_position.
        """

        # Define the new initial x_position
        if not self.animate:
            self.X[2] = -self.slider_z.value
            self.X_init[2] = -self.slider_z.value
            self.render()

        # The slider is disabled when the simulation is running
        else:
            self.slider_z.GetRepresentation().SetValue(self.X_init[2])


if __name__ == '__main__':
    SpringScenario()
