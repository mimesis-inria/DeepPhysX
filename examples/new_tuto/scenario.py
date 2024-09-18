from numpy import ndarray, array, zeros
from vedo import Plotter, Spring, Cube, Box, Text3D
from vedo.pyplot import plot


class SpringScenario:

    def __init__(self):
        """
        Render a Mass-Spring simulation with Vedo.
        Users can interactively change the position of the cube and the values of the stiffness, mass and friction.
        src: https://github.com/marcomusy/vedo/blob/master/examples/simulations/aspring.py
        """

        # Spring parameters
        self.spring_length: float = 1.
        self.spring_stiffness: float = 30.

        # Cube parameters
        self.cube_size: float = 0.2
        self.cube_mass: float = 10.5
        self.cube_friction: float = 5.

        # Simulation parameters
        self.dt: float = 0.1
        self.animate: bool = False

        # State vectors of the tip of the spring
        self.A: ndarray = zeros(3)
        self.V: ndarray = zeros(3)
        self.X: ndarray = array([self.spring_length, 0.5 * self.cube_size, 0.])
        self.X_rest: ndarray = self.X.copy()

        # Create the Vedo objects (cube, spring, floor and wall)
        self.mesh_cube = Cube(pos=self.X_rest, side=self.cube_size, c='tomato')
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X_rest,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        self.mesh_floor = Box(pos=[-0.02, 1.75 * self.spring_length + 0.5 * self.cube_size,
                                   -0.01, 0., -self.cube_size, self.cube_size], c='grey')
        self.mesh_wall = Box(pos=[-0.02, 0.02, 0., 2 * self.cube_size, -self.cube_size, self.cube_size], c='green')

        # Create the Plotter
        self.plt = Plotter(size=(1200, 800))
        self.plt.add(self.mesh_cube, self.mesh_spring, self.mesh_floor, self.mesh_wall)

        # Add widgets to the Plotter to control the simulation
        self.button_play = self.plt.add_button(fnc=self.play_pause, states=['Start', 'Pause'], pos=(0.9, 0.15))
        self.button_reset = self.plt.add_button(fnc=self.reset, states=['Reset'], pos=(0.9, 0.1))
        self.slider_position = self.plt.add_slider(sliderfunc=self.change_position, title='x position',
                                                   xmin=0.25 * self.spring_length, xmax=1.75 * self.spring_length,
                                                   value=self.spring_length, pos=[(0.1, 0.075), (0.8, 0.075)],
                                                   title_size=0.75)
        self.slider_stiffness = self.plt.add_slider3d(sliderfunc=self.change_stiffness, show_value=False,
                                                      xmin=10., xmax=50., value=self.spring_stiffness,
                                                      pos1=[0.1, 0.55, -self.cube_size],
                                                      pos2=[0.8, 0.55, -self.cube_size])
        self.slider_mass = self.plt.add_slider3d(sliderfunc=self.change_mass, show_value=False,
                                                 xmin=1., xmax=20., value=self.cube_mass,
                                                 pos1=[0.1, 0.65, -self.cube_size],
                                                 pos2=[0.8, 0.65, -self.cube_size])
        self.slider_friction = self.plt.add_slider3d(sliderfunc=self.change_friction, show_value=False,
                                                     xmin=0., xmax=10., value=self.cube_friction,
                                                     pos1=[0.1, 0.75, -self.cube_size],
                                                     pos2=[0.8, 0.75, -self.cube_size])
        self.text_stiffness = Text3D(txt=f'spring stiffness = {self.spring_stiffness}',
                                     pos=[0.12, 0.575, -self.cube_size], s=0.025)
        self.text_mass = Text3D(txt=f'cube mass = {self.cube_mass}',
                                pos=[0.12, 0.675, -self.cube_size], s=0.025)
        self.text_friction = Text3D(txt=f'cube friction = {self.cube_friction}',
                                    pos=[0.12, 0.775, -self.cube_size], s=0.025)
        self.plt.add(self.text_stiffness, self.text_mass, self.text_friction)

        # Position plot
        self.curve_t = [0.]
        self.curve_x = [self.X[0]]
        self.curve = plot(self.curve_t, self.curve_x, c='r5', xlim=(0, 10), ylim=(0, 2), grid=False)
        self.curve.scale(0.075).shift(1., 0.2425, -self.cube_size)
        self.plt.add(self.curve)

        # Add a timer callback to loop on time steps
        self.plt.add_callback('timer', self.step, enable_picking=False)
        self.plt.timer_callback('start')
        self.plt.show(axes=2, zoom='tight').close()

    def play_pause(self, *args):
        """
        Switch button to play / pause the simulation.
        """

        # Update the switch button and the animate flag
        self.button_play.switch()
        self.animate = not self.animate

        # Update the last position value in the curve
        if self.animate:
            self.curve_x[-1] = self.X[0]

    def reset(self, *args):
        """
        Button to reset the simulation to the rest state.
        """

        # Stop the simulation if currently running
        if self.animate:
            self.button_play.switch()
            self.animate = False

        # Reset the state vectors
        self.A = zeros(3, )
        self.V = zeros(3, )
        self.X = self.X_rest.copy()

        # Reset the curve
        self.curve_t = [0.]
        self.curve_x = [self.X[0]]

        # Render the rest state
        self.render()

    def change_position(self, *args):
        """
        Slider to define the initial x_position.
        """

        # Define the new initial x_position
        if not self.animate:
            self.X[0] = self.slider_position.value
        self.render()

    def change_stiffness(self, *args):
        """
        Slider to define the spring stiffness.
        """

        # Define the new stiffness value
        if not self.animate:
            self.spring_stiffness = round(self.slider_stiffness.value, 2)
            self.text_stiffness.text(txt=f'spring stiffness = {self.spring_stiffness}')
        # Slider is locked during the animation
        else:
            self.slider_stiffness.value = self.spring_stiffness

    def change_mass(self, *args):
        """
        Slider to change the cube mass.
        """

        # Define the new mass value
        if not self.animate:
            self.cube_mass = round(self.slider_mass.value, 2)
            self.text_mass.text(txt=f'cube mass = {self.cube_mass}')
        # Slider is locked during the animation
        else:
            self.slider_mass.value = self.cube_mass

    def change_friction(self, *args):
        """
        Slider to change the cube friction.
        """

        # Define the new friction value
        if not self.animate:
            self.cube_friction = round(self.slider_friction.value, 2)
            self.text_friction.text(txt=f'cube friction = {self.cube_friction}')
        # Slider is locked during the animation
        else:
            self.slider_friction.value = self.cube_friction

    def step(self, *args):
        """
        Time callback to compute a time step during the animation loop.
        """

        if self.animate:

            # Compute the next position of the tip of the spring
            F = -self.spring_stiffness * (self.X - self.X_rest) - self.cube_friction * self.V
            self.A = F / self.cube_mass
            self.V = self.V + self.A * self.dt
            self.X = self.X + self.V * self.dt

            # Update the curve
            self.curve_t.append(self.curve_t[-1] + self.dt)
            self.curve_x.append(self.X[0])

            # Update the rendering view
            self.render()

    def render(self):
        """
        Render the current state of the simulation.
        """

        # Update the cube mesh
        self.mesh_cube.pos(self.X)

        # Update the spring mesh
        self.plt.remove(self.mesh_spring)
        self.mesh_spring = Spring(start_pt=[0., 0.5 * self.cube_size, 0.], end_pt=self.X,
                                  coils=int(15 * self.spring_length), r1=0.05, thickness=0.01)
        self.plt.add(self.mesh_spring)

        # Update the position slider
        self.slider_position.value = self.X[0]

        # Update the plot
        self.plt.remove(self.curve)
        self.curve = plot(self.curve_t[:100], self.curve_x[-100:], lc='r5', xlim=(0, 10), ylim=(0, 2), grid=False)
        self.curve.scale(0.075).shift(1., 0.2425, -self.cube_size)
        self.plt.add(self.curve)

        # Finally render
        self.plt.render()


if __name__ == '__main__':
    SpringScenario()
