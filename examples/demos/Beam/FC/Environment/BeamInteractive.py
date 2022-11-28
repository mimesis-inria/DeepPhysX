"""
BeamInteractive
Simulation of a Beam with deformation predictions from a Fully Connected.
Training data are produced at each time step:
    * input: applied forces on each surface node
    * prediction: resulted displacement of each surface node
"""

# Python related imports
import os
import sys
import vtk

import numpy as np
from vedo import Mesh, Sphere, Arrows, Plotter, Text2D, Plane

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model


# Create an Environment as a BaseEnvironment child class
class Beam(BaseEnvironment):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1):

        BaseEnvironment.__init__(self,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 instance_id=instance_id,
                                 instance_nb=instance_nb)

        # Topologies & mappings
        self.mesh = None
        self.mesh_init = None
        self.surface = None

        # Plotter
        self.plotter = None
        self.selected = None
        self.interactive_window = True
        self.mouse_factor = 10

        # Force fields
        self.arrows = None
        self.areas = []
        self.sphere = lambda center: Sphere(center, r=1, c='lightgray').lighting('glossy')
        self.spheres = []
        self.spheres_init = []

        # Data sizes
        self.data_size = (p_model.nb_nodes, 3)

    def init_database(self):

        # Define the fields of the Training database
        self.define_training_fields(fields=[('input', np.ndarray), ('ground_truth', np.ndarray)])

    def create(self):

        # Load the meshes and the sparse grid
        self.mesh = Mesh(p_model.grid, c='o').lineWidth(0.1).lighting('ambient')
        self.mesh.shift(0, -7.5, -7.5)
        self.mesh_init = self.mesh.clone().points()

        # Get the surface points
        pts = self.mesh.points().copy()
        self.surface = np.concatenate((np.argwhere(pts[:, 0] == np.min(pts[:, 0])),
                                       np.argwhere(pts[:, 0] == np.max(pts[:, 0])),
                                       np.argwhere(pts[:, 1] == np.min(pts[:, 1])),
                                       np.argwhere(pts[:, 1] == np.max(pts[:, 1])),
                                       np.argwhere(pts[:, 2] == np.min(pts[:, 2])),
                                       np.argwhere(pts[:, 2] == np.max(pts[:, 2])))).reshape(-1)
        self.surface = np.unique(self.surface)

        # Define force fields at corners and edges
        bounds = self.mesh.bounds()
        sx, sy, sz = bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
        for x in np.arange(np.min(self.mesh_init[:, 0]), np.max(self.mesh_init[:, 0]) + sx / 8, sx / 4):
            if x != sx / 2:
                for y in (np.min(self.mesh_init[:, 1]), np.max(self.mesh_init[:, 1])):
                    for z in (np.min(self.mesh_init[:, 2]), np.max(self.mesh_init[:, 2])):
                        center = [x, y, z]
                        self.spheres_init.append(self.mesh_init.copy().tolist().index(center))
                        self.spheres.append(self.sphere(self.mesh.points()[self.spheres_init[-1]]))
                        x_min, x_max = x - sx / 4, x + sx / 4
                        y_min, y_max = y - sy / 2, y + sy / 2
                        z_min, z_max = z - sz / 2, z + sz / 2
                        zone = np.where((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (pts[:, 1] >= y_min) &
                                        (pts[:, 1] <= y_max) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max))
                        self.areas.append(np.array(list(set(zone[0].tolist()).intersection(set(list(self.surface))))))

        # Define force fields at faces
        for i in range(3):
            for a in (np.min(self.mesh_init[:, i]), np.max(self.mesh_init[:, i])):
                center = [sx / 2, sy / 2, sz / 2]
                center[i] = a
                self.spheres_init.append(self.mesh_init.copy().tolist().index(center))
                self.spheres.append(self.sphere(self.mesh.points()[self.spheres_init[-1]]))
                other = [0, 1, 2]
                other.remove(i)
                o0_min, o0_max = np.min(self.mesh_init[:, other[0]]), np.max(self.mesh_init[:, other[0]])
                o1_min, o1_max = np.min(self.mesh_init[:, other[1]]), np.max(self.mesh_init[:, other[1]])
                zone = np.where((pts[:, i] == a) & (pts[:, other[0]] >= o0_min) & (pts[:, other[0]] <= o0_max) &
                                (pts[:, other[1]] >= o1_min) & (pts[:, other[1]] <= o1_max))
                self.areas.append(np.array(list(set(zone[0].tolist()).intersection(set(list(self.surface))))))

        # Create plotter
        self.plotter = Plotter(title='Interactive Beam', N=1, interactive=True, offscreen=False, bg2='lightgray')
        self.plotter.render()
        self.plotter.add(*self.spheres)
        self.plotter.add(self.mesh)
        self.plotter.add(Plane(pos=[0., 0., 0.], normal=[1, 0, 0], s=(20, 20), c='darkred', alpha=0.2))
        self.plotter.add(Text2D("Press 'Alt' to interact with the object.\n"
                                "Left click to select a sphere.\n"
                                "Right click to unselect a sphere.", s=0.75))

        # Add callbacks
        self.plotter.addCallback('KeyPress', self.key_press)
        self.plotter.addCallback('KeyRelease', self.key_release)
        self.plotter.addCallback('LeftButtonPress', self.left_button_press)
        self.plotter.addCallback('RightButtonPress', self.right_button_press)
        self.plotter.addCallback('MouseMove', self.mouse_move)

    async def step(self):

        # Launch Vedo window
        self.plotter.show().close()
        # Smooth close
        self.set_training_data(input=np.zeros(self.data_size),
                               ground_truth=np.zeros(self.data_size))

    def key_press(self, evt):

        # Only react to an 'Alt' press
        if 'alt' in evt.keyPressed.lower():
            # Switch from environment to object interaction
            self.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyle3D())
            self.interactive_window = False

    def key_release(self, evt):

        # Only react with an 'Alt' release
        if 'alt' in evt.keyPressed.lower():
            # Switch from object to environment interaction
            self.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.interactive_window = True
            self.selected = None
            # Reset all
            self.update_mesh()
            self.update_arrows()
            self.update_spheres()

    def left_button_press(self, evt):

        # Select a sphere only in object interaction mode
        if not self.interactive_window:
            # Pick a unique sphere
            if evt.actor in self.spheres:
                self.selected = self.spheres.index(evt.actor)
                self.update_spheres(center=self.spheres_init[self.selected])

    def right_button_press(self, evt):

        # Unselect a sphere only in object interaction mode
        if not self.interactive_window:
            self.selected = None
            # Reset all
            self.update_mesh()
            self.update_arrows()
            self.update_spheres()

    def mouse_move(self, evt):

        # Only react on mouse movement in object interaction mode and if a sphere is picked
        if not self.interactive_window and self.selected is not None:

            # Compute input force vector
            mouse_3D = self.plotter.compute_world_position(evt.picked2d)
            move_3D = (mouse_3D - self.mesh_init[self.spheres_init[self.selected]]) / self.mouse_factor
            if np.linalg.norm(move_3D) > 10:
                move_3D = 10 * move_3D / np.linalg.norm(move_3D)
            F = np.zeros(self.data_size)
            F[self.areas[self.selected]] = move_3D * 2

            # Apply output displacement
            U = self.get_prediction(input=F)['prediction'].reshape(self.data_size)
            updated_grid = self.mesh_init + U

            # Update view
            self.update_mesh(positions=updated_grid)
            self.update_arrows(positions=updated_grid, vectors=F / 2)
            self.update_spheres(center=self.spheres_init[self.selected])

    def update_mesh(self, positions=None):

        # If no positions provided, reset the mesh
        if positions is None:
            self.mesh.points(self.mesh_init)
        # Otherwise, update the mesh positions
        else:
            self.mesh.points(positions)

    def update_arrows(self, positions=None, vectors=None):

        # Remove actual arrows
        if self.arrows is not None:
            self.plotter.remove(self.arrows)
        # If no positions provided, arrows are not created
        if positions is None or vectors is None:
            self.arrows = None
        # Otherwise, create new arrows
        else:
            self.arrows = Arrows(positions, positions + vectors, c='green')
            self.plotter.add(self.arrows)

    def update_spheres(self, center=None):

        # Remove actual spheres
        self.plotter.remove(*self.spheres)
        # If no center provided, reset all the spheres
        if center is None:
            self.spheres = [self.sphere(self.mesh_init[c]) for c in self.spheres_init]
        # Otherwise, update the selected cell
        else:
            self.spheres = [self.sphere(self.mesh.points()[center]).alpha(0.5)]
        # Update the view
        self.plotter.add(*self.spheres)

    def close(self):

        # Shutdown message
        print("Bye!")
