import os
import sys
import vtk
import numpy as np
from vedo import Mesh, Points, Sphere, Arrows, Plotter, Text2D, Plane

from DeepPhysX.simulation import Simulation
from DeepPhysX.utils.grid.sparse_grid import get_regular_grid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .beam_simulation import grid


# Create an Environment as a BaseEnvironment child class
class BeamInteractive(Simulation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Load the meshes and the sparse grid
        pos, quads = get_regular_grid(grid_min=grid['min'], grid_max=grid['max'], grid_res=grid['res'])
        self.grid = Mesh([pos, quads]).c('orange').linewidth(0.1).lighting('ambient')
        self.grid_init = Points(pos)

        # Get the surface points
        pts = np.array(self.grid.vertices)
        self.surface = np.concatenate((np.argwhere(pts[:, 0] == np.min(pts[:, 0])),
                                       np.argwhere(pts[:, 0] == np.max(pts[:, 0])),
                                       np.argwhere(pts[:, 1] == np.min(pts[:, 1])),
                                       np.argwhere(pts[:, 1] == np.max(pts[:, 1])),
                                       np.argwhere(pts[:, 2] == np.min(pts[:, 2])),
                                       np.argwhere(pts[:, 2] == np.max(pts[:, 2])))).reshape(-1)
        self.surface = np.unique(self.surface)

        # Plotter
        self.plotter = Plotter(title='Interactive Beam', N=1, interactive=True, offscreen=False, bg2='lightgray')
        self.object_mode = False

        # Force fields
        self.areas = []
        self.arrows = []
        self.spheres = []
        self.spheres_idx = []
        self.selected = None

    def init_database(self):

        for field in ('forces', 'displacement'):
            self.add_data_field(field_name=field, field_type=np.ndarray)

    def create(self):

        grid_min, grid_max = np.min(self.grid.vertices, axis=0), np.max(self.grid.vertices, axis=0)
        grid_size = grid_max - grid_min

        # Define force fields at corners and edges
        for x in np.arange(grid_min[0], grid_max[0] + grid_size[0] / 8, grid_size[0] / 4):
            if x != grid_size[0] / 2:
                for y in (grid_min[1], grid_max[1]):
                    for z in (grid_min[2], grid_max[2]):
                        center = [x, y, z]
                        self.spheres.append(Sphere(pos=center, r=1, c='lightgrey', alpha=0.5).lighting('glossy'))
                        self.spheres_idx.append(self.grid_init.closest_point(pt=center, return_point_id=True))
                        self.areas.append(np.array(self.grid_init.closest_point(pt=center, radius=20,
                                                                                return_point_id=True)))
        # Define force fields at faces
        for i in range(3):
            for a in (grid_min[i], grid_max[i]):
                center = 0.5 * grid_size
                center[i] = a
                self.spheres.append(Sphere(pos=center, r=1, c='lightgrey', alpha=0.5).lighting('glossy'))
                self.spheres_idx.append(self.grid_init.closest_point(pt=center, return_point_id=True))
                self.areas.append(np.array(self.grid_init.closest_point(pt=center, radius=20,
                                                                        return_point_id=True)))
        self.plotter.add(self.grid)
        self.plotter.add(*self.spheres)
        self.plotter.add(Plane(pos=0.5 * (grid_min + grid_max), s=2 * (grid_max - grid_min)[1:],
                               normal=[1, 0, 0], c='darkred', alpha=0.2).x(0))
        self.plotter.add(Text2D("Press 'b' to interact with the spheres / with the environment.\n"
                                "Left click to select / unselect a sphere.", s=0.75))

        # Add callbacks
        self.plotter.add_callback('KeyPress', self.key_press)
        self.plotter.add_callback('LeftButtonPress', self.left_button_press)
        self.plotter.add_callback('RightButtonPress', self.right_button_press)
        self.plotter.add_callback('MouseMove', self.mouse_move)

    def step(self):

        # Launch Vedo window
        self.plotter.show().close()

        # Smooth close
        self.set_data(forces=np.zeros((grid['n'], 3)), displacement=np.zeros((grid['n'], 3)))

    def reset_all(self):

        self.selected = None
        self.update_mesh()
        self.update_arrows()
        self.update_spheres()

    def key_press(self, evt):

        # Only react to an 'Alt' press
        if 'b' in evt.keypress.lower():
            self.object_mode = not self.object_mode

            # Switch from environment to object interaction
            if self.object_mode:
                self.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyle3D())
                self.update_spheres()

            # Switch from object to environment interaction
            else:
                self.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
                self.reset_all()

            self.plotter.render()

    def left_button_press(self, evt):

        # Select a sphere only in object interaction mode
        if self.object_mode and evt.actor in self.spheres:
            self.selected = self.spheres.index(evt.actor)
            self.update_spheres(selected=self.selected)
            self.plotter.render()

    def right_button_press(self, evt):

        if self.object_mode:
            self.reset_all()
            self.plotter.render()

    def mouse_move(self, evt):

        # Only react on mouse movement in object interaction mode and if a sphere is picked
        if self.object_mode and self.selected is not None:

            # Compute input force vector
            pos = self.plotter.compute_world_coordinate(evt.picked2d)
            d_pos = (pos - self.grid.vertices[self.spheres_idx[self.selected]])
            F = np.zeros((grid['n'], 3))
            F[self.areas[self.selected]] = d_pos
            F = 50 * F / np.linalg.norm(F)

            # Get the predicted displacement and update meshes
            new_pos = self.grid_init.vertices + self.get_prediction(forces=F)['displacement']
            self.update_mesh(positions=new_pos)
            self.update_arrows(positions=new_pos, vectors=F)
            self.update_spheres(selected=self.selected)
            self.plotter.render()

    def update_mesh(self, positions=None):

        # If no positions provided, reset the mesh
        if positions is None:
            self.grid.vertices = self.grid_init.vertices
        # Otherwise, update the mesh positions
        else:
            self.grid.vertices = positions

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

    def update_spheres(self, selected: int = -1):

        # In environment mode, reset all the spheres
        if not self.object_mode:
            for idx, sphere in zip(self.spheres_idx, self.spheres):
                sphere.pos(self.grid_init.vertices[idx]).alpha(0.5)

        # In object mode, hide all the spheres and update the selected one
        else:
            for idx, sphere in zip(self.spheres_idx, self.spheres):
                sphere.pos(self.grid_init.vertices[idx]).alpha(1 if selected == -1 else 0)
            if selected != -1:
                self.spheres[selected].pos(self.grid.vertices[self.spheres_idx[selected]]).alpha(1.)
