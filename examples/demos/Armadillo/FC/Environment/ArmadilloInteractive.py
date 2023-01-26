"""
Armadillo
Simulation of an Armadillo with deformation predictions from a Fully Connected.
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
from math import pow

# DeepPhysX related imports
from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment
from DeepPhysX.Core.Utils.Visualizer.GridMapping import GridMapping

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parameters import p_model, p_forces


# Create an Environment as a BaseEnvironment child class
class Armadillo(BaseEnvironment):

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
        self.mesh_coarse = None
        self.sparse_grid = None
        self.mapping = None
        self.mapping_coarse = None

        # Plotter
        self.plotter = None
        self.selected = None
        self.interactive_window = True
        self.mouse_factor = 10 * p_model.scale
        self.key_on = False
        self.click_on = False

        # Force fields
        self.arrows = None
        self.areas = []
        self.sphere = lambda center: Sphere(center, r=0.5 * p_model.scale, c='lightgray').lighting('glossy')
        self.spheres = []
        self.spheres_init = []

        # Data sizes
        self.input_size = (p_model.nb_nodes_mesh, 3)
        self.output_size = (p_model.nb_nodes_grid, 3)

    def init_database(self):

        # Define the fields of the Training database
        self.define_training_fields(fields=[('input', np.ndarray), ('ground_truth', np.ndarray)])

    def create(self):

        # Load the meshes and the sparse grid
        self.mesh = Mesh(p_model.mesh, c='o').scale(p_model.scale).lighting('plastic')
        self.mesh_init = self.mesh.clone().points()
        self.mesh_coarse = Mesh(p_model.mesh_coarse).scale(p_model.scale)
        self.sparse_grid = Mesh(p_model.sparse_grid)

        # Init the mappings
        self.mapping = GridMapping(self.sparse_grid, self.mesh)
        self.mapping_coarse = GridMapping(self.sparse_grid, self.mesh_coarse)

        # Define force fields
        sphere = lambda x, y: sum([pow(x_i - y_i, 2) for x_i, y_i in zip(x, y)])
        for zone in p_forces.zones:
            self.areas.append([])
            # Find the spherical area
            for i, pts in enumerate(self.mesh_coarse.points()):
                if sphere(pts, p_forces.centers[zone]) <= pow(p_forces.radius[zone], 2):
                    self.areas[-1].append(i)
            # Create sphere at initial state
            self.spheres_init.append(self.mesh_coarse.points(self.areas[-1]).mean(axis=0))
            self.spheres.append(self.sphere(self.spheres_init[-1]).alpha(0.5))

        # Define fixed plane
        mesh_y = self.mesh.points()[:, 1]
        fixed = np.where(mesh_y <= np.min(mesh_y) + 0.05 * (np.max(mesh_y) - np.min(mesh_y)))
        plane_origin = [np.mean(self.mesh.points()[:, 0][fixed]),
                        np.min(mesh_y),
                        np.mean(self.mesh.points()[:, 2][fixed])]

        # Create plotter
        self.plotter = Plotter(title='Interactive Armadillo', N=1, interactive=True, offscreen=False, bg2='lightgray')
        self.plotter.render()
        self.plotter.add(*self.spheres)
        self.plotter.add(self.mesh)
        self.plotter.add(Plane(pos=plane_origin, normal=[0, 1, 0], s=(10 * p_model.scale, 10 * p_model.scale),
                               c='darkred', alpha=0.2))
        self.plotter.add(Text2D("Press 'b' to interact with the spheres / with the environment.\n"
                                "Left click to select / unselect a sphere.", s=0.75))

        # Add callbacks
        self.plotter.add_callback('KeyPress', self.key_press)
        self.plotter.add_callback('LeftButtonPress', self.left_button_press)
        self.plotter.add_callback('MouseMove', self.mouse_move)

    async def step(self):

        # Launch Vedo window
        self.plotter.show().close()
        # Smooth close
        self.set_training_data(input=np.zeros(self.input_size),
                               ground_truth=np.zeros(self.output_size))

    def key_press(self, evt):

        # Only react to an 'Alt' press
        if 'b' in evt.keyPressed.lower():
            self.key_on = not self.key_on
            if self.key_on:
                # Switch from environment to object interaction
                self.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyle3D())
                self.interactive_window = False
                self.update_spheres()
            else:
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
            self.click_on = not self.click_on
            if self.click_on:
                # Pick a unique sphere
                if evt.actor in self.spheres:
                    self.selected = self.spheres.index(evt.actor)
                    self.update_spheres(center=self.spheres_init[self.selected])
            else:
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
            move_3D = (mouse_3D - self.spheres_init[self.selected]) / self.mouse_factor
            if np.linalg.norm(move_3D) > 2:
                move_3D = 2 * move_3D / np.linalg.norm(move_3D)
            F = np.zeros(self.input_size)
            amp = p_forces.amplitude[p_forces.zones[self.selected]]
            F[self.areas[self.selected]] = move_3D * amp

            # Apply output displacement
            U = self.get_prediction(input=F)['prediction'].reshape(self.output_size)
            updated_grid = self.sparse_grid.points().copy() + U
            updated_coarse = self.mapping_coarse.apply(updated_grid)

            # Update view
            self.update_mesh(positions=self.mapping.apply(updated_grid))
            self.update_arrows(positions=updated_coarse, vectors=0.1 * (F / amp) * self.mouse_factor)
            self.update_spheres(center=updated_coarse[self.areas[self.selected]].mean(axis=0))

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
        if self.interactive_window:
            self.spheres = [self.sphere(c).alpha(0.5) for c in self.spheres_init]
        elif center is None:
            self.spheres = [self.sphere(c) for c in self.spheres_init]
        # Otherwise, update the selected cell
        else:
            self.spheres = [self.sphere(center).alpha(0.5)]
        # Update the view
        self.plotter.add(*self.spheres)

    def close(self):

        # Shutdown message
        print("Bye!")
