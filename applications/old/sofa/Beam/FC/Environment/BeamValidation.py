"""
BeamValidation
Simulation of a Beam with FEM computed simulations.
The SOFA simulation contains two models of a Beam:
    * one to apply forces and compute deformations
    * one to apply the networks predictions
Training data are produced at each time step :
    * input : applied forces on each surface node
    * output : resulted displacement of each surface node
"""

# Python related imports
import os
import sys
import numpy as np

# Session related imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BeamTraining import BeamTraining, p_grid


class BeamValidation(BeamTraining):

    def __init__(self,
                 as_tcp_ip_client=True,
                 instance_id=1,
                 instance_nb=1,
                 compute_sample=True):

        BeamTraining.__init__(self,
                              as_tcp_ip_client=as_tcp_ip_client,
                              instance_id=instance_id,
                              instance_nb=instance_nb)

        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.compute_sample = compute_sample

    def createFEM(self):
        """
        FEM model of Beam. Used to apply forces and compute deformations.
        """

        # To compute samples, use default scene graph creation
        if self.compute_sample:
            BeamTraining.createFEM(self)

        # To read Dataset samples, no need for physical laws or solvers
        else:
            # Create FEM node
            self.root.addChild('fem')

            # Grid topology of the model
            self.f_grid_topo = self.root.fem.addObject('RegularGridTopology', name='GridTopo', min=p_grid.min.tolist(),
                                                       max=p_grid.max.tolist(), nx=p_grid.res[0], ny=p_grid.res[1],
                                                       nz=p_grid.res[2])
            self.f_grid_mo = self.root.fem.addObject('MechanicalObject', name='GridMO', src='@GridTopo')
            self.root.fem.addObject('HexahedronSetTopologyContainer', name='HexaTopo', src='@GridTopo')
            self.root.fem.addObject('HexahedronSetGeometryAlgorithms', template='Vec3d')
            self.root.fem.addObject('HexahedronSetTopologyModifier')

            # Surface
            self.f_surface_topo = self.root.fem.addObject('QuadSetTopologyContainer', name='QuadTopo', src='@HexaTopo')
            self.root.fem.addObject('QuadSetGeometryAlgorithms', template='Vec3d')
            self.root.fem.addObject('QuadSetTopologyModifier')
            self.root.fem.addObject('Hexa2QuadTopologicalMapping', input='@HexaTopo', output='@QuadTopo')

            # Fixed section
            self.root.fem.addObject('BoxROI', name='FixedBox', box=p_grid.fixed_box.tolist(), drawBoxes=True,
                                    drawSize=3.)
            self.root.fem.addObject('FixedConstraint', indices='@FixedBox.indices')

            # Forces
            self.cff_box = self.root.fem.addObject('BoxROI', name='ForceBox', box=p_grid.size, drawBoxes=False)
            self.cff = self.root.fem.addObject('ConstantForceField', name='CFF', showArrowSize=0.1,
                                               force=[0., 0., 0.], indices=list(iter(range(p_grid.nb_nodes))))

            # Visual
            self.root.fem.addChild('visual')
            self.f_visu = self.root.fem.visual.addObject('OglModel', src='@../GridTopo', color='green')
            self.root.fem.visual.addObject('BarycentricMapping', input='@../GridMO', output='@./')

    def onAnimateBeginEvent(self, event):
        """
        Called within the Sofa pipeline at the beginning of the time step. Define force vector.
        """

        if self.compute_sample:
            # Reset positions and compute new forces
            BeamTraining.onAnimateBeginEvent(self, event)

        else:
            # Reset positions
            self.f_grid_mo.position.value = self.f_grid_mo.rest_position.value
            self.n_grid_mo.position.value = self.n_grid_mo.rest_position.value

    def onAnimateEndEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the time step. Compute training data.
        """

        # Compute training data
        input_array = self.compute_input() if self.compute_sample else self.sample_training['input']
        output_array = self.compute_output() if self.compute_sample else self.sample_training['ground_truth']

        # Manually update FEM model if sample from Dataset
        if not self.compute_sample:
            U = np.reshape(self.sample_training['ground_truth'], self.data_size)
            self.f_grid_mo.position.value = self.f_grid_mo.rest_position.value + U

        # Send training data
        self.set_training_data(input=input_array,
                               ground_truth=output_array)

    def check_sample(self):
        """
        Check if the produced sample is correct. Automatically called by DeepPhysX to check sample validity.
        """

        # See the networks prediction even if the solver diverged.
        return True

    def apply_prediction(self, prediction):
        """
        Apply the predicted displacement to the NN model.
        """

        BeamTraining.apply_prediction(self, prediction)
        self.compute_metrics()

    def compute_metrics(self):
        """
        Compute L2 error and MSE for each sample.
        """

        pred = self.n_grid_mo.position.value - self.n_grid_mo.rest_position.value
        gt = self.f_grid_mo.position.value - self.f_grid_mo.rest_position.value

        # Compute metrics only for non-zero displacements
        if np.linalg.norm(gt) > 1:
            if self.solver is not None and not self.solver.converged.value:
                return
            error = (gt - pred).reshape(-1)
            self.l2_error.append(np.linalg.norm(error))
            self.MSE_error.append((error.T @ error) / error.shape[0])
            self.l2_deformation.append(np.linalg.norm(gt))
            self.MSE_deformation.append((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0])

    def close(self):

        if len(self.l2_error) > 0:
            print("\nL2 ERROR Statistics :")
            print(f"\t- Distribution : {np.mean(self.l2_error)} ± {np.std(self.l2_error)} mm")
            print(f"\t- Extrema : {np.min(self.l2_error)} -> {np.max(self.l2_error)} mm")
            relative_error = np.array(self.l2_error) / np.array(self.l2_deformation)
            print(f"\t- Relative Distribution : {1e2 * relative_error.mean()} ± {1e2 * relative_error.std()} %")
            print(f"\t- Relative Extrema : {1e2 * relative_error.min()} -> {1e2 * relative_error.max()} %")

            print("\nMSE Statistics :")
            print(f"\t- Distribution : {np.mean(self.MSE_error)} ± {np.std(self.MSE_error)} mm²")
            print(f"\t- Extrema : {np.min(self.MSE_error)} -> {np.max(self.MSE_error)} mm²")
            relative_error = np.array(self.MSE_error) / np.array(self.MSE_deformation)
            print(f"\t- Relative Distribution : {1e2 * relative_error.mean()} ± {1e2 * relative_error.std()} %")
            print(f"\t- Relative Extrema : {1e2 * relative_error.min()} -> {1e2 * relative_error.max()} %")
