from typing import Type
from socket import socket

from DeepPhysX.simulation.multiprocess.tcpip_object import TcpIpObject
from DeepPhysX.simulation.simulation_controller import Simulation, SimulationController


class TcpIpClient(TcpIpObject):

    def __init__(self,
                 simulation: Type[Simulation],
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 instance_id: int = 0,
                 instance_nb: int = 1):
        """
        TcpIpClient is a TcpIpObject which communicate with a TcpIpServer and manages a Simulation to compute data.

        :param simulation: Simulation class.
        :param ip_address: IP address of the TcpIpObject.
        :param port: Port number of the TcpIpObject.
        :param instance_id: Index of this instance.
        :param instance_nb: Number of simultaneously launched instances.
        """

        TcpIpObject.__init__(self)

        # Simulation instance
        self.simulation_class = simulation
        self.simulation_instance = (instance_id, instance_nb)
        self.simulation_controller: SimulationController

        # Bind to client address and send ID
        self.sock.connect((ip_address, port))
        self.send_labeled_data(data_to_send=instance_id, label="instance_ID",
                               receiver=self.sock, send_read_command=False)
        self.close_client: bool = False

    ###########################
    # Initializing Simulation #
    ###########################

    def initialize(self) -> None:
        """
        Receive parameters from the server to create simulation.
        """

        # Receive additional arguments
        env_kwargs = self.receive_dict(sender=self.sock)
        env_kwargs = env_kwargs['env_kwargs'] if 'env_kwargs' in env_kwargs else {}

        self.simulation_controller = SimulationController(simulation_class=self.simulation_class,
                                                          simulation_kwargs=env_kwargs,
                                                          manager=self,
                                                          simulation_id=self.simulation_instance[0],
                                                          simulation_nb=self.simulation_instance[1])

        # Receive prediction requests authorization
        self.allow_prediction_requests = self.receive_data(sender=self.sock)

        # Receive number of sub-steps
        self.simulations_per_step = self.receive_data(sender=self.sock)

        # Receive visualization database
        viewer_key = self.receive_data(sender=self.sock)
        viewer_key = None if viewer_key == 'None' else int(viewer_key)

        # Initialize the simulation
        self.simulation_controller.create_simulation(use_viewer=viewer_key is not None, viewer_key=viewer_key)

        # Initialization done
        self.send_data(data_to_send='done', receiver=self.sock)

        # Synchronize Database
        database_path = (self.receive_data(sender=self.sock), self.receive_data(sender=self.sock))
        normalize_data = self.receive_data(sender=self.sock)
        self.simulation_controller.connect_to_database(database_path=database_path, normalize_data=normalize_data)
        self.send_data(data_to_send='done', receiver=self.sock)

    ##################
    # Running Client #
    ##################

    def launch(self) -> None:
        """
        Trigger the main communication protocol with the server.
        """

        try:
            # Run the communication protocol with server while Client is not asked to shut down
            while not self.close_client:
                self.__communicate(server=self.sock)
        except KeyboardInterrupt:
            print("[TcpIpClient] KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            # Closing procedure when Client is asked to shut down
            self.__close()

    def __communicate(self, server: socket) -> None:
        """
        Communication protocol with a server. First receive a command from the client, then process the appropriate
        actions.

        :param server: TcpIpServer to communicate with.
        """

        self.listen_while_not_done(sender=server)

    def __close(self) -> None:
        """
        Close the simulation and shutdown the client.
        """

        # Close simulation
        self.simulation_controller.close()

        # Confirm exit command to the server
        self.send_command_exit(receiver=self.sock)

        # Close socket
        self.sock.close()

    ################################
    # Available requests to Server #
    ################################

    def get_prediction(self, *args, **kwargs) -> None:
        """
        Request a prediction from networks.

        :return: Prediction of the networks.
        """

        self.send_command_prediction()
        _ = self.receive_data(sender=self.sock)

    ##################################
    # Actions to perform on commands #
    ##################################

    def action_on_exit(self, client_id: int, sender: socket) -> None:
        """
        Action to run when receiving the 'exit' command.

        :param client_id: ID of the TcpIpClient.
        :param sender: TcpIpObject sender.
        """

        # Close client flag set to True
        self.close_client = True

    def action_on_prediction(self, client_id: int, sender: socket) -> None:
        """
        Action to run when receiving the 'prediction' command.

        :param client_id: ID of the TcpIpClient.
        :param sender: TcpIpObject sender.
        """

        # Receive prediction
        prediction = self.receive_data(sender=sender)
        # Apply the prediction in the simulation
        self.simulation_controller.simulation.apply_prediction(prediction)

    def action_on_sample(self, client_id: int, sender: socket) -> None:
        """
        Action to run when receiving the 'sample' command.

        :param client_id: ID of the TcpIpClient.
        :param sender: TcpIpObject sender.
        """

        dataset_batch = self.receive_data(sender=sender)
        self.simulation_controller.trigger_get_data(dataset_batch)

    def action_on_step(self, client_id: int, sender: socket) -> None:
        """
        Action to run when receiving the 'step' command.

        :param client_id: ID of the TcpIpClient.
        :param sender: TcpIpObject sender.
        """

        # Execute the required number of steps
        for step in range(self.simulations_per_step):
            # Compute data only on final step
            self.simulation_controller.compute_training_data = step == self.simulations_per_step - 1
            self.simulation_controller.simulation.step()

        # If produced sample is not usable, run again
        while not self.simulation_controller.simulation.check_sample():
            for step in range(self.simulations_per_step):
                # Compute data only on final step
                self.simulation_controller.compute_training_data = step == self.simulations_per_step - 1
                self.simulation_controller.simulation.step()

        # Sent training data to Server
        line = self.simulation_controller.trigger_send_data()
        self.simulation_controller.reset_data()
        self.send_command_done(receiver=sender)
        self.send_data(data_to_send=line, receiver=sender)
