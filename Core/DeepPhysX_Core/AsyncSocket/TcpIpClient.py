from typing import Any, Dict, Optional
from socket import socket
from asyncio import get_event_loop
from asyncio import AbstractEventLoop as EventLoop
from asyncio import run as async_run
from numpy import ndarray, array

from DeepPhysX_Core.AsyncSocket.TcpIpObject import TcpIpObject
from DeepPhysX_Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment


class TcpIpClient(TcpIpObject, AbstractEnvironment):
    """
    | TcpIpClient is both a TcpIpObject which communicate with a TcpIpServer and an AbstractEnvironment to compute
      simulated data.

    :param str ip_address: IP address of the TcpIpObject
    :param int port: Port number of the TcpIpObject
    :param as_tcp_ip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False
    :param int instance_id: ID of the instance
    :param int number_of_instances: Number of simultaneously launched instances
    """

    def __init__(self,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 as_tcp_ip_client: bool = True,
                 instance_id: int = 0,
                 number_of_instances: int = 1,):

        AbstractEnvironment.__init__(self,
                                     as_tcp_ip_client=as_tcp_ip_client,
                                     instance_id=instance_id,
                                     number_of_instances=number_of_instances)

        # Bind to client address
        if self.as_tcp_ip_client:
            TcpIpObject.__init__(self,
                                 ip_address=ip_address,
                                 port=port)
            self.sock.connect((ip_address, port))
            # Send ID
            self.sync_send_labeled_data(data_to_send=instance_id, label="instance_ID", receiver=self.sock,
                                        send_read_command=False)
        # Flag to trigger client's shutdown
        self.close_client: bool = False

    ##########################################################################################
    ##########################################################################################
    #                                 Initializing Environment                               #
    ##########################################################################################
    ##########################################################################################

    def initialize(self) -> None:
        """
        | Run __initialize method with asyncio.
        """

        async_run(self.__initialize())

    async def __initialize(self) -> None:
        """
        | Receive parameters from the server to create environment, send parameters to the server in exchange.
        """

        loop = get_event_loop()
        # Receive number of sub-steps
        self.simulations_per_step = await self.receive_data(loop=loop, sender=self.sock)
        # Receive parameters
        recv_param_dict = {}
        await self.receive_dict(recv_to=recv_param_dict, sender=self.sock, loop=loop)
        # Use received parameters
        if 'parameters' in recv_param_dict:
            self.recv_parameters(recv_param_dict['parameters'])

        # Create the environment
        self.create()
        self.init()
        # Send visualization
        visu_dict = self.send_visualization()
        self.send_visualization_data(visualization_data=visu_dict, receiver=self.sock)
        # Send parameters
        param_dict = self.send_parameters()
        await self.send_dict(name="parameters", dict_to_send=param_dict, loop=loop, receiver=self.sock)

        # Initialization done
        await self.send_command_done(loop=loop, receiver=self.sock)

    ##########################################################################################
    ##########################################################################################
    #                                      Running Client                                    #
    ##########################################################################################
    ##########################################################################################

    def launch(self) -> None:
        """
        | Run __launch method with asyncio.
        """

        async_run(self.__launch())

    async def __launch(self) -> None:
        """
        | Trigger the main communication protocol with the server.
        """

        try:
            # Run the communication protocol with server while Client is not asked to shut down
            while not self.close_client:
                await self.__communicate(server=self.sock)
        except KeyboardInterrupt:
            print(f"[{self.name}] KEYBOARD INTERRUPT: CLOSING PROCEDURE")
        finally:
            # Closing procedure when Client is asked to shut down
            await self.__close()

    async def __communicate(self, server: socket) -> None:
        """
        | Communication protocol with a server. First receive a command from the client, then process the appropriate
          actions.

        :param socket server: TcpIpServer to communicate with
        """

        loop = get_event_loop()
        await self.listen_while_not_done(loop=loop, sender=server, data_dict={})

    async def __close(self) -> None:
        """
        | Close the environment and shutdown the client.
        """

        # Close environment
        try:
            self.close()
        except NotImplementedError:
            pass
        # Confirm exit command to the server
        loop = get_event_loop()
        await self.send_command_exit(loop=loop, receiver=self.sock)
        # Close socket
        self.sock.close()

    ##########################################################################################
    ##########################################################################################
    #                                  Data sending to Server                                #
    ##########################################################################################
    ##########################################################################################

    async def __send_training_data(self, loop: Optional[EventLoop] = None, receiver: Optional[socket] = None) -> None:
        """
        | Send the training data to the TcpIpServer.

        :param Optional[EventLoop] loop: get_event_loop() return
        :param socket receiver: TcpIpObject receiver
        """

        loop = get_event_loop() if loop is None else loop
        receiver = self.sock if receiver is None else receiver

        # Send and reset network input
        if self.input.tolist():
            await self.send_labeled_data(data_to_send=self.input, label="input", loop=loop, receiver=receiver)
            self.input = array([])
        # Send network output
        if self.output.tolist():
            await self.send_labeled_data(data_to_send=self.output, label="output", loop=loop, receiver=receiver)
            self.output = array([])
        # Send loss data
        if self.loss_data:
            await self.send_labeled_data(data_to_send=self.loss_data, label='loss', loop=loop, receiver=receiver)
            self.loss_data = None
        # Send additional dataset fields
        for key in self.additional_fields.keys():
            await self.send_labeled_data(data_to_send=self.additional_fields[key], label='dataset_' + key,
                                         loop=loop, receiver=receiver)
        self.additional_fields = {}

    def send_prediction_data(self, network_input: ndarray, receiver: socket = None) -> ndarray:
        """
        | Request a prediction from the Environment.

        :param ndarray network_input: Data to send under the label 'input'
        :param socket receiver: TcpIpObject receiver
        :return: Prediction of the Network
        """

        receiver = self.sock if receiver is None else receiver
        # Send prediction command
        self.sync_send_command_prediction()
        # Send the network input
        self.sync_send_labeled_data(data_to_send=network_input, label='input', receiver=receiver)
        # Receive the network prediction
        _, pred = self.sync_receive_labeled_data()
        return pred

    def send_visualization_data(self, visualization_data: Dict[int, Dict[str, Any]], receiver: socket = None) -> None:
        """
        | Send the visualization data to TcpIpServer.

        :param Dict[int, Dict[str, Any]] visualization_data: Updated visualization data.
        :param socket receiver: TcpIpObject receiver
        """

        receiver = self.sock if receiver is None else receiver
        # Send 'visualization' command
        self.sync_send_command_visualisation()
        # Send visualization data
        self.sync_send_dict(name="visualisation", dict_to_send=visualization_data, receiver=receiver)

    ##########################################################################################
    ##########################################################################################
    #                              Available requests to Server                              #
    ##########################################################################################
    ##########################################################################################

    def request_get_prediction(self, input_array: ndarray) -> ndarray:
        """
        | Request a prediction from Network.

        :param ndarray input_array: Network input
        :return: Prediction of the Network
        """

        return self.send_prediction_data(network_input=input_array)

    def request_update_visualization(self, visu_dict: Dict[int, Dict[str, Any]]) -> None:
        """
        | Triggers the Visualizer update.

        :param Dict[int, Dict[str, Any]] visu_dict: Updated visualization data.
        """

        self.send_visualization_data(visualization_data=visu_dict)

    ##########################################################################################
    ##########################################################################################
    #                            Actions to perform on commands                              #
    ##########################################################################################
    ##########################################################################################

    async def action_on_exit(self, data: ndarray, client_id: int, sender: socket, loop: EventLoop) -> None:
        """
        | Action to run when receiving the 'exit' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param EventLoop loop: asyncio.get_event_loop() return
        :param socket sender: TcpIpObject sender
        """

        # Close client flag set to True
        self.close_client = True

    async def action_on_prediction(self, data: ndarray, client_id: int, sender: socket, loop: EventLoop) -> None:
        """
        | Action to run when receiving the 'prediction' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param EventLoop loop: asyncio.get_event_loop() return
        :param socket sender: TcpIpObject sender
        """

        # Receive prediction
        prediction = await self.receive_data(loop=loop, sender=sender)
        # Apply the prediction in Environment
        self.apply_prediction(prediction)

    async def action_on_sample(self, data: ndarray, client_id: int, sender: socket, loop: EventLoop) -> None:
        """
        | Action to run when receiving the 'sample' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param EventLoop loop: asyncio.get_event_loop() return
        :param socket sender: TcpIpObject sender
        """

        # Receive input sample
        if await self.receive_data(loop=loop, sender=sender):
            self.sample_in = await self.receive_data(loop=loop, sender=sender)
        # Receive output sample
        if await self.receive_data(loop=loop, sender=sender):
            self.sample_out = await self.receive_data(loop=loop, sender=sender)

        additional_fields = {}
        # Receive additional input sample if there are any
        if await self.receive_data(loop=loop, sender=sender):
            await self.receive_dict(recv_to=additional_fields, loop=loop, sender=sender)

        # Set the samples from Dataset
        self.additional_fields = additional_fields.get('additional_fields', {})

    async def action_on_step(self, data: ndarray, client_id: int, sender: socket, loop: EventLoop) -> None:
        """
        | Action to run when receiving the 'step' command

        :param dict data: Dict storing data
        :param int client_id: ID of the TcpIpClient
        :param EventLoop loop: asyncio.get_event_loop() return
        :param socket sender: TcpIpObject sender
        :return:
        """

        # Execute the required number of steps
        for step in range(self.simulations_per_step):
            # Compute data only on final step
            self.compute_essential_data = step == self.simulations_per_step - 1
            await self.step()

        # If produced sample is not usable, run again
        # Todo: add the max_rate here
        if self.sample_in is None and self.sample_out is None:
            while not self.check_sample():
                for step in range(self.simulations_per_step):
                    # Compute data only on final step
                    self.compute_essential_data = step == self.simulations_per_step - 1
                    await self.step()

        # Sent training data to Server
        await self.__send_training_data()
        await self.send_command_done()
