from typing import Any, Dict
from socket import socket
from asyncio import get_event_loop
from asyncio import AbstractEventLoop as EventLoop
from asyncio import run as async_run
from numpy import ndarray

from DeepPhysX.Core.AsyncSocket.TcpIpObject import TcpIpObject
from DeepPhysX.Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment, Database


class TcpIpClient(TcpIpObject, AbstractEnvironment):

    def __init__(self,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 as_tcp_ip_client: bool = True,
                 instance_id: int = 0,
                 number_of_instances: int = 1,):
        """
        TcpIpClient is both a TcpIpObject which communicate with a TcpIpServer and an AbstractEnvironment to compute
        simulated data.

        :param ip_address: IP address of the TcpIpObject.
        :param port: Port number of the TcpIpObject.
        :param as_tcp_ip_client: Environment is a TcpIpObject if True, is owned by an EnvironmentManager if False.
        :param instance_id: ID of the instance.
        :param number_of_instances: Number of simultaneously launched instances.
        """

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
        Receive parameters from the server to create environment.
        """

        async_run(self.__initialize())

    async def __initialize(self) -> None:
        """
        Receive parameters from the server to create environment.
        """

        loop = get_event_loop()

        # Receive number of sub-steps
        self.simulations_per_step = await self.receive_data(loop=loop, sender=self.sock)

        # Create the environment
        self.create()
        self.init()
        self.init_database()
        self.init_visualization()

        # Initialization done
        await self.send_command_done(loop=loop, receiver=self.sock)

    ##########################################################################################
    ##########################################################################################
    #                                      Running Client                                    #
    ##########################################################################################
    ##########################################################################################

    def launch(self) -> None:
        """
        Run __launch method with asyncio.
        """

        async_run(self.__launch())

    async def __launch(self) -> None:
        """
        Trigger the main communication protocol with the server.
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

    async def __communicate(self,
                            server: socket) -> None:
        """
        Communication protocol with a server. First receive a command from the client, then process the appropriate
        actions.

        :param server: TcpIpServer to communicate with.
        """

        loop = get_event_loop()
        await self.listen_while_not_done(loop=loop, sender=server, data_dict={})

    async def __close(self) -> None:
        """
        Close the environment and shutdown the client.
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

    def send_prediction_data(self,
                             network_input: ndarray,
                             receiver: socket = None) -> ndarray:
        """
        Request a prediction from the Environment.

        :param network_input: Data to send under the label 'input'.
        :param receiver: TcpIpObject receiver.
        :return: Prediction of the Network.
        """

        receiver = self.sock if receiver is None else receiver
        # Send prediction command
        self.sync_send_command_prediction()
        # Send the network input
        self.sync_send_labeled_data(data_to_send=network_input, label='input', receiver=receiver)
        # Receive the network prediction
        _, pred = self.sync_receive_labeled_data()
        return pred

    ##########################################################################################
    ##########################################################################################
    #                              Available requests to Server                              #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self,
                       **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from Network.

        :param input_array: Network input.
        :return: Prediction of the Network.
        """

        # Get a prediction
        self.database.update(table_name='Prediction',
                             data=kwargs,
                             line_id=self.instance_id)
        self.sync_send_command_prediction()
        _ = self.sync_receive_data()
        data_pred = self.database.get_line(table_name='Prediction',
                                           line_id=self.instance_id)
        del data_pred['id']
        return data_pred

    def request_update_visualization(self) -> None:
        """
        Triggers the Visualizer update.
        """

        self.sync_send_command_visualisation()
        self.sync_send_labeled_data(data_to_send=self.instance_id,
                                    label='instance')

    ##########################################################################################
    ##########################################################################################
    #                            Actions to perform on commands                              #
    ##########################################################################################
    ##########################################################################################

    async def action_on_exit(self,
                             data: ndarray,
                             client_id: int,
                             sender: socket,
                             loop: EventLoop) -> None:
        """
        Action to run when receiving the 'exit' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param loop: Asyncio event loop.
        :param sender: TcpIpObject sender.
        """

        # Close client flag set to True
        self.close_client = True

    async def action_on_prediction(self,
                                   data: ndarray,
                                   client_id: int,
                                   sender: socket,
                                   loop: EventLoop) -> None:
        """
        Action to run when receiving the 'prediction' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param loop: Asyncio event loop.
        :param sender: TcpIpObject sender.
        """

        # Receive prediction
        prediction = await self.receive_data(loop=loop, sender=sender)
        # Apply the prediction in Environment
        self.apply_prediction(prediction)

    async def action_on_sample(self,
                               data: ndarray,
                               client_id: int,
                               sender: socket,
                               loop: EventLoop) -> None:
        """
        Action to run when receiving the 'sample' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param loop: Asyncio event loop.
        :param sender: TcpIpObject sender.
        """

        dataset_batch = await self.receive_data(loop=loop, sender=sender)
        self._get_training_data(dataset_batch)

    async def action_on_step(self,
                             data: ndarray,
                             client_id: int,
                             sender: socket,
                             loop: EventLoop) -> None:
        """
        Action to run when receiving the 'step' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param loop: Asyncio event loop.
        :param sender: TcpIpObject sender.
        """

        # Execute the required number of steps
        for step in range(self.simulations_per_step):
            # Compute data only on final step
            self.compute_training_data = step == self.simulations_per_step - 1
            await self.step()

        # If produced sample is not usable, run again
        while not self.check_sample():
            for step in range(self.simulations_per_step):
                # Compute data only on final step
                self.compute_training_data = step == self.simulations_per_step - 1
                await self.step()

        # Sent training data to Server
        if self.update_line is None:
            line = self._send_training_data()
        else:
            self._update_training_data(self.update_line)
            line = self.update_line
        self._reset_training_data()
        await self.send_command_done()
        await self.send_data(data_to_send=line, loop=loop, receiver=sender)

    async def action_on_change_db(self, 
                                  data: Dict[Any, Any], 
                                  client_id: int, sender: socket,
                                  loop: EventLoop) -> None:
        """
        Action to run when receiving the 'step' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param loop: Asyncio event loop.
        :param sender: TcpIpObject sender.
        """
        
        new_database = (await self.receive_data(loop=loop, sender=sender),
                        await self.receive_data(loop=loop, sender=sender))
        self.database = Database(database_dir=new_database[0],
                                 database_name=new_database[1]).load()
