from typing import Any, Dict, Type, Optional, Union, Tuple
from socket import socket
from asyncio import get_event_loop
from asyncio import AbstractEventLoop as EventLoop
from asyncio import run as async_run
from numpy import ndarray

from DeepPhysX.Core.AsyncSocket.TcpIpObject import TcpIpObject
from DeepPhysX.Core.AsyncSocket.AbstractEnvironment import AbstractEnvironment


class TcpIpClient(TcpIpObject):

    def __init__(self,
                 environment: Type[AbstractEnvironment],
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 instance_id: int = 0,
                 instance_nb: int = 1,
                 visualization_db: Optional[Union[Tuple[str, str]]] = None):
        """
        TcpIpClient is a TcpIpObject which communicate with a TcpIpServer and manages an Environment to compute data.

        :param environment: Environment class.
        :param ip_address: IP address of the TcpIpObject.
        :param port: Port number of the TcpIpObject.
        :param instance_id: Index of this instance.
        :param instance_nb: Number of simultaneously launched instances.
        :param visualization_db: Path to the visualization Database.
        """

        TcpIpObject.__init__(self,
                             ip_address=ip_address,
                             port=port)

        # Create instance
        self.environment = environment(as_tcp_ip_client=True,
                                       instance_id=instance_id,
                                       instance_nb=instance_nb,
                                       visualization_db=visualization_db)
        self.environment.tcp_ip_client = self

        # Bind to client address and send ID
        self.sock.connect((ip_address, port))
        self.sync_send_labeled_data(data_to_send=instance_id, label="instance_ID", receiver=self.sock,
                                    send_read_command=False)
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

        # Receive prediction requests authorization
        self.allow_prediction_requests = await self.receive_data(loop=loop, sender=self.sock)

        # Receive number of sub-steps
        self.simulations_per_step = await self.receive_data(loop=loop, sender=self.sock)

        # Receive partitions
        partitions_list = await self.receive_data(loop=loop, sender=self.sock)
        partitions_list, exchange = partitions_list.split('%%%')
        partitions = [[partitions_list.split('///')[0], partition_name]
                      for partition_name in partitions_list.split('///')[1:]]
        exchange = [exchange.split('///')[0], exchange.split('///')[1]]
        self.environment.get_database_handler().init_remote(storing_partitions=partitions,
                                                            exchange_db=exchange)

        # Initialize the environment
        self.environment.create()
        self.environment.init()
        self.environment.init_database()
        self.environment.init_visualization()

        # Initialization done
        await self.send_data(data_to_send='done', loop=loop, receiver=self.sock)

        # Synchronize Database
        _ = await self.receive_data(loop=loop, sender=self.sock)
        self.environment.get_database_handler().load()

    ##########################################################################################
    ##########################################################################################
    #                                      Running Client                                    #
    ##########################################################################################
    ##########################################################################################

    def launch(self) -> None:
        """
        Trigger the main communication protocol with the server.
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
            self.environment.close()
        except NotImplementedError:
            pass
        # Confirm exit command to the server
        loop = get_event_loop()
        await self.send_command_exit(loop=loop, receiver=self.sock)
        # Close socket
        self.sock.close()

    ##########################################################################################
    ##########################################################################################
    #                              Available requests to Server                              #
    ##########################################################################################
    ##########################################################################################

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]:
        """
        Request a prediction from Network.

        :return: Prediction of the Network.
        """

        # Get a prediction
        self.environment.get_database_handler().update(table_name='Exchange',
                                                       data=kwargs,
                                                       line_id=self.environment.instance_id)
        self.sync_send_command_prediction()
        _ = self.sync_receive_data()
        data_pred = self.environment.get_database_handler().get_line(table_name='Exchange',
                                                                     line_id=self.environment.instance_id)
        del data_pred['id']
        return data_pred

    def request_update_visualization(self) -> None:
        """
        Triggers the Visualizer update.
        """

        self.sync_send_command_visualisation()
        self.sync_send_labeled_data(data_to_send=self.environment.instance_id,
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
        self.environment.apply_prediction(prediction)

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
        self.environment._get_training_data(dataset_batch)

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
            await self.environment.step()

        # If produced sample is not usable, run again
        while not self.environment.check_sample():
            for step in range(self.simulations_per_step):
                # Compute data only on final step
                self.compute_training_data = step == self.simulations_per_step - 1
                await self.environment.step()

        # Sent training data to Server
        if self.environment.update_line is None:
            line = self.environment._send_training_data()
        else:
            self.environment._update_training_data(self.environment.update_line)
            line = self.environment.update_line
        self.environment._reset_training_data()
        await self.send_command_done(loop=loop, receiver=sender)
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

        # Update the partition list in the DatabaseHandler
        new_database = await self.receive_data(loop=loop, sender=sender)
        self.environment.get_database_handler().update_list_partitions_remote(new_database.split('///'))
