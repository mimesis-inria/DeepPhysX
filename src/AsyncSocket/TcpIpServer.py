from typing import Any, Dict, List, Optional
from asyncio import get_event_loop, gather
from asyncio import AbstractEventLoop as EventLoop
from asyncio import run as async_run
from socket import socket
from queue import SimpleQueue

from DeepPhysX.Core.AsyncSocket.TcpIpObject import TcpIpObject
from DeepPhysX.Core.Database.DatabaseHandler import DatabaseHandler


class TcpIpServer(TcpIpObject):

    def __init__(self,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 nb_client: int = 5,
                 max_client_count: int = 10,
                 batch_size: int = 5,
                 manager: Optional[Any] = None):
        """
        TcpIpServer is used to communicate with clients associated with Environment to produce batches for the
        EnvironmentManager.

        :param ip_address: IP address of the TcpIpObject.
        :param port: Port number of the TcpIpObject.
        :param nb_client: Number of expected client connections.
        :param max_client_count: Maximum number of allowed clients.
        :param batch_size: Number of samples in a batch.
        :param manager: EnvironmentManager that handles the TcpIpServer.
        """

        super(TcpIpServer, self).__init__(ip_address=ip_address,
                                          port=port)

        # Bind to server address
        print(f"[{self.name}] Binding to IP Address: {ip_address} on PORT: {port} with maximum client count: "
              f"{max_client_count}")
        self.sock.bind((ip_address, port))
        self.sock.listen(max_client_count)
        self.sock.setblocking(False)

        # Expect a defined number of clients
        self.clients: List[List[int, socket]] = []
        self.nb_client: int = min(nb_client, max_client_count)

        # Init data to communicate with EnvironmentManager and Clients
        self.batch_size: int = batch_size
        self.data_fifo: SimpleQueue = SimpleQueue()
        self.data_dict: Dict[Any, Any] = {}
        self.sample_to_client_id: List[int] = []
        self.batch_from_dataset: Optional[List[int]] = None
        self.first_time: bool = True
        self.data_lines: List[List[int]] = []

        # Reference to EnvironmentManager
        self.environment_manager: Optional[Any] = manager

        # Connect the Server to the Database
        self.database_handler = DatabaseHandler(on_partitions_handler=self.__database_handler_partitions)
        self.environment_manager.data_manager.connect_handler(self.database_handler)

    ##########################################################################################
    ##########################################################################################
    #                              DatabaseHandler management                                #
    ##########################################################################################
    ##########################################################################################

    def get_database_handler(self) -> DatabaseHandler:
        """
        Get the DatabaseHandler of the TcpIpServer.
        """

        return self.database_handler

    def __database_handler_partitions(self) -> None:
        """
        Partition update event of the DatabaseHandler.
        """

        # Send the new partition to every Client
        for _, client in self.clients:
            self.sync_send_command_change_db(receiver=client)
            new_partition = self.database_handler.get_partitions()[-1]
            self.sync_send_data(data_to_send=f'{new_partition.get_path()[0]}///{new_partition.get_path()[1]}',
                                receiver=client)

    ##########################################################################################
    ##########################################################################################
    #                                     Connect Clients                                    #
    ##########################################################################################
    ##########################################################################################

    def connect(self) -> None:
        """
        Accept connections from clients.
        """

        print(f"[{self.name}] Waiting for clients...")
        async_run(self.__connect())

    async def __connect(self) -> None:
        """
        Accept connections from clients.
        """

        loop = get_event_loop()
        # Accept clients connections one by one
        for _ in range(self.nb_client):
            # Accept connection
            client, _ = await loop.sock_accept(self.sock)
            # Get the instance ID
            label, client_id = await self.receive_labeled_data(loop=loop, sender=client)
            print(f"[{self.name}] Client n°{client_id} connected: {client}")
            self.clients.append([client_id, client])

    ##########################################################################################
    ##########################################################################################
    #                                 Initialize Environment                                 #
    ##########################################################################################
    ##########################################################################################

    def initialize(self) -> None:
        """
        Send parameters to the clients to create their environments.
        """

        print(f"[{self.name}] Initializing clients...")
        async_run(self.__initialize())

    async def __initialize(self) -> None:
        """
        Send parameters to the clients to create their environments.
        """

        loop = get_event_loop()

        # Initialisation process for each client
        for client_id, client in self.clients:

            # Send prediction request authorization
            await self.send_data(data_to_send=self.environment_manager.allow_prediction_requests,
                                 loop=loop, receiver=client)

            # Send number of sub-steps
            nb_steps = self.environment_manager.simulations_per_step if self.environment_manager else 1
            await self.send_data(data_to_send=nb_steps, loop=loop, receiver=client)

            # Send partitions
            partitions = self.database_handler.get_partitions()
            if len(partitions) == 0:
                partitions_list = 'None'
            else:
                partitions_list = partitions[0].get_path()[0]
                for partition in partitions:
                    partitions_list += f'///{partition.get_path()[1]}'
            partitions_list += '%%%'
            exchange = self.database_handler.get_exchange()
            if exchange is None:
                partitions += 'None'
            else:
                partitions_list += f'{exchange.get_path()[0]}///{exchange.get_path()[1]}'
            await self.send_data(data_to_send=partitions_list, loop=loop, receiver=client)

            # Wait Client init
            await self.receive_data(loop=loop, sender=client)
            print(f"[{self.name}] Client n°{client_id} initialisation done")

    ##########################################################################################
    ##########################################################################################
    #                          Data: produce batch & dispatch batch                          #
    ##########################################################################################
    ##########################################################################################

    def get_batch(self,
                  animate: bool = True) -> List[List[int]]:
        """
        Build a batch from clients samples.

        :param animate: If True, triggers an environment step.
        """

        # Trigger communication protocol
        async_run(self.__request_data_to_clients(animate=animate))
        return self.data_lines

    async def __request_data_to_clients(self,
                                        animate: bool = True) -> None:
        """
        Trigger a communication protocol for each client. Wait for all clients before to launch another communication
        protocol while the batch is not full.

        :param animate: If True, triggers an environment step
        """

        nb_sample = 0
        self.data_lines = []
        # Launch the communication protocol while the batch needs to be filled
        while nb_sample < self.batch_size:
            # Run communicate protocol for each client and wait for the last one to finish
            await gather(*[self.__communicate(client=client,
                                              client_id=client_id,
                                              animate=animate) for client_id, client in self.clients])
            nb_sample += len(self.clients)

    async def __communicate(self,
                            client: Optional[socket] = None,
                            client_id: Optional[int] = None,
                            animate: bool = True) -> None:
        """
        Communication protocol with a client.

        :param client: TcpIpObject client to communicate with.
        :param client_id: Index of the client.
        :param animate: If True, triggers an environment step.
        """

        loop = get_event_loop()

        # 1. Send a sample to the Client if a batch from the Dataset is given
        if self.batch_from_dataset is not None:
            # Check if there is remaining samples, otherwise the Client is not used
            if len(self.batch_from_dataset) == 0:
                return
            # Send the sample to the Client
            await self.send_command_sample(loop=loop, receiver=client)
            line = self.batch_from_dataset.pop(0)
            await self.send_data(data_to_send=line, loop=loop, receiver=client)

        # 2. Execute n steps, the last one send data computation signal
        if animate:
            await self.send_command_step(loop=loop, receiver=client)
            # Receive data
            await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict,
                                             client_id=client_id)
            line = await self.receive_data(loop=loop, sender=client)
            self.data_lines.append(line)

    def set_dataset_batch(self,
                          data_lines: List[int]) -> None:
        """
        Receive a batch of data from the Dataset. Samples will be dispatched between clients.

        :param data_lines: Batch of indices of samples.
        """

        # Define batch from dataset
        self.batch_from_dataset = data_lines.copy()

    ##########################################################################################
    ##########################################################################################
    #                                 Server & Client shutdown                               #
    ##########################################################################################
    ##########################################################################################

    def close(self) -> None:
        """
        Run __close method with asyncio.
        """

        print(f"[{self.name}] Closing clients...")
        async_run(self.__close())

    async def __close(self) -> None:
        """
        Run server shutdown protocol.
        """

        # Send all exit protocol and wait for the last one to finish
        await gather(*[self.__shutdown(client=client, idx=client_id) for client_id, client in self.clients])
        # Close socket
        self.sock.close()

    async def __shutdown(self,
                         client: socket, idx: int) -> None:
        """
        Send exit command to all clients.

        :param client: TcpIpObject client.
        :param idx: Client index.
        """

        loop = get_event_loop()
        print(f"[{self.name}] Sending exit command to", idx)
        # Send exit command
        await self.send_command_exit(loop=loop, receiver=client)
        await self.send_command_done(loop=loop, receiver=client)
        # Wait for exit confirmation
        data = await self.receive_data(loop=loop, sender=client)
        if data != b'exit':
            raise ValueError(f"Client {idx} was supposed to exit.")

    ##########################################################################################
    ##########################################################################################
    #                            Actions to perform on commands                              #
    ##########################################################################################
    ##########################################################################################

    async def action_on_prediction(self,
                                   data: Dict[Any, Any],
                                   client_id: int,
                                   sender: socket,
                                   loop: EventLoop) -> None:
        """
        Action to run when receiving the 'prediction' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param loop: asyncio.get_event_loop() return.
        :param sender: TcpIpObject sender.
        """

        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        self.environment_manager.data_manager.get_prediction(client_id)
        await self.send_data(data_to_send=True, receiver=sender)

    async def action_on_visualisation(self,
                                      data: Dict[Any, Any],
                                      client_id: int,
                                      sender: socket,
                                      loop: EventLoop) -> None:
        """
        Action to run when receiving the 'visualisation' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param loop: asyncio.get_event_loop() return.
        :param sender: TcpIpObject sender.
        """

        _, idx = await self.receive_labeled_data(loop=loop, sender=sender)
        self.environment_manager.update_visualizer(idx)
