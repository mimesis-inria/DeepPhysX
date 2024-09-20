from typing import Any, Dict, List, Optional, Tuple
from asyncio import get_event_loop, gather
from asyncio import AbstractEventLoop as EventLoop
from asyncio import run as async_run
from socket import socket
from queue import SimpleQueue

from DeepPhysX.simulation.utils.tcpip_object import TcpIpObject
from DeepPhysX.database.database_handler import DatabaseHandler


class TcpIpServer(TcpIpObject):

    def __init__(self,
                 nb_client: int = 5,
                 max_client_count: int = 10,
                 batch_size: int = 5,
                 manager: Optional[Any] = None,
                 debug: bool = True):
        """
        TcpIpServer is used to communicate with clients associated with Environment to produce batches for the
        EnvironmentManager.

        :param nb_client: Number of expected client connections.
        :param max_client_count: Maximum number of allowed clients.
        :param batch_size: Number of samples in a batch.
        :param manager: EnvironmentManager that handles the TcpIpServer.
        """

        super(TcpIpServer, self).__init__()

        self.debug = debug

        # Bind to server address
        self.sock.bind((self.ip_address, self.port))
        self.port = self.sock.getsockname()[1]
        self.message(f"[{self.name}] Binding to IP '{self.ip_address}' on PORT '{self.port}'")
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

    def message(self, txt: str):

        if self.debug:
            print(txt)

    ##########################################################################################
    ##########################################################################################
    #                              DatabaseHandler management                                #
    ##########################################################################################
    ##########################################################################################

    # def get_database_handler(self) -> DatabaseHandler:
    #     """
    #     Get the DatabaseHandler of the TcpIpServer.
    #     """
    #
    #     return self.database_handler

    # def __database_handler_partitions(self) -> None:
    #     """
    #     Partition update event of the DatabaseHandler.
    #     """
    #
    #     # Send the new partition to every Client
    #     for _, client in self.clients:
    #         self.sync_send_command_change_db(receiver=client)
    #         new_partition = self.database_handler.get_partitions()[-1]
    #         self.sync_send_data(data_to_send=f'{new_partition.get_path()[0]}///{new_partition.get_path()[1]}',
    #                             receiver=client)

    ##########################################################################################
    ##########################################################################################
    #                                     Connect Clients                                    #
    ##########################################################################################
    ##########################################################################################

    def connect(self) -> None:
        """
        Accept connections from clients.
        """

        self.message(f"[{self.name}] Waiting for clients...")
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

    def initialize(self,
                   env_kwargs: Dict[str, Any],
                   visualization_db: Optional[Tuple[str, str]] = None) -> None:
        """
        Send parameters to the clients to create their environments.

        :param env_kwargs: Additional arguments to pass to the Environment.
        :param visualization_db: Path to the visualization Database to connect to.
        """

        print(f"[{self.name}] Initializing clients...")
        async_run(self.__initialize(env_kwargs, visualization_db))

    async def __initialize(self,
                           env_kwargs: Dict[str, Any],
                           visualization_db: Optional[Tuple[str, str]] = None) -> None:
        """
        Send parameters to the clients to create their environments.

        :param env_kwargs: Additional arguments to pass to the Environment.
        :param visualization_db: Path to the visualization Database to connect to.
        """

        loop = get_event_loop()

        # Initialisation process for each client
        for client_id, client in self.clients:

            # Send additional arguments
            await self.send_dict(name='env_kwargs', dict_to_send=env_kwargs, loop=loop, receiver=client)

            # Send prediction request authorization
            await self.send_data(data_to_send=self.environment_manager.allow_prediction_requests,
                                 loop=loop, receiver=client)

            # Send number of sub-steps
            nb_steps = self.environment_manager.simulations_per_step if self.environment_manager else 1
            await self.send_data(data_to_send=nb_steps, loop=loop, receiver=client)

            # Send partitions
            # partitions = self.database_handler.get_partitions()
            # if len(partitions) == 0:
            #     partitions_list = 'None'
            # else:
            #     partitions_list = partitions[0].get_path()[0]
            #     for partition in partitions:
            #         partitions_list += f'///{partition.get_path()[1]}'
            # partitions_list += '%%%'
            # exchange = self.database_handler.get_exchange()
            # if exchange is None:
            #     partitions += 'None'
            # else:
            #     partitions_list += f'{exchange.get_path()[0]}///{exchange.get_path()[1]}'
            # await self.send_data(data_to_send=partitions_list, loop=loop, receiver=client)

            # Send visualization Database
            visualization = 'None' if visualization_db is None else f'{visualization_db[0]}///{visualization_db[1]}'
            await self.send_data(data_to_send=visualization, loop=loop, receiver=client)

            # Wait Client init
            await self.receive_data(loop=loop, sender=client)
            print(f"[{self.name}] Client n°{client_id} initialisation done")

        # Synchronize Clients
        # for client_id, client in self.clients:
        #     await self.send_data(data_to_send='sync', loop=loop, receiver=client)

    def connect_to_database(self,
                            database: Tuple[str, str],
                            exchange_db: Tuple[str, str]):

        async_run(self.__connect_to_database(database, exchange_db))

    async def __connect_to_database(self,
                                    database: Tuple[str, str],
                                    exchange_db: Tuple[str, str]):

        loop = get_event_loop()
        for client_id, client in self.clients:
            await self.send_data(data_to_send=database[0], loop=loop, receiver=client)
            await self.send_data(data_to_send=database[1], loop=loop, receiver=client)
            await self.send_data(data_to_send=exchange_db[0], loop=loop, receiver=client)
            await self.send_data(data_to_send=exchange_db[1], loop=loop, receiver=client)
            await self.receive_data(loop=loop, sender=client)

    def connect_visualization(self) -> None:
        """
        Connect the Factories of the Clients to the Visualizer.
        """

        async_run(self.__connect_visualization())

    async def __connect_visualization(self):
        """
        Connect the Factories of the Clients to the Visualizer.
        """

        loop = get_event_loop()
        for _, client in self.clients:
            await self.send_data(data_to_send='conn', loop=loop, receiver=client)

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
            clients = self.clients[:min(len(self.clients), self.batch_size - nb_sample)]
            # Run communicate protocol for each client and wait for the last one to finish
            await gather(*[self.__communicate(client=client,
                                              client_id=client_id,
                                              animate=animate) for client_id, client in clients])
            nb_sample += len(clients)

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
