from typing import Any, Dict, List, Optional, Tuple
from asyncio import get_event_loop, run as async_run
from socket import socket
from queue import SimpleQueue
from threading import Thread

from DeepPhysX.simulation.multiprocess.tcpip_object import TcpIpObject
from SimRender.core import ViewerBatch


class TcpIpServer(TcpIpObject):

    def __init__(self,
                 nb_client: int = 5,
                 max_client_count: int = 10,
                 batch_size: int = 5,
                 manager: Optional[Any] = None,
                 debug: bool = True,
                 use_viewer: bool = False):
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
        self.simulation_manager: Optional[Any] = manager

        # Create ViewerBatch
        self.viewer_batch: Optional[ViewerBatch] = ViewerBatch() if use_viewer else None

    def message(self, txt: str):

        if self.debug:
            print(txt)

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
        self.clients = [None] * self.nb_client
        for _ in range(self.nb_client):
            # Accept connection
            client, _ = await loop.sock_accept(self.sock)
            # Get the instance ID
            label, client_id = self.receive_labeled_data(sender=client)
            print(f"[{self.name}] Client n°{client_id} connected: {client}")
            self.clients[client_id - 1] = [client_id, client]
            # self.clients.append([client_id, client])

    ##########################################################################################
    ##########################################################################################
    #                                 Initialize Environment                                 #
    ##########################################################################################
    ##########################################################################################

    def initialize(self, env_kwargs: Dict[str, Any]) -> None:
        """
        Send parameters to the clients to create their simulations.

        :param env_kwargs: Additional arguments to pass to the Environment.
        """

        print(f"[{self.name}] Initializing clients...")

        # Init ViewerBatch
        viewer_keys = None if self.viewer_batch is None else self.viewer_batch.start(nb_view=self.nb_client)

        # Initialisation process for each client
        for client_id, client in self.clients:

            # Send additional arguments
            self.send_dict(name='env_kwargs', dict_to_send=env_kwargs, receiver=client)

            # Send prediction request authorization
            self.send_data(data_to_send=self.simulation_manager.allow_prediction_requests, receiver=client)

            # Send number of sub-steps
            nb_steps = self.simulation_manager.simulations_per_step if self.simulation_manager else 1
            self.send_data(data_to_send=nb_steps, receiver=client)

            # Send visualization Database
            visualization = 'None' if viewer_keys is None else f'{viewer_keys[client_id - 1]}'
            self.send_data(data_to_send=visualization, receiver=client)

            # Wait Client init
            self.receive_data(sender=client)
            print(f"[{self.name}] Client n°{client_id} initialisation done")

        # Synchronize Clients
        # for client_id, client in self.clients:
        #     await self.send_data(data_to_send='sync', loop=loop, receiver=client)

    def connect_to_database(self,
                            database_path: Tuple[str, str],
                            normalize_data: bool):

        for client_id, client in self.clients:
            self.send_data(data_to_send=database_path[0], receiver=client)
            self.send_data(data_to_send=database_path[1], receiver=client)
            self.send_data(data_to_send=normalize_data, receiver=client)
            self.receive_data(sender=client)

    def connect_visualization(self) -> None:
        """
        Connect the Factories of the Clients to the Visualizer.
        """

        for _, client in self.clients:
            self.send_data(data_to_send='conn', receiver=client)

    ##########################################################################################
    ##########################################################################################
    #                          Data: produce batch & dispatch batch                          #
    ##########################################################################################
    ##########################################################################################

    def get_batch(self, animate: bool = True) -> List[int]:

        nb_sample = 0
        self.data_lines = []

        # Launch the communication protocol while the batch needs to be filled
        while nb_sample < self.batch_size:
            clients = self.clients[:min(len(self.clients), self.batch_size - nb_sample)]
            # Run communicate protocol for each client and wait for the last one to finish
            threads = [Thread(target=self.communicate, args=(client, client_id, animate)) for client_id, client in clients]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            nb_sample += len(clients)

        return self.data_lines

    def communicate(self,
                         client: Optional[socket] = None,
                         client_id: Optional[int] = None,
                         animate: bool = True) -> None:

            # 1. Send a sample to the Client if a batch from the Dataset is given
            if self.batch_from_dataset is not None:
                # Check if there is remaining samples, otherwise the Client is not used
                if len(self.batch_from_dataset) == 0:
                    return
                # Send the sample to the Client
                self.send_command_sample(receiver=client)
                line = self.batch_from_dataset.pop(0)
                self.send_data(data_to_send=line, receiver=client)

            # 2. Execute n steps, the last one send data computation signal
            if animate:
                self.send_command_step(receiver=client)
                # Receive data
                self.listen_while_not_done(sender=client, data_dict=self.data_dict, client_id=client_id)
                line = self.receive_data(sender=client)
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

        # Send all exit protocol and wait for the last one to finish
        for client_id, client in self.clients:
            self.__shutdown(client=client, idx=client_id)
        # Close socket
        self.sock.close()

        if self.viewer_batch is not None:
            self.viewer_batch.stop()

    def __shutdown(self, client: socket, idx: int) -> None:
        """
        Send exit command to all clients.

        :param client: TcpIpObject client.
        :param idx: Client index.
        """

        print(f"[{self.name}] Sending exit command to", idx)
        # Send exit command
        self.send_command_exit(receiver=client)
        self.send_command_done(receiver=client)
        # Wait for exit confirmation
        data = self.receive_data(sender=client)
        if data != b'exit':
            raise ValueError(f"Client {idx} was supposed to exit.")

    ##########################################################################################
    ##########################################################################################
    #                            Actions to perform on commands                              #
    ##########################################################################################
    ##########################################################################################

    def action_on_prediction(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None:
        """
        Action to run when receiving the 'prediction' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param sender: TcpIpObject sender.
        """

        self.simulation_manager.get_prediction_from_simulation(client_id)
        self.send_data(data_to_send=True, receiver=sender)

    def action_on_visualisation(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None:
        """
        Action to run when receiving the 'visualisation' command.

        :param data: Dict storing data.
        :param client_id: ID of the TcpIpClient.
        :param sender: TcpIpObject sender.
        """

        _, idx = self.receive_labeled_data(sender=sender)
        self.simulation_manager.update_visualizer(idx)
