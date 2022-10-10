from typing import Any, Dict, List, Optional, Union
from asyncio import get_event_loop, gather
from asyncio import AbstractEventLoop as EventLoop
from asyncio import run as async_run
from socket import socket
from numpy import ndarray
from queue import SimpleQueue

from DeepPhysX.Core.AsyncSocket.TcpIpObject import TcpIpObject


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
        self.batch_from_dataset: Optional[Dict[str, ndarray]] = None
        self.first_time: bool = True

        # Reference to EnvironmentManager
        self.environment_manager: Optional[Any] = manager

    ##########################################################################################
    ##########################################################################################
    #                                     Connect Clients                                    #
    ##########################################################################################
    ##########################################################################################

    def connect(self) -> None:
        """
        Run __connect method with asyncio.
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

    def initialize(self,
                   param_dict: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Run __initialize method with asyncio. Manage parameters exchange.

        :param param_dict: Dictionary of parameters to send to the client's environment.
        :return: Dictionary of parameters for each environment to send the manager.
        """

        print(f"[{self.name}] Initializing clients...")
        async_run(self.__initialize(param_dict))
        # Return param dict
        param_dict = {}
        for client_id in self.data_dict:
            if 'parameters' in self.data_dict[client_id]:
                param_dict[client_id] = self.data_dict[client_id]['parameters']
        return param_dict

    async def __initialize(self, param_dict: Dict[Any, Any]) -> None:
        """
        Send parameters to the clients to create their environments, receive parameters from clients in exchange.

        :param param_dict: Dictionary of parameters to send to the client's environment.
        """

        loop = get_event_loop()
        # Empty dictionaries for received parameters from clients
        self.data_dict = {client_ID: {} for client_ID in range(len(self.clients))}
        # Initialisation process for each client
        for client_id, client in self.clients:
            # Send number of sub-steps
            nb_steps = self.environment_manager.simulations_per_step if self.environment_manager else 1
            await self.send_data(data_to_send=nb_steps, loop=loop, receiver=client)
            # Send parameters to client
            await self.send_dict(name="parameters", dict_to_send=param_dict, loop=loop, receiver=client)
            # Receive visualization data and parameters
            await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict, client_id=client_id)
            print(f"[{self.name}] Client n°{client_id} initialisation done")

    ##########################################################################################
    ##########################################################################################
    #                          Data: produce batch & dispatch batch                          #
    ##########################################################################################
    ##########################################################################################

    def get_batch(self,
                  get_inputs: bool = True,
                  get_outputs: bool = True,
                  animate: bool = True) -> Dict[str, Union[ndarray, dict]]:
        """
        Build a batch from clients samples.

        :param get_inputs: If True, compute and return input.
        :param get_outputs: If True, compute and return output.
        :param animate: If True, triggers an environment step.
        :return: Batch (list of samples) & additional data in a dictionary.
        """

        # Trigger communication protocol
        async_run(self.__request_data_to_clients(get_inputs=get_inputs, get_outputs=get_outputs, animate=animate))

        # Sort stored data between following fields
        data_sorter = {'input': [], 'output': [], 'additional_fields': {}, 'loss': []}
        list_fields = [key for key in data_sorter.keys() if type(data_sorter[key]) == list]
        # Map produced samples with clients ID
        self.sample_to_client_id = []

        # Process while queue is empty or batch is full
        while max([len(data_sorter[key]) for key in list_fields]) < self.batch_size and not self.data_fifo.empty():
            # Get data dict from queue
            data = self.data_fifo.get()
            # Network in / out / loss
            for field in ['input', 'output', 'loss']:
                if field in data:
                    data_sorter[field].append(data[field])
            # Additional fields
            field = 'additional_fields'
            if field in data:
                for key in data[field]:
                    if key not in data_sorter[field].keys():
                        data_sorter[field][key] = []
                    data_sorter[field][key].append(data[field][key])
            # ID of client
            if 'ID' in data:
                self.sample_to_client_id.append(data['ID'])
        return data_sorter

    async def __request_data_to_clients(self,
                                        get_inputs: bool = True,
                                        get_outputs: bool = True,
                                        animate: bool = True) -> None:
        """
        Trigger a communication protocol for each client. Wait for all clients before to launch another communication
        protocol while the batch is not full.

        :param get_inputs: If True, compute and return input
        :param get_outputs: If True, compute and return output
        :param animate: If True, triggers an environment step
        """

        client_launched = 0
        # Launch the communication protocol while the batch needs to be filled
        while client_launched < self.batch_size:
            # Run communicate protocol for each client and wait for the last one to finish
            await gather(*[self.__communicate(client=client, client_id=client_id, get_inputs=get_inputs,
                                              get_outputs=get_outputs, animate=animate)
                           for client_id, client in self.clients])
            client_launched += len(self.clients)

    async def __communicate(self,
                            client: Optional[socket] = None,
                            client_id: Optional[int] = None,
                            get_inputs: bool = True,
                            get_outputs: bool = True,
                            animate: bool = True) -> None:
        """
        | Communication protocol with a client. It goes through different steps:
        |   1) Eventually send samples to Client
        |   2) Running steps & Receiving training data
        |   3) Add data to the Queue

        :param client: TcpIpObject client to communicate with.
        :param client_id: Index of the client.
        :param get_inputs: If True, compute and return input.
        :param get_outputs: If True, compute and return output.
        :param animate: If True, triggers an environment step.
        """

        loop = get_event_loop()

        # 1) If a sample from Dataset is given, sent it to the TcpIpClient
        if self.batch_from_dataset is not None:
            # Check if there is remaining samples, otherwise client is not used
            if len(self.batch_from_dataset['input']) == 0 or len(self.batch_from_dataset['output']) == 0:
                return
            # Send the sample to the TcpIpClient
            await self.send_command_sample(loop=loop, receiver=client)
            # Pop the first sample of the numpy batch for network in / out
            for field in ['input', 'output']:
                # Tell if there is something to read
                await self.send_data(data_to_send=field in self.batch_from_dataset, loop=loop, receiver=client)
                if field in self.batch_from_dataset:
                    # Pop sample from array if there are some
                    sample = self.batch_from_dataset[field][0]
                    self.batch_from_dataset[field] = self.batch_from_dataset[field][1:]
                    # Keep the sample in memory
                    self.data_dict[client_id][field] = sample
                    # Send network in / out sample
                    await self.send_data(data_to_send=sample, loop=loop, receiver=client)
            # Pop the first sample of the numpy batch for each additional dataset field
            field = 'additional_fields'
            # Tell TcpClient if there is additional data for this field
            await self.send_data(data_to_send=field in self.batch_from_dataset, loop=loop, receiver=client)
            if field in self.batch_from_dataset:
                sample = {}
                # Get each additional data field
                for key in self.batch_from_dataset[field]:
                    # Pop sample from array
                    sample[key] = self.batch_from_dataset[field][key][0]
                    self.batch_from_dataset[field][key] = self.batch_from_dataset[field][key][1:]
                    # Keep the sample in memory
                    self.data_dict[client_id][field + '_' + key] = sample[key]
                # Send additional in / out sample
                await self.send_dict(name="additional_fields", dict_to_send=sample, loop=loop, receiver=client)

        # 2) Execute n steps, the last one send data computation signal
        if animate:
            await self.send_command_step(loop=loop, receiver=client)
            # Receive data
            await self.listen_while_not_done(loop=loop, sender=client, data_dict=self.data_dict,
                                             client_id=client_id)

        # 3.1) Add all received in / out data to queue
        data = {}
        for get_data, net_field in zip([get_inputs, get_outputs], ['input', 'output']):
            if get_data:
                # Add network field
                data[net_field] = self.data_dict[client_id][net_field]
        # 3.2) Add loss data if provided
        if 'loss' in self.data_dict[client_id]:
            data['loss'] = self.data_dict[client_id]['loss']
        # 3.3) Add additional fields (transform key from 'dataset_{FIELD}' to '{FIELD}')
        additional_fields = [key for key in self.data_dict[client_id].keys() if key.__contains__('dataset_')]
        data['additional_fields'] = {}
        for field in additional_fields:
            data['additional_fields'][field[len('dataset_'):]] = self.data_dict[client_id][field]
        # 3.4) Identify sample
        data['ID'] = client_id
        # 3.5) Add data to the Queue
        self.data_fifo.put(data)

    def set_dataset_batch(self,
                          batch: Dict[str, Union[ndarray, Dict]]) -> None:
        """
        Receive a batch of data from the Dataset. Samples will be dispatched between clients.

        :param batch: Batch of data.
        """

        # Check batch size
        if len(batch['input']) != self.batch_size:
            raise ValueError(f"[{self.name}] The size of batch from Dataset is {len(batch['input'])} while the batch "
                             f"size was set to {self.batch_size}.")
        # Define batch from dataset
        self.batch_from_dataset = batch.copy()

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

        # Receive network input
        label, network_input = await self.receive_labeled_data(loop=loop, sender=sender)

        # Check that manager hierarchy is well-defined
        if self.environment_manager.data_manager is None:
            raise ValueError("Cannot request prediction if DataManager does not exist")
        elif self.environment_manager.data_manager.manager is None:
            raise ValueError("Cannot request prediction if Manager does not exist")
        elif not hasattr(self.environment_manager.data_manager.manager, 'network_manager'):
            raise AttributeError("Cannot request prediction if NetworkManager does not exist. If using a data "
                                 "generation pipeline, please disable get_prediction requests.")
        elif self.environment_manager.data_manager.manager.network_manager is None:
            raise ValueError("Cannot request prediction if NetworkManager does not exist")

        # Get the prediction from NetworkPrediction
        prediction = self.environment_manager.data_manager.get_prediction(network_input=network_input[None, ])
        # Send back the prediction to the Client
        await self.send_labeled_data(data_to_send=prediction, label="prediction", receiver=sender,
                                     send_read_command=False)

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
