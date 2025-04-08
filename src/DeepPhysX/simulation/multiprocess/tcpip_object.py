from typing import Dict, Any, List, Union, Tuple, Optional
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
from numpy import ndarray

from DeepPhysX.simulation.multiprocess.bytes_converter import BytesConverter

Convertible = Union[type(None), bytes, str, bool, int, float, List, ndarray]


class TcpIpObject:

    def __init__(self):
        """
        TcpIpObject defines communication protocols to send and receive data and commands.
        """

        # Define socket
        self.sock: socket = socket(AF_INET, SOCK_STREAM)
        self.sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

        # Register IP and PORT
        self.ip_address: str = 'localhost'
        self.port: int = 0

        # Create data converter
        self.data_converter: BytesConverter = BytesConverter()

        # Available commands
        self.command_dict: Dict[str, bytes] = {'exit': b'exit',
                                               'step': b'step',
                                               'done': b'done',
                                               'prediction': b'pred',
                                               'read': b'read',
                                               'sample': b'samp'}
        self.action_on_command: Dict[bytes, Any] = {self.command_dict['exit']: self.action_on_exit,
                                                    self.command_dict['step']: self.action_on_step,
                                                    self.command_dict['prediction']: self.action_on_prediction,
                                                    self.command_dict['sample']: self.action_on_sample}

    ################################################
    # LOW level of send & receive data on networks #
    ################################################

    def send_data(self,
                  data_to_send: Convertible,
                  receiver: Optional[socket] = None) -> None:
        """
        Send data through the given socket.

        :param data_to_send: Data that will be sent on socket.
        :param receiver: Socket receiver.
        """

        receiver = self.sock if receiver is None else receiver

        # Cast data to bytes fields
        data_as_bytes = self.data_converter.data_to_bytes(data_to_send)

        # Send the whole message
        receiver.sendall(data_as_bytes)

    def receive_data(self, sender: Optional[socket] = None) -> Convertible:
        """
        Receive data from a socket.

        :param sender: Socket sender.
        """

        sender = self.sock if sender is None else sender
        sender.setblocking(True)

        # Receive the number of fields to receive
        nb_bytes_fields_b = sender.recv(self.data_converter.int_size)
        nb_bytes_fields = self.data_converter.size_from_bytes(nb_bytes_fields_b)

        # Receive the sizes in bytes of all the relevant fields
        sizes_b = [sender.recv(self.data_converter.int_size) for _ in range(nb_bytes_fields)]
        sizes = [self.data_converter.size_from_bytes(size_b) for size_b in sizes_b]

        # Receive each byte field
        bytes_fields = [self.read_data(sender, size) for size in sizes]

        # Return the data in the expected format
        return self.data_converter.bytes_to_data(bytes_fields)

    def read_data(self,
                  sender: socket,
                  read_size: int) -> bytes:
        """
        Read the data on the socket with value of buffer size as relatively small powers of 2.

        :param sender: Socket sender
        :param read_size: Amount of data to read on the socket.
        """

        sender = self.sock if sender is None else sender

        # Maximum read sizes array
        read_sizes = [8192, 4096]
        bytes_field = b''

        while read_size > 0:

            # Select the good amount of bytes to read
            read_size_idx = 0
            while read_size_idx < len(read_sizes) and read_size < read_sizes[read_size_idx]:
                read_size_idx += 1

            # If the amount of bytes to read is too small then read it all
            chunk_size_to_read = read_size if read_size_idx >= len(read_sizes) else read_sizes[read_size_idx]

            # Try to read at most chunk_size_to_read bytes from the socket
            data_received_as_bytes = sender.recv(chunk_size_to_read)

            # Accumulate the data
            bytes_field += data_received_as_bytes
            read_size -= len(data_received_as_bytes)

        return bytes_field

    ######################################
    # Send & receive abstract named data #
    ######################################

    def send_labeled_data(self,
                          data_to_send: Convertible,
                          label: str,
                          receiver: Optional[socket] = None,
                          send_read_command: bool = True) -> None:
        """
        Send data with an associated label.

        :param data_to_send: Data that will be sent on socket.
        :param label: Associated label.
        :param receiver: TcpIpObject receiver.
        :param send_read_command: If True, the command 'read' is sent before sending data.
        """

        receiver = self.sock if receiver is None else receiver

        # Send a 'read' command before data if specified
        if send_read_command:
            self.send_command_read(receiver=receiver)

        # Send label then data
        self.send_data(data_to_send=label, receiver=receiver)
        self.send_data(data_to_send=data_to_send, receiver=receiver)

    def receive_labeled_data(self, sender: socket) -> Tuple[str, Convertible]:
        """
        Receive data and an associated label.

        :param sender: Socket sender.
        """

        # Listen to sender
        recv = self.receive_data(sender)

        # 'recv' can be either a 'read' command either the label
        if recv in self.command_dict.values():
            label = self.receive_data(sender)
        else:
            label = recv

        # Receive data
        data = self.receive_data(sender)
        return label, data

    def send_dict(self,
                  name: str,
                  dict_to_send: Dict[Any, Any],
                  receiver: Optional[socket] = None) -> None:
        """
        Send a whole dictionary field by field as labeled data.

        :param name: Name of the dictionary.
        :param dict_to_send: Dictionary to send.
        :param receiver: TcpIpObject receiver.
        """

        receiver = self.sock if receiver is None else receiver

        # If dict is empty, the sending is done
        if dict_to_send is None or dict_to_send == {}:
            self.send_command_done(receiver=receiver)
            return

        # Sends to make the listener start the receive_dict routine
        self.send_labeled_data(data_to_send=name, label="::dict::", receiver=receiver)

        # Treat the dictionary field by field
        for key in dict_to_send:

            # If data is another dict, send as an unnamed dictionary
            if type(dict_to_send[key]) == dict:
                # Send key then data
                self.send_labeled_data(data_to_send=key, label="dict_id", receiver=receiver)
                self.send_unnamed_dict(dict_to_send=dict_to_send[key], receiver=receiver)

            # If data is not a dict, send as labeled data
            else:
                # Send key then data
                self.send_labeled_data(data_to_send=dict_to_send[key], label=key, receiver=receiver)

        # The sending is done
        self.send_command_done(receiver=receiver)
        self.send_command_done(receiver=receiver)

    def send_unnamed_dict(self,
                          dict_to_send: Dict[Any, Any],
                          receiver: Optional[socket] = None) -> None:
        """
        Send a whole dictionary field by field as labeled data. Dictionary will be unnamed.

        :param dict_to_send: Dictionary to send.
        :param receiver: TcpIpObject receiver.
        """

        receiver = self.sock if receiver is None else receiver

        # Treat the dictionary field by field
        for key in dict_to_send:

            # If data is another dict, send as an unnamed dictionary
            if type(dict_to_send[key]) == dict:
                # Send key then dict
                self.send_labeled_data(data_to_send=key, label="dict_id", receiver=receiver)
                self.send_unnamed_dict(dict_to_send=dict_to_send[key], receiver=receiver)

            # If data is not a dict, send as labeled data
            else:
                # Send key then data
                self.send_labeled_data(data_to_send=dict_to_send[key], label=key, receiver=receiver)

        # The sending is done
        self.send_command_done(receiver=receiver)

    def receive_dict(self, sender: Optional[socket] = None) -> Dict[Any, Any]:
        """
        Receive a whole dictionary field by field as labeled data.

        :param sender: TcpIpObject sender.
        """

        sender = self.sock if sender is None else sender
        recv_to = {}

        # Receive data while command 'done' is not received
        while self.receive_data(sender) != self.command_dict['done']:

            # Receive field as a labeled data
            label, data = self.receive_labeled_data(sender)

            # If label refers to dict keyword, receive an unnamed dict
            if label in ["::dict::", "dict_id"]:
                recv_to[data] = self.receive_dict(sender=sender)
            # Otherwise, set the dict field directly
            else:
                recv_to[label] = data

        return recv_to

    #########################
    # Command related sends #
    #########################

    def __send_command(self,
                       receiver: socket,
                       command: str) -> None:
        """
        Send a bytes command among the available commands.

        :param command: Name of the command to send.
        :param receiver: TcpIpObject receiver.
        """

        # Check if the command exists
        try:
            cmd = self.command_dict[command]
        except KeyError:
            raise KeyError(f"\"{command}\" is not a valid command. Use {self.command_dict.keys()} instead.")

        # Send command as a byte data
        self.send_data(data_to_send=cmd, receiver=receiver)

    def send_command_done(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'done' command.

        :param receiver: TcpIpObject receiver.
        """

        self.__send_command(receiver=receiver, command='done')

    def send_command_exit(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'exit' command.

        :param receiver: TcpIpObject receiver.
        """

        self.__send_command(receiver=receiver, command='exit')

    def send_command_prediction(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'prediction' command.

        :param receiver: TcpIpObject receiver.
        """

        self.__send_command(receiver=receiver, command='prediction')

    def send_command_read(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'read' command.

        :param receiver: TcpIpObject receiver.
        """

        self.__send_command(receiver=receiver, command='read')

    def send_command_sample(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'sample' command.

        :param receiver: TcpIpObject receiver.
        """

        self.__send_command(receiver=receiver, command='sample')

    def send_command_step(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'step' command.

        :param receiver: TcpIpObject receiver.
        """

        self.__send_command(receiver=receiver, command='step')

    ##################################
    # Actions to perform on commands #
    ##################################

    def listen_while_not_done(self, sender: socket, client_id: int = None) -> None:

        # Compute actions until 'done' command is received
        while (cmd := self.receive_data(sender)) != self.command_dict['done']:

            # Compute the associated action
            if cmd in self.command_dict.values():
                self.action_on_command[cmd](client_id=client_id, sender=sender)

    def action_on_exit(self, client_id: int, sender: socket) -> None: ...

    def action_on_prediction(self, client_id: int, sender: socket) -> None: ...

    def action_on_sample(self, client_id: int, sender: socket) -> None: ...

    def action_on_step(self, client_id: int, sender: socket) -> None: ...
