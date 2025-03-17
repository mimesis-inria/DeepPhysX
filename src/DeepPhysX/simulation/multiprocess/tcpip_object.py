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

        self.name: str = self.__class__.__name__

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
                                               'finished': b'fini',
                                               'prediction': b'pred',
                                               'read': b'read',
                                               'sample': b'samp',
                                               'visualisation': b'visu'}
        self.action_on_command: Dict[bytes, Any] = {
            self.command_dict["exit"]: self.action_on_exit,
            self.command_dict["step"]: self.action_on_step,
            self.command_dict["done"]: self.action_on_done,
            self.command_dict["finished"]: self.action_on_finished,
            self.command_dict["prediction"]: self.action_on_prediction,
            self.command_dict["read"]: self.action_on_read,
            self.command_dict["sample"]: self.action_on_sample,
            self.command_dict["visualisation"]: self.action_on_visualisation
        }

    ##########################################################################################
    ##########################################################################################
    #                      LOW level of send & receive data on networks                       #
    ##########################################################################################
    ##########################################################################################

    def send_data(self, data_to_send: Convertible, receiver: Optional[socket] = None) -> None:
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

        :return: Converted data.
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

    def read_data(self, sender: socket, read_size: int) -> bytes:
        """
        Read the data on the socket with value of buffer size as relatively small powers of 2.

        :param read_size: Amount of data to read on the socket.
        :return: Bytes field with 'read_size' length.
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

    ##########################################################################################
    ##########################################################################################
    #                            Send & receive abstract named data                          #
    ##########################################################################################
    ##########################################################################################

    def send_labeled_data(self, data_to_send: Convertible, label: str, receiver: Optional[socket] = None,
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
        # Send label
        self.send_data(data_to_send=label, receiver=receiver)
        # Send data
        self.send_data(data_to_send=data_to_send, receiver=receiver)

    def receive_labeled_data(self, sender: socket) -> Tuple[str, Convertible]:
        """
        Receive data and an associated label.

        :return: Label, Data.
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

    def send_dict(self, name: str, dict_to_send: Dict[Any, Any], receiver: Optional[socket] = None) -> None:
        """
        Send a whole dictionary field by field as labeled data.

        :param name: Name of the dictionary.
        :param dict_to_send: Dictionary to send.
        :param receiver: TcpIpObject receiver.
        """

        receiver = self.sock if receiver is None else receiver

        # If dict is empty, the sending is finished
        if dict_to_send is None or dict_to_send == {}:
            self.send_command_finished(receiver=receiver)
            return

        # Sends to make the listener start the receive_dict routine
        self.send_command_read()
        self.send_labeled_data(data_to_send=name, label="::dict::", receiver=receiver)

        # Treat the dictionary field by field
        for key in dict_to_send:
            # If data is another dict, send as an unnamed dictionary
            if type(dict_to_send[key]) == dict:
                # Send key
                self.send_labeled_data(data_to_send=key, label="dict_id", receiver=receiver)
                # Send data
                self.send_unnamed_dict(dict_to_send=dict_to_send[key], receiver=receiver)
            # If data is not a dict, send as labeled data
            else:
                # Send key and data
                self.send_labeled_data(data_to_send=dict_to_send[key], label=key, receiver=receiver)

        # The sending is finished
        self.send_command_finished(receiver=receiver)
        self.send_command_finished(receiver=receiver)

    def send_unnamed_dict(self, dict_to_send: Dict[Any, Any], receiver: Optional[socket] = None) -> None:
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
                # Send key
                self.send_labeled_data(data_to_send=key, label="dict_id", receiver=receiver,
                                            send_read_command=True)
                # Send data
                self.send_unnamed_dict(dict_to_send=dict_to_send[key], receiver=receiver)
            # If data is not a dict, send as labeled data
            else:
                # Send key and data
                self.send_labeled_data(data_to_send=dict_to_send[key], label=key, receiver=receiver,
                                            send_read_command=True)

        # The sending is finished
        self.send_command_finished(receiver=receiver)

    def receive_dict(self, recv_to: Dict[Any, Any], sender: Optional[socket] = None) -> None:
        """
        Receive a whole dictionary field by field as labeled data.

        :param recv_to: Dictionary to fill with received fields.
        :param sender: TcpIpObject sender.
        """

        sender = self.sock if sender is None else sender

        # Receive data while command 'finished' is not received
        while self.receive_data(sender) != self.command_dict['finished']:
            # Receive field as a labeled data
            label, param = self.receive_labeled_data(sender)
            # If label refers to dict keyword, receive an unnamed dict
            if label in ["::dict::", "dict_id"]:
                recv_to[param] = {}
                self.receive_dict(recv_to=recv_to[param], sender=sender)
            # Otherwise, set the dict field directly
            else:
                recv_to[label] = param

    ##########################################################################################
    ##########################################################################################
    #                                 Command related sends                                  #
    ##########################################################################################
    ##########################################################################################

    def send_command(self, receiver: socket, command: str = '') -> None:
        """
        Send a bytes command among the available commands.
        Do not use this one. Use the dedicated function 'send_command_{cmd}(...)'.

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

    def send_command_compute(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'compute' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='compute')

    def send_command_done(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'done' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='done')

    def send_command_exit(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'exit' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='exit')

    def send_command_finished(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'finished' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='finished')

    def send_command_prediction(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'prediction' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='prediction')

    def send_command_read(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'read' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='read')

    def send_command_sample(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'sample' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='sample')

    def send_command_step(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'step' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='step')

    def send_command_visualisation(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'visualisation' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='visualisation')

    def send_command_change_db(self, receiver: Optional[socket] = None) -> None:
        """
        Send the 'change_database' command.

        :param receiver: TcpIpObject receiver.
        """

        self.send_command(receiver=receiver, command='db')

    ##########################################################################################
    ##########################################################################################
    #                            Actions to perform on commands                              #
    ##########################################################################################
    ##########################################################################################

    def listen_while_not_done(self, sender: socket, data_dict: Dict[Any, Any], client_id: int = None) -> Dict[Any, Any]:

        # Compute actions until 'done' command is received
        while (cmd := self.receive_data(sender)) != self.command_dict['done']:
            # Compute the associated action
            if cmd in self.command_dict.values():
                self.action_on_command[cmd](data=data_dict, client_id=client_id, sender=sender)
        # Return collected data
        return data_dict

    def action_on_compute(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...

    def action_on_done(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...

    def action_on_exit(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...

    def action_on_finished(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...

    def action_on_prediction(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...

    def action_on_read(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None:

        # Receive labeled data
        label, param = self.receive_labeled_data(None)
        # If data to receive appears to be a dict, receive dict
        if label == "::dict::":
            data[client_id][param] = {}
            self.receive_dict(recv_to=data[client_id][param], sender=sender)
        # Otherwise add labeled data to data dict
        else:
            data[client_id][label] = param

    def action_on_sample(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...

    def action_on_step(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...

    def action_on_visualisation(self, data: Dict[Any, Any], client_id: int, sender: socket) -> None: ...
