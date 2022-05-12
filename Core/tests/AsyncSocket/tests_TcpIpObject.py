from unittest import TestCase
import threading
from numpy import array, ndarray
from asyncio import run, get_event_loop

from DeepPhysX_Core.AsyncSocket.TcpIpServer import TcpIpServer
from DeepPhysX_Core.AsyncSocket.TcpIpClient import TcpIpClient

samples = [None, b'test', 'test', True, False, 1, -1, 1., -1., [0.1, 0.1], [[-1, 0], [0, 1]],
           array([0.1, 0.1], dtype=float), array([[-1, 0], [0, 1]], dtype=int)]


class TestTcpIpObjects(TestCase):

    # ___________
    # SETUP TESTS

    def setUp(self):
        self.server = TcpIpServer(ip_address='localhost', port=11111, nb_client=1, batch_size=5)
        self.client = TcpIpClient(ip_address='localhost', port=11111, instance_id=0, number_of_instances=1)
        self.result = []

    def tearDown(self):
        self.server.close()
        self.server.sock.close()
        self.client.sock.close()
        print("")

    # _______________________
    # DATA EXCHANGE PROTOCOLS

    async def server_exchange(self, func, data):
        if func == 'data':
            await self.server.send_data(data_to_send=data, receiver=self.server.clients[0][1])
            return await self.server.receive_data(loop=get_event_loop(), sender=self.server.clients[0][1])
        elif func == 'labeled':
            await self.server.send_labeled_data(label=data[0], data_to_send=data[1], receiver=self.server.clients[0][1])
            return await self.server.receive_labeled_data(loop=get_event_loop(), sender=self.server.clients[0][1])
        elif func == 'dict':
            print("Server sending", data)
            await self.server.send_dict(name='test', dict_to_send=data, receiver=self.server.clients[0][1])
            d = {}
            await self.server.receive_dict(recv_to=d, loop=get_event_loop(), sender=self.server.clients[0][1])
            return d['test'] if 'test' in d else {}

    def sync_server_exchange(self, func, data):
        if func == 'data':
            self.server.sync_send_data(data_to_send=data, receiver=self.server.clients[0][1])
            return self.server.sync_receive_data()

    async def client_exchange(self, func):
        if func == 'data':
            recv = await self.client.receive_data(loop=get_event_loop(), sender=self.client.sock)
            await self.client.send_data(data_to_send=recv)
        elif func == 'labeled':
            label, recv = await self.client.receive_labeled_data(loop=get_event_loop(), sender=self.client.sock)
            await self.client.send_labeled_data(data_to_send=recv, label=label, receiver=self.client.sock)
        elif func == 'dict':
            d = {}
            await self.client.receive_dict(recv_to=d, loop=get_event_loop(), sender=self.client.sock)
            d = d['test'] if 'test' in d else {}
            await self.client.send_dict(name='test', dict_to_send=d, receiver=self.client.sock)

    def sync_client_exchange(self, func):
        if func == 'data':
            recv = self.client.sync_receive_data()
            self.client.sync_send_data(data_to_send=recv)

    # ________________________________
    # SERVER & CLIENT THREADS LAUNCHER

    def launch_threads(self, server_target, server_args, client_target, client_args):
        server_thread = threading.Thread(target=server_target, args=server_args)
        server_thread.start()
        client_thread = threading.Thread(target=client_target, args=client_args)
        client_thread.start()
        server_thread.join()

    # _____
    # TESTS

    def test_actions(self):
        res = True
        for cmd in self.server.command_dict.values():
            if cmd not in self.server.action_on_command.keys():
                res = False
                break
        self.assertTrue(res)

    def test_connection(self):
        self.launch_threads(server_target=self.server.connect, server_args=(),
                            client_target=self.client.launch, client_args=())
        self.assertEqual(len(self.server.clients), 1)

    def test_send_data(self):
        self.launch_threads(server_target=self.thread_send_data_server, server_args=(False,),
                            client_target=self.thread_send_data_client, client_args=(False,))
        self.assertTrue(False not in self.result)

    def thread_send_data_server(self, sync):
        self.server.connect()
        for data in samples:
            recv = self.sync_server_exchange('data', data) if sync else run(self.server_exchange('data', data))
            result = recv == data if type(data) != ndarray else (recv == data).all()
            self.result.append(result)

    def thread_send_data_client(self, sync):
        for _ in range(len(samples)):
            recv = self.sync_client_exchange('data') if sync else run(self.client_exchange('data'))
        self.client.launch()

    def test_send_labeled_data(self):
        self.launch_threads(server_target=self.thread_send_labeled_data_server, server_args=(False,),
                            client_target=self.thread_send_labeled_data_client, client_args=(False,))
        self.assertTrue(False not in self.result)

    def thread_send_labeled_data_server(self, sync):
        self.server.connect()
        data = ('field', 10.5)
        recv = self.sync_server_exchange('labeled', data) if sync else run(self.server_exchange('labeled', data))
        self.result.append(recv == data)

    def thread_send_labeled_data_client(self, sync):
        self.sync_client_exchange('labeled') if sync else run(self.client_exchange('labeled'))
        self.client.launch()

    def test_send_dict(self):
        self.launch_threads(server_target=self.thread_send_dict_server, server_args=(False,),
                            client_target=self.thread_send_dict_client, client_args=(False,))
        self.assertTrue(False not in self.result)

    def thread_send_dict_server(self, sync):
        self.server.connect()
        data = {'field': 10.5}
        recv = self.sync_server_exchange('dict', data) if sync else run(self.server_exchange('dict', data))
        self.result.append(recv == data)

    def thread_send_dict_client(self, sync):
        self.sync_client_exchange('dict') if sync else run(self.client_exchange('dict'))
        self.client.launch()
