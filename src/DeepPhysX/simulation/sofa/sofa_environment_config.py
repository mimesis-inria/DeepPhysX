from typing import Type, Optional, Any, Dict
from os.path import join, dirname
from sys import modules, executable
from subprocess import run

from DeepPhysX.simulation.core.base_environment_config import BaseEnvironmentConfig
from DeepPhysX.simulation.sofa.sofa_environment import SofaEnvironment


class SofaEnvironmentConfig(BaseEnvironmentConfig):

    def __init__(self,
                 environment_class: Type[SofaEnvironment],
                 as_tcp_ip_client: bool = True,
                 number_of_thread: int = 1,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 simulations_per_step: int = 1,
                 max_wrong_samples_per_step: int = 10,
                 load_samples: bool = False,
                 only_first_epoch: bool = True,
                 always_produce: bool = False,
                 visualizer: Optional[str] = None,
                 record_wrong_samples: bool = False,
                 env_kwargs: Optional[Dict[Any, Any]] = None):
        """
        SofaEnvironmentConfig is a configuration class to parameterize and create a SofaEnvironment for the
        EnvironmentManager.

        :param environment_class: Class from which an instance will be created.
        :param as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False.
        :param number_of_thread: Number of thread to run.
        :param ip_address: IP address of the TcpIpObject.
        :param port: Port number of the TcpIpObject.
        :param simulations_per_step: Number of iterations to compute in the Environment at each time step.
        :param max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step.
        :param load_samples: If True, the dataset will always be used in the environment.
        :param only_first_epoch: If True, data will always be created from environment. If False, data will be created
                                 from the environment during the first epoch and then re-used from the Dataset.
        :param always_produce: If True, data will always be produced in Environment(s).
        :param visualizer: Backend of the Visualizer to use.
        :param record_wrong_samples: If True, wrong samples are recorded through Visualizer.
        :param env_kwargs: Additional arguments to pass to the Environment.
        """

        BaseEnvironmentConfig.__init__(self,
                                       environment_class=environment_class,
                                       as_tcp_ip_client=as_tcp_ip_client,
                                       number_of_thread=number_of_thread,
                                       ip_address=ip_address,
                                       port=port,
                                       simulations_per_step=simulations_per_step,
                                       max_wrong_samples_per_step=max_wrong_samples_per_step,
                                       load_samples=load_samples,
                                       only_first_epoch=only_first_epoch,
                                       always_produce=always_produce,
                                       visualizer=visualizer,
                                       record_wrong_samples=record_wrong_samples,
                                       env_kwargs=env_kwargs)

        self.environment_class: Type[SofaEnvironment] = environment_class

    def start_client(self,
                     idx: int = 1) -> None:
        """
        Run a subprocess to start a TcpIpClient.

        :param idx: Index of client.
        """

        script = join(dirname(modules[SofaEnvironment.__module__].__file__), 'launcher_sofa_environment.py')
        run([executable, script, self.environment_file, self.environment_class.__name__,
             self.ip_address, str(self.port), str(idx), str(self.number_of_thread)])

    def create_environment(self) -> SofaEnvironment:
        """
        Create an Environment that will not be a TcpIpObject.

        :return: Environment object.
        """

        # Create instance
        environment = self.environment_class(as_tcp_ip_client=False,
                                             **self.env_kwargs)
        if not isinstance(environment, SofaEnvironment):
            raise TypeError(f"[{self.name}] The given 'environment_class'={self.environment_class} must be a "
                            f"SofaEnvironment.")
        return environment
