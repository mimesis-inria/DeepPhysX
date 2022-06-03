from numpy import array
from numpy.random import randn

from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


class TestEnvironment(BaseEnvironment):

    def __init__(self, ip_address='localhost', port=10000, instance_id=1, number_of_instances=1,
                 as_tcp_ip_client=True, environment_manager=None):
        BaseEnvironment.__init__(self, ip_address=ip_address, port=port, instance_id=instance_id,
                                 number_of_instances=number_of_instances, as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)
        # Environment parameters that must be learned by network
        self.p = [round(randn(), 2) for _ in range(4)]
        self.truth = None
        self.idx_step = 0

    def recv_parameters(self, param_dict):
        if 'interpolation' in param_dict:
            self.p = param_dict['interpolation']

    def create(self):
        self.truth = lambda x: self.p[0] + self.p[1] * x + self.p[2] * (x ** 2) + self.p[3] * (x ** 3)

    async def step(self):
        self.idx_step += 1
        # Without dataset sample
        if not (self.sample_in or self.sample_out):
            my_input = randn(1).round(2)
            self.set_training_data(input_array=my_input,
                                   output_array=self.truth(my_input))
            self.set_loss_data(self.idx_step)
            self.set_additional_dataset('step', array([self.idx_step]))
            self.set_additional_dataset('step_1', array([self.idx_step]))
        # With dataset sample
        else:
            self.set_training_data(input_array=self.sample_in * 2,
                                   output_array=self.sample_out * 2)
            full_dataset = 'step' in self.additional_fields and 'step_1' in self.additional_fields
            self.set_loss_data(100)
            self.reset_additional_datasets()
            self.set_additional_dataset('full', array([full_dataset]))

    def check_sample(self, check_input=True, check_output=True):
        return self.idx_step % 2 == 0 if not (self.sample_in or self.sample_out) else True
