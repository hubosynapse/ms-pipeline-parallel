import mindspore as ms
from queue import Queue
from mindspore import nn, ops, Tensor


class PipelinedMlpTrainer:
    def __init__(self, input_shape, device_nodes, optimizer_cls, loss_fn, activation=nn.Tanh()):
        self.model = PipelineMlp(inpute_size, hidden_size, output_size,
                                  n_layers, num_pipeline_ranks=2)
        self.async_host = AsyncHost(self.model)

        self.optimizer0 = optimizer_cls(self.model0.parameters())
        self.optimizer1 = optimizer_cls(self.model1.parameters())

        self.loss_and_grads = ms.value_and_grad(lambda predict, label: loss_fn(predict, label))
        self.losses = []

        self.queue_predict = Queue()
        self.queue_host0_fvjp = Queue()
        self.queue_host1_fvjp = Queue()


    def train_with_pipeline(self, input, target):
        # Split data for pipelining
        input_split = ops.split(input, input.shape[0] // 2)
        target_split = ops.split(target, target.shape[0] // 2)

        for fwd in range(2):
            input_part = input_split[fwd]
            target_part = target_split[fwd]
            f_vjp0 = self.async_host.forward(input_part)
            predict, f_vjp1 = self.async_host.forward()
            self.queue_fvjp0.put(f_vjp0)
            self.queue_fvjp1.put(f_vjp1)
            self.queue_pred.put((predict, target_splict))

        avg_loss = 0.0
        for bwd in range(2):
            loss, grads = self.loss_and_grads(*self.queue_pred.get())
            grads1 = self.async_host.backward(self.queue_fvjp1.get())
            grads0 = self.async_host.backward(self.queue_fvjp0.get())
            avg_loss += loss

        self.w0, self.opt0 = self.update0(grad_collect0, w0, opt0)
        self.w1, self.opt1 = self.update1(grad_collect1, w1, opt1)
        self.losses.append(avg_loss)
