import logging
import mindspore as ms
from queue import Queue
from mindspore import nn, ops, Tensor
from src.model import PipelinedMlp
from src.device_host import AsyncHost


class PipelinedMlpTrainer:
    def __init__(self, input_size, hidden_size, output_size, n_layers,
                 optimizer_cls, loss_fn, activation):
        self.model = PipelinedMlp(input_size, hidden_size, output_size,
                                 n_layers, num_pipeline_ranks=2)
        self.async_host = AsyncHost(self.model, optimizer_cls)

        self.loss_and_grads = ops.value_and_grad(lambda predict, label: loss_fn(predict, label))
        self.losses = []

        self.queue_predict = Queue()
        self.queue_host0_fvjp = Queue()
        self.queue_host1_fvjp = Queue()
        self.queue_receive_shape = Queue()
        

    def train_with_pipeline(self, input, target):
        self.model.set_train(True)
        # Split data for pipelining
        input_split = ops.split(input, input.shape[0] // 2)
        target_split = ops.split(target, target.shape[0] // 2)

        for fwd in range(2):
            input_part = input_split[fwd]
            target_part = target_split[fwd]
            if self.model.pipeline_rank == 0:
                logging.debug("Stage 0 forward")
                outputs, f_vjp0 = self.async_host.forward(forward_inputs=input_part)
                self.queue_fvjp0.put(f_vjp0)
                self.queue_receive_shape(outputs.shape)
            elif self.model.pipeline_rank == 1:
                logging.debug("Stage 1 forward")
                receive_shape = self.queue_receive_shape.get()
                predict, f_vjp1 = self.async_host.forward(receive_shape=receive_shape)
                logging.debug(predict.shape)
                self.queue_fvjp1.put(f_vjp1)
                self.queue_pred.put((predict, target_split))
            else:
                raise ValueError("Pipeline rank should be 0 or 1")

        avg_loss = 0.0
        for bwd in range(2):
            loss, grads = self.loss_and_grads(*self.queue_pred.get())
            grads1 = self.async_host.backward(self.queue_fvjp1.get())
            grads0 = self.async_host.backward(self.queue_fvjp0.get())
            avg_loss += loss

        self.async_host.update()
        self.losses.append(avg_loss)
        return avg_loss
