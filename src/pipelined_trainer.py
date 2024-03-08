import logging
import mindspore as ms
from queue import Queue
from mindspore import nn, ops, Tensor
from src.model import PipelinedMlp
from src.device_host import AsyncHost


class PipelinedMlpTrainer:
    def __init__(self, input_size, hidden_size, output_size, n_layers,
                 batch_size, micro_batch_size,
                 optimizer_cls, loss_fn, activation):
        self.NUM_STAGES = 2
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.model = PipelinedMlp(input_size, hidden_size, output_size,
                                 n_layers, num_pipeline_ranks=self.NUM_STAGES)

        self.async_host = AsyncHost(self.model, optimizer_cls, batch_size=batch_size)

        self.loss_and_grads = ops.value_and_grad(lambda predict, label: loss_fn(predict, label))
        self.losses = []

        if self.model.pipeline_rank == 0:
            self.queue_fvjp0 = Queue()
        elif self.model.pipeline_rank == 1:
            self.queue_predict = Queue()
            self.queue_fvjp1 = Queue()
        else:
            raise ValueError("Pipeline rank should be 0 or 1")
        

    def train_with_pipeline(self, input, target):
        self.model.set_train(True)


        # Split data for pipelining
        input_split = ops.split(input, self.micro_batch_size)
        target_split = ops.split(target, self.micro_batch_size)
        num_splits = len(input_split)

        for fwd in range(num_splits):
            input_part = input_split[fwd]
            target_part = target_split[fwd]

            if self.model.pipeline_rank == 0:
                outputs, f_vjp0 = self.async_host.forward(forward_inputs=input_part)
                self.queue_fvjp0.put(f_vjp0)
            elif self.model.pipeline_rank == 1:
                predict, f_vjp1 = self.async_host.forward()
                self.queue_fvjp1.put(f_vjp1)
                self.queue_predict.put((predict, target_split))
            else:
                raise ValueError("Pipeline rank should be 0 or 1")

        avg_loss = 0.0
        for bwd in range(num_splits):
            if self.model.pipeline_rank == 1:
                loss, grads = self.loss_and_grads(*self.queue_predict.get())
                grads = self.async_host.backward(self.queue_fvjp1.get())
            elif self.model.pipeline_rank == 0:
                grads = self.async_host.backward(self.queue_fvjp0.get())
            else:
                raise ValueError("Pipeline rank should be 0 or 1")
            avg_loss += loss

        avg_loss = avg_loss / num_splits
        self.async_host.update()
        self.losses.append(avg_loss)
        return avg_loss
