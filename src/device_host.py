import logging
import mindspore as ms
from mindspore import ops
from mindspore.ops.operations._inner_ops import Send, Receive

world_group = 'nccl_world_group'

class AsyncHost:
    def __init__(self, model, optimizer_cls, micro_batch_size):
        self.model = model
        self.optimizer = optimizer_cls(self.model.trainable_params())
        self.micro_batch_size = micro_batch_size
        self.grads_collection = []
        self.depend = ops.Depend()
        
    def forward(self, forward_inputs=None):
        logging.debug(f"Forward with rank {self.model.pipeline_rank}")

        # Receive inputs from the previous stage
        if self.model.pipeline_rank > 0:
            logging.debug("Receiving...")
            receive_shape = [self.micro_batch_size, self.model.hidden_size]
            recv = Receive(sr_tag=0,
                           src_rank=self.model.pipeline_rank - 1,
                           shape=receive_shape,
                           dtype=ms.float32,
                           group=world_group)
            forward_inputs = recv()
            logging.debug("Received")
        else:
            if forward_inputs is None:
                raise ValueError("For stage rank equals 0, argument forward_inputs is required.")


        # Compute outputs and the vector-Jacobian product function
        outputs, f_vjp = ms.vjp(self.model, forward_inputs, weights=self.optimizer.parameters)

        # Send outputs to the next stage
        if self.model.pipeline_rank < self.model.num_pipeline_ranks - 1:
            logging.debug("Sending...")
            send = Send(sr_tag=0,
                        dest_rank=self.model.pipeline_rank + 1,group=world_group)
            outputs = self.depend(outputs, send(outputs))
            logging.debug("Sent")


        # Store f_vjp in the current stage for backward
        # TODO How to make sure that the current vjp is computed from the relevant step
        return outputs, f_vjp


    def backward(self, f_vjp, backward_inputs=None):
        logging.debug(f"Backward with rank {self.model.pipeline_rank}")

        if self.model.pipeline_rank < self.model.num_pipeline_ranks-1:
            logging.debug("Receiving...")
            receive_shape = [self.micro_batch_size, self.model.hidden_size]

            # Receive backward_inputs from next stage
            recv = Receive(sr_tag=0,
                           src_rank=self.model.pipeline_rank + 1,
                           shape=receive_shape,
                           dtype=ms.float32,
                           group=world_group)
            backward_inputs = recv()
            logging.debug("Received")
        else:
            if backward_inputs is None:
                raise ValueError("For the last stage, argument forward_inputs is required.")


        # Compute gradients
        grads_wrt_foward_inputs, grads_wrt_weights = f_vjp(backward_inputs)

        # Accumulate gradients w.r.t weights
        self.grads_collection.append(grads_wrt_weights)

        outputs = grads_wrt_foward_inputs

        if self.model.pipeline_rank > 0:
            # Send gradients to the previous stage
            logging.debug("Sending...")
            send = Send(sr_tag=0,
                        dest_rank=self.model.pipeline_rank - 1,group=world_group)
            outputs = self.depend(outputs, send(outputs))
            logging.debug("Sent")
            

    def update(self):
        logging.debug(f"Update with rank {self.model.pipeline_rank}")

        # Accumulate gradients from all sources
        for param in self.optimizer.parameters:
            summed_grads = sum(grads[param.name] for grads in self.grads_collection)
    
        # Perform optimization step
        self.optimizer(summed_grads)

        # Empty the gradients collection after updating
        self.grads_collection = []