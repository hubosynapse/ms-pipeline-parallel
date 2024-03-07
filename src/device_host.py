import logging
import mindspore as ms
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.ops._primitive_cache import _get_cache_prim

world_group = 'nccl_world_group'

class AsyncHost:
    def __init__(self, model, optimizer_cls):
        self.model = model
        self.optimizer = optimizer_cls(self.model.trainable_params())
        self.grads_collection = []
        
    def forward(self, forward_inputs=None, receive_shape=None):
        logging.debug(f"Forward with rank {self.model.pipeline_rank}")

        # Receive inputs from the previous stage
        if self.model.pipeline_rank > 0:
            if receive_shape is None:
                raise ValueError("For stage rank > 0, argument receive_shape is required.")
            logging.debug("Receiving...")
            recv = _get_cache_prim(Receive)(sr_tag=0,
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
        logging.debug(outputs.shape)

        # Send outputs to the next stage
        if self.model.pipeline_rank < self.model.num_pipeline_ranks - 1:
            logging.debug("Sending...")
            send = _get_cache_prim(Send)(sr_tag=0,
                                         dest_rank=self.model.pipeline_rank + 1,
                                         group=world_group)
            send(outputs)
            logging.debug("Sent")


        # Store f_vjp in the current stage for backward
        # TODO How to make sure that the current vjp is computed from the relevant step
        return outputs, f_vjp


    def backward(self, f_vjp, backward_inputs=None):
        if self.model.pipeline_rank < self.model.num_pipeline_ranks:
            # Receive backward_inputs from next stage
            recv = _get_cache_prim(Receive)(sr_tag=0,
                                            src_rank=self.model.pipeline_rank + 1,
                                            group=world_group)
            backward_inputs = recv()

        # Compute gradients
        grads_wrt_weights, grads_wrt_foward_inputs = f_vjp(backward_inputs)

        # Accumulate gradients w.r.t weights
        self.grads.append(grads_wrt_weights)

        outputs = grads_wrt_foward_inputs
        if self.model.pipeline_rank > 0:
            # Send gradients to the previous stage
            send = _get_cache_prim(Send)(sr_tag=0,
                                         dest_rank=self.model.pipeline_rank - 1,
                                         group=world_group)
            send(outputs)


    def update(self):
        # Manually sum gradients from different sources
        for param in self.optimizer.parameters:
            # Accumulate gradients from all sources
            summed_grads = sum(grads[param.name] for grads in self.grads_collection)
    
        # Perform optimization step
        self.optimizer(summed_grads)

        # Zero the gradients after updating
        self.optimizer.zero_grad()


    # def update(grads_collection, weights, optimizer_state):
    #     grads = tree_multimap(lambda x, y: x + y, *grads_collection)
    #     updates, optimizer_state = optimizer.update(grads, optimizer_state)
    #     weights = optax.apply_updates(updates, weights)
    #     return weights, optimizer_state
    #     # if self.grads is not None:
    #     #     for param, grad in zip(self.model.parameters(), self.grads):
    #     #         if grad is not None:
    #     #             param.grad = grad.to(self.device)
    #     #     self.optimizer.step()
    #     #     self.grads = None  # Reset gradients after update
    #     if self.pipeline_rank == self.num_pipeline_ranks - 1:
    #         send = _get_cache_prim(Send)(sr_tag=1, dest_rank=0, group=world_group)
    #         send(outs)
    #     elif self.pipeline_rank == 0:
    #         recv = _get_cache_prim(Receive)(sr_tag=1, src_rank=self.num_pipeline_ranks - 1,
    #                                         shape=outs.shape, dtype=outs.dtype, group=world_group)
    #         outs = recv(outs)
