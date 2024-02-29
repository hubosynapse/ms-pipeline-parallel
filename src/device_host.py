import mindspore as ms
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.ops._primitive_cache import _get_cache_prim

world_group = 'nccl_world_group'

class AsyncHost:
    def __init__(self, model, optimizer_cls, device):
        self.model = model
        self.optimizer = optimizer_cls(self.model.parameters())
        self.model.set_train(True)
        self.grads = []

    def forward(self, forward_inputs=None):
        # Receive inputs from the previous stage
        if self.model.pipeline_rank > 0:
            recv = _get_cache_prim(Receive)(sr_tag=0,
                                            src_rank=self.model.pipeline_rank - 1,
                                            group=world_group)
            forward_inputs = recv()

        # Compute outputs and the vector-Jacobian product function
        outputs, f_vjp = ms.vjp(self.model, forward_inputs)

        # Send outputs to the next stage
        if self.model.pipeline_rank < self.model.num_pipeline_ranks - 1:
            send = _get_cache_prim(Send)(sr_tag=0,
                                         dest_rank=self.model.pipeline_rank + 1,
                                         group=world_group)
            send(outputs)

        # Store f_vjp in the current stage for backward
        # TODO How to make sure that the current vjp is computed from the relevant step
        return f_vjp


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


    def update(grads_collection, weights, optimizer_state):
        grads = tree_multimap(lambda x, y: x + y, *grads_collection)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        weights = optax.apply_updates(updates, weights)
        return weights, optimizer_state

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
