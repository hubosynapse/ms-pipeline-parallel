import logging
import mindspore as ms
from typing import Tuple, List
from mindspore import Tensor, nn
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.ops._primitive_cache import _get_cache_prim


class PipelinedMlp(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size, n_layers, pipeline_rank=0, num_pipeline_ranks=2):
        super().__init__()
        if num_pipeline_ranks > 1:
            self.pipeline_rank = ms.communication.get_rank()
        else:
            self.pipeline_rank = 0
        logging.debug(f"Init PipelinedMlp with rank {pipeline_rank}")
        self.num_pipeline_ranks = num_pipeline_ranks

        layers = []
        size = input_size if pipeline_rank == 0 else hidden_size
        num_layers_per_rank = n_layers // num_pipeline_ranks

        for _ in range(num_layers_per_rank):
            layers.append(nn.Dense(size, hidden_size))
            layers.append(nn.ReLU())
            size = hidden_size

        if pipeline_rank == num_pipeline_ranks - 1:
            layers.append(nn.Dense(hidden_size, output_size))

        self.layers = nn.SequentialCell(*layers)

    def construct_partial(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def construct(self, x):
        h = self.construct_partial(x)
        return h
