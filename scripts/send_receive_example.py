import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.communication import init
from mindspore import Tensor

init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pipeline_rank = ms.communication.get_rank()
        self.depend = ops.Depend()
        print(f"Net with rank {self.pipeline_rank}")
        if self.pipeline_rank == 0:
            print("Init send")
            self.send = Send(sr_tag=0, dest_rank=1, group="nccl_world_group")
        else:
            print("Init receive")
            self.receive = Receive(sr_tag=0, src_rank=0, shape=[2, 8], dtype=ms.float32, group="nccl_world_group")

    def construct(self, x):
        print(f"Construct with rank {self.pipeline_rank}")

        if self.pipeline_rank == 0:
            print("Execute send")
            out = self.depend(x, self.send(x))
        else:
            print("Execute receive")
            ut = self.receive()
            out = 0

        print(out)
        print(f"End construct with rank {self.pipeline_rank}")
        return out
        
if __name__ == "__main__":
    input_ = Tensor(np.ones([2, 8]), dtype=ms.float32)
    net = Net()
    output = net(input_)