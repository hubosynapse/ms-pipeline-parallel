"""
Benchmark model/pipline parallelism on MNIST data

We select a computing-bound setup (a MLP with memory size around 40MB)
"""
import sys
import time
import logging
import numpy as np
import mindspore as ms
import mindspore.dataset as ds

from pathlib import Path
from itertools import cycle
from queue import Queue
from typing import Iterable
from mindspore import nn, ops, context
from mindspore.communication.management import init

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.pipelined_trainer import PipelinedMlpTrainer
from examples.load_data_ms import create_dataset


def random_shuffle(images, labels):
    assert images.shape[0] == labels.shape[0]
    n_data = images.shape[0]
    indices = np.random.permutation(n_data)
    return images[indices, ...], labels[indices, ...]


def main(num_epochs: int=50, batch_size: int=32, micro_batch_size: int=16, model_dir=""):
    logging.debug("Set context")
    ms.set_context(pynative_synchronize=True)

    logging.debug("Init nccl")
    init("nccl")

    loss_fn = ops.cross_entropy
    optimizer_cls = nn.Adam

    data_path = project_root.joinpath("examples").joinpath("data").joinpath("MNIST")
    ds_train = create_dataset(str(data_path.joinpath("train")), batch_size=batch_size)
    # ds_train = ds_train.filter(predicate=lambda x: x[0]==1, input_columns = ["label"])

    ds_train_iter = ds_train.create_dict_iterator()

    trainer = PipelinedMlpTrainer(
        input_size=28*28,
        hidden_size=1024,
        output_size=10,
        n_layers=8,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        optimizer_cls=optimizer_cls, loss_fn=loss_fn)
    
    
    model_path = Path(model_dir)
    for epoch in range(num_epochs):
        for num_iter, row in enumerate(ds_train_iter):
            start = time.time()
            loss = trainer.train_with_pipeline(row["image"], row["label"])
            elapsed = time.time() - start

            # test_loss, acc = test(imgs_test, labels_test, pipe_state)
            if trainer.model.pipeline_rank == 1:
                if num_iter % 20 == 0:
                    logging.info(f"Epoch {epoch} Iteration {num_iter} {elapsed:.4f} sec): Train loss {loss.asnumpy():.4f}")#, Test loss {test_loss:.4f}, Acc On Test: {acc:.3f}")

            # if num_iter > 10:
            #     break

        # Save checkpoint
        # ms.save_checkpoint(save_obj=trainer.model,
        #                    ckpt_file_name=model_path.joinpath(f"rank_{trainer.model.pipeline_rank}checkpoint_{epoch}"))



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(num_epochs=30, batch_size=32, micro_batch_size=16, model_dir="/data1/hubo/models/pipelinedMlP")
