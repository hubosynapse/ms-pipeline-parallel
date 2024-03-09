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


def main(num_epochs: int=1, batch_size: int=32, micro_batch_size: int=16):
    logging.debug("Set context")
    ms.set_context(pynative_synchronize=True)

    logging.debug("Init nccl")
    init("nccl")

    logging.debug("Loss and opt")
    loss_fn = ops.cross_entropy
    optimizer_cls = nn.Adam

    logging.debug("Load dataset")
    data_path = project_root.joinpath("examples").joinpath("data").joinpath("MNIST")
    ds_train = create_dataset(str(data_path.joinpath("train")), batch_size=batch_size)

    ds_train_iter = ds_train.create_dict_iterator(num_epochs=num_epochs)

    logging.debug("Init trainer")
    trainer = PipelinedMlpTrainer(
        input_size=28*28,
        hidden_size=1024,
        output_size=10,
        n_layers=20,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        optimizer_cls=optimizer_cls, loss_fn=loss_fn, activation=nn.ReLU)
    
    for num_iter, row in enumerate(ds_train_iter):
        start = time.time()
        loss = trainer.train_with_pipeline(row["image"], row["label"])
        elapsed = time.time() - start

        # test_loss, acc = test(imgs_test, labels_test, pipe_state)
        logging.info(f"Iteration {num_iter} {elapsed:.4f} sec): Train loss {avg_loss:.4f}")#, Test loss {test_loss:.4f}, Acc On Test: {acc:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
