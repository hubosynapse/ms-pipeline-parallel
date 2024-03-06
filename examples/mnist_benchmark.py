"""
Benchmark model/pipline parallelism on MNIST data

We select a computing-bound setup (a MLP with memory size around 40MB)
"""
import sys
import time
from pathlib import Path
from itertools import cycle
from queue import Queue
from typing import Iterable

from mindspore import nn, ops

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.pipelined_trainer import PipelinedMlpTrainer
from examples.load_data import MNIST



def random_shuffle(images, labels):
    assert images.shape[0] == labels.shape[0]
    n_data = images.shape[0]
    indices = np.random.permutation(n_data)
    return images[indices, ...], labels[indices, ...]


def main(epoch: int=50, batch: int=4096):
    loss_fn = ops.SoftmaxCrossEntropyWithLogits()
    optimizer_cls = nn.Adam

    trainer = PipelinedMlpTrainer(
        input_size=28*28,
        hidden_size=1024,
        output_size=10,
        n_layers=20,
        optimizer_cls=optimizer_cls, loss_fn=loss_fn, activation=nn.ReLU)

    imgs, labels = MNIST.get_all_train()
    imgs_test, labels_test = MNIST.get_all_test()
    imgs_test, labels_test = imgs_test.reshape(imgs_test.shape[0], -1), labels_test.reshape(labels_test.shape[0], -1)

    print("\n")
    batch_per_epoch = imgs.shape[0] // batch
    for e in range(epoch):
        imgs, labels = random_shuffle(imgs, labels)
        avg_loss = 0.0
        start = time.time()
        for b in range(batch_per_epoch):
            x, y = imgs[b * batch: (b + 1) * batch, ...], labels[b * batch: (b + 1) * batch, ...]
            trainer.train_with_pipeline(x, y, pipe_state)
            avg_loss += loss
            elapsed = time.time() - start

            test_loss, acc = test(imgs_test, labels_test, pipe_state)
            avg_loss /= batch_per_epoch
            print(f"Epoch {e} ({batch_per_epoch * batch} images/{elapsed:.4f} sec): Train loss {avg_loss:.4f}, Test loss {test_loss:.4f}, Acc On Test: {acc:.3f}")


if __name__ == "__main__":
    main()
