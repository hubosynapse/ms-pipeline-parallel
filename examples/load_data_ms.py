import mindspore.dataset as ds
import mindspore.dataset.transforms as trans
from mindspore.common import dtype as mstype
from mindspore import ops

def create_dataset(data_path, batch_size=32, num_parallel_workers=1):
    """ create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define map operations
    type_cast_op = trans.TypeCast(mstype.float32)
    
    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="image", operations=type_cast_op, num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds
