# <center> ms-pipeline-parallel

### Introduction
ms-pipeline-parallel gives a minimalistic example of training models with pipeline parallelism in MindSpore.

Pipeline parallelism works by partitioning a model's layers into stages that can operate concurrently, thereby optimizing both memory and computational resources. 

Here pipeline parallelism through gradient accumulation. Training data is broken down into micro-batches, allowing for simultaneous processing across different pipeline stages. As each stage completes its forward pass, it passes activation memory to the subsequent stage. Then, following the completion of a backward pass on a micro-batch, gradients are communicated back through the pipeline, with each stage accumulating gradients locally. All data parallel groups concurrently perform gradient reductions, leading up to the optimizer updating the model weights.

For a more comprehensive understanding of pipeline parallelism, please refer to the tutorials https://www.deepspeed.ai/tutorials/pipeline/ provided by DeepSpeed.

The layout of this demo is inspired by [pipax](https://github.com/MingRuey/pipax/tree/main/src), which is another repository using JAX.

The model to be trained is a multi-layer perceptron. It is further divided into two stages, which operates on two GPUs. The first GPU processes the forward pass (F) of a micro-batch and then sends the resulting activation data to the second GPU for its forward pass. After completing the forward passes, the process reverses for the backward pass (B), where gradients are calculated and communicated back to the first GPU. Both GPUs accumulate gradients from their respective backward passes, and once all micro-batches are processed, the accumulated gradients are used to update the model's weights.

![Alt text]("docs/pipeline_parallel.png")


### How to run
Before running the demo script, there are a few prerequisites to be fullfilled.

#### Prerequisites
##### MindSpore
To install MindSpore, please refer to the [installation guide](https://www.mindspore.cn/install/en) from MindSpore website. Since this demo only supports GPU version of MindSpore, remember to select `GPU CUDA 11.6` for installation.

##### Configuration of distributed environment
In addition, we need to configure the environment for distributed training. The procedure is elaborated in MindSpore [programming guide](https://www.mindspore.cn/docs/programming_guide/en/r1.3/distributed_training_gpu.html). Since this demo requires multi-GPUs within a single host, but not multi-host, we only need to install [OpenMPI-4.0.2](https://www.open-mpi.org/faq/?category=building#easy-build) and [NCCL-2.7.6](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian) for our environment.

#### Run the benchmark example
After fullfilling the prerequites, we could run the demo now.
Go to the examples/ folder
```bash
cd examples 
```
We first need to download the MNIST dataset.
```
python  download_dataset.py
```
Then we could run the training by
```
./run_mnist_benckmark.sh
```

#### Version Compatibility
This demo is tested on Python 3.8, MindSpore 2.2.11 (GPU CUDA 11.6), Open MPI 4.0.2

### License
This project is released under the [Apache 2.0 license](LICENSE).
