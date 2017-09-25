# Tensorflow
TensorFlow is an open-source library for machine learning across a range of tasks. It was developed by the Google Brain team for internal Google use, 
but was later released under the Apache 2.0 open source license for the purpose of conducting 
neural network and deep learning research. TensorFlow's great for research, bit it's ready for use in real products too, 
as it was built from the ground up to be fast, portable and ready for production service.

TensorFlow is uses dataflow graphs for numerical computation, and allows you to deploy to CPUs or GPUs on desktops or servers with a single API, and also has a lot of tools to make building neural networks and optimizers easy.

## Installation
TensorFlow can be installed to run either on the CPU or on the GPU. Installing on the CPU is in most cases trivial, but
a GPU installation will be a lot more efficient. TensorFlow requires a CUDA compatibility of at least 3.0 to be able to
use the GPU.

### Installing on Linux (GPU)
A prerequisite for the GPU installation is to have the proprietary graphics drivers for your video card. 

Additional requiered packages are:
- `numpy`
- `swig`

#### Checking CUDA compatibility of the GPU
You can use `lspci | grep -i nvidia` to see which Graphic Card you have. Then you can search [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to find which version your GPU supports. As said, you need **at least version 3.0**.

#### Installing CUDA 
CUDA is one of the required libraries for TensorFlow. You can install it with `pacman` if you have an Archlinux distribution. For other distributions, check if the default package manager can install cuda. 

Otherwise you have to manually download an installer from the official site - [Link](https://developer.nvidia.com/cuda-downloads) - *Please note, that you need an Nvidia Developer account in order to download CUDA and Cudnn in the next step*

A guide for installing CUDA on Ubuntu, which you may find helpful - [Link](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)

#### Installing CUDNN
CUDnn is another dependency for the TensorFlow project. You can download a tarball from the Nvidia website [here...](https://developer.nvidia.com/cudnn-download) You have to complete a short survey and then you can choose a download.
Be careful to download the appropriate tarball for your CUDA version. Currently TensorFlow seems to support **CUDnn 6.0**.

After completing the download unpack the tarball and copy the files to the CUDA install location. For example on Archlinux CUDA is installed in `/opt/cuda/`, so copy the tarball's 
`.../cudnn/lib64/` to `/opt/cuda/lib64` and `.../cudnn/include/` to `/opt/cuda/include`. 

After that it's important to add your CUDA and CUDnn install location to your `$LD_LIBRARY_PATH`. 
For example add your following line to your `~/.bashrc`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<cuda/install/dir>`

#### Installing TensorFlow
Depending if you want to use Python2 or Python3, you can either use `pip` or `pip3` and do `pip install tensorflow-gpu`. 

That also works in a virtual environment with `virtualenv`, but you have to explicitly point to the CUDA install location 
or copy the CUDA files to the virtual environment's `lib` folder. 

### Installing on Linux (CPU)
Installing on the CPU is as easy as doing `pip install tensorflow`. 

However your CPU may have additional instruction, which will opimize how TensorFlow is running. But to do that, you need to compile TensorFlow from source. [Compiling TensorFlow](https://www.tensorflow.org/install/install_sources)

### Installing on Windows
*???*

### Additional Install Guides
- [Install on Archlinux](https://github.com/ddigiorg/AI-TensorFlow/blob/master/install/install-TF_2016-02-27.md)
- [Official TensorFlow Install Documentation](https://www.tensorflow.org/install/)


### Validating your TensorFlow installation
A simple TensorFlow program to verify if it's properly installed.

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

## Terminology
- **Tensor** - it's like a matrix. An array of primitive types, shaped into any number of dimensions.

- **Dataflow Graph** - each node is an operations between zero or more tensors as inputs and produces a tensor as output.

- **Estimator** - a high-level API. Estimators encapsulate training, evaluation, prediction and exporting for serving

See the [TensorFlow Glossary](https://www.tensorflow.org/versions/r0.12/resources/glossary) for more terms.


## APIs
TensorFlow provides multiple APIs:

### TensorFlow Core 
- the lowest level API
- provides the user with complete control 
- officially recommended for machine learning researchers and experts, who need fine control over their program

### Higher level APIs
- easier to learn and to use 
- make repetitve task easier

### Examples for high-level APIs
- `tf.estimator` - makes running training and evaluation loops easier and provides dataset management tools
- `tf.layers` - more layers for neural networks
- `tf.nn` - tools for neural networks - convolutional, dropout, pooling layers, activation and loss functions
- `tf.train` - provides optimizers for the weights in the model
- and many more...

### tf.contrib
Contains contributions to the TensorFlow, which have yet to be merged. The code there is not supported by the TensorFlow team and may require some more testing. However the APIs here are still useful. 

- `tf.contrib.cloud` - a module for cloud computing
- `tf.contrib.keras` - a neural networks API with a focus on enabling fast experimentation - [External Docs](https://keras.io/)
*Note that Keras will be integrated in TensorFlow in the near future, so it's better to use *`tf.contrib keras`
- `tf.contrib.learn` - API inspired from scikit-learn
- `tf.contrib.slim` - 'syntactic sugar' to simplify writing TensorFlow code
- `tf.contrib.tensorboard` - a tool, which creates visualisations of the dataflow graph of a program
- and many more...

## Tutorials
There are very good tutorials on the TensorFlow website

- [Basic NN with the MNIST Dataset](https://www.tensorflow.org/get_started/mnist/beginners) - *This tutorial is intended for reader who are new to both machine learning and tensorflow. If you know what MNIST and softmax regression are, then the fast paced tutorial*

- [Faster Paced MNIST tutorial for Machine Learning experts](https://www.tensorflow.org/get_started/mnist/pros) - *This introduction assumes familiarity with neural networks and the MNIST dataset.*

- [tf.estimator Quickstart](https://www.tensorflow.org/get_started/estimator) - *A high level API which makes it easy to configure, train and evaluate a variety of machine learning models*

- [Input Pipelines with tf.estimator](https://www.tensorflow.org/get_started/input_fn) - *This tutorial introduces creating input functions to preprocess and feed data into the model*

- [Using Tensorboard to visualize learning](https://www.tensorflow.org/get_started/summaries_and_tensorboard) - *Serializing data and visualizing progress with TensorBoard*

- [Programmer's Guide](https://www.tensorflow.org/programmers_guide/) - *Details about different data structures and concepts from the TensorFlow codebase, such as Estimators, Tensors, Sessions, Threading, etc.*

- [More Tutorials](https://www.tensorflow.org/tutorials/) - *Problem-specific tutorials on topics such as image recognition, natural language processing and linear models*

- [Performance Guide](https://www.tensorflow.org/performance/performance_guide) - *Cotains a collection of best practices for optimizing TensorFlow code*

- [High-Performance Models](https://www.tensorflow.org/performance/performance_models) - *These Documents detail how to build highly scalable models that target a variety of system types and network topologies. Contains a lot of low-level TensorFlow code*

## Useful Links
- The TensorFlow API Documentation - [Link](https://www.tensorflow.org/api_docs/python/)
- Basic implementations of different machine learning models - [Link](https://hackernoon.com/tensorflow-in-a-nutshell-part-three-all-the-models-be1465993930)

