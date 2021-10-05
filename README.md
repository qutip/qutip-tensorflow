qutip-tensorflow: TensorFlow backend for QuTiP
==============================================

A plug-in for [QuTiP](https://qutip.org) providing a [TensorFlow](https://www.tensorflow.org/) linear-algebra backend.
Backing the linear algebra operations with TensorFlow extends QuTiP's
capability to work with a GPU. Furthermore, it allows QuTiP's `Qobj` class to
benefit from [auto differentiation](https://www.tensorflow.org/guide/autodiff).

How to use qutip-tensorflow
---------------------------

To use qutip-tensorflow you only need to include the import statement.
```python
import qutip_tensorflow
```
Once qutip-tensorflow is imported, it hooks into QuTiP adding a new data backed
based on TensorFlow's Tensor. It is hence not necessary to use any of
qutip-tensorflow's functions explicitly.

The main class implemented in qutip-tensorflow is `TfTensor` that
wraps around a `Tensor` to provide compatibility between QuTiP and TensorFlow.
It is possible to instantiate a new `Qobj` backed with a `TfTensor` using:
```python
import qutip
import tensorflow as tf
qobj = qutip.Qobj(tf.constant([1, 2]))
qobj.data  # Instance of TfTensor
```

You can still access the underlying `Tensor` with the attribute `_tf`.
```python
qobj.data._tf  # Instance of tf.Tensor
```

QuTiP provides several useful functions for array creation. These return by
default a `Qobj` backed with either a `Dense` or `CSR` data container. To
obtain a `Qobj` backed with a `TfTensor` it suffices to use the `to` method:
```python
sx = qutip.sigmax()  # Pauli X matrix
sx.data  # Instance of `CSR`
sx = sx.to('tftensor') # 'TfTensor' also works
sx.data  # Instance of `TfTensor`
```

When importing qutip-tensorflow, operations are done using the default detected
device. Hence, if a GPU is configured by TensorFlow, it will use make use of it.

qutip-tensorflow also works with TensorFlow's `GradientTape` for auto
differentiation:
```python
sz = qt.sigmaz().to('tftensor')

# It is very common to express your variables as being real
variable = tf.Variable(10, dtype=tf.float64)

state = qutip.basis(2, 0).to('tftensor')

with tf.GradientTape() as tape:
    # Tensorflow does not support automatic casting by default.
    x = tf.cast(variable, tf.complex128)

    # The operation computed is <0|x*sz|0> = x <0|sz|0> = x
    y = qutip.expect(x*sz, state)

# dy/dx = 1
tape.gradient(y, variable)  # 1
```

For a more involved example of how to use `GradientTape` for optimization
purposes see the example notebook in `qutip_tensorflow/examples`, which can be
run in [colab](https://colab.research.google.com/) using a GPU. To configure
the GPU in colab see [here](https://colab.research.google.com/notebooks/gpu.ipynb).

What is not supported yet
-------------------------

There are some features from TensorFlow that are not supported yet:
- function tracing with `tf.function`: see progress in issue #30.
- Support for keras models: see progress in issue #31.
- Support for batched operations: see progress in issue #29.
- There are still a few functions that do not relay in TensorFlow for the
  computation. This means auto differentiation and GPU operations are not
  possible with them. See progress in issue #28.


Installation (Linux)
--------------------

At this moment it is only possible to install qutip-tensorflow from source.

_It is strongly recommended to install qutip-tensorflow in  a [virtual
environment](https://docs.python.org/3/tutorial/venv.html) so that it does not
conflict with your local python installation._

First install QuTiP 5.0. Note that this version of QuTiP is still in
development, so it is necessary to install it from source:
```
pip install git+https://github.com/qutip/qutip@dev.major
```
To install qutip-tensorflow from source:
```
pip install git+https://github.com/qutip/qutip-tensorflow
```

Benchmarks
----------

If you aim to use qutip-tensorflow to speed up your code by computing with a
GPU, it is possible to run a set of benchmarks that have been prepared to help
assessing when GPU operations are faster than CPU ones. It is expected that for
small system sizes CPU operations will be faster, whereas for larger system
sizes GPU operations may posses an advantage depending on your hardware.

To run the benchmarks first clone the repository and install the package.
```
git clone https://github.com/qutip/qutip-tensorflow.git
cd qutip-tensorflow
pip install git+https://github.com/qutip/qutip@dev.major
pip install ".[full]"
```

To run the benchmarks use
```
python benchmarks/benchmarks.py
```

This will store the resulting data and figures in the folder `.benchmarks/`.

The benchmarks consist on a set of operations, such as matrix multiplication,
that are tested for each of the specialisations in QuTiP. Some of the
benchmarks also include similar operations using pure NumPy, TensorFlow or
SciPy implementations of the same operation for comparison. The benchmarks run
the same operations for different hermitian matrix sizes that can either be
dense or sparse (tridiagonal).  The script also includes a few other options.
You can get a description of the arguments with `python
benchmarks/benchmarks.py --help`. It also accepts any argument that
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) accepts.
Examples:

-`python benchmarks/benchmarks.py -k"test_linear_algebra" --collect-only`:
Shows all the available benchmarks. Useful to filter them with the `-k`
argument. 

-`python benchmarks/benchmarks.py -k"matmul"`: Runs only the benchmarks for
`matmul`.

-`python benchmarks/benchmarks.py -k"add and -dense-"`: Runs only the
benchmarks for `add` (addition) with dense random matrices. 

-`python benchmarks/benchmarks.py -k"add and -dense- and qutip_dense"`: runs only the
benchmarks for `add` with dense random matrices and only for the `qutip_dense`
data type. 

-`python benchmarks/benchmarks.py -k"add and -dense- and qutip_"`: runs only the
benchmarks for `add` with dense random matrices for all the specialisations in
QuTiP. 

-`python benchmarks/benchmarks.py -k"expm and -512-"`: Runs only the
benchmarks for `expm` for a matrix of size 512x512 (the size can only be
2,4,8...,512,1024).

-`python benchmarks/benchmarks.py -k"(tensorflow or numpy or qutip_dense) and
-2-"`: Runs the benchmarks for every operation with hermitian
matrices of size 2x2 represented with either `tensorflow`, `numpy` or the
`qutip_dense` data type.


Support
-------

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)

We are proud to be affiliated with [Unitary Fund](https://unitary.fund) and
[NumFOCUS](https://numfocus.org).  QuTiP development is supported by [Nori's
lab](https://dml.riken.jp/) at RIKEN, by the University of Sherbrooke, and by
Aberystwyth University, [among other supporting
organizations](https://qutip.org/#supporting-organizations).  Initial work on
this project was sponsored by [Google Summer of Code
2021](https://summerofcode.withgoogle.com).


