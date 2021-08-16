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

Installation (Linux)
--------------------

At this moment it is only possible to install qutip-tensorflow from source.

_It is strongly recommended to install qutip-tensorflow in  a [virtual
environment](https://docs.python.org/3/tutorial/venv.html) so that it does not
conflict with your local installation of python._

First install QuTiP 5.0. Note that this version of QuTiP is still in
development, so it is necessary to install it from source:
```
pip install git+https://github.com/qutip/qutip@dev.major
```
To install qutip-tensorflow from source:
```
pip install git+https://github.com/qutip/qutip-tensorflow
```

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


