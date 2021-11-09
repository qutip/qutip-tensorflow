import qutip
from .tftensor import TfTensor128, TfTensor64
import tensorflow as tf

# Conversion function
def _tf128_from_dense(dense):
    return TfTensor128(dense.to_array())

def _tf64_from_dense(dense):
    return TfTensor64(dense.to_array())

def _tf_to_dense(tftensor):
    return qutip.data.Dense(tftensor.to_array(), copy=False)

def _tf64_from_tf128(tftensor):
    # Posibly raise a warning?
    return TfTensor64._fast_constructor(tf.cast(tftensor._tf, tf.complex64),
                                       shape=tftensor.shape)

def _tf64_to_tf128(tftensor):
    return TfTensor128._fast_constructor(tf.cast(tftensor._tf, tf.complex128),
                                       shape=tftensor.shape)

def is_tftensor128(data):
    return isinstance(data, tf.Tensor) and data.dtype is tf.complex128

def is_tftensor64(data):
    return isinstance(data, tf.Tensor) and data.dtype is tf.complex64

# Register the data layer
qutip.data.to.add_conversions(
    [
        (TfTensor128, qutip.data.Dense, _tf128_from_dense),
        (qutip.data.Dense, TfTensor128, _tf_to_dense),

        (TfTensor64, qutip.data.Dense, _tf64_from_dense),
        (qutip.data.Dense, TfTensor64, _tf_to_dense),

        (TfTensor64, TfTensor128, _tf64_from_tf128),
        (TfTensor128, TfTensor64, _tf64_to_tf128),
    ]
)

# User friendly name for conversion with `to` or Qobj creation functions:
qutip.data.to.register_aliases(["tftensor128", "tftensor", "TfTensor"],
                               TfTensor128)
qutip.data.to.register_aliases(["tftensor64"], TfTensor64)

qutip.data.create.add_creators(
    [
        (is_tftensor64, TfTensor64, 85),
        (is_tftensor128, TfTensor128, 85),
    ]
)
