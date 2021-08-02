import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

import qutip
from .core.data import TfTensor

def _get_tensor(qobj):
    """This function returns the ``tf.Tensor`` or ``tf.Variable`` that  represents a
    Qobj. If ``qobj.data`` is not represented with a ``TfTensor``, it raises a
    ValueError."""

    if not isinstance(qobj.data, TfTensor):
        raise ValueError(f"The provided qobj is not backed with a TfTensor"
                         "data. You may want to use Qobj.to('tftensor')"
                         "before operating with it.") from None
    return qobj.data._tf


class GradientTape(tf.GradientTape):


    def watch(self, other):
        if isinstance(other, qutip.Qobj):
            other = _get_tensor(other)

        if isinstance(other, list):
            other = [_get_tensor(obj) if isinstance(obj, qutip.Qobj)
                        else obj for obj in other]


        return super().watch(other)

    def gradient(self, target, sources, output_gradients=None,
                 unconnected_gradients=tf.UnconnectedGradients.NONE):
        if isinstance(target, qutip.Qobj):
            target = _get_tensor(target)

        if isinstance(sources, list):
            sources = [_get_tensor(source) if isinstance(source, qutip.Qobj)
                        else source for source in sources]
        elif isinstance(sources, qutip.Qobj):
            sources = _get_tensor(sources)

        return super().gradient(target, sources, output_gradients,
                                unconnected_gradients)
    def jacobian(self, target, sources,
                 unconnected_gradients=tf.UnconnectedGradients.NONE,
                 parallel_iterations=None, experimental_use_pfor=True):

        if isinstance(target, qutip.Qobj):
            target = _get_tensor(target)

        if isinstance(sources, list):
            sources = [_get_tensor(source) if isinstance(source, qutip.Qobj)
                        else source for source in sources]
        elif isinstance(sources, qutip.Qobj):
            sources = _get_tensor(sources)

        return super().jacobian(target, sources,
                                unconnected_gradients=tf.UnconnectedGradients.NONE,
                                parallel_iterations=None,
                                experimental_use_pfor=True)
