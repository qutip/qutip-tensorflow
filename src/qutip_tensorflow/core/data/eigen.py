import qutip
from .tftensor import TfTensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf


__all__ = ["eigs_tftensor"]


def _eigs_check_shape(data):
    if data.shape[0] != data.shape[1]:
        raise TypeError("Can only diagonalize square matrices")


def _eigs_check_sort(sort):
    if sort not in ("low", "high"):
        raise ValueError("'sort' must be 'low' or 'high'")


def _eigs_check_eigvals(eigvals, maximum_eigvals):
    """Raise ValueError if number of requested eigvals is greater than the
    possible number of eigenvalues."""
    if eigvals > maximum_eigvals:
        raise ValueError(
            f"""It is only possible to compute {maximum_eigvals}
                         eigenvalues but you requested {eigvals}"""
        )


def eigs_tftensor(data, isherm=None, vecs=True, sort="low", eigvals=0):
    _eigs_check_shape(data)
    _eigs_check_sort(sort)
    eigvals = eigvals if eigvals != 0 else data.shape[0]
    _eigs_check_eigvals(eigvals, data.shape[0])

    if vecs:
        driver = tf.linalg.eigh if isherm else tf.linalg.eig
        eigenvalues, eigenvectors = driver(data._tf)

    else:
        driver = tf.linalg.eigvalsh if isherm else tf.linalg.eigvals
        eigenvalues = driver(data._tf)

    if not isherm:
        # Eigenvalues are sorted by 'abs' value if not hermitian.
        # Change to real value sort.
        mask = tf.argsort(tf.math.real(eigenvalues))
        eigenvalues = tf.gather(eigenvalues, mask)
        if vecs:
            eigenvectors = tf.gather(eigenvectors, mask)

    if sort == "high":
        eigenvalues = tf.reverse(eigenvalues, [-1])
        if vecs:
            eigenvectors = tf.reverse(eigenvectors, [-1])

    if vecs:
        return eigenvalues[:eigvals], eigenvectors[..., :eigvals]
    else:
        return eigenvalues[:eigvals]


qutip.data.eigs.add_specialisations([(TfTensor, eigs_tftensor)])
