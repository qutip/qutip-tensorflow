import qutip
from .tftensor import TfTensor
from .adjoint import adjoint_tftensor
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

__all__ = ["project_tftensor"]


def project_tftensor(state):
    """
    Calculate the projection |state><state|.  The shape of `state` will be used
    to determine if it has been supplied as a ket or a bra.  The result of this
    function will be identical is passed `state` or `adjoint(state)`.
    """
    isket = state.shape[1] == 1
    isbra = state.shape[0] == 1
    if isket:
        state_dag = adjoint_tftensor(state)

    elif isbra:
        state_dag = state
        state = adjoint_tftensor(state)

    else:
        raise ValueError(
            "state must have one column or one row but instead"
            f"has shape: {state.shape}."
        )

    out_shape = (state.shape[0], state.shape[0])
    return TfTensor._fast_constructor(state._tf @ state_dag._tf, shape=out_shape)


qutip.core.data.project.add_specialisations(
    [
        (TfTensor, TfTensor, project_tftensor),
    ]
)
