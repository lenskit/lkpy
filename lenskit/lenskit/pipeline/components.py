"Definition of the component interfaces."

from typing_extensions import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from lenskit.data.dataset import Dataset

# COut is only return, so Component[U] can be assigned to Component[T] if U ≼ T.
COut = TypeVar("COut", covariant=True)


class Component(Protocol, Generic[COut]):
    """
    Interface (protocol) for pipeline components: functions from inputs to outputs.

    Most components will implement additional component protocols, such as:

    * :class:`ConfigurableComponent`
    * :class:`TrainableComponent`

    .. note::

        This protocol is equivalent to ``Callable[..., COut]`` but is defined as a
        protocol so we can define sub-protocols with additional methods.
    """

    def __call__(self, **kwargs: Any) -> COut:
        """
        Apply this component to its input data.

        .. note::

            The protocol definition allows arbitrary keyword arguments (and no
            positional arguments), to work with Python's type system limitations
            and the impossibility of manually writing :class:`~typing.ParamSpec`
            declarations, but component implementations are expected to declare
            specific input arguments with type annotations.
        """
        raise NotImplementedError()


@runtime_checkable
class ConfigurableComponent(Generic[COut], Component[COut], Protocol):
    """
    Interface for configurable pipeline components (those that have
    hyperparameters).  A configurable component supports two additional
    operations:

    * saving its configuration with :meth:`get_config`.
    * creating a new instance from a saved configuration with the class method
      :meth:`from_config`.

    A component must implement both of these methods to be considered
    configurable.

    .. note::

        This is a subtype of :class:`Component`, so implementations must also
        implement ``__call__`` as specified there.
    """

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> dict[str, object]:
        """
        Reinstantiate this component from configuration values.
        """
        ...

    def get_config(self) -> dict[str, object]:
        """
        Get this component's configured hyperparameters.
        """
        ...


@runtime_checkable
class TrainableComponent(Generic[COut], Component[COut], Protocol):
    """
    Interface for pipeline components that can learn parameters from training
    data.

    .. note::

        This is a subtype of :class:`Component`, so implementations must also
        implement ``__call__`` as specified there.
    """

    def train(self, data: Dataset) -> Self:
        """
        Train the pipeline component to learn its parameters from a training
        dataset.

        Args:
            data:
                The training dataset.
        Returns:
            The component.
        """
        raise NotImplementedError()
