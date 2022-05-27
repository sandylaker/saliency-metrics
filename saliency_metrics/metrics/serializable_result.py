from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class SerializableResult(Protocol):
    """Final evaluation result that can be serialized to a JSON or yaml file."""

    @abstractmethod
    def dump(self, file_path: str) -> None:
        """Save the result to a JSON or yaml file.

        Args:
            file_path: Output file path.

        Returns:
            None
        """
        raise NotImplementedError
