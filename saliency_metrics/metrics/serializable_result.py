from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SerializableResult(Protocol):
    """Final evaluation result that can be serialized to a JSON or yaml file."""

    def add_single_result(self, single_result: Any) -> None:
        """Add single evaluation result.

        Args:
            single_result: The result on a single sample or the result under a single re-training setting.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def dump(self, file_path: str) -> None:
        """Save the result to a JSON or yaml file.

        Args:
            file_path: Output file path.

        Returns:
            None
        """
        raise NotImplementedError
