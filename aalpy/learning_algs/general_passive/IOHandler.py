from abc import abstractmethod, ABC
from typing import Generic, TypeVar

T = TypeVar("T")

class IOHandler(Generic[T]):
    @abstractmethod
    def init(self, data, output_format, data_format):
        ...

    def init_merge(self, red: 'GsmNode[T]', blue: 'GsmNode[T]'):
        pass

    @abstractmethod
    def abstract(self, in_val, out_val):
        ...

    @abstractmethod
    def init_data(self) -> T:
        ...

    @abstractmethod
    def aggregate_data(self, src_node: 'GsmNode[T]', in_value, out_value, dst_node: 'GsmNode[T]'):
        ...

    @abstractmethod
    def merge(self, x: T, y: T) -> T:
        ...

    @abstractmethod
    def copy(self, x: T) -> T:
        ...

class NoAbstractionIOHandler(IOHandler[T], ABC):
    def init(self, data, output_format, data_format):
        pass

    def abstract(self, in_val, out_val):
        return in_val, out_val

class NoIOHandler(NoAbstractionIOHandler[T]):
    def init_data(self) -> T:
        return None

    def aggregate_data(self, src_node: 'GsmNode[T]', in_sym, out_value, dst_node: 'GsmNode[T]'):
        pass

    def merge(self, x: T, y: T) -> T:
        return None

    def copy(self, x: T) -> T:
        return None
