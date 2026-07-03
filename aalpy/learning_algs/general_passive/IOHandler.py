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

class CopyOnWriteIOHandler(IOHandler[T], ABC):
    def __init__(self):
        self.copied_on_write = set()

    def init_merge(self, red: 'GsmNode[T]', blue: 'GsmNode[T]'):
        self.copied_on_write.clear()

    def merge(self, x: T, y: T) -> T:
        if id(x) not in self.copied_on_write:
            x = self.copy_on_write(x)
            self.copied_on_write.add(id(x))
        self.merge_into_x(x, y)
        return x

    @abstractmethod
    def merge_into_x(self, x: T, y: T):
        pass

    @abstractmethod
    def copy_on_write(self, x: T) -> T:
        pass

    def copy(self, x: T) -> T:
        return x
