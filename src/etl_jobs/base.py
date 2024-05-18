from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Literal
from loguru import logger
StepNum = Literal[f'step_{int}']

class Read(ABC):
    """
    Abstract base class for different reading strategies.
    Each subclass should implement the `read` method.
    """

    @abstractmethod
    def read(self, **kwargs) -> Dict[StepNum, Any]:
        """
        Implement this method to read data from the specific source.
        """
        pass

class ReadChain(Read):
    """
    Abstract base class for chaining multiple readers together.
    Each subclass should implement the `transform` method.
    """

    def __init__(self, readers: List[Read]):
        """
        Initialize the chained reader with a list of readers.
        """
        self.readers = readers
        self.data = {}

    def read(self) -> Dict[StepNum, Any]:
        """
        Chain the read methods together and apply transformations.
        """
        for i, reader in enumerate(self.readers):
            self.data[f'step_{i}'] = reader.read()  #
        return self.data



class Do(ABC):
    """
    Abstract base class for different processing strategies.
    Each subclass should implement the `process` method.
    """

    @abstractmethod
    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        """
        Implement this method to process the input data.

        :param data: The input data to be processed.
        :return: Processed data.
        """
        pass

class DoChain(Do):
    def __init__(self, processors: List[Do]):
        self.processors = processors
        self.data = {}

    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        for i, processor in enumerate(self.processors):
            self.data[f'step_{i}'] = processor.process(data=data.get(f'step_{i}'))
        return self.data


class Write(ABC):
    """
    Abstract base class for different writing strategies.
    Each subclass should implement the `write` method.
    """

    @abstractmethod
    def write(self, data: Dict[StepNum, Any]) -> None:
        """
        Implement this method to write data to the specific target.

        :param data: The data to be written.
        """
        pass

class WriteChain(Write):
    def __init__(self, writers: List[Write], ):
        self.writers = writers

    def write(self, data: Dict[StepNum, Any]) -> None:
        assert isinstance(data, dict)
        for i, writer in enumerate(self.writers):
            writer.write(data=data.get(f'step_{i}'))

class Job:
    """
    A Job class that wraps a combination of Read, Do, and Write operations.
    Allows for easy composition and execution of ETL jobs.
    """

    def __init__(self, reader: Read = None, processor: Do = None, writer: Write = None):
        """
        Initialize the Job with specific Read, Do, and Write implementations.

        :param reader: The reading strategy to use.
        :param processor: The processing strategy to use.
        :param writer: The writing strategy to use.
        """
        self.reader = reader
        self.processor = processor
        self.writer = writer
        self.data = {}
        self.processed_data = {}

    def run(self, data=None, processed_data=None) -> None:
        """
        Execute the ETL job by reading, processing, and writing data.
        """
        if self.reader:
            logger.info('start READ')
            data = self.reader.read()
            self.data.update(data)
            logger.info('end READ')
        if self.processor:
            logger.info('start DO')
            processed_data = self.processor.process(data)  # Dict[StepNum, List[Dict[str, List[Dict]]]]
            self.processed_data.update(processed_data)
            logger.info('end DO')
        if self.writer:
            logger.info('start WRITE')
            if self.processor:
                self.writer.write(data=self.processed_data)
            else:
                self.writer.write(data=self.data)
            logger.info('end WRITE')
