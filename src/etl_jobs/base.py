from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import logger

class Read(ABC):
    """
    Abstract base class for different reading strategies.
    Each subclass should implement the `read` method.
    """

    @abstractmethod
    def read(self) -> Any:
        """
        Implement this method to read data from the specific source.
        """
        pass


class Do(ABC):
    """
    Abstract base class for different processing strategies.
    Each subclass should implement the `process` method.
    """

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Implement this method to process the input data.

        :param data: The input data to be processed.
        :return: Processed data.
        """
        pass


class Write(ABC):
    """
    Abstract base class for different writing strategies.
    Each subclass should implement the `write` method.
    """

    @abstractmethod
    def write(self, data: Any) -> None:
        """
        Implement this method to write data to the specific target.

        :param data: The data to be written.
        """
        pass


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

    def run(self, data=None, processed_data=None) -> None:
        """
        Execute the ETL job by reading, processing, and writing data.
        """
        if self.reader:
            logger.info('start READ')
            data = self.reader.read()
            logger.info('end READ')
        if self.processor:
            logger.info('start DO')
            processed_data = self.processor.process(data)
            logger.info('end DO')
        if self.writer:
            logger.info('start WRITE')
            if self.processor:
                self.writer.write(processed_data)
            else:
                self.writer.write(data)
            logger.info('end WRITE')
