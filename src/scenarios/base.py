from abc import abstractmethod, ABC
from typing import List

class BaseScenario(ABC):
    def handler_query(self, user_input: str,):
        raise NotImplementedError