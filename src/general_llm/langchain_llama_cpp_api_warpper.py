import sys
sys.path.append('/home/amstel/llm/src')

from langchain_community.llms import LlamaCpp
from general_llm.llm_endpoint import call_generation_api
from typing import Dict, Any
from langchain_core.callbacks.manager import Callbacks
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.language_models.llms import LLM


class LlamaCppApiWrapper(LLM,):
    model_path = '.'
    def _call(self, prompt: str, stream: bool = False, **kwargs):
        assert stream is False  # later
        if not stream:
            generation_str = call_generation_api(prompt=prompt,)
        # adapt for parent class ._call
        # return {"choices": [{"text": generation_str}]}
        return generation_str

    def _llm_type(self) -> str:
        return 'llama3_api_wrapper'

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        object.__setattr__(self, 'callbacks', None)
        object.__setattr__(self, 'verbose', False)
        object.__setattr__(self, 'tags', None)
        object.__setattr__(self, 'metadata', None)
        object.__setattr__(self, 'model_path', '.')
        object.__setattr__(self, 'cache', None)
        object.__setattr__(self, 'streaming', False)
        object.__setattr__(self, 'stop', ['<|eot_id|>'])



    # @property
    # def _default_params(self) -> Dict[str, Any]:
    #     """Get the default parameters for calling llama_cpp."""
    #     pass
    #     return {"stop_sequences": None}
    #
    # @root_validator()
    # def validate_environment(cls, values: Dict) -> Dict:
    #     pass
    #
    # @root_validator()
    # def build_model_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    #     pass

