import re
from typing import List
from langchain_text_splitters import TextSplitter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd
import requests
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

class MarkdownTextSplitter(TextSplitter):
    def __init__(self, patterns = [r"\*\*(.*?)\*\*"]):
        self.patterns = patterns

    def split_text(self, text: str) -> List[str]:
        chunks = []
        matches = []
        for pattern in self.patterns:
            matches.extend(list(re.finditer(pattern, text)))
        n_matches = len(matches)
        for i, element in enumerate(matches):
            if i < n_matches-1:
                start_pos = element.span()[0]
                end_pos = matches[i+1].span()[0]
                chunks.append(text[start_pos:end_pos].strip())
            else:
                start_pos = element.span()[0]
                end_pos = -1
                chunks.append(text[start_pos:end_pos].strip())
        return chunks

from FlagEmbedding import FlagReranker

class BGEDocumentCompressor(BaseDocumentCompressor):
    model_name_or_path: str = 'BAAI/bge-reranker-v2-m3'
    use_fp16: bool = False
    device: str = 'cpu'
    top_n: int = 5
    top_n = max(1, top_n)
    client = FlagReranker(
        model_name_or_path=model_name_or_path,
        use_fp16=use_fp16,
        device=device,
    )
    elbow: bool = False

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        res = {}
        for i, document in enumerate(documents):
            res[i] = self.client.compute_score([query, document.page_content], normalize=True)
        sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        output = []
        if self.elbow:
            # self.top_n is ignored
            scores = pd.Series(data=[x[1] for x in sorted_res], index=[x[0] for x in sorted_res])
            argmin = scores.diff().argmin()
            self.top_n = argmin
        for i, _score in sorted_res[:self.top_n]:
            output.append(documents[i])
        return output

