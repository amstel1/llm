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
from loguru import logger

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
    elbow: bool = True
    most_relevant_at_the_top = False

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
        if max(res.values()) <= 0.09:
            # it's unreasonable to use reranking when the reranking model considers the docs so irreleavant
            logger.debug(f'emergency exit from reranking, max score is: {max(res.values())}')
            return documents
        sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        output = []
        if self.elbow:
            # self.top_n is ignored
            scores = pd.Series(data=[x[1] for x in sorted_res], index=[x[0] for x in sorted_res])
            logger.debug(f'reranking before elbow: {scores}')
            argmin = scores.diff().argmin()
            logger.debug(f'reranking argmin, remain: {argmin} out of {len(scores)}')
            self.top_n = argmin
        for i, _score in sorted_res[:self.top_n]:
            output.append(documents[i])
        if not self.most_relevant_at_the_top:
            output = output[::-1]
        return output

from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pymilvus import AnnSearchRequest, Collection
from pymilvus.client.abstract import BaseRanker, SearchResult  # type: ignore
import pickle
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from langchain_milvus.utils.sparse import BM25SparseEmbedding


class ExtendedMilvusCollectionHybridSearchRetriever(BaseRetriever):
    """This is a hybrid search retriever
    that uses Milvus Collection to retrieve documents based on multiple fields.
    For more information, please refer to:
    https://milvus.io/docs/release_notes.md#Multi-Embedding---Hybrid-Search
    """

    collection: Collection
    """Milvus Collection object."""
    rerank: BaseRanker
    """Milvus ranker object. Such as WeightedRanker or RRFRanker."""
    anns_fields: List[str]
    """The names of vector fields that are used for ANNS search."""
    field_embeddings: List[Union[Embeddings, BaseSparseEmbedding]]
    """The embedding functions of each vector fields, 
    which can be either Embeddings or BaseSparseEmbedding."""
    field_search_params: Optional[List[Dict]] = None
    """The search parameters of each vector fields. 
    If not specified, the default search parameters will be used."""
    field_limits: Optional[List[int]] = None
    """Limit number of results for each ANNS field. 
    If not specified, the default top_k will be used."""
    field_exprs: Optional[List[Optional[str]]] = None
    """The boolean expression for filtering the search results."""
    top_k: int = 4
    """Final top-K number of documents to retrieve."""
    text_field: str = "text"
    """The text field name, 
    which will be used as the `page_content` of a `Document` object."""
    output_fields: Optional[List[str]] = None
    """Final output fields of the documents. 
    If not specified, all fields except the vector fields will be used as output fields,
    which will be the `metadata` of a `Document` object."""
    use_elbow_for_embedding: bool = True
    most_relevant_at_the_top: bool = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # If some parameters are not specified, set default values
        if self.field_search_params is None:
            default_search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            self.field_search_params = [default_search_params] * len(self.anns_fields)
        if self.field_limits is None:
            self.field_limits = [self.top_k] * len(self.anns_fields)
        if self.field_exprs is None:
            self.field_exprs = [None] * len(self.anns_fields)

        # Check the fields
        self._validate_fields_num()
        self.output_fields = self._get_output_fields()
        self._validate_fields_name()

        # Load collection
        self.collection.load()

    def _validate_fields_num(self) -> None:
        assert (
            len(self.anns_fields) >= 1
        ), "At least two fields are required for hybrid search."
        lengths = [len(self.anns_fields)]
        if self.field_limits is not None:
            lengths.append(len(self.field_limits))
        if self.field_exprs is not None:
            lengths.append(len(self.field_exprs))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All field-related lists must have the same length.")

        if len(self.field_search_params) != len(self.anns_fields):  # type: ignore[arg-type]
            raise ValueError(
                "field_search_params must have the same length as anns_fields."
            )

    def _validate_fields_name(self) -> None:
        collection_fields = [x.name for x in self.collection.schema.fields]
        for field in self.anns_fields:
            assert (
                field in collection_fields
            ), f"{field} is not a valid field in the collection."
        assert (
            self.text_field in collection_fields
        ), f"{self.text_field} is not a valid field in the collection."
        for field in self.output_fields:  # type: ignore[union-attr]
            assert (
                field in collection_fields
            ), f"{field} is not a valid field in the collection."

    def _get_output_fields(self) -> List[str]:
        if self.output_fields:
            return self.output_fields
        output_fields = [x.name for x in self.collection.schema.fields]
        for field in self.anns_fields:
            if field in output_fields:
                output_fields.remove(field)
        if self.text_field not in output_fields:
            output_fields.append(self.text_field)
        return output_fields

    def _build_ann_search_requests(self, query: str) -> List[AnnSearchRequest]:
        search_requests = []
        for ann_field, embedding, param, limit, expr in zip(
            self.anns_fields,
            self.field_embeddings,
            self.field_search_params,  # type: ignore[arg-type]
            self.field_limits,  # type: ignore[arg-type]
            self.field_exprs,  # type: ignore[arg-type]
        ):
            request = AnnSearchRequest(
                data=[embedding.embed_query(query)],
                anns_field=ann_field,
                param=param,
                limit=limit,
                expr=expr,
            )
            search_requests.append(request)
        return search_requests

    def _parse_document(self, data: dict) -> Document:
        return Document(
            page_content=data.pop(self.text_field),
            metadata=data,
        )

    def _process_search_result(
        self, search_results: List[SearchResult]
    ) -> List[Document]:
        documents = []
        for result in search_results[0]:
            data = {x: result.entity.get(x) for x in self.output_fields}  # type: ignore[union-attr]
            data.update({'distance': result.distance})
            doc = self._parse_document(data)
            documents.append(doc)
        return documents

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        requests = self._build_ann_search_requests(query)
        # if 'distance' not in self.output_fields:
        #     self.output_fields.append('distance')
        search_result = self.collection.hybrid_search(
            requests, self.rerank, limit=self.top_k, output_fields=self.output_fields
        )
        documents = self._process_search_result(search_result)
        if self.use_elbow_for_embedding:
            # elbow embedding
            scores = pd.Series(data=[x.metadata.get('distance') for x in documents], index=[i for i, _ in enumerate(documents)])
            logger.debug(f'embedding before elbow: {scores}')
            argmin = scores.diff().argmin()
            logger.debug(f'embedding argmin, remain: {argmin} out of {len(scores)}')
            if scores.diff().min() <= -0.05:
                # enough of a gap to use elbow
                documents = documents[:argmin]
                logger.debug('embedding, the gap is enough to apply elbow')
            else:
                logger.debug('embedding, the gap is NOT enough to apply elbow')

            logger.debug(f'embedding, n documents: {len(documents)}')
        if not self.most_relevant_at_the_top:
            documents = documents[::-1]
        return documents

class BM25:

    def fit(self, data=None):
        if not data:
            data = self.data
        self.bse = BM25SparseEmbedding(corpus=data, language='ru')

    def transform(self, data:str) -> Dict:
        '''return bse'''
        assert self.bse
        return self.bse.embed_query(data)  # dict

    def load_data_one(self, pkl_input_filepath: str):
        with open(pkl_input_filepath, 'rb') as f:
            data = pickle.load(f)
        self.data = data

    def load_data_many(self, pkl_input_filepaths:list[str]):
        results = []
        for filepath in pkl_input_filepaths:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            results.append(data)
        self.data = results  # must be list[str

    def save_bse_model(self, filepath='router_bm25_model.pkl'):
        assert self.bse
        with open(filepath, 'wb') as f:
            pickle.dump(self.bse, f)

    def load_bse_model(self, filepath='router_bm25_model.pkl'):
        with open(filepath, 'rb') as f:
            self.bse = pickle.load(f)
