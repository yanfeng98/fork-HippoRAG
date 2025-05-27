import re
import logging
import numpy as np
from hashlib import md5
from dataclasses import dataclass
from argparse import ArgumentTypeError
from typing import Dict, Any, List, Literal, Optional

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)


@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]] if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }


@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def reformat_openie_results(
        corpus_openie_results: list[dict[str, Any]]) -> tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:

    ner_output_dict: dict[str, NerRawOutput] = {
        chunk_item['idx']:
            NerRawOutput(chunk_id=chunk_item['idx'],
                         response=None,
                         metadata={},
                         unique_entities=list(np.unique(chunk_item['extracted_entities'])))
        for chunk_item in corpus_openie_results
    }
    triple_output_dict: dict[str, TripleRawOutput] = {
        chunk_item['idx']:
            TripleRawOutput(chunk_id=chunk_item['idx'],
                            response=None,
                            metadata={},
                            triples=filter_invalid_triples(triples=chunk_item['extracted_triples']))
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict


def text_processing(text: str | list[str]) -> str | list[str]:
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text: str = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()


def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> tuple[List[str], List[List[str]]]:
    chunk_triple_entities: list[list[str]] = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities: set[list[str]] = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes: list[str] = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities


def flatten_facts(chunk_triples: List[List[Triple]]) -> List[Triple]:
    graph_triples: list[tuple[str, str, str]] = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    min_val: float = np.min(x)
    max_val: float = np.max(x)
    range_val: float = max_val - min_val

    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x

    return (x - min_val) / range_val


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive).")
