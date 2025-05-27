import os
import logging
import numpy as np
import pandas as pd
from typing import List
from copy import deepcopy

from .embedding_model import BaseEmbeddingModel
from .utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class EmbeddingStore:

    def __init__(self, embedding_model: BaseEmbeddingModel, db_filename: str, batch_size: int, namespace: str):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model: BaseEmbeddingModel = embedding_model
        self.batch_size: int = batch_size
        self.namespace: str = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename: str = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(
            ), df["embedding"].values.tolist()
            self.hash_id_to_idx: dict[str, int] = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row: dict[str, dict[str, str]] = {
                h: {
                    "hash_id": h,
                    "content": t
                } for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text: dict[str, str] = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id: dict[str, str] = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]) -> None:
        nodes_dict: dict[str, dict[str, str]] = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids: list[str] = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing: list[str] = self.hash_id_to_row.keys()
        missing_ids: list[str] = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return {}

        texts_to_encode: list[str] = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        missing_embeddings: np.ndarray = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _upsert(self, hash_ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

        logger.info(f"Saving new records.")
        self._save_data()

    def _save_data(self):
        data_to_save: pd.DataFrame = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row: dict[str, dict[str, str]] = {
            h: {
                "hash_id": h,
                "content": t
            } for h, t in zip(self.hash_ids, self.texts)
        }
        self.hash_id_to_idx: dict[str, int] = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text: dict[str, str] = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id: dict[str, str] = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def get_all_id_to_rows(self) -> dict[str, dict[str, str]]:
        return deepcopy(self.hash_id_to_row)

    def get_embeddings(self, hash_ids: list[str], dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            return []

        indices: np.ndarray = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings: np.ndarray = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings

    def get_all_ids(self) -> list[str]:
        return deepcopy(self.hash_ids)

    def get_rows(self, hash_ids: list[str], dtype=np.float32) -> dict[str, dict[str, str]]:
        if not hash_ids:
            return {}

        results: dict[str, dict[str, str]] = {id: self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_row(self, hash_id: str) -> dict[str, str]:
        return self.hash_id_to_row[hash_id]

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
