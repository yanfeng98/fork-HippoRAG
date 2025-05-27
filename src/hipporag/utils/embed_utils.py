import torch
from tqdm import tqdm
from typing import List, Generator, Tuple


def retrieve_knn(query_ids: List[str],
                 key_ids: List[str],
                 query_vecs,
                 key_vecs,
                 k=2047,
                 query_batch_size=1000,
                 key_batch_size=10000) -> dict[str, tuple[list[str], list[float]]]:
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.
    Args:
        query_ids:
        key_ids:
        k: top-k
        query_batch_size:
        key_batch_size:

    Returns:

    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(key_vecs) == 0:
        return {}

    query_vecs: torch.Tensor = torch.tensor(query_vecs, dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)
    key_vecs: torch.Tensor = torch.tensor(key_vecs, dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results: dict[str, tuple[list[str], list[float]]] = {}

    def get_batches(vecs: torch.Tensor, batch_size: int) -> Generator[Tuple[torch.Tensor, int], None, None]:
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries"):
        query_batch: torch.Tensor = query_batch.clone().detach()
        query_batch = query_batch.to(device)

        batch_topk_sim_scores: list[torch.Tensor] = []
        batch_topk_indices: list[torch.Tensor] = []

        offset_keys: int = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            key_batch: torch.Tensor = key_batch.to(device)
            actual_key_batch_size: int = key_batch.size(0)

            similarity: torch.Tensor = torch.mm(query_batch, key_batch.T)
            topk_sim_scores, topk_indices = torch.topk(similarity,
                                                       min(k, actual_key_batch_size),
                                                       dim=1,
                                                       largest=True,
                                                       sorted=True)

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            key_batch = key_batch.cpu()
            torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size
        # end for each kb batch

        batch_topk_sim_scores: torch.Tensor = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices: torch.Tensor = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(batch_topk_sim_scores,
                                                               min(k, batch_topk_sim_scores.size(1)),
                                                               dim=1,
                                                               largest=True,
                                                               sorted=True)
        final_topk_sim_scores: torch.Tensor = final_topk_sim_scores.cpu()
        final_topk_indices: torch.Tensor = final_topk_indices.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx: int = query_batch_start_idx + i
            query_idx: str = query_ids[query_relative_idx]

            final_topk_indices_i: torch.Tensor = final_topk_indices[i]
            final_topk_sim_scores_i: torch.Tensor = final_topk_sim_scores[i]

            query_to_topk_key_relative_ids: torch.Tensor = batch_topk_indices[i][final_topk_indices_i]
            query_to_topk_key_ids: list[str] = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
            results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

        query_batch = query_batch.cpu()
        torch.cuda.empty_cache()
    # end for each query batch

    return results
