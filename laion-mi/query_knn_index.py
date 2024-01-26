import torch
import os

from typing import Tuple, List, Union

from cld3 import get_language
from clip_retrieval.clip_client import ClipClient

import pyarrow as pa
import pyarrow.parquet as pq

from utils import QueryThread, QueryProcess

from config import AESTHETIC_OUTPUT_IMGS_DIR, SEARCH_QUERY_RESULTS_DIR

CHECKPOINT_EVERY = 3_000

get_laion5b_search_client = lambda: ClipClient(
    url="https://knn.laion.ai/knn-service",  # url may change, check github.com/rom1504/clip-retrieval
    indice_name="laion5B-L-14",
    num_images=40,
    deduplicate=False,
)


def get_embeddings_urls() -> Tuple[torch.Tensor, list]:
    with open(os.path.join(AESTHETIC_OUTPUT_IMGS_DIR, "embeddings.pt"), "rb") as f_obj:
        embeddings = torch.load(f_obj)
    with open(os.path.join(AESTHETIC_OUTPUT_IMGS_DIR, "urls.txt"), "r") as f_obj:
        urls = []
        for line in f_obj.readlines():
            urls.append(line[:-1])  # rm \n
    return embeddings, urls


def get_response(embedding: torch.Tensor, url: str, search_client: ClipClient):
    results_raw = search_client.query(embedding_input=embedding.tolist())
    results = list()
    for raw_result in results_raw:
        if get_language(raw_result["caption"]).language != "en":
            continue
        result = {"url_sim": raw_result["url"], "similarity": raw_result["similarity"]}
        results.append(result)
    return {"url": url, "results": results}


def get_split_indices(arr: list, num_splits: int) -> List[List[int]]:
    indices = []
    multi = len(arr) // num_splits
    lb = 0
    rb = -1
    r = 1
    while r < num_splits:
        rb = r * multi
        indices.append([lb, rb])
        lb = rb
        r += 1
    indices.append([lb, -1])
    return indices


def create_threads(
    num_splits: int, embeddings: torch.Tensor, urls: List[str], procID: str = -1
) -> List[QueryThread]:
    threads = list()
    emb_list = list(embeddings)
    split_indices = get_split_indices(urls, num_splits)
    idx = 0
    for lb, rb in split_indices:
        thread = QueryThread(
            int(procID),
            idx,
            emb_list[lb:rb],
            urls[lb:rb],
            get_laion5b_search_client,
            get_response,
            CHECKPOINT_EVERY,
        )
        threads.append(thread)
        idx += 1
    return threads


def create_processes(
    num_processes: int, threads_cnt: int, embeddings: torch.Tensor, urls: List[str]
):
    processes = list()
    split_indices = get_split_indices(urls, num_processes)
    idx = 0
    for lb, rb in split_indices:
        process = QueryProcess(
            procID=idx,
            embeddings=embeddings[lb:rb, :],
            urls=urls[lb:rb],
            threads_cnt=threads_cnt,
            create_threads=create_threads,
            run_threads=run_threads,
        )
        processes.append(process)
        idx += 1
    return processes


def run_threads(threads: List[QueryThread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def run_processes(processes: List[QueryProcess]) -> None:
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def save_results_threads(threads: List[QueryThread]) -> None:
    save_results(threads)


def save_results_processes(processes: List[QueryProcess]) -> None:
    save_results(processes)


def save_results(workers: Union[List[QueryThread], List[QueryProcess]]) -> None:
    final_results: dict() = {
        "url_orig": [],
        "url_sim": [],
    }
    for worker in workers:
        for results in worker.results:
            if len(results["results"]) == 0:
                new_results = {
                    "url_orig": [results["url"]],
                    "url_sim": ["NO DUPLICATES"],
                }
            else:
                new_results = {
                    "url_orig": [results["url"]] * len(results["results"]),
                    "url_sim": [
                        results["results"][idx]["url_sim"]
                        for idx in range(len(results["results"]))
                    ],
                }
            for key, values in new_results.items():
                final_results[key] += values
    final_results = pa.table(
        [values for values in final_results.values()],
        names=[key for key in final_results.keys()],
    )
    os.makedirs(SEARCH_QUERY_RESULTS_DIR, exist_ok=True)
    pq.write_table(
        final_results,
        os.path.join(SEARCH_QUERY_RESULTS_DIR, "search_query_urls.parquet"),
    )


def main():
    embeddings, urls = get_embeddings_urls()
    print("embeddings size", embeddings.shape, "urls cnt", len(urls))
    processes = create_processes(
        num_processes=1,  # unfortunately API is a bottleneck, not processor
        threads_cnt=8,
        embeddings=embeddings,
        urls=urls,
    )
    # threads = create_threads(2, embeddings, urls)
    print("workers start")
    run_processes(processes)
    # run_threads(threads)
    print("workers finished")
    save_results_processes(processes)
    # save_results(threads)
    print("results saved, script end")


if __name__ == "__main__":
    main()
