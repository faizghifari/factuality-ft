import json
import utils
import time
import concurrent.futures
import argparse

from tqdm import tqdm
from factscore.atomic_facts import AtomicFactGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--bio_path", help="bio path", required=True, type=str)
parser.add_argument(
    "--facts_path",
    help="result output fact path",
    required=True,
    type=str,
)
parser.add_argument(
    "--total_data",
    help="total data to be extracted",
    required=True,
    type=int,
)
parser.add_argument(
    "--batch_size",
    help="size of batch for processing",
    default=10,
    type=int,
)
args = parser.parse_args()

count = 0
generator = AtomicFactGenerator(
    "api.key", "demos", gpt3_cache_file=f"./.cache/gpt_cache_{count}.pkl"
)
annotated_bio = utils.read_jsonl_file(args.facts_path)


def process_data_single(dp):
    if not {"topic": dp["topic"], "num_response": dp["num_response"]} in annotated_bio:
        atomic_facts, _ = generator.run(dp["output"])
        annotations = []
        for f in atomic_facts:
            annotations.append(
                {"text": f[0], "model-atomic-facts": [{"text": t} for t in f[1]]}
            )
        dp["annotations"] = annotations

        return dp
    else:
        return None


def process_data_multi(data):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_data_single, d): d for d in data}

        concurrent.futures.wait(futures)

        for future in futures:
            annotated_dp = future.result()
            if annotated_dp is not None:
                utils.append_dict_to_jsonl(args.facts_path, annotated_dp)
                results.append(annotated_dp)

    return results


if __name__ == "__main__":
    while len(annotated_bio) < args.total_data:
        bio_data = []
        with open(args.bio_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                bio_data.append(dp)

        batch_data = utils.split_list_into_batches(bio_data, args.batch_size)
        for batch in tqdm(batch_data):
            try:
                results = process_data_multi(batch)
            except Exception as e:
                print(f"Exception: {e}")
                count += 1
                generator = AtomicFactGenerator(
                    "api.key",
                    "demos",
                    gpt3_cache_file=f"./.cache/gpt_cache_{count}.pkl",
                )
                results = True
            if results:
                print("Wait for 1 min to prevent rate limit . . .")
                time.sleep(60)
        annotated_bio = utils.read_jsonl_file(args.facts_path)
