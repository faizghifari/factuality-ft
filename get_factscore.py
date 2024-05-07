import json
import utils
import numpy as np

from tqdm import tqdm

from factscore.factscorer import FactScorer


def remove_key_from_dicts(dict_list, key_to_remove):
    """
    Removes a specified key from all dictionaries in a list of dictionaries.
    """
    cleaned_dict_list = []
    for d in dict_list:
        cleaned_dict = {k: v for k, v in d.items() if k != key_to_remove}
        cleaned_dict_list.append(cleaned_dict)
    return cleaned_dict_list


fs = FactScorer()
annotated_fs = utils.read_jsonl_file("./data/Llama-1-7B-factscore.jsonl")
annotated_fs = remove_key_from_dicts(annotated_fs, "factscore")

if __name__ == "__main__":
    while len(annotated_fs) < 3550:
        facts_data = []
        with open("./data/Llama-1-7B-facts.jsonl") as f:
            for line in f:
                dp = json.loads(line)
                facts_data.append(dp)

        batch_data = utils.split_list_into_batches(facts_data, 10)
        for batch in tqdm(batch_data):
            dps, topics, generations, atomic_facts = [], [], [], []
            for dp in batch:
                if (
                    not {"topic": dp["topic"], "num_response": dp["num_response"]}
                    in annotated_fs
                ):
                    topics.append(dp["topic"])
                    generations.append(dp["output"])
                    atomic_facts.append(
                        [
                            atom["text"]
                            for sent in dp["annotations"]
                            for atom in sent["model-atomic-facts"]
                        ]
                    )
                    dps.append(dp)
            if len(dps) > 0:
                out = fs.get_score(
                    topics=topics,
                    generations=generations,
                    atomic_facts=atomic_facts,
                    verbose=True,
                )

                assert len(out["decisions"]) == len(dps)

                for idx, decision in enumerate(out["decisions"]):
                    s = dps[idx].copy()
                    count_atoms = 0
                    for annotation in s["annotations"]:
                        for atom in annotation["model-atomic-facts"]:
                            atom["is_supported"] = decision[count_atoms]
                    if decision:
                        s["factscore"] = np.mean([d["is_supported"] for d in decision])
                    else:
                        s["factscore"] = 0.0
                    utils.append_dict_to_jsonl("./data/Llama-1-7B-factscore.jsonl", s)
        annotated_fs = utils.read_jsonl_file("./data/Llama-1-7B-factscore.jsonl")
