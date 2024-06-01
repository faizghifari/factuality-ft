import torch
import utils
import argparse

from tqdm import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lora_dir", help="path to the lora directory", required=True, type=str
)
parser.add_argument(
    "--output_path",
    help="result output path",
    default=None,
    type=str,
)
parser.add_argument(
    "--names_path",
    help="path to name list to generate",
    default="./test_names.txt",
    type=str,
)
args = parser.parse_args()

config = PeftConfig.from_pretrained(args.lora_dir)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16,
    use_cache=True,
)
model = PeftModel.from_pretrained(model, args.lora_dir)
merged_model = model.merge_and_unload()
model.eval()


def generate_samples(
    prompt, bio_prompt, ent, fpath, num_samples=10, max_new_tokens=256
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        num_return_sequences=num_samples,
        use_cache=True,
    )
    results = []
    for idx, output in enumerate(output_ids):
        sample = tokenizer.decode(output, skip_special_tokens=True)
        sample = sample.split(bio_prompt)[-1]
        sample = sample.split("\n\n")[1]

        result = {
            "input": bio_prompt,
            "output": sample,
            "topic": ent,
            "num_response": idx,
        }
        utils.append_dict_to_jsonl(
            fpath,
            result,
        )
        results.append(result)

    return results


few_shot = """Write a short biography of George Washington.

George Washington (1732-1799) was the first President of the United States and one of the nation's founding fathers. He commanded the Continental Army during the American Revolutionary War, leading the colonies to victory over Britain. Born in Virginia, Washington worked as a surveyor before the war. After serving as president from 1789-1797, establishing key institutions, he retired to Mount Vernon. Widely hailed as the "Father of His Country," Washington played an indispensable role in the founding of the United States.

Write a short biography of Son Heung-Min.

Son Heung-Min (born 1992) is a South Korean professional soccer player and forward for Tottenham Hotspur and the national team. He began his career at Hamburg SV in Germany before transferring to Bayer Leverkusen. Tottenham signed Son in 2015 for £22 million. At Spurs, he has scored over 100 goals, leading the Premier League in scoring in 2022-23. Son has represented South Korea at two World Cups and won gold at the 2018 Asian Games. He is the highest-scoring South Korean in Premier League history.

Write a short biography of Narendra Modi.

Narendra Modi (born 1950) is the current Prime Minister of India and leader of the Bharatiya Janata Party. He joined the Hindu nationalist RSS youth wing as a young man. Modi served as Chief Minister of Gujarat from 2001-2014, facing criticism over the 2002 riots. Elected Prime Minister in 2014, he has promoted Hindu nationalist policies and economic development initiatives. Modi's supporters praise his leadership, while critics accuse him of eroding India's secular traditions and discriminating against minorities. He won re-election in 2019 on a nationalist platform.

"""


if __name__ == "__main__":
    named_entities = utils.read_file_lines(args.names_path)
    generated_entities = []
    generated_prompts = utils.read_jsonl_file(args.output_path)
    if generated_prompts:
        for g in generated_prompts:
            generated_entities.append(g["topic"])
        generated_entities = list(set(generated_entities))

    for ent in tqdm(named_entities):
        if ent not in generated_entities:
            bio_prompt = f"Write a short biography of {ent}."
            prompt = f"{few_shot}{bio_prompt}\n\n"
            try:
                results = generate_samples(prompt, bio_prompt, ent, args.output_path)
            except IndexError:
                results = generate_samples(prompt, bio_prompt, ent, args.output_path)
