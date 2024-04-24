import json
import torch
import random

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=True,
)

model.eval()


def read_file_lines(filename):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            # Stripping newline characters from the end of each line
            lines = [line.strip() for line in lines]
            return lines
    except FileNotFoundError:
        print("File not found.")
        return []


def write_list_to_file(file_path, text_list):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in text_list:
            file.write(str(item) + "\n")


def take_random_elements(input_list, num_elements):
    if num_elements > len(input_list):
        print("Error: Number of elements to take exceeds the length of the list.")
        return []
    else:
        random_elements = random.sample(input_list, num_elements)
        return random_elements


def write_jsonl(list_of_dicts, filename):
    try:
        # Open a file for writing
        with open(filename, "w") as f:
            # Iterate over the list of dictionaries
            for item in list_of_dicts:
                # Convert each dictionary to a JSON string and write it to the file
                json_string = json.dumps(item)
                f.write(json_string + "\n")
        print(f"Data written to {filename} successfully.")
    except Exception as e:
        print(f"Error occurred while writing to {filename}: {e}")


def append_dict_to_jsonl(file_path, dictionary):
    """
    Appends a dictionary as a new JSON object to an existing .jsonl file.
    If the file doesn't exist, it creates a new file.

    Args:
        dictionary (dict): The dictionary to be appended as a JSON object.
        file_path (str): The path to the .jsonl file.
    """
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            json_object = json.dumps(dictionary)
            file.write(json_object + "\n")
        print(f"Dictionary appended to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def generate_samples(
    prompt, bio_prompt, ent, fpath, num_samples=10, max_new_tokens=256
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_samples,
        use_cache=True,
    )
    for idx, output in enumerate(output_ids):
        sample = tokenizer.decode(output, skip_special_tokens=True)
        sample = sample.split(bio_prompt)[-1]
        sample = sample.split("\n\n")[1]
        # sample = sample.replace(prompt, "").strip()

        # if "Answer: " in sample:
        #     sample = sample.split("Answer: ")[1].strip()
        # if "Question: " in sample:
        #     sample = sample.split("Question: ")[0].strip()
        # if "Q: " in sample:
        #     sample = sample.split("Q: ")[0].strip()
        # samples.append(sample)
        append_dict_to_jsonl(
            fpath,
            {
                "input": bio_prompt,
                "output": sample,
                "topic": ent,
                "num_response": idx,
            },
        )

    # return samples


few_shot = """Write a short biography of George Washington.

George Washington (1732-1799) was the first President of the United States and one of the nation's founding fathers. He commanded the Continental Army during the American Revolutionary War, leading the colonies to victory over Britain. Born in Virginia, Washington worked as a surveyor before the war. After serving as president from 1789-1797, establishing key institutions, he retired to Mount Vernon. Widely hailed as the "Father of His Country," Washington played an indispensable role in the founding of the United States.

Write a short biography of Son Heung-Min.

Son Heung-Min (born 1992) is a South Korean professional soccer player and forward for Tottenham Hotspur and the national team. He began his career at Hamburg SV in Germany before transferring to Bayer Leverkusen. Tottenham signed Son in 2015 for £22 million. At Spurs, he has scored over 100 goals, leading the Premier League in scoring in 2022-23. Son has represented South Korea at two World Cups and won gold at the 2018 Asian Games. He is the highest-scoring South Korean in Premier League history.

Write a short biography of Narendra Modi.

Narendra Modi (born 1950) is the current Prime Minister of India and leader of the Bharatiya Janata Party. He joined the Hindu nationalist RSS youth wing as a young man. Modi served as Chief Minister of Gujarat from 2001-2014, facing criticism over the 2002 riots. Elected Prime Minister in 2014, he has promoted Hindu nationalist policies and economic development initiatives. Modi's supporters praise his leadership, while critics accuse him of eroding India's secular traditions and discriminating against minorities. He won re-election in 2019 on a nationalist platform.

"""
named_entities = read_file_lines("./data/unlabeled/prompt_entities.txt")
random_entities = take_random_elements(named_entities, 355)

write_list_to_file("./chosen_names.txt", random_entities)

model_results = []
for ent in tqdm(random_entities):
    bio_prompt = f"Write a short biography of {ent}."
    prompt = f"{few_shot}{bio_prompt}\n\n"
    generate_samples(prompt, bio_prompt, ent, "Llama-1-7B.jsonl")
    # for idx, sample in enumerate(samples):
    #     model_results.append(
    #         {
    #             "input": bio_prompt,
    #             "output": sample,
    #             "topic": ent,
    #             "num_response": idx,
    #         }
    #     )

# write_jsonl(model_results, "Llama-1-7B.jsonl")

# while True:
#     prompt = input("Enter your prompt: ")
#     samples = generate_samples(prompt)
#     for sample in samples:
#         print(sample)
#         print("-" * 50)
