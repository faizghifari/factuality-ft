from fuzzywuzzy import fuzz
import torch
from torch import bfloat16
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed, BitsAndBytesConfig
from tqdm import tqdm

from collections import defaultdict
import re

import utils

def get_llm():
    set_seed(42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    model = AutoModelForCausalLM.from_pretrained(
        "huggyllama/llama-7b",
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16,
        # use_cache=True,
        quantization_config=bnb_config,
    )

    # model = model.cuda()
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, batch_size=2)
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    return pipe, tokenizer, model

class PromptDataset(Dataset):
    def __init__(self, response, questions, few_shot):
        self.response = response
        self.few_shot = few_shot
        self.questions = questions

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        ques = self.questions[index]
        prompt = self.few_shot
        prompt = prompt.replace("[NAME]", self.response["topic"])
        prompt = prompt.replace("[QUESTION]", f'{ques["atomic-question"]}')
        return prompt

class SampleAnswer():
    def __init__(
            self, 
            questions_path,
            save_path, 
            pipeline,
            temperature=1.0,
            num_samples=20,
            max_new_tokens=16):
        
        self.datalist = utils.read_atomic_facts(questions_path)
        self.save_path = save_path
        self.pipeline = pipeline
        self.temperature = temperature
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def get_system_prompt(self):
        with open('prompts/answer_prompt.txt', 'r') as file:
            self.few_shot = file.read()
    
    @staticmethod
    def extract_first_text_in_quotation_marks(text):
        match = re.search(r'"(.*?)"', text)
        if match:
            return match.group(1)
        else:
            return text
    
    @staticmethod
    def exact_match(str1, str2):
        return str1 == str2
    
    @staticmethod
    def bin_matching_strings(strings, threshold=70):
        groups = defaultdict(int)
        for string in strings:
            matched = False
            for group in groups:
                if fuzz.ratio(string, group) > threshold:
                    groups[string] += 1
                    matched = True
                    break
            if not matched:
                groups[string] +=1
            
        return groups
    
    def get_answer(self, prompts, batch_size):
        return self.pipeline(
            prompts, 
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=int(self.num_samples),
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            batch_size=batch_size,
            )
    
    def get_question_factscore(self, sequences, threshold=70):
        """Get factscore for each question"""
        answers = list(self.extract_first_text_in_quotation_marks(s['generated_text'].split("\n\n")[0]) for s in sequences)
        groups = self.bin_matching_strings(answers, threshold=threshold)
        factscore = max(groups.values()) / self.num_samples
        return factscore
    
    def get_response_factscore(self, res):
        """Get factscore for each response"""
        res_factscore = 0
        prompt_dataset = PromptDataset(res, res["atomic-questions"], self.few_shot)
        for i, seq in tqdm(enumerate(self.get_answer(prompt_dataset, batch_size=4)), desc="Getting factscore for each question"):
            factscore = self.get_question_factscore(seq, threshold=70)
            res["atomic-questions"][i]["factscore"] = factscore
            res_factscore += factscore / len(res["atomic-questions"])
        res["factscore"] = res_factscore

    def sampler(self):
        self.get_system_prompt()
        for res in tqdm(self.datalist, desc="response"):
            self.get_response_factscore(res)
        utils.write_jsonl(self.datalist, self.save_path)

if __name__ == '__main__':
    questions_path = "/root/factuality-ft/data/Llama-1-7B-facts-questions-2.jsonl"
    save_path = "data/Llama-1-7B-facts-questions-factscore.jsonl"
    pipe, tokenizer, model = get_llm()
    sample_answer = SampleAnswer(
        questions_path=questions_path,
        save_path=save_path,
        pipeline=pipe,
        temperature=0.7,
        num_samples=20,
        max_new_tokens=16,
    )
    sample_answer.sampler()
    
