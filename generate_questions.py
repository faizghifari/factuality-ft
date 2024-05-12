import openai
from tqdm import tqdm

import time

import utils

MODEL = "gpt-3.5-turbo"
with open('api.key', "r") as f:
    API_KEY = f.read().strip()

class GenerateQuestions():
    def __init__(
            self,
            save_path,
            responses,
            temp=0.7,
            max_tokens=1024,
            seed=0):
        self.save_path = save_path
        self.responses = responses
        self.temp = temp
        self.max_tokens = max_tokens
        self.seed = seed

    def get_system_prompt(self):
        with open('prompt/question_prompt.txt', 'r') as file:
            self.system_prompt = file.read()

    def get_question(
            self,
            system_prompt: str,
            atomic_fact: str,
            reponse: dict,
            temp=0.7, 
            max_tokens=1024, 
            seed=0):
        ques = {
            'atomic_fact': atomic_fact
        }
        prompt = system_prompt
        for k, v in zip(['[NAME]', '[STATEMENT]'], [reponse['topic'], atomic_fact]):
            prompt = prompt.replace(k, v)
        for _ in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL, 
                    messages = [
                        {"role": "user", "content" : prompt},
                    ],
                    seed = seed,
                    temperature =  temp,
                    max_tokens = max_tokens
                )
                ques['atomic-question'] = response['choices'][0]['message']['content']
                return ques
            except Exception as e:
                print('[ERROR]', e)
                ques['atomic-question'] = '#ERROR#'
                time.sleep(1)
        # ques['atomic-question'] = 'gen_ques'
        
        return ques
    
    def get_all_questions(self, responses):
        for res in tqdm(responses, desc='responses'):
            ques_list = []
            facts = res['atomic-facts']
            for fact in facts:
                atomic_fact = fact['text']
                ques = self.get_question(
                    system_prompt=self.system_prompt, 
                    atomic_fact=atomic_fact, 
                    reponse=res, 
                    temp=self.temp, 
                    max_tokens=self.max_tokens, 
                    seed=self.seed)
                
                ques_list.append(ques)
            res['atomic-questions'] = ques_list
            res.pop('atomic-facts')

        return responses

    def generator(self):
        self.get_system_prompt()
        gen_reponses = self.get_all_questions(self.responses)
        utils.write_jsonl(gen_reponses, self.save_path)


if __name__ == '__main__':
    openai.api_key = API_KEY
    save_path = 'data/Llama-1-7B-facts-questions-2.jsonl'
    responses = utils.read_atomic_facts("/root/factuality-ft/data/Llama-1-7B-facts.jsonl")
    generate_questions = GenerateQuestions(save_path, responses)
    generate_questions.generator()