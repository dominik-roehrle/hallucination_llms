import os
import torch
from openai import OpenAI
import pandas as pd
import transformers
from peft import AutoPeftModelForCausalLM

transformers.logging.set_verbosity_error()

with open("api.key", "r") as file:
    api_key = file.read().strip() 

os.environ["OPENAI_API_KEY"] = api_key
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LLMInteraction:
    """ this class interacts with the LLMs"""
    def __init__(self, llama_model_path=None, fine_tuned_version=False, few_shot=False, use_cache=False):
        self.few_shot = few_shot
        if llama_model_path:
            if fine_tuned_version:
                self.llama_model = AutoPeftModelForCausalLM.from_pretrained(
                    llama_model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    load_in_4bit=True
                )
                self.llama_tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
            else:
                self.llama_tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path, local_files_only=True)
                self.llama_model = transformers.AutoModelForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    load_in_4bit=True,
                    local_files_only=True,
                    use_cache=use_cache
                )

            

        self.correction_instruction = """Please correct the statements marked as FALSE based only on the provided SOURCE. 
Do not use information from any other source. If a FALSE statement cannot be corrected because no information from the SOURCE contradicts the statement, output REMOVE. 
If only a part of a statement is incorrect, correct the incorrect part while keeping the rest of the statement intact. Ensure that every given wrong fact is either corrected (Corrected) or removed (REMOVE) as provided in the examples below."""


        mini_facts_with_labels_false_sample1 = ["- 'Angry Dad: The Movie' is the 22nd episode of the 22nd season of The Simpsons.",
            "- The Simpsons began airing on September 30, 2018.", 
            "- Todd Solondz is an American film director, screenwriter, and producer."]
        mini_facts_with_labels_corrected_sample1 = ["- Angry Dad: The Movie is the fourteenth episode of The Simpsons twenty-second season. (Corrected)", 
            "- The Simpsons began airing on September 30, 2018. (REMOVE)", 
            "- Todd Solondz is an American independent film screenwriter and director. (Corrected)"]
        self.ground_truth_source_sample1 = "Angry Dad: The Movie is the fourteenth episode of The Simpsons twenty-second season. It originally aired on the Fox network in the United States on February 20, 2011. Todd Solondz (born October 15, 1959) is an American independent film screenwriter and director known for his style of dark, thought-provoking, socially conscious satire."

        mini_facts_with_labels_false_sample2 = ["- Vice Principals premiered in 2016."]
        mini_facts_with_labels_corrected_sample2 = ["- Vice Principals premiered in 2016. (REMOVE)"]
        self.ground_truth_source_sample2 = "American Dad! is an American adult animated sitcom created by Seth MacFarlane, Mike Barker, and Matt Weitzman for the Fox Broadcasting Company."

        mini_facts_with_labels_false_sample3 = [
            "- Buddymoon is a 2014 American comedy film.",
            "- Buddymoon was written by Simmons, Peter John, Flula Borg and David Giuntoli.",
            "- Buddymoon has won several awards.",
            "- Dirk Nowitzki is a German basketball player and producer.",
            "- Buddymoon stars Alex Takeuchi, Robert Kendall, and Addison Timbers."]
        mini_facts_with_labels_corrected_sample3 = ["- Buddymoon is a 2016 American independent comedy film. (Corrected)",
            "- Buddymoon was written by Simmons, Flula Borg, and David Giuntoli. (Corrected)",
            "- Buddymoon has won several awards. (REMOVE)",
            "- Dirk Nowitzki is a German professional basketball player. (Corrected)",
            "- Buddymoon stars Borg and Giuntoli. (Corrected)",]
        self.ground_truth_source_sample3 = 'Dirk Werner Nowitzki (] ) (born June 19, 1978) is a German professional basketball player for the Dallas Mavericks of the National Basketball Association (NBA). Buddymoon (previously known as Honey Buddies) is a 2016 American independent comedy film directed by Alex Simmons; written by Simmons, Flula Borg, and David Giuntoli; and starring Borg and Giuntoli.'

        self.mini_facts_with_labels_false1 = "\n".join(mini_facts_with_labels_false_sample1)
        self.mini_facts_with_labels_corrected1 = "\n".join(mini_facts_with_labels_corrected_sample1)

        self.mini_facts_with_labels_false2 = "\n".join(mini_facts_with_labels_false_sample2)
        self.mini_facts_with_labels_corrected2 = "\n".join(mini_facts_with_labels_corrected_sample2)

        self.mini_facts_with_labels_false3 = "\n".join(mini_facts_with_labels_false_sample3)
        self.mini_facts_with_labels_corrected3 = "\n".join(mini_facts_with_labels_corrected_sample3)


    def call_openai_llm(self, messages, response_format):
        """ calls the OpenAI LLM"""
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": response_format
            }
        )
        return response.choices[0].message.content
    
    def call_llama_llm(self, messages):
        """ calls the Llama LLM"""
        input_ids = self.llama_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.llama_model.device)

        terminators = [
            self.llama_tokenizer.eos_token_id,
            self.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llama_model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.llama_tokenizer.decode(response, skip_special_tokens=True)
    

    def correct_mini_facts_prompt(self, mini_facts_with_labels_false, ground_truth_source, correction_evidence=None, is_training=False):
        """ generates the correction prompt for the LLMs"""
        if self.few_shot:
            messages = [
                {"role": "system", "content": "Please correct the statements marked as FALSE based only on the provided SOURCE. Do not use information from any other source. If a FALSE statement cannot be corrected because no information from the SOURCE contradicts the statement, output REMOVE. If only a part of a statement is incorrect, correct the incorrect part while keeping the rest of the statement intact. Ensure that every given wrong fact is either corrected (Corrected) or removed (REMOVE) as provided in the examples below."},
                {"role": "user", "content": f"WRONG STATEMENTS: {self.mini_facts_with_labels_false1} \n\n SOURCE:{self.ground_truth_source_sample1}"},
                {"role": "assistant", "content": f"{self.mini_facts_with_labels_corrected1}"},
                {"role": "user", "content": f"WRONG STATEMENTS: {self.mini_facts_with_labels_false2} \n\n SOURCE:{self.ground_truth_source_sample2}"},
                {"role": "assistant", "content": f"{self.mini_facts_with_labels_corrected2}"},
                {"role": "user", "content": f"WRONG STATEMENTS: {self.mini_facts_with_labels_false3} \n\n SOURCE:{self.ground_truth_source_sample3}"},
                {"role": "assistant", "content": f"{self.mini_facts_with_labels_corrected3}"},
                {"role": "user", "content": f"WRONG STATEMENTS: {mini_facts_with_labels_false} \n\n SOURCE: {ground_truth_source}"},
            ]
        else:
            messages = messages = [
                {"role": "system", "content": "Please correct the statements marked as FALSE based only on the provided SOURCE. Do not use information from any other source. If a FALSE statement cannot be corrected because no information from the SOURCE contradicts the statement, output REMOVE. If only a part of a statement is incorrect, correct the incorrect part while keeping the rest of the statement intact. Ensure that every given wrong fact is either corrected (Corrected) or removed (REMOVE) as provided in the examples below."},
                {"role": "user", "content": f"WRONG STATEMENTS: {mini_facts_with_labels_false} \n\n SOURCE: {ground_truth_source}"},
            ]

        if is_training:
            messages.append({"role": "assistant", "content": f"{correction_evidence}"})
        return messages
    

