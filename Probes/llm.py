import transformers
import torch
import numpy as np
from stoppingCriteria import TokenStoppingCriteria

transformers.logging.set_verbosity_error()

class LLM:
    """ this class interacts with the LLMs"""
    def __init__(self, model_path, model_name):

        if model_name == "llama":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True,
                local_files_only=True,
            )
        elif model_name == "phi":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, 
                                              torch_dtype="auto", 
                                                device_map="cuda",
                                                local_files_only=True,
                                                trust_remote_code=True)
            

    def call_text_llm(self, prompt):
        """ calls the LLM to get the generated text, tokens, pe, probs"""
        sentinel_token = "###"
        number_examples = prompt.count(sentinel_token)
        sentinel_token_ids = self.tokenizer(sentinel_token, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        stopping_criteria_list = transformers.StoppingCriteriaList([
            TokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=0, counter=0, stop_counter=number_examples)
        ])

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            stopping_criteria=stopping_criteria_list,
            do_sample=False,                     
            return_dict_in_generate=True,
            output_scores=True,                   
            temperature=1.0,
            top_p=1.0,
        )

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        token_details = []

        for idx, (token_id, transition_score) in enumerate(zip(generated_tokens[0], transition_scores[0])):
            token_str = self.tokenizer.decode(token_id)

            generated_token_prob_from_transition_score = np.exp(transition_score.cpu().numpy())

            logits = outputs.scores[idx].detach().cpu().numpy().flatten()  
            probs = np.exp(logits) / np.sum(np.exp(logits))

            generated_token_prob_from_distribution = probs[token_id.item()]

            token_wise_predictive_entropy = -np.sum(probs * np.log(probs + 1e-12))

            token_details.append({
                "token": token_str,
                "token_wise_predictive_entropy": token_wise_predictive_entropy,
                "generated_token_prob_from_transition_score": generated_token_prob_from_transition_score,
                "generated_token_prob_from_distribution": generated_token_prob_from_distribution
            })

        decoded_output = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return decoded_output, token_details
