import transformers
import torch
import numpy as np
from stoppingCriteria import TokenStoppingCriteria

transformers.logging.set_verbosity_error()

class LLM:
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
            

    def call_text_llm(self, new_prompt):
        # 1. Set up stopping criteria based on the occurrence of the sentinel token "###"
        sentinel_token = "###"
        number_examples = new_prompt.count(sentinel_token)
        sentinel_token_ids = self.tokenizer(sentinel_token, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        stopping_criteria_list = transformers.StoppingCriteriaList([
            TokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=0, counter=0, stop_counter=number_examples)
        ])

        # 2. Tokenize the input prompt
        inputs = self.tokenizer(new_prompt, return_tensors="pt").to("cuda")

        # 3. Generate output with specified generation parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            stopping_criteria=stopping_criteria_list,
            do_sample=False,                      # Greedy decoding
            return_dict_in_generate=True,
            output_scores=True,                    # Return logits at each step
            temperature=1.0,
            top_p=1.0,
        )

        # 4. Calculate transition scores (log-probabilities) for each generated token
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        # 5. Extract only the newly generated tokens (excluding input tokens)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        # 6. Initialize a list to store token details
        token_details = []

        # 7. Iterate over each generated token and its corresponding transition score
        for idx, (token_id, transition_score) in enumerate(zip(generated_tokens[0], transition_scores[0])):
            # Decode the current token
            token_str = self.tokenizer.decode(token_id)

            # Method 1: Calculate the probability of the generated token using the transition score
            generated_token_prob_from_transition_score = np.exp(transition_score.cpu().numpy())

            # Method 2: Calculate the full probability distribution and get the generated token's probability
            logits = outputs.scores[idx].detach().cpu().numpy().flatten()  # Flatten to ensure 1D array
            probs = np.exp(logits) / np.sum(np.exp(logits))

            # Access the generated token's probability directly from the distribution
            generated_token_prob_from_distribution = probs[token_id.item()]

            # Calculate Predictive Entropy (PE) over all possible tokens at this step
            token_wise_predictive_entropy = -np.sum(probs * np.log(probs + 1e-12))

            # Store the details of the current token
            token_details.append({
                "token": token_str,
                "token_wise_predictive_entropy": token_wise_predictive_entropy,
                "generated_token_prob_from_transition_score": generated_token_prob_from_transition_score,
                "generated_token_prob_from_distribution": generated_token_prob_from_distribution
            })

        # Return the decoded sequence and detailed information about each token
        decoded_output = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return decoded_output, token_details
