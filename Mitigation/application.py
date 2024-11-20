import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
device = torch.device(f"cuda:7")


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging
from peft import AutoPeftModelForCausalLM


class LlamaModelLoader:
    """class to load and manage base and fine-tuned LLaMA model."""
    def __init__(self, model_path, finetuned_model_dir):
        self.device = torch.device("cuda:7")
        self.model_path = model_path
        self.finetuned_model_dir = finetuned_model_dir
        self.tokenizer = None
        self.model = None
        self.tokenizer_finetuned = None
        self.model_finetuned = None

        # Configure logging
        logging.set_verbosity_error()
        
        # Configure 4-bit quantization
        self.bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    def load_base_model(self):
        """Load the base model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
            local_files_only=True,
        )

    def load_finetuned_model(self):
        """Load the fine-tuned model and tokenizer."""
        self.tokenizer_finetuned = AutoTokenizer.from_pretrained(self.finetuned_model_dir)
        self.model_finetuned = AutoPeftModelForCausalLM.from_pretrained(
            self.finetuned_model_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True
        )

    def get_base_model(self):
        """Return the base model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            self.load_base_model()
        return self.model, self.tokenizer

    def get_finetuned_model(self):
        """Return the fine-tuned model and tokenizer."""
        if self.model_finetuned is None or self.tokenizer_finetuned is None:
            self.load_finetuned_model()
        return self.model_finetuned, self.tokenizer_finetuned
    

class ProbeNN(nn.Module):
    """ Neural network model for probing with three hidden layers and an output layer."""
    def __init__(self, input_dim):
        super(ProbeNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x
    

class TokenStoppingCriteria(transformers.StoppingCriteria):
    """Stopping criteria based on the occurrence of a sentinel token."""
    def __init__(self, sentinel_token_ids: torch.Tensor, 
                 starting_idx: int, counter: int, stop_counter: int):
        super().__init__()
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx
        self.counter = counter
        self.stop_counter = stop_counter

    def __call__(self, input_ids: torch.Tensor, _scores: torch.Tensor) -> bool:
        self.counter = 0
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            if trimmed_sample.shape[0] < 1:
                continue

            for token_id in trimmed_sample:
                if token_id in self.sentinel_token_ids:
                    if self.counter == self.stop_counter:
                        return True
                    else:
                        self.counter += 1
        return False
    

def call_text_llm(new_prompt, model, tokenizer):
    """Calling text LLm to get token details such as pe, probs, tokens."""
    sentinel_token = "###"
    number_examples = new_prompt.count(sentinel_token)
    sentinel_token_ids = tokenizer(sentinel_token, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    stopping_criteria_list = transformers.StoppingCriteriaList([
        TokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=0, counter=0, stop_counter=number_examples)
    ])

    inputs = tokenizer(new_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        stopping_criteria=stopping_criteria_list,
        do_sample=False,                     
        return_dict_in_generate=True,
        output_scores=True,                   
        temperature=1.0,
        top_p=1.0,
    )
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    token_details = []

    for idx, (token_id, transition_score) in enumerate(zip(generated_tokens[0], transition_scores[0])):
        token_str = tokenizer.decode(token_id)
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
    decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return decoded_output, token_details


def call_chat_llm(messages, model, tokenizer):
    """Calling chat LLm to get the response."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def get_embeddings(prompt, layer):
    """Get embeddings for the prompt."""
    prompt = prompt.rstrip(". ")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
    embeddings = {}
    last_hidden_state = outputs.hidden_states[0][layer][0][-1]
    last_hidden_state = last_hidden_state.to(torch.float32)
    embeddings[layer] = [last_hidden_state.cpu().numpy().tolist()]
    return embeddings


def correct_mini_facts_prompt(mini_facts_with_labels_false, ground_truth_source):
    """Instruction Prompt to correct the mini facts."""
    messages = messages = [
        {"role": "system", "content": "Please correct the statements marked as FALSE based only on the provided SOURCE. Do not use information from any other source. If a FALSE statement cannot be corrected because no information from the SOURCE contradicts the statement, output REMOVE. If only a part of a statement is incorrect, correct the incorrect part while keeping the rest of the statement intact. Ensure that every given wrong fact is either corrected (Corrected) or removed (REMOVE) as provided in the examples below."},
        {"role": "user", "content": f"WRONG STATEMENTS: {mini_facts_with_labels_false} \n\n SOURCE: {ground_truth_source}"},
    ]
    return messages

def get_prompt_mini_facts(gen):
    """Prompt to generate mini facts."""
    mini_facts_instruction = f"""Your task is to breakdown claims/sentences into independant statements. 
You must NEVER correct or comment the original claims/sentences even if something of the original claims/sentences is incorrect.
Do NEVER generate statements that are not in the original claims/sentences. Every statement must start with an entity that specifies the topic (e.g. **The Fox Broadcasting Company** and not **The company**)."""
            
    mini_facts_samples = ["The Hunger Games is a 2012 American science fiction film directed by John Peter and based on the novel of the same name by Suzanne Collins. Matt Lucena is an American former professional tennis player.",
    """Owen Wilson starred in the film "The Karate Kid" (2010) alongside martial arts expert Tom Wu. Owen Wilson voiced Lightning McQueen in the "Cars" franchise, not "The Royal Tenenbaums" franchise.""",
    "Feels So Good is a song by the American R&B group Tony! Toni! TonÃ. The song was written by the group's lead vocalist Raphael Saadiq and producer Tony! Toni! TonÃ's lead vocalist Dwayne Wimberly."]
            
            
    mini_facts_sample_outputs = ["""- **The Hunger Games** is a 2012 American science fiction film.
    - **The Hunger Games** was directed by John Peter.
    - **The Hunger Games** is based on a novel of the same name by Suzanne Collins.
    - **Matt Lucena** is an American former professional tennis player.""",
    """- **Owen Wilson** starred in the film The Karate Kid (2010) alongside martial arts expert Tom Wu.
    - **Owen Wilson** voiced Lightning McQueen in the Cars franchise.
    - **Owen Wilson** did not voice Lightning McQueen in the The Royal Tenenbaums franchise.""",
    """- **Feels So Good** is a song by the American R&B group Tony! Toni! TonÃ.
    - **Feels So Good** was written by the group's lead vocalist Raphael Saadiq and producer Tony! Toni! TonÃ's lead vocalist Dwayne Wimberly."""]
        

    messages = [{"role": "system", "content" : f"{mini_facts_instruction}"},
        {"role": "user", "content": f"{mini_facts_samples[0]}"},
        {"role": "assistant", "content": f"{mini_facts_sample_outputs[0]}"},
        {"role": "user", "content":  f"{mini_facts_samples[1]}"},
        {"role": "assistant", "content":  f"{mini_facts_sample_outputs[1]}"},
        {"role": "user", "content":  f"{mini_facts_samples[2]}"},
        {"role": "assistant", "content":  f"{mini_facts_sample_outputs[2]}"},
        {"role": "user", "content":  f"{gen}"}
        ]
    return messages

def generate_sentence_by_sentence(row, prompt, source, model, tokenizer, model_finetuned, tokenizer_finetuned, probe, layer):
    """Generates evidence sentence-by-sentence and performs correction after each sentence."""
    generation = ""
    new_prompt = f"{prompt} CLAIM: {row['claim']}"
    max_number_of_sentences = 3
    number_of_sentences = 0

    while number_of_sentences < max_number_of_sentences:
        response, _ = call_text_llm(new_prompt, model, tokenizer)
        response_clean = response.replace("EVIDENCE: ", "").replace("EVIDENCE_END.", "").replace("###", "")
        print("Sentence: ", response_clean)
        messages = get_prompt_mini_facts(response_clean)
        mini_facts = call_chat_llm(messages, model, tokenizer).replace("**", "").split("\n")
        mini_facts_list = [mini_fact.replace("- ", "").strip() for mini_fact in mini_facts]
        test_pred_probs = []
        for mini_fact in mini_facts_list:
            embedding = get_embeddings(mini_fact, layer)
            embedding = np.array(embedding[layer][0])
            with torch.no_grad():
                test_pred_probs.append(probe(torch.tensor(embedding, dtype=torch.float32)).cpu().numpy()[0])

        false_pred_mini_facts = ["- " + mf for mf, prob in zip(mini_facts_list, test_pred_probs) if prob < 0.5]
        print("Prediction False: ", false_pred_mini_facts)
        true_pred_mini_facts = ["- " + mf for mf, prob in zip(mini_facts_list, test_pred_probs) if prob >= 0.5]
        corrected_mini_facts = []
        if false_pred_mini_facts:
            messages = correct_mini_facts_prompt("\n".join(false_pred_mini_facts), source)
            output = call_chat_llm(messages, model_finetuned, tokenizer_finetuned)
            print("Corrected: ", output)
            corrected_mini_facts = [fact.replace("(Corrected)", "").strip() for fact in output.split("\n") if "Corrected" in fact]

        final_mini_facts = true_pred_mini_facts + corrected_mini_facts
        final_mini_facts_sentence = " ".join([mf.replace("\n", "").replace("- ", "").replace(".", ",").strip() for mf in final_mini_facts])
        final_mini_facts_sentence = final_mini_facts_sentence[:-1] + ' .' + "###"
        print("Corrected Sentence: ", final_mini_facts_sentence)

        generation += " " + final_mini_facts_sentence
        if "EVIDENCE_END" in response:
            print("Generation: ", generation)
            break
        new_prompt += " " + final_mini_facts_sentence
        number_of_sentences += 1

    return generation.strip()

def generate_whole_evidence(row, prompt, source, model, tokenizer, model_finetuned, tokenizer_finetuned, probe, layer):
    """Generates the evidence based on the few-shot prompt and performs correction on the whole evidence."""
    new_prompt = f"{prompt} CLAIM: {row['claim']}"
    response, _ = call_text_llm(new_prompt, model, tokenizer)
    response_clean = response.replace("EVIDENCE: ", "").replace("EVIDENCE_END.", "").replace("###", "")
    print("Generated Evidence: ", response_clean)
    messages = get_prompt_mini_facts(response_clean)
    mini_facts = call_chat_llm(messages, model, tokenizer).replace("**", "").split("\n")
    mini_facts_list = [mini_fact.replace("- ", "").strip() for mini_fact in mini_facts]
    test_pred_probs = []
    for mini_fact in mini_facts_list:
        embedding = get_embeddings(mini_fact, layer)
        embedding = np.array(embedding[layer][0])
        with torch.no_grad():
            test_pred_probs.append(probe(torch.tensor(embedding, dtype=torch.float32)).cpu().numpy()[0])

    false_pred_mini_facts = ["- " + mf for mf, prob in zip(mini_facts_list, test_pred_probs) if prob < 0.5]
    print("Prediction False: ", false_pred_mini_facts)
    true_pred_mini_facts = ["- " + mf for mf, prob in zip(mini_facts_list, test_pred_probs) if prob >= 0.5]
    corrected_mini_facts = []
    if false_pred_mini_facts:
        messages = correct_mini_facts_prompt("\n".join(false_pred_mini_facts), source)
        output = call_chat_llm(messages, model_finetuned, tokenizer_finetuned)
        print("Corrected: ", output)
        corrected_mini_facts = [fact.replace("(Corrected)", "").strip() for fact in output.split("\n") if "Corrected" in fact]
    final_mini_facts = true_pred_mini_facts + corrected_mini_facts
    final_evidence = " ".join([mf.replace("\n", "").replace("- ", "").replace(".", ",").strip() for mf in final_mini_facts]) 
    final_evidence = final_evidence[:-1] + " .###"
    print("Final Generation: ", final_evidence)
    return final_evidence

def generate_baseline_evidence(row, prompt, model, tokenizer):
    """Generates the evidence based on the few-shot prompt without any corrections."""
    new_prompt = f"{prompt} CLAIM: {row['claim']}"
    response, _ = call_text_llm(new_prompt, model, tokenizer)
    response_clean = response.replace("EVIDENCE: ", "").replace("EVIDENCE_END.", "").replace("###", "")
    print(response_clean)
    return response_clean.strip()

if __name__ == "__main__":
    model_loader = LlamaModelLoader(
        # insert the path to the base model
        model_path="/home/wombat_share/llms/llama/Meta-Llama-3-8B-Instruct",
        # insert the path to the fine-tuned model
        finetuned_model_dir="llama_finetuned_model"
    )
    
    model, tokenizer = model_loader.get_base_model()
    model_finetuned, tokenizer_finetuned = model_loader.get_finetuned_model()

    prompt_sentence_by_sentence = """CLAIM: Alexander Glazunov served as director of the Saint Petersburg Conservatory when he started composing Symphony No 9 in D minor. EVIDENCE: Glazunov's Symphony No. 9 in D minor was begun in 1910, but was still unfinished by the time of Glazunov's death in 1936 .### Alexander Glazunov served as director of the Saint Petersburg Conservatory between 1905 and 1928 and was instrumental in the reorganization of the Saint Petersburg Conservatory into the Petrograd Conservatory, then the Leningrad Conservatory, following the Bolshevik Revolution. EVIDENCE_END .### CLAIM: In regards to Value premium the expert who argued that no value premium exists, did not found The Vanguard Group. EVIDENCE: Other experts, such as John C. Bogle, have argued that no value premium exists, claiming that Fama and French's research is period dependent .### John C. Bogle is the founder and retired chief executive of The Vanguard Group. EVIDENCE_END .### CLAIM: Besides Bamburgh Castle, Lindisfarne Castle is located on the northeast coast of England. EVIDENCE: Fenwick, Northumberland's close proximity to Lindisfarne Castle, Bamburgh Castle and Chillingham Castle means Fenwick, Northumberland is an ideal base from which to explore the rich history of Northumberland and the Farne Islands .### Bamburgh Castle is a castle on the northeast coast of England, by the village of Bamburgh in Northumberland. EVIDENCE_END .### CLAIM: After succeeding minister Arnold Burns, Edwin Meese remained in office from 1985-1988. EVIDENCE: Meese resigned from office later in July 1988, shortly after Arnold Burns and Weld appeared before Congress .### Edwin Meese is an American attorney, law professor, author and member of the Republican Party who served in official capacities within the Ronald Reagan Gubernatorial Administration (1967–1974), the Ronald Reagan Presidential Transition Team (1980) and the Ronald Reagan White House (1981–1985), eventually rising to hold the position of the 75th Attorney General of the United States (1985–1988). EVIDENCE_END .###"""
    prompt_whole_evidence = """CLAIM: Alexander Glazunov served as director of the Saint Petersburg Conservatory when he started composing Symphony No 9 in D minor. EVIDENCE: Glazunov's Symphony No. 9 in D minor was begun in 1910, but was still unfinished by the time of Glazunov's death in 1936.Alexander Glazunov served as director of the Saint Petersburg Conservatory between 1905 and 1928 and was instrumental in the reorganization of the Saint Petersburg Conservatory into the Petrograd Conservatory, then the Leningrad Conservatory, following the Bolshevik Revolution. EVIDENCE_END .### CLAIM: In regards to Value premium the expert who argued that no value premium exists, did not found The Vanguard Group. EVIDENCE: Other experts, such as John C. Bogle, have argued that no value premium exists, claiming that Fama and French's research is period dependent.John C. Bogle is the founder and retired chief executive of The Vanguard Group. EVIDENCE_END .### CLAIM: Besides Bamburgh Castle, Lindisfarne Castle is located on the northeast coast of England. EVIDENCE: Fenwick, Northumberland's close proximity to Lindisfarne Castle, Bamburgh Castle and Chillingham Castle means Fenwick, Northumberland is an ideal base from which to explore the rich history of Northumberland and the Farne Islands. Bamburgh Castle is a castle on the northeast coast of England, by the village of Bamburgh in Northumberland. EVIDENCE_END .### CLAIM: After succeeding minister Arnold Burns, Edwin Meese remained in office from 1985-1988. EVIDENCE: Meese resigned from office later in July 1988, shortly after Arnold Burns and Weld appeared before Congress. Edwin Meese is an American attorney, law professor, author and member of the Republican Party who served in official capacities within the Ronald Reagan Gubernatorial Administration (1967–1974), the Ronald Reagan Presidential Transition Team (1980) and the Ronald Reagan White House (1981–1985), eventually rising to hold the position of the 75th Attorney General of the United States (1985–1988). EVIDENCE_END .###"""
    
    # Load the test data
    df_test = pd.read_pickle("application/application_data.pkl").reset_index(drop=True)
    print(len(df_test))
    layer = -16
    probe = ProbeNN(4096)
    # Load the probe model
    probe.load_state_dict(torch.load(f"../Probes/processed_datasets_llama_layer-16/mini_fact_embeddings{layer}_hover.pth"))
    probe.eval()

    # select the application scenario (whole_evidence, sentence_by_sentence, baseline)
    application = "whole_evidence"

    df_results = pd.DataFrame()

    for index, row in df_test.iterrows():
        print(f"Processing claim {index}")
        source = row['ground_truth_source']
        docs = row['docs']
        if application == "baseline":
            generation_result = generate_baseline_evidence(
                row, prompt_whole_evidence, model, tokenizer
            )
        
        elif application == "sentence_by_sentence":
            generation_result = generate_sentence_by_sentence(
                row, prompt_sentence_by_sentence, source, model, tokenizer, model_finetuned, tokenizer_finetuned, probe, layer
            )
        
        elif application == "whole_evidence":
            generation_result = generate_whole_evidence(
                row, prompt_whole_evidence, source, model, tokenizer, model_finetuned, tokenizer_finetuned, probe, layer
            )

        df_results = pd.concat([
            df_results,
            pd.DataFrame({
                "gen_evidence": [generation_result],
                "Source": [source],
                "docs": [docs]
            })
        ], ignore_index=True)
        df_results.to_pickle(f"application/{application}.pkl")

    print("All claims processed.")