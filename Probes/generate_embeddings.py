from llm import LLM
import pandas as pd
import torch
import os
import numpy as np


class Embeddings:
    """ this class generates embeddings for the datasets"""
    def __init__(self, llm, remove_period):
        self.llm = llm
        self.remove_period = remove_period

    def load_data(self, input_file, embeddings_name):
        """ loads the dataset"""
        print(f"Loading dataset from {input_file}")
        try:
            print(f"Loading {input_file}")
            df = pd.read_pickle(input_file)
            df.reset_index(drop=True, inplace=True)
        except FileNotFoundError:
            print(f"File {input_file} not found")

        df[str(embeddings_name)] = pd.Series(dtype='object')
        return df

    def process_row(self, prompt, layer):
        """ processes the row of each mini-fact or sentence and generates the embeddings"""
        if self.remove_period:
            prompt = prompt.rstrip(". ")
        inputs = self.llm.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.llm.model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
        embeddings = {}
        last_hidden_state = outputs.hidden_states[0][layer][0][-1]
        last_hidden_state = last_hidden_state.to(torch.float32)
        embeddings[layer] = [last_hidden_state.cpu().numpy().tolist()]
        return embeddings

    def save_data(self, df, output_file):
        """ saves the dataset with the embeddings"""
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        df.to_pickle(output_file)


if __name__ == "__main__":

    # Set the model path
    model_path = ""
    
    llm = LLM(model_path)
    datasets = ["hover", "fever"]
    layers = [-1, -8, -24, 1]
    remove_period = True
    model_name = "llama"

    for dataset in datasets:
        if not os.path.exists(f"recreate_datasets_{dataset}"):
            raise FileNotFoundError(f"Directory datasets_{dataset} not found: Please run generate_mini_facts_and_sentences.py first")
        
        input_file_mini_facts = f"recreate_datasets_{dataset}_{model_name}/mini_fact_and_sentence_with_bart.pkl"
        input_file_sentences = f"recreate_datasets_{dataset}_{model_name}/sentence_with_bart.pkl"
        for layer in layers:
            if model_name == "phi":
                output_file_mini_facts = f"processed_datasets_{model_name}_with_bart/mini_fact_{dataset}_layer{layer}_test_unbalanced.pkl"
                output_file_sentences = f"processed_datasets_{model_name}_with_bart/sentence_{dataset}_layer{layer}_test_unbalanced.pkl"
            elif model_name == "llama":
                output_file_mini_facts = f"processed_datasets_with_bart_{dataset}_layer{layer}/mini_fact_{dataset}.pkl"
                output_file_sentences = f"processed_datasets_with_bart_{dataset}_layer{layer}/sentence_{dataset}.pkl"
            print(f"Processing layer {layer} for dataset {dataset}")
            for input_file, output_file in [(input_file_mini_facts, output_file_mini_facts), (input_file_sentences, output_file_sentences)]:
                print(f"Processing {input_file}")
                if "mini_fact" in input_file:
                    embeddings_name = f"embeddings{layer}_mini_fact"
                else:
                    embeddings_name = f"embeddings{layer}_sentence"

                embeddings = Embeddings(llm, remove_period)
                df = embeddings.load_data(input_file=input_file, embeddings_name=embeddings_name)
                df[str(embeddings_name)] = None
                df[str(embeddings_name)] = df[str(embeddings_name)].astype(object)

                for index, row in df.iterrows():
                    if index % 100 == 0:
                        print(f"Processing row {index}")
                    if "mini_fact" in input_file:
                        prompt = row['output_mini_fact']
                    else:
                        prompt = row['output_sentence']
                    embeddings_list = embeddings.process_row(prompt, layer)
                    df.at[index, str(embeddings_name)] = embeddings_list[layer][0]

                print(np.array(df[str(embeddings_name)].tolist()).shape)
                embeddings.save_data(df=df, output_file=output_file)




