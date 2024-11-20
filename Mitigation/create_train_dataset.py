import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import re
import os
import ast


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NLPProcessor:
    """ Class to process the NLP tasks using BART model."""
    def __init__(self, bart_model_path):
        print("Loading BART model...")
        self.bart_model = AutoModelForSequenceClassification.from_pretrained(bart_model_path, local_files_only=True)
        self.bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_path, local_files_only=True)
        self.bart_model.to(device)
    
    def split_source_to_fit_with_hypothesis(self, source, hypothesis, bart_tokenizer, max_length=1024):
        """Splits the source into chunks so that each chunk, when combined with the hypothesis, fits within the token limit."""
        original_max_length = bart_tokenizer.model_max_length
        bart_tokenizer.model_max_length = int(1e12)  
        hypothesis_tokens = bart_tokenizer.encode(hypothesis, add_special_tokens=False)
        hypothesis_length = len(hypothesis_tokens)
        num_special_tokens = bart_tokenizer.num_special_tokens_to_add(pair=True)
        max_source_length = max_length - hypothesis_length - num_special_tokens
        if max_source_length <= 0:
            bart_tokenizer.model_max_length = original_max_length
            raise ValueError("The hypothesis is too long to fit within the max_length limit.")
        source_tokens = bart_tokenizer.encode(source, add_special_tokens=False)
        bart_tokenizer.model_max_length = original_max_length
        token_chunks = [source_tokens[i:i+max_source_length] for i in range(0, len(source_tokens), max_source_length)]
        text_chunks = [bart_tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]
        return text_chunks
    
    def call_bart_model(self, source, statement):
        """Calls the BART model to predict the label."""
        source_chunks = self.split_source_to_fit_with_hypothesis(source, statement, self.bart_tokenizer, max_length=1024)
        entailment_probs = []
        pred_labels = []
        for idx, chunk in enumerate(source_chunks):
            inputs = self.bart_tokenizer(
                chunk,
                statement,
                return_tensors='pt',
                truncation=True,
                max_length=1024,
                add_special_tokens=True
            )
            input_length = inputs['input_ids'].shape[1]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bart_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                dominating_class = probs.argmax(dim=1).item()

            class_names = ["Contradiction", "Neutral", "Entailment"]
            prob_entailment = probs[:, 2].item()
            entailment_probs.append(prob_entailment)
            pred_labels.append(class_names[dominating_class])

        filtered_labels = [label for label in pred_labels if label != "Neutral"]
        if filtered_labels:
            final_label = max(set(filtered_labels), key=pred_labels.count)
        else:
            final_label = max(set(pred_labels), key=pred_labels.count)
        return final_label
    

class DatasetBuilder:
    """Class to build the training dataset for the LLM model."""
    def __init__(self, dataset_name, nlp_processor, llm_name):
        self.dataset_name = dataset_name
        self.nlp_processor = nlp_processor
        self.llm_name = llm_name

    def remove_trailing_brackets(self, text):
        """ Removes the trailing brackets from the corrected mini-fact."""
        match = re.search(r'\s*\([^)]*\)\s*$', text)
        if match:
            removed_brackets = match.group()  
            cleaned_text = re.sub(r'\s*\([^)]*\)\s*$', '', text) 
        else:
            removed_brackets = None
            cleaned_text = text
        return cleaned_text, removed_brackets

    def split_mini_facts(self, df_corrections_with_evidence):
        """Split the evidence into a df with mini-facts"""
        df_mini_facts_correction = pd.DataFrame()
        for index, row in df_corrections_with_evidence.iterrows():
            correction_facts = row[f'correction_evidence_{self.llm_name}'].split("\n")
            ground_truth_source = row["ground_truth_source"]
            mini_facts_with_labels_false = row['mini_facts_with_labels_false']

            correction_facts = list(dict.fromkeys(correction_facts))
            correction_facts = [fact for fact in correction_facts if "(REMOVE)" in fact or "(Corrected)" in fact]
            while len(correction_facts) < len(mini_facts_with_labels_false):
                correction_facts.append("")

            for fact, wrong_fact in zip(correction_facts, mini_facts_with_labels_false):
                df_mini_facts_correction = pd.concat([df_mini_facts_correction, 
                                                      pd.DataFrame({f"correction_fact_{self.llm_name}": [fact], 
                                                                    "gen_evidence": [row['gen_evidence']], 
                                                                    "ground_truth_source": [ground_truth_source], 
                                                                    "wrong_fact": [wrong_fact]})], ignore_index=True)
        
        df_mini_facts_correction.reset_index(drop=True, inplace=True)
        return df_mini_facts_correction
    
    def build_or_evaluate_dataset(self, df_corrections_with_evidence, df_mini_facts_correction, build_dataset):
        """Builds or evaluates the dataset. Corrected is when BART model predicts Entailment, and Remove is when BART model predicts anything other than Entailment."""
        counter_removed = 0
        counter_removed_true = 0
        counter_corrected = 0
        counter_corrected_true = 0
        for index, row in df_mini_facts_correction.iterrows():
            if index % 100 == 0:
                print(f"Processed {index} samples")
            correction_fact = row[f'correction_fact_{self.llm_name}']
            ground_truth_source = row['ground_truth_source']
            gen_evidence = row['gen_evidence']
            wrong_fact = row['wrong_fact']
            clean_fact, bracket_gpt = self.remove_trailing_brackets(correction_fact)


            if bracket_gpt is None or clean_fact == "REMOVE" or str(clean_fact).lower() == 'nan' or clean_fact == "" or wrong_fact == "":
                df_corrections_with_evidence = df_corrections_with_evidence[df_corrections_with_evidence['gen_evidence'] != gen_evidence]
                continue
            else:
                bart_label = self.nlp_processor.call_bart_model(ground_truth_source, clean_fact)
            if "REMOVE" in bracket_gpt:
                counter_removed += 1
                if bart_label != "Entailment":
                    counter_removed_true += 1
                else:
                    df_corrections_with_evidence = df_corrections_with_evidence[df_corrections_with_evidence['gen_evidence'] != gen_evidence]
            elif "Corrected" in bracket_gpt:
                counter_corrected += 1
                if bart_label == "Entailment":
                    counter_corrected_true += 1
                else:
                    df_corrections_with_evidence = df_corrections_with_evidence[df_corrections_with_evidence['gen_evidence'] != gen_evidence]
        if build_dataset:
            df_corrections_with_evidence.rename(columns={f"correction_evidence_{self.llm_name}": "correction_evidence"}, inplace=True)
            return df_corrections_with_evidence
        else:
            return counter_removed, counter_removed_true, counter_corrected, counter_corrected_true, df_corrections_with_evidence
    
    def balance_remove_corrected(self, df_corrections_with_evidence):
        """Balances the dataset (3:1) by removing the samples with REMOVE label."""
        corrected_samples = df_corrections_with_evidence['correction_evidence'].str.contains('(Corrected)', regex=False).sum()
        remove_samples = df_corrections_with_evidence['correction_evidence'].str.contains('(REMOVE)', regex=False).sum()
        print(f"Initial number of Corrected samples: {corrected_samples}")
        print(f"Initial number of REMOVE samples: {remove_samples}")
        grouped = df_corrections_with_evidence.groupby('gen_evidence')
        corrected_groups = []
        remove_groups = []
        for name, group in grouped:
            corrected_count = group['correction_evidence'].str.contains('(Corrected)', regex=False).sum()
            remove_count = group['correction_evidence'].str.contains('(REMOVE)', regex=False).sum()
            if corrected_count > remove_count:
                corrected_groups.append(group)
            else:
                remove_groups.append(group)
        corrected_count = len(corrected_groups)
        remove_needed = min(len(remove_groups), corrected_count // 3)
        remove_sampled = pd.concat(remove_groups).sample(n=remove_needed, random_state=42)
        df_corrections_with_evidence_balanced = pd.concat([pd.concat(corrected_groups), remove_sampled])
        df_corrections_with_evidence_balanced = df_corrections_with_evidence_balanced.sample(frac=1).reset_index(drop=True)
        corrected_samples = df_corrections_with_evidence_balanced['correction_evidence'].str.contains('(Corrected)', regex=False).sum()
        remove_samples = df_corrections_with_evidence_balanced['correction_evidence'].str.contains('(REMOVE)', regex=False).sum()
        print(f"Final Number of Corrected samples: {corrected_samples}")
        print(f"Final Number of REMOVE samples: {remove_samples}")
        return df_corrections_with_evidence_balanced

if __name__ == "__main__":

    # insert BART model path
    bart_model_path = ""

    datasets = ["fever", "hover"]
    nlp_processor = NLPProcessor(bart_model_path)
    llm_name = "openai"
    create_combined_dataset = True

    for dataset_name in datasets:
        file_names = [f"corrections_evidence_{dataset_name}_train", f"corrections_evidence_{dataset_name}_dev"]
        for file_name in file_names:
            df_corrections_with_evidence = pd.read_pickle(f"{llm_name}_corrections_{dataset_name}/{file_name}.pkl")
            dataset_builder = DatasetBuilder(dataset_name, nlp_processor, llm_name)

            df_mini_facts_correction = dataset_builder.split_mini_facts(df_corrections_with_evidence)

            df_corrections_with_evidence = dataset_builder.build_or_evaluate_dataset(df_corrections_with_evidence, df_mini_facts_correction, build_dataset=True)

            df_corrections_with_evidence_balanced = dataset_builder.balance_remove_corrected(df_corrections_with_evidence)

            if not os.path.exists(f"train_datasets_{dataset_name}"):
                os.makedirs(f"train_datasets_{dataset_name}")
            df_corrections_with_evidence_balanced.reset_index(drop=True, inplace=True)
            df_corrections_with_evidence_balanced.to_pickle(f"train_datasets_{dataset_name}/{file_name}_balanced.pkl")

        
    if create_combined_dataset:
        if not os.path.exists("train_datasets_combined"):
            os.makedirs("train_datasets_combined")
        df_train_fever = pd.read_pickle(f"train_datasets_fever/corrections_evidence_fever_train_balanced.pkl")
        df_train_hover = pd.read_pickle(f"train_datasets_hover/corrections_evidence_hover_train_balanced.pkl")
        df_train_combined = pd.concat([df_train_fever, df_train_hover], ignore_index=True)
        df_train_combined.to_pickle("train_datasets_combined/corrections_evidence_combined_train_balanced.pkl")

        df_dev_fever = pd.read_pickle(f"train_datasets_fever/corrections_evidence_fever_dev_balanced.pkl")
        df_dev_hover = pd.read_pickle(f"train_datasets_hover/corrections_evidence_hover_dev_balanced.pkl")
        df_dev_combined = pd.concat([df_dev_fever, df_dev_hover], ignore_index=True)
        df_dev_combined.to_pickle("train_datasets_combined/corrections_evidence_combined_dev_balanced.pkl")