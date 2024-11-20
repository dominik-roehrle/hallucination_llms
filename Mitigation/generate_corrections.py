import os
import torch
import pandas as pd
from llm_interaction import LLMInteraction


class GenerateCorrections:
    """ this class generates corrections for false mini facts"""
    def __init__(self, llm_name, llm_interaction, save_corrections_path):
        self.llm_name = llm_name
        self.llm_interaction = llm_interaction
        self.save_corrections_path = save_corrections_path

    def group_mini_facts(self, df_mini_facts):
        """ groups the mini facts by the generated evidence"""
        df_mini_facts_grouped = df_mini_facts.groupby("gen_evidence")
        df_corrections_evidence = pd.DataFrame(columns=["gen_evidence", "ground_truth_source", "mini_facts_with_labels_false"])
        for name, group in df_mini_facts_grouped:
            docs = group['docs'].values[0]
            mini_facts_with_labels_true = []
            mini_facts_with_labels_false = []
            for mini_fact, label in zip(group['output_mini_fact'].tolist(), group['label_mini_fact'].tolist()):
                if label == 1:
                    mini_facts_with_labels_true.append("- " + mini_fact)
                else:
                    mini_facts_with_labels_false.append("- " + mini_fact)

            if mini_facts_with_labels_false:
                df_corrections_evidence = pd.concat([df_corrections_evidence, 
                                                     pd.DataFrame({"gen_evidence": [group['gen_evidence'].values[0]], 
                                                                                            "ground_truth_source": [group['ground_truth_source'].values[0]], 
                                                                                            "mini_facts_with_labels_false": [mini_facts_with_labels_false], 
                                                                                            "mini_facts_with_labels_true": [mini_facts_with_labels_true], 
                                                                                            "docs": [docs]})]) 
        df_corrections_evidence.reset_index(drop=True, inplace=True)
        return df_corrections_evidence

    def generate_corrections(self, df_corrections_evidence):
        """ generates corrections for the false mini facts"""
        df_corrections_evidence_model = pd.DataFrame()
        print(len(df_corrections_evidence))
        for index, row in df_corrections_evidence.iterrows():
            print(index)
            mini_facts_with_labels_false = row['mini_facts_with_labels_false']
            mini_facts_with_labels_true = row['mini_facts_with_labels_true']
            ground_truth_source = row['ground_truth_source']
            gen_evidence = row['gen_evidence']
            docs = row['docs']

            if mini_facts_with_labels_false:
                messages = self.llm_interaction.correct_mini_facts_prompt("\n".join(mini_facts_with_labels_false), ground_truth_source)
    
                if self.llm_name == "openai":
                    response = self.llm_interaction.call_openai_llm(messages, response_format="text")
                elif self.llm_name == "llama" or self.llm_name == "llama_finetuned":
                    response = self.llm_interaction.call_llama_llm(messages)
                    print(response)
                
                df_corrections_evidence_model = pd.concat([df_corrections_evidence_model, 
                                                               pd.DataFrame({"gen_evidence": [gen_evidence], 
                                                                             f"correction_evidence_{llm_name}": [response], 
                                                                             "ground_truth_source": [ground_truth_source], 
                                                                             "mini_facts_with_labels_false": [mini_facts_with_labels_false],
                                                                             "mini_facts_with_labels_true": [mini_facts_with_labels_true],
                                                                             "docs": [docs]})]) 
        
        df_corrections_evidence_model.reset_index(drop=True, inplace=True)
        if self.save_corrections_path:
            if not os.path.exists(os.path.dirname(self.save_corrections_path)):
                os.makedirs(os.path.dirname(self.save_corrections_path))
            df_corrections_evidence_model.to_pickle(self.save_corrections_path)

if __name__ == "__main__":
    
    # select the LLM model
    llm_name = "llama_finetuned"
    dataset_name = "fever"

    if llm_name == "openai":
        openai = LLMInteraction(fine_tuned_version=False, few_shot=True, use_cache=False) 
        df_mini_facts = pd.read_pickle(f"mini_facts_{dataset_name}_gen_evidence.pkl")
        save_corrections_path = f"recreate_openai_corrections_{dataset_name}/corrections_evidence_{dataset_name}.pkl"
        corrections = GenerateCorrections(llm_name, openai, save_corrections_path)
        df_corrections_evidence = corrections.group_mini_facts(df_mini_facts)
        corrections.generate_corrections(df_corrections_evidence)

    elif llm_name == "llama":
        # insert the llama model path
        llama_model_path = ""
        save_corrections_path = f"llama_corrections/corrections_evidence_{dataset_name}_test.pkl"
        
        llama = LLMInteraction(llama_model_path, fine_tuned_version=False, few_shot=True, use_cache=False)
        df_corrections_evidence = pd.read_pickle(f"corrections_evidence_for_evaluation/corrections_evidence_{dataset_name}_test.pkl")
        corrections = GenerateCorrections(llm_name, llama, save_corrections_path)
        corrections.generate_corrections(df_corrections_evidence)

    elif llm_name == "llama_finetuned":
        llama_model_path = "llama_finetuned_model"
        save_corrections_path = f"llama_finetuned_corrections/corrections_evidence_{dataset_name}_test.pkl"

        llama = LLMInteraction(llama_model_path, fine_tuned_version=True, few_shot=False, use_cache=False)
        df_corrections_evidence = pd.read_pickle(f"corrections_evidence_for_evaluation/corrections_evidence_{dataset_name}_test.pkl").iloc[:10]
        corrections = GenerateCorrections(llm_name, llama, save_corrections_path)
        corrections.generate_corrections(df_corrections_evidence)