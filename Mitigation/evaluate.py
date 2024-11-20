from create_train_dataset import DatasetBuilder, NLPProcessor
import pandas as pd

""" this file evaluates the LLMs to get the correction rates"""

if __name__ == "__main__":

    # insert the path to the BART model
    bart_model_path = ""
    datasets = ["fever"]
    nlp_processor = NLPProcessor(bart_model_path)
    llm_names = ["llama_70B", "llama_finetuned", "openai", "llama"]
    df_results = pd.DataFrame(columns=["Dataset", "LLM", 
                                        "Facts",
                                       "Removed Facts", 
                                       "True Removed Facts", 
                                       "Score Removed",
                                       "Corrected Facts", 
                                       "True Corrected Facts", 
                                       "Score Corrected",
                                       "Total Score"])
    
    for dataset_name in datasets:
        for llm_name in llm_names:
            if llm_name == "openai":
                file_name = f"openai_corrections_{dataset_name}/corrections_evidence_{dataset_name}_test"
            else:
                file_name = f"{llm_name}_corrections_v2/corrections_evidence_{dataset_name}_test"
        
            df_corrections_with_evidence = pd.read_pickle(file_name + ".pkl")
            evaluate = DatasetBuilder(dataset_name, nlp_processor, llm_name)
            print(len(df_corrections_with_evidence))
            df_mini_facts_correction = evaluate.split_mini_facts(df_corrections_with_evidence)
            len_mini_facts = len(df_mini_facts_correction)
            print(len_mini_facts)

            counter_removed, counter_removed_true, counter_corrected, counter_corrected_true, \
            df_corrections_with_evidence = evaluate.build_or_evaluate_dataset(df_corrections_with_evidence, df_mini_facts_correction, build_dataset=False)
            
            print(f"Removed Facts: {counter_removed}")
            print(f"True Removed Facts: {counter_removed_true}")
            print(f"Score Removed: {(counter_removed_true) / counter_removed}")
            
            print(f"Corrected Facts: {counter_corrected}")
            print(f"True Corrected Facts: {counter_corrected_true}")
            print(f"Score Corrected: {(counter_corrected_true) / counter_corrected}")

            print(f"Total Score: {(counter_removed_true + counter_corrected_true) / len_mini_facts}")

            
            df_results = pd.concat([df_results if not df_results.empty else None, pd.DataFrame({"Dataset": [dataset_name], 
                                                                                                    "LLM": [llm_name],
                                                                                                    "Facts": [len_mini_facts],
                                                                                                    "Removed Facts": [counter_removed], 
                                                                                                    "True Removed Facts": [counter_removed_true], 
                                                                                                    "Score Removed": [(counter_removed_true) / counter_removed],
                                                                                                    "Corrected Facts": [counter_corrected], 
                                                                                                    "True Corrected Facts": [counter_corrected_true], 
                                                                                                    "Score Corrected": [(counter_corrected_true) / counter_corrected],
                                                                                                    "Total Score": [(counter_removed_true + counter_corrected_true) / len_mini_facts]})], ignore_index=True)
                                                                                               
            
    #df_results.reset_index(drop=True, inplace=True)
    #df_results.to_csv("evaluation_results.csv", index=False)