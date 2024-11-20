import pandas as pd
import numpy as np  # Use NumPy for random operations
from itertools import chain
import ast


def balance_dataframe(df, label_name):
    """ Balance the DataFrame by downsampling the majority class """
    df_label_1 = df[df[str(label_name)] == 1]
    df_label_0 = df[df[str(label_name)] == 0]
    min_class_count = min(len(df_label_1), len(df_label_0))
    df_label_1_downsampled = df_label_1.sample(min_class_count, random_state=42)
    df_label_0_downsampled = df_label_0.sample(min_class_count, random_state=42)
    balanced_df = pd.concat([df_label_1_downsampled, df_label_0_downsampled])
    return balanced_df.reset_index(drop=True)

def read_and_preprocess_dataset(probe_method, dataset_folder, dataset_name):
    """ Read and preprocess the dataset """
    df = pd.read_pickle(f"{dataset_folder}/{dataset_name}.pkl")
    print(f"Total samples: {len(df)}")
    df.drop_duplicates(subset=[f"output_{probe_method}"], inplace=True, ignore_index=True)
    df['docs'] = df['docs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df.dropna(subset=['docs'], inplace=True)
    print(f"Total samples after dropping duplicates and nan docs: {len(df)}")
    return df

def remove_conflicting_samples(df, conflicting_docs):
    """ Remove samples that still contain conflicting documents """
    def has_conflicting_docs(docs):
        return any(doc in conflicting_docs for doc in docs)
    df_cleaned = df[~df['docs'].apply(has_conflicting_docs)]
    return df_cleaned

def split_dataset(df, train_ratio, dev_ratio, test_ratio):
    """ Split the dataset into train, dev, and test so that no doc occurs in multiple datasets """
    doc_to_dataset = {}
    train_samples = []
    dev_samples = []
    test_samples = []
    train_docs = set()
    dev_docs = set()
    test_docs = set()
    total_samples = len(df)

    train_target_count = total_samples * train_ratio
    dev_target_count = total_samples * dev_ratio
    test_target_count = total_samples * test_ratio

    train_sample_count = 0
    dev_sample_count = 0
    test_sample_count = 0
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for index, row in df_shuffled.iterrows():
        document_ids = row['docs']
        assigned_datasets = set()

        for doc_id in document_ids:
            if doc_id in doc_to_dataset:
                assigned_datasets.add(doc_to_dataset[doc_id])
        if not assigned_datasets:
            available_datasets = []
            if train_sample_count < train_target_count:
                available_datasets.append('train')
            if dev_sample_count < dev_target_count:
                available_datasets.append('dev')
            if test_sample_count < test_target_count:
                available_datasets.append('test')
            if not available_datasets:
                continue
            assigned_dataset = rng.choice(sorted(available_datasets))
        elif len(assigned_datasets) == 1:
            assigned_dataset = sorted(assigned_datasets)[0]
        else:
            dataset_doc_counts = {'train': 0, 'dev': 0, 'test': 0}
            for doc_id in document_ids:
                if doc_id in doc_to_dataset:
                    dataset_doc_counts[doc_to_dataset[doc_id]] += 1
            sorted_datasets = sorted(dataset_doc_counts.items())
            assigned_dataset = max(sorted_datasets, key=lambda x: x[1])[0]

        if assigned_dataset == 'train':
            train_samples.append(row)
            train_sample_count += 1
            train_docs.update(document_ids)
        elif assigned_dataset == 'dev':
            dev_samples.append(row)
            dev_sample_count += 1
            dev_docs.update(document_ids)
        elif assigned_dataset == 'test':
            test_samples.append(row)
            test_sample_count += 1
            test_docs.update(document_ids)
        for doc_id in document_ids:
            doc_to_dataset[doc_id] = assigned_dataset

    train_df = pd.DataFrame(train_samples)
    dev_df = pd.DataFrame(dev_samples)
    test_df = pd.DataFrame(test_samples)
    conflicting_docs = (train_docs & dev_docs) | (train_docs & test_docs) | (dev_docs & test_docs)
    print(f"Documents in multiple datasets: {len(conflicting_docs)}")
    return train_df, dev_df, test_df, conflicting_docs
    

def process_sentence(dataset_folder, sentence_dataset_name, train_ratio, dev_ratio, test_ratio):
    probe_method = "sentence"
    df_sentence = read_and_preprocess_dataset(probe_method, dataset_folder, sentence_dataset_name)
    
    # Split the dataset
    train_df_sentence, dev_df_sentence, test_df_sentence, conflicting_docs = split_dataset(df_sentence, train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio)
    
    # Remove conflicting samples from each set
    train_df_sentence_cleaned = remove_conflicting_samples(train_df_sentence, conflicting_docs)
    dev_df_sentence_cleaned = remove_conflicting_samples(dev_df_sentence, conflicting_docs)
    test_df_sentence_cleaned = remove_conflicting_samples(test_df_sentence, conflicting_docs)
    
    # Balance each set
    print(f"Training samples Sentences before balancing: {len(train_df_sentence_cleaned)}, Validation samples: {len(dev_df_sentence_cleaned)}, Test samples: {len(test_df_sentence_cleaned)}")
    train_df_sentence_balanced = balance_dataframe(train_df_sentence_cleaned, label_name="label_sentence")
    dev_df_sentence_balanced = balance_dataframe(dev_df_sentence_cleaned, label_name="label_sentence")
    test_df_sentence_balanced = balance_dataframe(test_df_sentence_cleaned, label_name="label_sentence")
    print(f"Training samples Sentences after balancing: {len(train_df_sentence_balanced)}, Validation samples: {len(dev_df_sentence_balanced)}, Test samples: {len(test_df_sentence_balanced)}")
    
    train_df_sentence_balanced.reset_index(drop=True, inplace=True)
    dev_df_sentence_balanced.reset_index(drop=True, inplace=True)
    test_df_sentence_cleaned.reset_index(drop=True, inplace=True)
    test_df_sentence_balanced.reset_index(drop=True, inplace=True)

    train_df_sentence_balanced.to_pickle(f"{dataset_folder}/{sentence_dataset_name}_train.pkl")
    dev_df_sentence_balanced.to_pickle(f"{dataset_folder}/{sentence_dataset_name}_dev.pkl")
    test_df_sentence_balanced.to_pickle(f"{dataset_folder}/{sentence_dataset_name}_test_balanced.pkl")
    test_df_sentence_cleaned.to_pickle(f"{dataset_folder}/{sentence_dataset_name}_test_unbalanced.pkl")

    train_sentence_docs = list(set(chain.from_iterable(train_df_sentence_cleaned['docs'])))
    dev_sentence_docs = list(set(chain.from_iterable(dev_df_sentence_balanced['docs'])))
    test_sentence_docs = list(set(chain.from_iterable(test_df_sentence_cleaned['docs'])))

    for train_sentence_doc_t in train_sentence_docs:
        if train_sentence_doc_t in test_sentence_docs:
            print("Duplicates FOUND!")

    # Extract outputs from the datasets
    train_sentence_outputs = train_df_sentence_cleaned['output_sentence'].tolist()
    dev_sentence_outputs = dev_df_sentence_balanced['output_sentence'].tolist()
    test_sentence_outputs = test_df_sentence_cleaned['output_sentence'].tolist()
    return train_sentence_outputs, dev_sentence_outputs, test_sentence_outputs, train_sentence_docs, dev_sentence_docs, test_sentence_docs


def process_mini_facts(dataset_folder, mini_fact_dataset_name, train_outputs, dev_outputs, test_outputs, train_docs, dev_docs, test_docs):
    probe_method = "mini_fact"
    df_mini_facts = read_and_preprocess_dataset(probe_method, dataset_folder, mini_fact_dataset_name)

    # Create the datasets by filtering the original DataFrame
    train_df_mini_facts = df_mini_facts.loc[(df_mini_facts['gen_sentence'].isin(train_outputs)) & (df_mini_facts['docs'].apply(lambda x: all(doc in train_docs for doc in x)))]
    dev_df_mini_facts = df_mini_facts.loc[(df_mini_facts['gen_sentence'].isin(dev_outputs)) & (df_mini_facts['docs'].apply(lambda x: all(doc in dev_docs for doc in x)))]
    test_df_mini_facts = df_mini_facts.loc[(df_mini_facts['gen_sentence'].isin(test_outputs)) & (df_mini_facts['docs'].apply(lambda x: all(doc in test_docs for doc in x)))]
    
    # Balance the datasets
    train_df_mini_facts_balanced = balance_dataframe(train_df_mini_facts, label_name='label_mini_fact')
    dev_df_mini_facts_balanced = balance_dataframe(dev_df_mini_facts, label_name='label_mini_fact')
    test_df_mini_facts_balanced = balance_dataframe(test_df_mini_facts, label_name='label_mini_fact')
    print(f"Training samples Mini facts: {len(train_df_mini_facts_balanced)}, Validation samples: {len(dev_df_mini_facts_balanced)}, Test samples: {len(test_df_mini_facts_balanced)}")
    
    # Save the datasets
    train_df_mini_facts_balanced.reset_index(drop=True, inplace=True)
    dev_df_mini_facts_balanced.reset_index(drop=True, inplace=True)
    test_df_mini_facts.reset_index(drop=True, inplace=True)
    test_df_mini_facts_balanced.reset_index(drop=True, inplace=True)

    train_df_mini_facts_balanced.to_pickle(f"{dataset_folder}/{mini_fact_dataset_name}_train.pkl")
    dev_df_mini_facts_balanced.to_pickle(f"{dataset_folder}/{mini_fact_dataset_name}_dev.pkl")
    test_df_mini_facts.to_pickle(f"{dataset_folder}/{mini_fact_dataset_name}_test_unbalanced.pkl")
    test_df_mini_facts_balanced.to_pickle(f"{dataset_folder}/{mini_fact_dataset_name}_test_balanced.pkl")

    train_mini_facts_docs = list(set(chain.from_iterable(train_df_mini_facts_balanced['docs'])))
    dev_mini_facts_docs = list(set(chain.from_iterable(dev_df_mini_facts_balanced['docs'])))
    test_mini_facts_docs = list(set(chain.from_iterable(test_df_mini_facts_balanced['docs'])))

    for train_mini_facts_doc_t in train_mini_facts_docs:
        if train_mini_facts_doc_t in test_mini_facts_docs:
            print(train_mini_facts_doc_t)


if __name__ == "__main__":
    dataset = "hover"
    layers = [-1, -4, -8, -16, -24, 1]

    if dataset == "hover":
        train_ratio= 0.93
        dev_ratio= 0.02
        test_ratio= 0.05
    elif dataset == "fever":
        train_ratio= 0.96
        dev_ratio= 0.01
        test_ratio= 0.02


    for layer in layers:
        rng = np.random.RandomState(42)
        dataset_folder = f"processed_datasets_llama_{dataset}_layer{layer}"
        sentence_dataset_name = f"sentence_{dataset}"
        mini_fact_dataset_name = f"mini_fact_{dataset}"
        

        train_outputs, dev_outputs, test_outputs, train_docs, dev_docs, test_docs = process_sentence(dataset_folder, 
                                                                                                     sentence_dataset_name, 
                                                                                                     train_ratio, 
                                                                                                     dev_ratio, 
                                                                                                     test_ratio)
        
        process_mini_facts(dataset_folder, mini_fact_dataset_name, 
                            train_outputs, dev_outputs, test_outputs, train_docs, dev_docs, test_docs)


    