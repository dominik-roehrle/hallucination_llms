import pandas as pd
import random
from itertools import chain
import numpy as np
import os

def read_and_preprocess_dataset(df):
    """ Read and preprocess the dataset """
    print(f"Total samples: {len(df)}")
    df.drop_duplicates(subset=["gen_evidence"], inplace=True, ignore_index=True)
    df.dropna(subset=["ground_truth_source"], inplace=True)
    print(f"Total samples after dropping duplicates and nan source: {len(df)}")
    df['docs'] = df['docs'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df.dropna(subset=['docs'], inplace=True)
    print(f"Total samples after dropping nan docs: {len(df)}")
    return df

def split_dataset(df, train_ratio=0.85, dev_ratio=0.05, test_ratio=0.1):
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
            assigned_dataset = rng.choice(available_datasets)
        elif len(assigned_datasets) == 1:
            assigned_dataset = assigned_datasets.pop()
        else:
            dataset_doc_counts = {'train': 0, 'dev': 0, 'test': 0}
            for doc_id in document_ids:
                if doc_id in doc_to_dataset:
                    dataset_doc_counts[doc_to_dataset[doc_id]] += 1
            assigned_dataset = max(dataset_doc_counts, key=dataset_doc_counts.get)
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

def remove_conflicting_samples(df, conflicting_docs):
    """ Remove samples if there are still conflicting docs left"""
    def has_conflicting_docs(docs):
        return any(doc in conflicting_docs for doc in docs)
    df_cleaned = df[~df['docs'].apply(has_conflicting_docs)]
    return df_cleaned


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    dataset_name = "hover"
    llm_name = "openai"
    df_corrections = pd.read_pickle(f"{llm_name}_corrections_{dataset_name}/corrections_evidence_{dataset_name}.pkl")

    df_corrections = read_and_preprocess_dataset(df_corrections)

    # Split the dataset
    if dataset_name == "fever":
        train_ratio = 0.93
        dev_ratio = 0.03
        test_ratio = 0.04
    elif dataset_name == "hover":
        train_ratio = 0.67
        dev_ratio = 0.15
        test_ratio = 0.18

    train_df, dev_df, test_df, conflicting_docs = split_dataset(df_corrections, train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio)

    train_df_cleaned = remove_conflicting_samples(train_df, conflicting_docs)
    dev_df_cleaned = remove_conflicting_samples(dev_df, conflicting_docs)
    test_df_cleaned = remove_conflicting_samples(test_df, conflicting_docs)

    train_df_cleaned.reset_index(drop=True, inplace=True)
    dev_df_cleaned.reset_index(drop=True, inplace=True)
    test_df_cleaned.reset_index(drop=True, inplace=True)
    train_df_cleaned.to_pickle(f"{llm_name}_corrections_{dataset_name}/corrections_evidence_{dataset_name}_train.pkl")
    dev_df_cleaned.to_pickle(f"{llm_name}_corrections_{dataset_name}/corrections_evidence_{dataset_name}_dev.pkl")
    test_df_cleaned.to_pickle(f"{llm_name}_corrections_{dataset_name}/corrections_evidence_{dataset_name}_test.pkl")

    test_df_cleaned_without_llm = test_df_cleaned.copy()
    del test_df_cleaned_without_llm[f'correction_evidence_{llm_name}']
    if not os.path.exists("corrections_evidence_for_evaluation"):
        os.makedirs("corrections_evidence_for_evaluation")
    test_df_cleaned_without_llm.reset_index(drop=True, inplace=True)
    test_df_cleaned_without_llm.to_pickle(f"corrections_evidence_for_evaluation/corrections_evidence_{dataset_name}_test.pkl")
    

    print(len(train_df_cleaned), len(dev_df_cleaned), len(test_df_cleaned))

    train_docs = list(set(chain.from_iterable(train_df_cleaned['docs'])))
    dev_docs = list(set(chain.from_iterable(dev_df_cleaned['docs'])))
    test_docs = list(set(chain.from_iterable(test_df_cleaned['docs'])))
    for train_doc_t in train_docs:
        if train_doc_t in test_docs:
            print("Duplicates FOUND!")