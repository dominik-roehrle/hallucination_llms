import pandas as pd
import json
from transformers import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import openai
import spacy
from fastcoref import spacy_component
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os 
import torch

# Suppress transformer logging
logging.set_verbosity_error()

with open("../api.key", "r") as file:
    api_key = file.read().strip() 

os.environ["OPENAI_API_KEY"] = api_key
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OpenAIInteraction:
    """ this class interacts with the OpenAI API"""
    def __init__(self, maximum_mini_facts_per_evidence):

        # optionally set the maximum number of mini facts per evidence
        self.maximum_mini_facts_per_evidence = 8

        self.mini_facts_instruction = f"""Your task is to breakdown claims/sentences into independant statements. 
You must NEVER correct or comment the original claims/sentences even if something of the original claims/sentences is incorrect.
Do NEVER generate statements that are not in the original claims/sentences. Every statement must start with an entity that specifies the topic (e.g. **The Fox Broadcasting Company** and not **The company**)."""
        
        self.mini_facts_samples = ["The Hunger Games is a 2012 American science fiction film directed by John Peter and based on the novel of the same name by Suzanne Collins. Matt Lucena is an American former professional tennis player.",
"""Owen Wilson starred in the film "The Karate Kid" (2010) alongside martial arts expert Tom Wu. Owen Wilson voiced Lightning McQueen in the "Cars" franchise, not "The Royal Tenenbaums" franchise.""",
"Feels So Good is a song by the American R&B group Tony! Toni! TonÃ. The song was written by the group's lead vocalist Raphael Saadiq and producer Tony! Toni! TonÃ's lead vocalist Dwayne Wimberly."]
        
        
        self.mini_facts_sample_outputs = ["""- **The Hunger Games** is a 2012 American science fiction film.
- **The Hunger Games** was directed by John Peter.
- **The Hunger Games** is based on a novel of the same name by Suzanne Collins.
- **Matt Lucena** is an American former professional tennis player.""",
"""- **Owen Wilson** starred in the film The Karate Kid (2010) alongside martial arts expert Tom Wu.
- **Owen Wilson** voiced Lightning McQueen in the Cars franchise.
- **Owen Wilson** did not voice Lightning McQueen in the The Royal Tenenbaums franchise.""",
"""- **Feels So Good** is a song by the American R&B group Tony! Toni! TonÃ.
- **Feels So Good** was written by the group's lead vocalist Raphael Saadiq and producer Tony! Toni! TonÃ's lead vocalist Dwayne Wimberly."""]
        
        self.labeling_instruction = """Please label each statement (the statements are seperated by a bullet point '- ') for truthfulness (True or False) based on the given context. 
Return the output in the JSON format where each statement is a key and the value is either True or False. The final format should be e.g.: {{'Mount Everest is a mountain': True, 'Mount Everest is the second highest mountain': False, 'Mount Everest is in the Himalaya': True}}"""
        
        self.replace_pronouns_instruction = """Replace pronouns and general words with the entity such that each sentence can be read independantly. 
Do not change the sentences except the replacement even if they are not correct."""

        self.replace_pronoun_samples = ["""Beautiful is a song by American singer-songwriter Christina Aguilera, featuring rapper Daughtry. It was released in 2002 as the second single from Aguilera's self-titled album Christina Aguilera. The song peaked at number 5 on the US Billboard Hot 100 chart.""",
                                """Mount Everest is a mountain. It reaches more than 8000 meters. The mountain is in the Himalayas."""]


        self.replace_pronoun_sample_outputs = ["""Beautiful is a song by American singer-songwriter Christina Aguilera, featuring rapper Daughtry. Beautiful was released in 2002 as the second single from Aguilera's self-titled album Christina Aguilera. Beautiful peaked at number 5 on the US Billboard Hot 100 chart.""",
                                       """Mount Everest is a mountain. Mount Everest reaches more than 8000 meters. Mount Everest is in the Himalayas."""]



    def call_llm(self, messages, response_format):
        """ calls the OpenAI API"""
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
    

    def get_prompt_mini_facts(self, gen_evidence):
        """ generates the prompt for the mini facts"""
        messages = [{"role": "system", 
                "content" : [{"type": "text", 
                            "text": f"{self.mini_facts_instruction}"}]},
            {"role": "user", "content": [{"type": "text", "text": f"{self.mini_facts_samples[0]}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{self.mini_facts_sample_outputs[0]}"}]},
            {"role": "user", "content": [{"type": "text", "text": f"{self.mini_facts_samples[1]}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{self.mini_facts_sample_outputs[1]}"}]},
            {"role": "user", "content": [{"type": "text", "text": f"{self.mini_facts_samples[2]}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{self.mini_facts_sample_outputs[2]}"}]},
            {"role": "user", "content": [{"type": "text", "text": f"{gen_evidence}"}]}]
        return messages
    
    
    def get_prompt_labeling(self, mini_facts, ground_truth_source):
        """ generates the prompt for the labeling of the mini facts"""
        messages = [{"role": "system", 
                    "content" : [{"type": "text", 
                                "text":  f"{self.labeling_instruction}"}]},
                    {"role": "user", "content": [{"type": "text", "text": f" Context: {ground_truth_source} \nStatements: {mini_facts}"}]}]
        return messages
    

    def get_prompt_replace_pronouns(self, gen_evidence):
        """ generates the prompt for the replacement of pronouns"""
        messages = [{"role": "system", 
                 "content" : [{"type": "text", 
                               "text": f"{self.replace_pronouns_instruction}"}]},
                {"role": "user", "content": [{"type": "text", "text": f"{self.replace_pronoun_samples[0]}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"{self.replace_pronoun_sample_outputs[0]}"}]},
                {"role": "user", "content": [{"type": "text", "text": f"{self.replace_pronoun_samples[1]}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"{self.replace_pronoun_sample_outputs[1]}"}]},
                {"role": "user", "content": [{"type": "text", "text": f"{gen_evidence}"}]}]
        return messages


class NLPProcessor:
    """ this class processes the text with the NLP models (sentence split and BART)"""
    def __init__(self, bart_model_path):
        self.nlp = spacy.load("en_core_web_sm")
        #self.nlp.add_pipe(
        #    "fastcoref",
        #    config={
        #        'model_architecture': 'LingMessCoref',
        #        'model_path': 'biu-nlp/lingmess-coref',
        #        'device': 'cuda'
        #    }
        #)

        print("Loading BART model...")
        self.bart_model = AutoModelForSequenceClassification.from_pretrained(bart_model_path, local_files_only=True)
        self.bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_path, local_files_only=True)
        self.bart_model.to(device)

    def convert_text_to_sentences(self, text):
        """ converts the text into sentences with spacy"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences
    
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
        """ calls the BART model to predict entailment, contradiction, or neutral"""
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
    """ this class builds the dataset with the mini facts and sentences"""
    def __init__(self, dataset_name, openai_interaction, nlp_processor, save_path_mini_facts_and_evidence, 
                 save_path_mini_facts_and_sentences, save_path_mini_facts_and_sentences_bart, save_path_mini_facts_and_evidence_bart,
                 save_path_sentences):
        self.dataset_name = dataset_name
        self.openai_interaction = openai_interaction
        self.nlp_processor = nlp_processor
        self.vectorizer = TfidfVectorizer()
        self.save_path_mini_facts_and_evidence = save_path_mini_facts_and_evidence
        self.save_path_mini_facts_and_sentences = save_path_mini_facts_and_sentences
        self.save_path_mini_facts_and_sentences_bart = save_path_mini_facts_and_sentences_bart
        self.save_path_mini_facts_and_evidence_bart = save_path_mini_facts_and_evidence_bart
        self.save_path_sentences = save_path_sentences


    def create_mini_facts_and_evidence(self, df_gen_evidence):
        """ creates the mini facts and evidence dataset"""
        df_mini_facts_and_evidence = pd.DataFrame()
        for index, row in df_gen_evidence.iterrows():
            print(index)
            gen_evidence = row["gen_evidence"]
            ground_truth_source = row["ground_truth_source"]
            docs = row["docs"]

            messages = self.openai_interaction.get_prompt_mini_facts(gen_evidence)
            response = self.openai_interaction.call_llm(messages, response_format="text")
            response = response.replace("**", "")

            messages = self.openai_interaction.get_prompt_labeling(response, ground_truth_source)
            response = self.openai_interaction.call_llm(messages, response_format="json_object")

            try:
                response_json = json.loads(response)
                response_data = [(key.strip(), int(value)) for key, value in response_json.items()]
            except Exception as e:
                print(f"Error parsing JSON at index {index}: {e}. The response is skipped.")
                continue

            new_row = pd.DataFrame(response_data, columns=["output_mini_fact", "label_mini_fact"])
            new_row['ground_truth_source'] = [ground_truth_source] * len(new_row)
            new_row['gen_evidence'] = [gen_evidence] * len(new_row)
            new_row['docs'] = [docs] * len(new_row)
        
            df_mini_facts_and_evidence = pd.concat([new_row, df_mini_facts_and_evidence], axis=0, ignore_index=True)

        if self.save_path_mini_facts_and_evidence:
            df_mini_facts_and_evidence.to_pickle(self.save_path_mini_facts_and_evidence)
        return df_mini_facts_and_evidence

    def create_mini_facts_and_sentences(self, df_mini_facts_and_evidence):
        """ creates the mini facts and sentences dataset"""
        df_mini_facts_and_sentences = pd.DataFrame()
        df_grouped_evidence = df_mini_facts_and_evidence.groupby("gen_evidence")

        for gen_evidence, group in df_grouped_evidence:
            messages = self.openai_interaction.get_prompt_replace_pronouns(gen_evidence)
            gen_evidence_with_coref = self.openai_interaction.call_llm(messages, response_format="text")
            sentences = self.nlp_processor.convert_text_to_sentences(gen_evidence_with_coref)

            outputs_mini_facts = group["output_mini_fact"].tolist()
            labels_mini_facts = group["label_mini_fact"].tolist()
            ground_truth_sources = group["ground_truth_source"].tolist()
            docs = group["docs"].tolist()

            for output_mini_fact, label_mini_fact, ground_truth_source, doc in zip(outputs_mini_facts, labels_mini_facts, ground_truth_sources, docs):
                if len(sentences) > 1:
                    cosine_scores = []
                    for sentence in sentences:
                        tfidf_matrix = self.vectorizer.fit_transform([output_mini_fact, sentence])
                        cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        cosine_scores.append(cosine_score)
                    index_cosine = cosine_scores.index(max(cosine_scores))
                else:
                    index_cosine = 0

                df_mini_facts_and_sentences = pd.concat([df_mini_facts_and_sentences, pd.DataFrame(
                    {"output_mini_fact": [output_mini_fact], "label_mini_fact" : [label_mini_fact], "gen_sentence": [sentences[index_cosine]], 
                     "gen_evidence" : [gen_evidence], 
                     "gen_evidence_with_coref" : [gen_evidence_with_coref],
                     "ground_truth_source": [ground_truth_source],
                    "docs": [doc]})], axis=0, ignore_index=True)
        return df_mini_facts_and_sentences

    def remove_duplicates(self, df_mini_facts_and_sentences):
        """ removes duplicates from mini-facts and sentence dataset"""
        duplicates = df_mini_facts_and_sentences[df_mini_facts_and_sentences.duplicated(subset='output_mini_fact', keep='first')]
        values_to_remove = duplicates['gen_sentence'].unique()
        df_mini_facts_and_sentences = df_mini_facts_and_sentences[~df_mini_facts_and_sentences['gen_sentence'].isin(values_to_remove)]
        if self.save_path_mini_facts_and_sentences:
            df_mini_facts_and_sentences.to_pickle(self.save_path_mini_facts_and_sentences)
        return df_mini_facts_and_sentences

    def get_sentences_only(self, df_mini_facts_and_sentences):
        """ gets the sentences only dataset"""
        def count_false_facts(x):
            return (x == 0).sum()

        def aggregate_label(x):
            return 0 if (x == 0).any() else 1

        aggregated_df = df_mini_facts_and_sentences.groupby('gen_sentence').agg(
            false_facts=('label_mini_fact', count_false_facts),
            label_sentence=('label_mini_fact', aggregate_label),
            gen_evidence=('gen_evidence', 'first'),
            docs=('docs', 'first'),
            gen_evidence_with_coref=('gen_evidence_with_coref', 'first')
        ).reset_index()

        aggregated_df.rename(columns={"gen_sentence": "output_sentence"}, inplace=True)

        if self.save_path_sentences:
            aggregated_df.to_pickle(self.save_path_sentences)
        return aggregated_df
    
    def match_gpt4_with_bart(self, df, remove_sentence_or_gen_evidence):
        """ matches the GPT-4 labels with BART model"""
        df_bart = df.copy()
        for index, row in df.iterrows():
            ground_truth_source = row["ground_truth_source"]
            statement = row["output_mini_fact"]
            label = row["label_mini_fact"]
            try:
                pred_label = self.nlp_processor.call_bart_model(ground_truth_source, statement)
                if pred_label == "Entailment" and label == 1:
                    pass
                elif pred_label == "Contradiction" and label == 0:
                    pass
                elif pred_label == "Entailment" and label == 0:
                    df_bart = df_bart[df_bart[remove_sentence_or_gen_evidence] != row[remove_sentence_or_gen_evidence]]
                elif pred_label == "Contradiction" and label == 1:
                    df_bart = df_bart[df_bart[remove_sentence_or_gen_evidence] != row[remove_sentence_or_gen_evidence]]

            except Exception as e:
                print(f"Error processing index {index}: {e}")
                df_bart = df_bart[df_bart[remove_sentence_or_gen_evidence] != row[remove_sentence_or_gen_evidence]]

        if remove_sentence_or_gen_evidence == "gen_sentence":
            if self.save_path_mini_facts_and_sentences_bart:
                df_bart.to_pickle(self.save_path_mini_facts_and_sentences_bart)

        elif remove_sentence_or_gen_evidence == "gen_evidence":
            if self.save_path_mini_facts_and_evidence_bart:
                df_bart.to_pickle(self.save_path_mini_facts_and_evidence_bart)
        return df_bart

if __name__ == "__main__":

    # Initialize classes
    dataset_name = 'hover' 
    model_name = "llama"
    if not os.path.exists(f"recreate_datasets_{dataset_name}_{model_name}"):
        os.makedirs(f"recreate_datasets_{dataset_name}_{model_name}")
    save_path_mini_facts_and_evidence = f"recreate_datasets_{dataset_name}_{model_name}/mini_fact_and_evidence.pkl"
    save_path_mini_facts_and_sentences = f"recreate_datasets_{dataset_name}_{model_name}/mini_fact_and_sentence.pkl"
    save_path_mini_facts_and_sentences_bart = f"recreate_datasets_{dataset_name}_{model_name}/mini_fact_and_sentence_with_bart.pkl"
    save_path_mini_facts_and_evidence_bart = f"recreate_datasets_{dataset_name}_{model_name}/mini_fact_and_evidence_with_bart.pkl"
    save_path_sentences = f"recreate_datasets_{dataset_name}_{model_name}/sentence_with_bart.pkl"

    openai_interaction = OpenAIInteraction(maximum_mini_facts_per_evidence=6)
 
    # Set the model path
    bart_model_path = ""
    nlp_processor = NLPProcessor(bart_model_path=bart_model_path)
  
    dataset_builder = DatasetBuilder(dataset_name, openai_interaction, nlp_processor, 
                                     save_path_mini_facts_and_evidence, save_path_mini_facts_and_sentences, 
                                     save_path_mini_facts_and_sentences_bart, save_path_mini_facts_and_evidence_bart,
                                     save_path_sentences)

    try:
        df_gen_evidence = pd.read_pickle(f'recreate_datasets_{dataset_name}_{model_name}/gen_evidence_{dataset_name}.pkl')
        #pd.read_pickle(f'test_evidence_with_claims/test_{dataset_name}_phi.pkl')
    except FileNotFoundError:
        print(f"File new_datasets_probs/{dataset_name}_probs.pkl not found: Please run generate_evidence.py first")
        
    df_gen_evidence.drop_duplicates(subset=['gen_evidence'], inplace=True)

    df_mini_facts_and_evidence = dataset_builder.create_mini_facts_and_evidence(df_gen_evidence)

    df_mini_facts_and_sentences = dataset_builder.create_mini_facts_and_sentences(df_mini_facts_and_evidence)

    df_mini_facts_and_sentences = dataset_builder.remove_duplicates(df_mini_facts_and_sentences)

    df_mini_facts_and_sentences_bart = dataset_builder.match_gpt4_with_bart(df_mini_facts_and_sentences, remove_sentence_or_gen_evidence="gen_sentence")

    # for correction dataset
    #df_mini_facts_and_evidence_bart = dataset_builder.match_gpt4_with_bart(df_mini_facts_and_sentences, remove_sentence_or_gen_evidence="gen_evidence") 

    df_sentences = dataset_builder.get_sentences_only(df_mini_facts_and_sentences_bart)
 

    
    


