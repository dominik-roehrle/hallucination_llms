import pandas as pd
from transformers import logging
import spacy
from fastcoref import spacy_component
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer
import torch
import os

logging.set_verbosity_error()


class DatasetBuilder:
    """ this class builds the dataset for the probabilities and token importance"""
    def __init__(self, nlp_processor, vectorizer, gen_evidence_file, sentence_file, model_name):
        self.vectorizer = vectorizer
        self.nlp_processor = nlp_processor 
        self.gen_evidence_file = gen_evidence_file
        self.sentence_file = sentence_file
        self.model_name = model_name

    def match_tokens_llama(self, start_index, token_list, sentence):
        """ matches the tokens with the sentence for llama"""
        sentence_list = sentence.split()
        len_sentence_list = len(sentence_list)
        matched_probs = []
        for word, tokens, probs, pe in token_list[start_index:start_index + len_sentence_list]: 
            matched_probs.append((word, tokens, probs, pe))
            sentence_list.pop(0)
            sentence_list = sentence_list[len("".join(tokens)):]
            if sentence_list == []:
                break
        return start_index + len(matched_probs), matched_probs
    
    def match_tokens_phi(self, start_index, token_list, sentence):
        """ matches the tokens with the sentence for phi"""
        sentence_list = "".join(sentence)
        len_sentence_list = len(sentence_list)
        matched_probs = []
        dot = "."
        count = sentence_list.count(dot)
        for tokens, probs, pe in token_list[start_index:start_index + len_sentence_list]: 
            matched_probs.append((tokens, probs, pe))
            if dot in tokens:
                count -= 1
            if count == 0:
                break
        return start_index + len(matched_probs), matched_probs
    
    def merge_with_probs(self):
        """ merges the sentence with the probabilities"""
        df_gen_evidence = pd.read_pickle(self.gen_evidence_file)
        df_gen_evidence.drop_duplicates(subset="gen_evidence", inplace=True)
        df_sentence = pd.read_pickle(self.sentence_file)
        df_sentence = pd.merge(df_sentence, df_gen_evidence, on="gen_evidence", how="left").iloc[:10]
        df_sentence.rename(columns={"docs_x": "docs"}, inplace=True)
        #df_sentence.rename(columns={"token_details": "concat_probs"}, inplace=True)
        print(df_sentence['gen_evidence'].isna().sum())
        return df_sentence
    
    def concatenate_tokens_with_probs(self, tokens_probs):
        """ concatenates the tokens with the probabilities"""
        concatenated = []
        current_word = ""
        current_tokens = []  
        current_probs = []  
        current_pe = []
        for token, prob, pe in tokens_probs:
            if token.startswith(" "):  
                if current_word:
                    concatenated.append((current_word, current_tokens, current_probs, current_pe)) 
                current_word = token.strip() 
                current_tokens = [token]  
                current_probs = [prob]  
                current_pe = [pe]
            else:
                current_word += token  
                current_tokens.append(token)  
                current_probs.append(prob)  
                current_pe.append(pe)

        if current_word:
            concatenated.append((current_word, current_tokens, current_probs, current_pe))
        return concatenated
    
    def create_probs(self, df_sentence):
        """ gets the probabilities for the sentences"""
        df_grouped = df_sentence.groupby("gen_evidence")
        df_probs_sentences = pd.DataFrame(columns=["output_sentence", "label_sentence", "gen_evidence", "concat_probs_sentence", "docs"])

        index = 0
        for gen_evidence, group in df_grouped:
            output_list = group['output_sentence'].tolist()
            
            #concat_probs = group['token_details'].tolist()
            if group['concat_probs'].isna().sum() > 0:
                continue

            concat_probs = [[item['token'], item['generated_token_prob_from_transition_score'], item['token_wise_predictive_entropy']] for item in group['concat_probs'].tolist()[0]]
            
            # llama
            if self.model_name == "llama":
                concat_probs = concat_probs[4:-6]
                concat_probs = [tuple(item) for item in concat_probs]
                concat_probs = self.concatenate_tokens_with_probs(concat_probs)
            # phi
            elif self.model_name == "phi":
                concat_probs = [tuple(d) for d in concat_probs]
                concat_probs = concat_probs[5:-8]

            
            label_list = group['label_sentence'].tolist()
            docs = group['docs'].tolist()[0]
            claim = group['claim'].tolist()[0]
            sentences = self.nlp_processor.convert_text_to_sentences(gen_evidence)
            for output, label in zip(output_list, label_list):
                if len(sentences) > 1:
                    matched_probs_list = []
                    cosine_scores = []
                    start_index = 0
                    for sentence in sentences:
                        # llama
                        if self.model_name == "llama":
                            start_index, matched_probs = self.match_tokens_llama(start_index, concat_probs, sentence)
                        # phi
                        elif self.model_name == "phi":
                            start_index, matched_probs = self.match_tokens_phi(start_index, concat_probs, sentence)

                        matched_probs_list.append(matched_probs)
                        tfidf_matrix = self.vectorizer.fit_transform([output, sentence])
                        cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                        cosine_scores.append(cosine_score)

                    index_cosine = cosine_scores.index(max(cosine_scores))
                    concat_probs_sentence = matched_probs_list[index_cosine]
                else:
                    concat_probs_sentence = concat_probs
                
                df_probs_sentences = pd.concat([df_probs_sentences, pd.DataFrame({"output_sentence": [output], "label_sentence" : [label], "gen_evidence" : [gen_evidence], "concat_probs_sentence" : [concat_probs_sentence], "docs" : [docs], 'claim' : [claim]})], axis=0, ignore_index=True)
            index += 1

        df_probs_sentences.reset_index(drop=True, inplace=True)
        return df_probs_sentences
    
    # inspirings from https://github.com/jinhaoduan/SAR 
    def get_token_importance(self, df_probs_sentences, batch_size=32):
        """ get token importance of the tokens by removing tokens and calculating the similarity"""

        df_token_importance = df_probs_sentences.copy()
        df_token_importance['token_importance'] = None
        df_token_importance['token_importance'] = df_token_importance['token_importance'].astype(object)

        for sample_idx, row in df_probs_sentences.iterrows():
            generated_text = row['output_sentence']
            concat_probs_sentence = row['concat_probs_sentence']
          
          
            # llama
            if self.model_name == "llama":
                tokens = [tokens[1] for tokens in concat_probs_sentence]
                tokens = [item for sublist in tokens for item in sublist]
                words = [tokens[0] for tokens in concat_probs_sentence]
                
            # phi 
            elif self.model_name == "phi":
                tokens = [tokens[0] for tokens in concat_probs_sentence]

            token_importance = []
            replaced_sentences = [
                generated_text.replace(token, '') for token in tokens
            ]

            # Batch predictions
            for i in range(0, len(replaced_sentences), batch_size):
                batch = replaced_sentences[i:i + batch_size]
                similarity_to_original = self.nlp_processor.roberta_measure_model.predict(
                    [(generated_text, sentence) for sentence in batch]
                )

                # Calculate token importance
                batch_importance = [1 - torch.tensor(similarity) for similarity in similarity_to_original]
                token_importance.extend(batch_importance)

            token_importance = torch.tensor(token_importance).reshape(-1)
            df_token_importance.at[sample_idx, 'token_importance'] = token_importance.tolist()
            print(df_token_importance.at[sample_idx])
        df_token_importance.reset_index(drop=True, inplace=True)
        return df_token_importance
    
class NLP:
    def __init__(self, roberta_model_path, roberta_measure_model):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe(
            "fastcoref",
            config={
                'model_architecture': 'LingMessCoref',
                'model_path': 'biu-nlp/lingmess-coref',
                'device': 'cuda'
            }
        )

        self.roberta_model_path = roberta_model_path
        self.roberta_measure_model = roberta_measure_model


    def convert_text_to_sentences(self, text):
        """ converts the text to sentences"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences
        

if __name__ == "__main__":
    # Set the model path
    roberta_model_path = "" 
    roberta_measure_model = CrossEncoder(model_name=roberta_model_path, num_labels=1)
    vectorizer = TfidfVectorizer()
    nlp_processor = NLP(roberta_model_path, roberta_measure_model)

    dataset = "hover"
    model_name = "llama"
    # we only generate the probs for the test claims
    dataset_builder = DatasetBuilder(nlp_processor, vectorizer, f"test_evidence_with_claims/test_{dataset}_{model_name}.pkl", 
                                     f"datasets_{dataset}_{model_name}/sentence_with_bart_{dataset}.pkl", model_name=model_name)
    
    df_sentence = dataset_builder.merge_with_probs()
    df_probs_sentences = dataset_builder.create_probs(df_sentence)
    
    #if not os.path.exists(f"datasets_{dataset}"):
    #    os.makedirs(f"datasets_{dataset}")
    #df_probs_sentences.to_pickle(f"probs_test_{model_name}/probs_sentence_{dataset}.pkl")

    df_token_importance = dataset_builder.get_token_importance(df_probs_sentences)
    #df_token_importance.to_pickle(f"probs_test_{model_name}/probs_sentence_{dataset}_with_token_importance.pkl")



