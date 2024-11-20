import pandas as pd
from llm import LLM  
import sqlite3
import unicodedata2
import os

class HoverEvidenceGenerator:
    """ this class generates evidence for the Hover dataset"""
    def __init__(self, llm, db_path, hover_files, number_few_shot, max_number_claim_evidence, model_name, only_test_claims=False):
        self.llm = llm
        self.db_path = db_path
        self.number_few_shot = number_few_shot
        self.max_number_claim_evidence = max_number_claim_evidence
        self.model_name = model_name
        self.only_test_claims = only_test_claims

        self.claim_evidence1_file = hover_files['claim_evidence1_file']
        self.claim_evidence2_file = hover_files['claim_evidence2_file']

        self.few_shot_data = pd.read_json(hover_files["few_shot"])
        self.hops_2_supports_few_shot = self.few_shot_data.loc[(self.few_shot_data['hops'] == 2) & (self.few_shot_data['label'] == 'SUPPORTS')]
        self.hops_3_supports_few_shot = self.few_shot_data.loc[(self.few_shot_data['hops'] == 3) & (self.few_shot_data['label'] == 'SUPPORTS')]
        self.hops_4_supports_few_shot = self.few_shot_data.loc[(self.few_shot_data['hops'] == 4) & (self.few_shot_data['label'] == 'SUPPORTS')]

        self.hops_2_refutes_few_shot = self.few_shot_data.loc[(self.few_shot_data['hops'] == 2) & (self.few_shot_data['label'] == 'REFUTES')]
        self.hops_3_refutes_few_shot = self.few_shot_data.loc[(self.few_shot_data['hops'] == 3) & (self.few_shot_data['label'] == 'REFUTES')]
        self.hops_4_refutes_few_shot = self.few_shot_data.loc[(self.few_shot_data['hops'] == 4) & (self.few_shot_data['label'] == 'REFUTES')]
        self.prompt_df = pd.DataFrame(columns=['claim', 'claim_label', 'evidence', 'hops'])

    def query_wiki(self, doc_title):
        """ queries the wiki database for the docs"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        para = (
            c.execute("SELECT text FROM documents WHERE id = ?", (unicodedata2.normalize('NFD', doc_title),)).fetchall()
        )[0][0]
        conn.close()
        return para
    
    def generate_prompt_option(self, supports_option, hops_option, old_hops, old_supports):
        """ generates the prompt options """
        if supports_option == "SUPPORTS+REFUTES":
            if old_supports == "SUPPORTS":
                new_supports = "REFUTES"
            else:
                new_supports = "SUPPORTS"
        else:
            new_supports = supports_option
        if hops_option == "mixed":
            if old_hops == 2:
                new_hops = 3
            elif old_hops == 3:
                new_hops = 4
            else:
                new_hops = 2
        else:
            new_hops = hops_option
        return new_supports, new_hops
    
    def select_row_from_prompt_df(self, supports, hops, ending):
        """ selects the row from the prompt dataframe"""
        df_name = f"hops_{hops}_{supports.lower()}_{ending}"
        df = getattr(self, df_name)
        selected_row = df.iloc[0:1]
        if df.empty:
            raise Exception("Not enough samples in the dataframe. Please choose a lower number of examples/generations")
        df = df.iloc[1:]
        setattr(self, df_name, df)
        return selected_row
    
    def get_prompt(self, supports_option, hops_option):
        """ generates the prompt"""
        if supports_option == "SUPPORTS+REFUTES":
            old_supports = "REFUTES"
        else:
            old_supports = supports_option
        if hops_option == "mixed": 
            old_hops = 4
        else:
            old_hops = hops_option
        for example_index in range(self.number_few_shot):
            new_supports, new_hops = self.generate_prompt_option(supports_option, hops_option, old_hops, old_supports)
            selected_row = self.select_row_from_prompt_df(new_supports, new_hops, ending="few_shot")
            old_hops, old_supports = new_hops, new_supports
            self.prompt_df = pd.concat([self.prompt_df, selected_row], ignore_index=True)
        prompt = ""
        for index, entry in self.prompt_df.iterrows():
            claim = entry['claim']
            if self.model_name == "phi":
                prompt += ' CLAIM: ' + str(claim) +  ' EVIDENCE: ' + " ".join(entry['evidence']) + ' EVIDENCE_END . ###' 
            else:
                prompt += ' CLAIM: ' + str(claim) +  ' EVIDENCE: ' + " ".join(entry['evidence']) + ' EVIDENCE_END .###' 
        return prompt


    def get_evidence(self, supports_option, hops_option, hops_data, used_docs, df_generated_evidence):
        """ generates the evidence of the claims"""
        prompt = self.get_prompt(supports_option=supports_option, hops_option=hops_option)
        for index, row in hops_data.iterrows():
            print(index)

            # Skip if the document has already been used
            if self.only_test_claims:
                pass
            else:
                skip = False
                docs = row['supporting_facts']
                docs = [item[0] for item in docs] 
                docs = list(set(docs))  
                for doc in docs:
                    if doc in used_docs:
                        skip = True
                if skip:
                    if self.max_number_claim_evidence // 4 == index:
                        break
                    else:
                        continue
                else:
                    used_docs.extend(docs) 

            new_prompt = prompt + " CLAIM: " + row["claim"]
            output, token_details = self.llm.call_text_llm(new_prompt)
            output = output.replace("EVIDENCE: ", "").replace("EVIDENCE_END.", "").replace("EVIDENCE_END .", "").replace("###", "").strip()
            print(output)
            if self.only_test_claims:
                doc_titles = [title for title in row['docs']]
                doc_titles = list(set(doc_titles))
                ground_truth_source = ""
            else:
                doc_titles = [title for title, index in row['supporting_facts']]
                doc_titles = list(set(doc_titles))
                ground_truth_source = [self.query_wiki(doc_title) for doc_title in doc_titles]
            df_generated_evidence = pd.concat([df_generated_evidence, pd.DataFrame([[output, " ".join(ground_truth_source), token_details, doc_titles]],
                                                           columns=["gen_evidence", "ground_truth_source", "token_details", "docs"])], 
                                                           ignore_index=True)
            
            if self.max_number_claim_evidence // 4 == index:
                break
        return df_generated_evidence, used_docs
    
    def run_hover_hops_2(self):
        """ generates evidence for the Hover dataset with 2 hops"""

        # if only testing claims and not all claims
        if self.only_test_claims:
            raw_data = pd.read_pickle(self.claim_evidence1_file)
            hops2 = raw_data[raw_data['docs'].apply(len) == 2].reset_index(drop=True)
            df_generated_evidence = pd.DataFrame()
            used_docs = []
            df_generated_evidence, used_docs = self.get_evidence(supports_option="SUPPORTS+REFUTES", hops_option=2, hops_data=hops2, used_docs=used_docs, df_generated_evidence=df_generated_evidence)
        else:
            raw_data = pd.read_json(self.claim_evidence1_file)
            hops2_1 = raw_data.loc[raw_data['num_hops'] == 2].reset_index(drop=True)
            
            raw_data = pd.read_json(self.claim_evidence2_file)
            hops2_2 = raw_data.loc[raw_data['num_hops'] == 2].reset_index(drop=True)
            
            df_generated_evidence = pd.DataFrame()
            used_docs = []
            df_generated_evidence, used_docs = self.get_evidence(supports_option="SUPPORTS+REFUTES", hops_option=2, hops_data=hops2_1, used_docs=used_docs, df_generated_evidence=df_generated_evidence)
            
            df_generated_evidence, used_docs = self.get_evidence(supports_option="SUPPORTS+REFUTES", hops_option=2, hops_data=hops2_2, used_docs=used_docs, df_generated_evidence=df_generated_evidence)
        return df_generated_evidence
    
    def run_hover_hops_3(self):
        """ generates evidence for the Hover dataset with 3 hops"""

        if self.only_test_claims:
            raw_data = pd.read_pickle(self.claim_evidence1_file)
            hops3 = raw_data[raw_data['docs'].apply(len) == 3].reset_index(drop=True)
            df_generated_evidence = pd.DataFrame()
            used_docs = []
            df_generated_evidence, used_docs = self.get_evidence(supports_option="SUPPORTS+REFUTES", hops_option=3, hops_data=hops3, used_docs=used_docs, df_generated_evidence=df_generated_evidence)
        else:
            raw_data = pd.read_json(self.claim_evidence1_file)
            hops3_1 = raw_data.loc[raw_data['num_hops'] == 3].reset_index(drop=True)

            raw_data = pd.read_json(self.claim_evidence2_file)
            hops3_2 = raw_data.loc[raw_data['num_hops'] == 3].reset_index(drop=True)

            df_generated_evidence = pd.DataFrame()
            used_docs = []
            df_generated_evidence, used_docs = self.get_evidence(supports_option="SUPPORTS+REFUTES", hops_option=3, hops_data=hops3_1, used_docs=used_docs, df_generated_evidence=df_generated_evidence)

            df_generated_evidence, used_docs = self.get_evidence(supports_option="SUPPORTS+REFUTES", hops_option=3, hops_data=hops3_2, used_docs=used_docs, df_generated_evidence=df_generated_evidence)
            df_generated_evidence.reset_index(drop=True, inplace=True)
        return df_generated_evidence


class FeverEvidenceGenerator:
    """ this class generates evidence for the Fever dataset"""
    def __init__(self, llm, fever_files, number_few_shot, number_claim_evidence, model_name, only_test_claims):
        self.llm = llm
        self.few_shot_file = fever_files['few_shot']
        self.claim_evidence_file = fever_files['claim_evidence']
        self.number_few_shot = number_few_shot
        self.number_claim_evidence = number_claim_evidence
        self.model_name = model_name
        self.only_test_claims = only_test_claims

    def get_fever_samples(self):
        """ gets the fever samples for the few shot"""
        df_few_shot = pd.read_json(self.few_shot_file, lines=True)
        df_few_shot.drop_duplicates(subset=["claim"], inplace=True)
        df_few_shot = df_few_shot.iloc[:self.number_few_shot]

        if self.only_test_claims:
            df_claim_evidence = pd.read_pickle(self.claim_evidence_file)
        else:
            df_claim_evidence = pd.read_json(self.claim_evidence_file, lines=True)
        df_claim_evidence.drop_duplicates(subset=["claim"], inplace=True, ignore_index=True)
        df_claim_evidence = df_claim_evidence.iloc[:self.number_claim_evidence]
        return df_few_shot, df_claim_evidence

    def generate_few_shot(self, df_few_shot):
        """ generates the few shot prompt"""
        prompt = ""
        for index, row in df_few_shot.iterrows():
            claim = row["claim"]
            resolved_evidence = row["resolved_evidence"]
            resolved_evidence = " ".join(resolved_evidence)
            if self.model_name == "phi":
                prompt += f" CLAIM: {claim} EVIDENCE: {resolved_evidence} EVIDENCE_END . ###"
            else:
                prompt += f" CLAIM: {claim} EVIDENCE: {resolved_evidence} EVIDENCE_END.###"
        return prompt.strip()

    def get_evidence(self, df_few_shot, df_claim_evidence):
        """ generates the evidence for the claims"""
        df_gen_evidence = pd.DataFrame(columns=["gen_evidence", "docs", "ground_truth_source", "token_details"])
        prompt = self.generate_few_shot(df_few_shot)
        for index, row in df_claim_evidence.iterrows():
            claim = row["claim"]
            if self.only_test_claims:
                ground_truth_source = ""
            else:
                ground_truth_source = " ".join(row["whole_sources"])
            docs = row["docs"]
            prompt_with_new_claim = prompt + f" CLAIM: {claim}"
            output, token_details = self.llm.call_text_llm(prompt_with_new_claim)
            output = output.replace("EVIDENCE_END.###", "").replace("EVIDENCE:", "").replace("###", "").replace("EVIDENCE_END .", "").strip()
            print(output)
            df_gen_evidence = pd.concat([df_gen_evidence, pd.DataFrame([[output, docs, ground_truth_source, token_details]], columns=["gen_evidence", "docs", "ground_truth_source", "token_details"])])
        return df_gen_evidence
    
    def run_fever(self):
        """ generates the evidence for the Fever dataset"""
        df_few_shot, df_claim_evidence = self.get_fever_samples()
        df_generated_evidence = self.get_evidence(df_few_shot, df_claim_evidence)
        df_generated_evidence.reset_index(drop=True, inplace=True)
        return df_generated_evidence



if __name__ == "__main__":
    
    # insert the model path and name
    model_path = ""
    model_name = "llama"

    llm = LLM(model_path, model_name=model_name)

    fever_files = {
        "few_shot": "datasets_llama_fever/fever_claim_evidence_few_shot.jsonl",
        "claim_evidence": "datasets_llama_fever/fever_claim_evidence_generate.jsonl"
    }
    fever_generator = FeverEvidenceGenerator(llm, fever_files, number_few_shot=5, number_claim_evidence=1, model_name=model_name, only_test_claims=True)
    df_generated_evidence = fever_generator.run_fever()
    #if not os.path.exists("recreate_datasets_llama_fever"):
    #    os.makedirs("recreate_datasets_llama_fever")
    #df_generated_evidence.to_pickle("recreate_datasets_llama_fever/gen_evidence_fever.pkl")


    # if only test claims are used
    """
    db_path = 'datasets_llama_hover/wiki_wo_links.db'
    hover_files = {
        'few_shot': 'datasets_hover/hover_claim_evidence_few_shot.json',
        'claim_evidence1_file': 'test_evidence_with_claims/test_hover.pkl',
        'claim_evidence2_file': None
        #'claim_evidence1_file': 'datasets_llama_hover/hover_claim_evidence_generate1.json',
        #'claim_evidence2_file': 'datasets_llama_hover/hover_claim_evidence_generate2.json'
    }
    """

    db_path = 'datasets_llama_hover/wiki_wo_links.db'
    hover_files = {
        'few_shot': 'datasets_hover/hover_claim_evidence_few_shot.json',
        'claim_evidence1_file': 'datasets_llama_hover/hover_claim_evidence_generate1.json',
        'claim_evidence2_file': 'datasets_llama_hover/hover_claim_evidence_generate2.json'
    }


    hover_generator = HoverEvidenceGenerator(llm, db_path, hover_files, number_few_shot=4, max_number_claim_evidence=1, model_name=model_name, only_test_claims=False)
    df_generated_evidence_hops2 = hover_generator.run_hover_hops_2()

    hover_generator = HoverEvidenceGenerator(llm, db_path, hover_files, number_few_shot=4, max_number_claim_evidence=1, model_name=model_name, only_test_claims=False)
    df_generated_evidence_hops3 = hover_generator.run_hover_hops_3()
    #if not os.path.exists("recreate_datasets_llama_hover"):
    #    os.makedirs("recreate_datasets_llama_hover")
    #df_generated_evidence.to_pickle("recreate_datasets_llama_hover/gen_evidence_hover.pkl")
   








