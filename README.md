This is the repository for the research: 
Addressing Hallucination in Large Language Models: Detection and
Mitigation Techniques for Claim-Based Verification

To download LLMs create an account on https://huggingface.co/
Maybe the access to Llama has to be requested first: https://huggingface.co/meta-llama and https://llama.meta.com/llama-downloads
Also the BART-Large-MNLI model has to be downloaded: https://huggingface.co/facebook/bart-large-mnli


To install pytorch go to https://pytorch.org/ and follow the instructions (2.5.1 cu118)
then install the other packages with: pip install -r requirements.txt
Note: for windows the library bitsandbytes does not work so remove it from requirements.txt first and -> go to: 
https://github.com/d8ahazard/sd_dreambooth_extension/issues/7 
and type pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl 
also for windows you need to install pip install unicodedata2==15.1.0 if you want to generate evidence 

The repository is divided into three parts: 
- Probes
- Mitigation 
- Entity Analysis

## Probes:
### To create the dataset for the probes go to Probes/

- download https://1drv.ms/u/s!AhVn8Lx_iIapipk3hFU1JRpMosF--g?e=llFsno (raw FEVER and HoVer datasets + already generated files)

- python generate_evidence.py 
(creates evidence text from HoVer and FEVER with an LLM)

- python generate_mini_facts_and_sentences.py 
(creates the labelled mini-facts from the evidence, generates the sentences and removes sentences that are not aligned with BART-large-MNLI)

- python generate_embeddings.py 
(generates the embeddings of mini-facts and sentences)

- python train_test_split_balance.py
(splits the datasets and balances them)

- python TrainProbes.py
(trains the probes with a feedforward neural network)

- go to model_test.ipynb to run the evaluation

### To run the results from the existing probes ...
- download the dataset with embeddings (8 GB): https://1drv.ms/u/s!AhVn8Lx_iIapipkfm8s7rjAHUdhQLQ?e=nX4arh 
(make sure the folders inside the zip are insides Probes/, the embeddings are indexed from the last hidden layer: -1, -8, -16, -24 and the first layer (1), this corresponds with 32, 25, 17, 9 and 1)

- download the probes (.pth files): https://1drv.ms/f/s!AhVn8Lx_iIapipkgKEms9E6yilRC5A?e=rwuBT8 (make sure the folder is inside Probes/)
- go to model_test.ipynb to run the evaluation

### To run the evaluation for the baselines
- download https://1drv.ms/u/s!AhVn8Lx_iIapipk2ydzSyocV3ZZhJg?e=YLcEo7 (make sure the folders are insides Probes/)
- go to evaluate_probs.ipynb


## Finetuning 
### To recreate the dataset for finetuning go to Mitigation/:
- download https://1drv.ms/f/s!AhVn8Lx_iIapipk6-VBvpNUn3pPbIw?e=NpvXF6 
(these are the labelled mini-facts with their associated evidence, make sure the files are inside Mitigation/)

- python generate_corrections.py 
(pre-generation of corrections with OpenAI)

- python train_test_split.py
(splits the dataset in train, dev and test)

- python create_train_dataset.py
(for train and dev: removes samples that are not aligned with BART-Large-MNLI and balances)

- go to finetuning.ipynb to finetune the LLM

- python evaluate.py 
(evaluation of LLMs and the finetuned LLM with BART-Large-MNLI)

### to get the already finetuned LLM (make sure to change the path to your base model in adapter_config.json)
- download https://1drv.ms/u/s!AhVn8Lx_iIapiplTSTcnfV0ZfQz2-g?e=9fUSoi 

### to get the already created datasets for finetuning
- download https://1drv.ms/u/s!AhVn8Lx_iIapipk9fqq85l-4SFZepg?e=5fA4mM (make sure every folder of the zip is inside Mitigation/)


## Application 
### to run the application scenario go to folder Mitigation/: 
- download https://1drv.ms/f/s!AhVn8Lx_iIapiplAmL3VAzH2Sxp0Rw?e=zQcpOZ (make sure folder application is inside Mitigation/)

- python application.py 
(make sure to select a probe and the fine-tuned LLM)

- go to evaluate_application.ipynb to evaluate the approaches

## Entity Analysis
### to recreate the real train samples
- go to gen_real_samples.ipnyb


### to run the evaluation of the probes on the test datasets go to Entity Analysis/
- download https://1drv.ms/u/s!AhVn8Lx_iIapiplQt1lFdOiEcJmxZw?e=DNyGw8 (make sure every folder is inside Entity Analysis)
- run model_test.ipynb if you want to test the probes on the test datasets
- run results.ipynb if you want to plot the probability distributions


### to get the already created dataset
- download https://1drv.ms/u/s!AhVn8Lx_iIapiplROt5adNLKBiQ-xw?e=a6TfhQ 



Note: Some coding was conducted with the help of Github Copilot, 
inspirings for finetuning was taken from https://www.kaggle.com/code/zivicmilos/llm-finetuning, 
for token sar from https://github.com/jinhaoduan/SAR,  
for the probes from https://github.com/balevinstein/Probes








