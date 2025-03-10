# Project Title

Official Implementation for relation extraction using ontology and large language models (LLMs).

## Description

This project focuses on enhancing relation extraction tasks by integrating ontology validation and LLM validation. The workflow involves generating schemas, validating predictions against ontological structures, refining predictions using LLMs, and evaluating the final outcomes.

## Dataset

The TACRED dataset can be obtained from [this link](https://nlp.stanford.edu/projects/tacred/). The Re-TACRED dataset can be obtained following the instructions in [Re-TACRED](https://github.com/gstoica27/Re-TACRED). The expected structure of files is:
```
Initial_Predictions
 |-- Dataset
 |    |-- tacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- retacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
```


## Pretrained models 

Initial predictions are generated using pretrained models trained on the **TACRED** and **ReTACRED** datasets, following the architecture and methodology from the [RE Improved Baseline](https://github.com/wzhouad/RE_improved_baseline) repository. If you wish to train new models, you can follow the instructions in that repository, or you can directly use the pretrained models provided in this [Google Drive link](https://github.com/wzhouad/RE_improved_baseline). These models are based on a **RoBERTa-large encoder** fine-tuned on TACRED and ReTACRED for relation extraction.The expected structure of files is:
```
Initial_Predictions
 |-- Models
 |    |-- tacred
 |    |    |-- RoBERTa_TACRED.json        
 |    |-- retacred
 |    |    |-- RoBERTa_ReTACRED.json        
```

## Code Structure

The repository is organized as follows:

- **Figures/**: Contains visual representations and figures related to the project.
- **Initial_Predictions/**:
  - **Dataset/**: Houses the initial datasets used for predictions.
  - **Predictions/**: Stores the initial prediction results.
- **Prompts/**: Includes prompt templates for LLM validation.
- **Revised_Predictions/**: Contains predictions refined after validation processes.
- **Schemas/**: Stores schema files generated and used during the project.
- **0_GenerateSchema.py**: Script to generate schema from the input dataset.
- **1_Ontology_Validation.py**: Validates initial predictions against the generated schema.
- **2_LLM_Validation.py**: Refines predictions using large language models.
- **Evaluation.py**: Evaluates the final predictions.
- **README.md**: This file, providing an overview of the project.
- **RelCheck_Example(RoBERTaModel).ipynb**: Jupyter notebook example using the RoBERTa model.
- **Requirements.txt**: Lists the necessary Python packages and dependencies.
- **scorer.py**: Contains scoring functions used in the evaluation process.

## How to Run Each Module

Before running the modules, ensure all dependencies are installed:

```bash
pip install -r Requirements.txt
```

### 1. Generating the Schema

```bash
python3 0_GenerateSchema.py --input_dataset Initial_Predictions/Dataset/ReTACRED/train.json --output_schema Schemas/ReTACRED_Schema.json
```

### 2. Ontology-Based Validation

```bash
python3 1_Ontology_Validation.py --input_predictions  ./Initial_Predictions/Predictions/Initial_predictions.csv                                  --ontology_schema Schemas/ReTACRED_Schema.json                                  --threshold 1                                  --output_validated  ./Revised_Predictions/Post-Ontology_Predictions.csv
```

### 3. LLM-Based Validation

```bash
python3 2_LLM_Validation.py --input_predictions Revised_Predictions/Post-Ontology_Predictions.csv                             --ontology_schema Schemas/ReTACRED_Schema.json                             --relations_spans Schemas/ReTACRED_Spans.json                             --prompt Prompts/qa4re_without_types.txt                             --api_key API_KEY.json                             --model gpt-4o                             --threshold 0.8                             --output_predictions Revised_Predictions/Final_predictions.csv
```

### 4. Evaluation of Final Predictions

```bash
python3 Evaluation.py --final_predictions Revised_Predictions/Final_predictions.csv
```



