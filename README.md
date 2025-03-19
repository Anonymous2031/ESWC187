# RelCheck

Official Implementation for RelCheck


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

Initial predictions are generated using pretrained models trained on the **TACRED** and **ReTACRED** datasets, following the architecture and methodology from the [RE Improved Baseline](https://github.com/wzhouad/RE_improved_baseline) repository. If you wish to train new models, you can follow the instructions in that repository, or you can directly use the pretrained models provided in this [Drive](https://drive.google.com/file/d/1z2Qmc1ikFdi0mGUx0jiL5IxU6g8UJo3E/view?usp=sharing). These models are based on a **RoBERTa-large encoder** fine-tuned on TACRED and ReTACRED for relation extraction.The expected structure of files is:
```
Initial_Predictions
 |-- Models
 |    |-- tacred
 |    |    |-- RoBERTa_TACRED.bin        
 |    |-- retacred
 |    |    |-- RoBERTa_ReTACRED.bin       
```


## How to Run Each Module

Before running the modules, ensure all dependencies are installed:

```bash
pip install -r Requirements.txt
```

### 1. Generating the Schema

```bash
python3 0_GenerateSchema.py 
        --input_dataset Initial_Predictions/Dataset/ReTACRED/train.json 
        --output_schema Schemas/ReTACRED_Schema.json
```

### 2. Ontology-Based Validation

```bash
python3 1_Ontology_Validation.py 
        --input_predictions  ./Initial_Predictions/Predictions/Initial_predictions.csv                                  
        --ontology_schema Schemas/ReTACRED_Schema.json                                  
        --threshold 1                                  
        --output_validated  ./Revised_Predictions/Post-Ontology_Predictions.csv
```

### 3. LLM-Based Validation

```bash
python3 2_LLM_Validation.py 
        --input_predictions Revised_Predictions/Post-Ontology_Predictions.csv                             
        --ontology_schema Schemas/ReTACRED_Schema.json                             
        --relations_spans Schemas/ReTACRED_Spans.json                             
        --prompt Prompts/qa4re_without_types.txt                             
        --api_key API_KEY.json                             
        --model gpt-4o                             
        --threshold 0.8                             
        --output_predictions Revised_Predictions/Final_predictions.csv
```

### 4. Evaluation of Final Predictions

```bash
python3 Evaluation.py 
        --final_predictions Revised_Predictions/Final_predictions.csv
```


## Remark  
We provided a **DATASET EXAMPLE** in this repository to illustrate the **data structure** at each stage. Only a small set of examples with a higher validation threshold is included—**only >99% confidence cases were kept, while others were re-evaluated.**



