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

