import pandas as pd
import json
import argparse
import time

def load_ontology(ontology_file):
    with open(ontology_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def check_ontology_validation(subject_type, object_type, prediction, ontology):
    if subject_type in ontology and object_type in ontology[subject_type]:
        return prediction in ontology[subject_type][object_type]
    return False  # Consider it a violation if not found in ontology

def validate_predictions(input_csv, ontology_file, output_csv, threshold):
    # Load ontology schema
    ontology = load_ontology(ontology_file)
    
    # Load initial predictions
    df = pd.read_csv(input_csv)
    
    # Initialize new columns
    df['Ontology_Violation'] = False
    df['Post-Ontology Predictions'] = df['Initial_Predictions']
    
    start_time = time.time()
    violations = 0
    
    # Validate predictions only if confidence is below threshold
    for index, row in df.iterrows():
        if row['Confidence'] < threshold:
            valid = check_ontology_validation(row['Subject_Type'], row['Object_Type'], str(row['Initial_Predictions']), ontology)
            df.at[index, 'Ontology_Violation'] = not valid
            if not valid:
                violations += 1
                df.at[index, 'Post-Ontology Predictions'] = 'no_relation'
    
    end_time = time.time()
    
    # Save results
    df.to_csv(output_csv, index=False)
    
    print(f"Finished with {violations} ontology violations detected")
    print(f"Time taken for validation: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ontology Validation for Predictions")
    parser.add_argument("--input_predictions", type=str, required=True, help="Path to input CSV file with predictions")
    parser.add_argument("--ontology_schema", type=str, required=True, help="Path to ontology schema JSON file")
    parser.add_argument("--output_validated", type=str, required=True, help="Path to output CSV file with ontology validation")
    parser.add_argument("--threshold", type=float, required=True, help="Confidence threshold for validation")
    args = parser.parse_args()
    
    validate_predictions(args.input_predictions, args.ontology_schema, args.output_validated, args.threshold)