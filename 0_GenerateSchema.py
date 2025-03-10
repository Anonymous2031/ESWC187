import json
import argparse

def construct_ontology_schema(input_file, output_file):
    # Step 1: Initialize an empty set to store unique triplets
    S = set()
    
    # Step 2-5: Read dataset and extract triplets
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
        for sample in dataset:
            subject_type = sample["subj_type"]
            object_type = sample["obj_type"]
            relation = sample["relation"]
            
            S.add((subject_type, object_type, relation))
    
    # Step 6: Initialize an empty dictionary for ontology
    ontology = {}
    
    # Step 7-17: Construct ontology from triplets
    for subject_type, object_type, relation in S:
        if subject_type not in ontology:
            ontology[subject_type] = {}
        
        if object_type not in ontology[subject_type]:
            ontology[subject_type][object_type] = []
        
        if relation not in ontology[subject_type][object_type]:
            ontology[subject_type][object_type].append(relation)
    
    # Step 18: Convert ontology to JSON format and save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ontology, f, indent=4)
    
    print(f"Ontology schema saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ontology schema from dataset")
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to input JSON dataset")
    parser.add_argument("--output_schema", type=str, required=True, help="Path to output JSON schema")
    args = parser.parse_args()
    
    construct_ontology_schema(args.input_dataset, args.output_schema)