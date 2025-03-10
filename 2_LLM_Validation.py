import json
import pandas as pd
import argparse
import time
import openai
import asyncio

# Function to load API key from a JSON file
def load_api_key(api_key_file):
    with open(api_key_file, 'r', encoding='utf-8') as file:
        api_data = json.load(file)
        return api_data.get("api_key", "")

# Function to load ontology schema
def load_ontology(ontology_file):
    with open(ontology_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to load relation spans
def load_relation_spans(spans_file):
    with open(spans_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to get possible relations from the schema
def get_possible_relations(subject_type, object_type, schema):
    return schema.get("relations", {}).get(subject_type, {}).get(object_type, ["no_relation"])

# Function to load prompt from a text file
def load_prompt(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Asynchronous function to validate relationships with GPT
async def validate_relationships_with_gpt_async(data, model, prompt, api_key):
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key) 
    responses = []

    async def query_gpt(item):
        try:
            formatted_prompt = prompt.format(
                sentence=item['sentence'],
                entity1=item['entity1'],
                relationship_span=item['relationship_span'],
                entity2=item['entity2']
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0,
                    max_tokens=10,
                    top_p=1.0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["\n"]
                )
            )
            answer = response.choices[0].message.content.strip().lower()
            return 1 if answer == "yes" else 0
        except Exception as e:
            print(f"Error querying GPT: {e}")
            return -1  # -1 indicates an error occurred

    responses = await asyncio.gather(*(query_gpt(item) for item in data))
    return responses

def validate_predictions_with_llm(input_csv, ontology_file, spans_file, prompt_file, api_key_file, model, threshold, output_csv):
    # Load ontology, spans, prompt, and API key
    ontology = load_ontology(ontology_file)
    rel2span = load_relation_spans(spans_file)
    prompt = load_prompt(prompt_file)
    api_key = load_api_key(api_key_file)
    
    # Load predictions
    df = pd.read_csv(input_csv)
    df['GPT_Validation'] = -1
    df['Final_Predictions'] = df['Post-Ontology Predictions']
    
    # Filter data for validation
    data_to_validate = [{
        "index": index,
        "sentence": row['Tokens'],
        "entity1": row['Subject_Entity'],
        "subject_type": row['Subject_Type'],
        "entity2": row['Object_Entity'],
        "object_type": row['Object_Type'],
        "relationship": row['Post-Ontology Predictions'],
        "relationship_span": rel2span.get(row['Post-Ontology Predictions'], ""),
        "possible_relations": get_possible_relations(row['Subject_Type'], row['Object_Type'], ontology)
    } for index, row in df.iterrows() if row['Post-Ontology Predictions'] != 'no_relation' and row['Confidence'] < threshold]
    
    print(f"Validating {len(data_to_validate)} predictions with LLM...")
    
    async def run_async():
        return await validate_relationships_with_gpt_async(data_to_validate, model, prompt, api_key)
    
    results = asyncio.run(run_async())
    
    # Update DataFrame based on LLM validation
    for result, data in zip(results, data_to_validate):
        original_index = data['index']
        df.at[original_index, 'GPT_Validation'] = result
        if result == 0:  # If LLM deems incorrect
            df.at[original_index, 'Final_Predictions'] = 'no_relation'
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"LLM validation completed. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Based Validation for Predictions")
    parser.add_argument("--input_predictions", type=str, required=True, help="Path to ontology validated predictions CSV file")
    parser.add_argument("--ontology_schema", type=str, required=True, help="Path to ontology schema JSON file")
    parser.add_argument("--relations_spans", type=str, required=True, help="Path to relation spans JSON file")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt text file")
    parser.add_argument("--api_key", type=str, required=True, help="Path to API key JSON file")
    parser.add_argument("--model", type=str, required=True, help="LLM model to use (e.g., 'gpt-4o')")
    parser.add_argument("--threshold", type=float, required=True, help="Confidence threshold for validation")

    parser.add_argument("--output_predictions", type=str, required=True, help="Path to output CSV file with final predictions")
    args = parser.parse_args()
    
    validate_predictions_with_llm(
        args.input_predictions,
        args.ontology_schema,
        args.relations_spans,
        args.prompt,
        args.api_key,
        args.model,
        args.threshold,
        args.output_predictions
    )
