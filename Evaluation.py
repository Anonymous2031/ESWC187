import pandas as pd
import scorer
import argparse

def evaluate_predictions(final_predictions_file):
    # Step 1: Load the data
    df = pd.read_csv(final_predictions_file)
    
    # Step 2: Extract gold labels, initial predictions, post-ontology predictions, and final predictions
    gold_labels = df['True_Labels'].tolist()
    initial_predictions = df['Initial_Predictions'].tolist()
    ontology_predictions = df['Post-Ontology Predictions'].tolist()
    final_predictions = df['Final_Predictions'].tolist()
    
    # Evaluate initial predictions
    p, r, f1_initial = scorer.score(gold_labels, initial_predictions, verbose=True)
    print(f"Initial predictions evaluation: Precision: {p:.2%}, Recall: {r:.2%}, F1: {f1_initial:.2%}")
    
    # Evaluate post-ontology predictions
    p, r, f1_ontology = scorer.score(gold_labels, ontology_predictions, verbose=True)
    print(f"Post-Ontology predictions evaluation: Precision: {p:.2%}, Recall: {r:.2%}, F1: {f1_ontology:.2%}")
    
    # Evaluate final predictions (after LLM validation)
    p, r, f1_final = scorer.score(gold_labels, final_predictions, verbose=True)
    print(f"Final predictions evaluation: Precision: {p:.2%}, Recall: {r:.2%}, F1: {f1_final:.2%}")
    
    # Calculate impact of Ontology and LLM validation
    impact_ontology = f1_ontology - f1_initial
    impact_llm = f1_final - f1_ontology
    
    print(f"\nImpact of Ontology Validation: {impact_ontology:.2%} F1 change")
    print(f"Impact of LLM Validation: {impact_llm:.2%} F1 change")
    print(f"Total Change in F1 Score: {f1_final - f1_initial:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate final predictions and measure component impact.")
    parser.add_argument("--final_predictions", type=str, required=True, help="Path to the final predictions CSV file")
    args = parser.parse_args()
    
    evaluate_predictions(args.final_predictions)
