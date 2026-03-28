import numpy as np

from core import text_classifier, encode_text, get_sigmoid, get_vocabulary
from data import training_data

# Create a vocabulary using training_data
vocabulary: dict[str, int] = get_vocabulary(training_data)

def main():
    # Train the classifier, and get weights
    weights: np.ndarray = text_classifier(training_data, vocabulary)
    
    # Test the classifier
    for sentence, truth in training_data:
        # Encode each sentence
        encoded_text: np.ndarray = encode_text(sentence, vocabulary)
        
        # Calculate first prediction
        first_prediction: float = get_sigmoid(np.dot(encoded_text, weights))
        
        # Print result
        print(f"Text: {sentence} — Truth: {truth} — Prediction: {first_prediction:.4f}")
        
        
main()