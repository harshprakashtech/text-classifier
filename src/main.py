import numpy as np
import torch

from core import text_classifier_with_torch, encode_text, get_sigmoid, get_vocabulary
from data import training_data

# Create a vocabulary using training_data
vocabulary: dict[str, int] = get_vocabulary(training_data)

def main():
    # Train the classifier, and get weights
    # weights: np.ndarray = text_classifier(training_data, vocabulary)
    model: torch.nn.Module = text_classifier_with_torch(training_data, vocabulary)
    
    # Test the classifier
    for sentence, truth in training_data:
        # Encode each sentence
        # encoded_text: np.ndarray = encode_text(sentence, vocabulary)
        encoded_text = torch.tensor(encode_text(sentence, vocabulary), dtype=torch.float32)

        # Calculate first prediction
        # first_prediction: float = get_sigmoid(np.dot(encoded_text, weights))
        prediction = torch.sigmoid(model(encoded_text))

        
        # Print result
        # print(f"Text: {sentence} — Truth: {truth} — Prediction: {first_prediction:.4f}")
        print(f"Text: {sentence} — Truth: {truth} — Prediction: {prediction.item():.4f}")
        
main()