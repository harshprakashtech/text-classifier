import numpy as np

from .encode import encode_text
from .vocabulary import get_vocabulary
from .sigmoid import get_sigmoid
from .loss_function import get_loss

# Create a text classifier
def text_classifier(training_data: list[tuple[str, int]], vocabulary: dict[str, int]) -> np.ndarray:
    # Initialize random weights
    weights: np.ndarray = np.random.randn(len(vocabulary)) * 0.01
    
    # Train the classifier over 100 times
    for epoch in range(100):
        total_loss: float = 0
        
        # Train the classifier over training data
        for sentence, truth in training_data:
            # Encode each sentence
            encoded_text: np.ndarray = encode_text(sentence, vocabulary)
            
            # Calculate first prediction
            prediction: float = get_sigmoid(np.dot(encoded_text, weights)) 
            
            # Calculate loss
            loss: float = get_loss(prediction, truth)
            
            # Calculate error
            error: float = prediction - truth
            
            # Update weights
            weights -= 0.1 * error * encoded_text
            
            # Update total loss
            total_loss += loss
            
        # Print loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch} — Loss: {total_loss/len(training_data):.4f}")
        
    return weights