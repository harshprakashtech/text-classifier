import numpy as np

import torch
from torch import nn

from .encode import encode_text
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


# Create a text classifier
def text_classifier_with_torch(training_data: list[tuple[str, int]], vocabulary: dict[str, int]) -> np.ndarray:
    # Initialize random weights
    # weights: np.ndarray = np.random.randn(len(vocabulary)) * 0.01
    
    # Initialize model
    model = nn.Linear(len(vocabulary), 1)
    
    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Train the classifier over 100 times
    for epoch in range(100):
        total_loss: float = 0
        
        # Train the classifier over training data
        for sentence, truth in training_data:
            # Encode each sentence
            # encoded_text: np.ndarray = encode_text(sentence, vocabulary)
            encoded_text = torch.tensor(encode_text(sentence, vocabulary), dtype=torch.float32)
            
            # Calculate first prediction
            # prediction: float = get_sigmoid(np.dot(encoded_text, weights))
            prediction = torch.sigmoid(model(encoded_text))
            
            # Calculate loss
            # loss: float = get_loss(prediction, truth)
            truth_tensor = torch.tensor([truth], dtype=torch.float32)
            loss = nn.BCELoss()(prediction, truth_tensor)
            
            # Calculate error
            # error: float = prediction - truth
            
            # Update weights
            # weights -= 0.1 * error * encoded_text
            
            # Calculate gradients and Update weights            
            optimizer.zero_grad()  # clear old gradients
            loss.backward()        # calculate gradients automatically
            optimizer.step()       # nudge weights
            
            # Update total loss
            total_loss += loss
            
            
        # Print loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch} — Loss: {total_loss/len(training_data):.4f}")
        
    # return weights
    return model