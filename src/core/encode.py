import numpy as np

# Encode text into a vector (0 and 1)
def encode_text(text: str, vocabulary: dict[str, int]) -> list[int]:
    vector: list[int] = np.zeros(len(vocabulary)) # [0, 0, 0, 0, 0, 0]

    # Flip 0 -> 1 for words in text
    for word in text.split():
        if word in vocabulary:
            vector[vocabulary[word]] = 1

    return vector
    
    