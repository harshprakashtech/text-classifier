# Helper function to create a vocabulary from training data
# Returns a dictionary of words and their indices

def get_vocabulary(training_data: list[tuple[str, int]]) -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    index: int = 0

    # Add unique words to vocabulary with index
    for sentence, label in training_data:
        for word in sentence.split():
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1 
    
    return vocabulary