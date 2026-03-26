from data import training_data

# Create a vocabulary from the training data
vocabulary = {}
index = 0

for sentence, label in training_data:
    for word in sentence.split():
        if word not in vocabulary:
            vocabulary[word] = index
            index += 1
        