# Text Classifier

A simple, lightweight text classification project implemented in Python using PyTorch. This project demonstrates basic sentiment analysis by training a neural network on a small dataset of labeled sentences.

## Features

- **Custom Neural Network**: Implemented using PyTorch `torch.nn`.
- **Vocabulary Generation**: Dynamically creates a mapping of unique words from the training dataset.
- **Bag-of-Words Encoding**: Converts text into numerical vectors that are suitable for model training.
- **Sentiment Analysis**: Classifies sentences as positive (1) or negative (0).

## Project Structure

```text
.
├── src/
│   ├── main.py              # Entry point to train and test the model
│   ├── core/
│   │   ├── text_classifier.py  # Model definition and training logic
│   │   ├── vocabulary.py       # Functions for vocabulary management
│   │   ├── encode.py           # Text-to-vector encoding functions
│   │   ├── sigmoid.py          # Custom sigmoid implementation (optional)
│   │   └── loss_function.py    # Custom loss function (optional)
│   └── data/
│       └── loader.py           # Sample training dataset
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/harshprakashtech/text-classifier.git
    cd text-classifier
    ```

2.  **Set up a virtual environment (optional but recommended)**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: Ensure you have `torch` installed as it's required for the model._

### Usage

Run the main script to train the model and see predictions on the training data:

```bash
python3 src/main.py
```

## Technologies Used

- **Python**: Core language.
- **NumPy**: For vector operations and numerical computing.
- **PyTorch**: For model definition and training logic.

_Created as part of a learning journey into Machine Learning and NLP._
