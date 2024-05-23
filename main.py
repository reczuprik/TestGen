import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# test


def prepare_data(csv_file):
    """
    Load and preprocess the data from a CSV file.

    Args:
    csv_file (str): Path to the CSV file containing requirement-test case pairs.

    Returns:
    tuple: tokenized_requirements, tokenized_test_cases, tokenizer
    """
    data = pd.read_csv(csv_file)
    requirements = data["requirement"].tolist()
    test_cases = data["test_case"].tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(requirements + test_cases)

    seq_requirements = tokenizer.texts_to_sequences(requirements)
    seq_test_cases = tokenizer.texts_to_sequences(test_cases)

    max_seq_len = max(
        max(len(seq) for seq in seq_requirements),
        max(len(seq) for seq in seq_test_cases),
    )

    pad_requirements = pad_sequences(
        seq_requirements, maxlen=max_seq_len, padding="post"
    )
    pad_test_cases = pad_sequences(seq_test_cases, maxlen=max_seq_len, padding="post")

    return pad_requirements, pad_test_cases, tokenizer


# Example usage:
# pad_requirements, pad_test_cases, tokenizer = prepare_data('requirements_test_cases.csv')
