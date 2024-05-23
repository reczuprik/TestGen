import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding


def prepare_data(csv_file):
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

    return pad_requirements, pad_test_cases, tokenizer, max_seq_len


def train_model(
    requirements, test_cases, tokenizer, max_seq_len, epochs=10, batch_size=32
):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 256
    latent_dim = 512

    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(vocab_size, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    target_data = test_cases[:, 1:]
    target_data = target_data.reshape(target_data.shape[0], target_data.shape[1], 1)

    model.fit(
        [requirements, test_cases[:, :-1]],
        target_data,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model


def generate_test_case(model, tokenizer, new_requirement, max_seq_len):
    seq_new_requirement = tokenizer.texts_to_sequences([new_requirement])
    pad_new_requirement = pad_sequences(
        seq_new_requirement, maxlen=max_seq_len, padding="post"
    )

    states_value = model.predict(pad_new_requirement)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get("startseq", 0)

    stop_condition = False
    generated_test_case = []

    while not stop_condition:
        output_tokens, h, c = model.layers[4].predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, "")

        if sampled_word == "endseq" or len(generated_test_case) > max_seq_len:
            stop_condition = True
        else:
            generated_test_case.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return " ".join(generated_test_case)


def post_process(test_case):
    lines = test_case.split("\n")
    processed_lines = []
    for line in lines:
        if not line.startswith(("Given", "When", "Then", "And", "But")):
            line = "Given " + line
        processed_lines.append(line)

    return "\n".join(processed_lines)


def main():
    root = tk.Tk()
    root.withdraw()

    old_data_file = filedialog.askopenfilename(
        title="Select the CSV file with requirement-test case pairs"
    )
    pad_requirements, pad_test_cases, tokenizer, max_seq_len = prepare_data(
        old_data_file
    )

    print("Training the model...")
    model = train_model(pad_requirements, pad_test_cases, tokenizer, max_seq_len)
    print("Model training complete.")

    new_data_file = filedialog.askopenfilename(
        title="Select the CSV file with new requirements"
    )
    new_data = pd.read_csv(new_data_file)
    new_requirements = new_data["requirement"].tolist()

    generated_test_cases = []
    for req in new_requirements:
        raw_test_case = generate_test_case(model, tokenizer, req, max_seq_len)
        processed_test_case = post_process(raw_test_case)
        generated_test_cases.append(processed_test_case)

    output_data = pd.DataFrame(
        {"requirement": new_requirements, "generated_test_case": generated_test_cases}
    )
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv", title="Save the generated test cases"
    )
    output_data.to_csv(save_path, index=False)
    print(f"Generated test cases saved to {save_path}")


if __name__ == "__main__":
    main()
