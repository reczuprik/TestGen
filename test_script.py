import unittest
import tempfile
import pandas as pd
from unittest.mock import patch, mock_open

# Import the necessary functions from your script
from ss import prepare_data, train_model, generate_test_case, post_process


class TestScriptFunctions(unittest.TestCase):
    def setUp(self):
        self.train_data = """requirement,test_case
"User logs into the system","Given the user is on the login page\\nWhen the user enters valid credentials\\nThen the user should be redirected to the dashboard"
"User resets the password","Given the user is on the forgot password page\\nWhen the user enters their registered email\\nAnd the user clicks on the reset button\\nThen the user should receive a password reset link in their email"
"User updates profile information","Given the user is logged in\\nWhen the user navigates to the profile page\\nAnd the user updates their profile information\\nThen the profile information should be saved successfully"
"""

    @patch("builtins.open", new_callable=mock_open, read_data=train_data)
    def test_prepare_data(self, mock_file):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write(self.train_data)
            temp_file_path = temp_file.name

        pad_requirements, pad_test_cases, tokenizer, max_seq_len = prepare_data(
            temp_file_path
        )

        self.assertIsNotNone(pad_requirements)
        self.assertIsNotNone(pad_test_cases)
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(max_seq_len)

    def test_train_model(self):
        # Prepare test data
        data = pd.read_csv(io.StringIO(self.train_data))
        requirements = data["requirement"].tolist()
        test_cases = [tc.replace("\\n", "\n") for tc in data["test_case"].tolist()]

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
        pad_test_cases = pad_sequences(
            seq_test_cases, maxlen=max_seq_len, padding="post"
        )

        model = train_model(
            pad_requirements,
            pad_test_cases,
            tokenizer,
            max_seq_len,
            epochs=1,
            batch_size=32,
        )

        self.assertIsNotNone(model)

    def test_generate_test_case(self):
        # Prepare test data
        data = pd.read_csv(io.StringIO(self.train_data))
        requirements = data["requirement"].tolist()
        test_cases = [tc.replace("\\n", "\n") for tc in data["test_case"].tolist()]

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
        pad_test_cases = pad_sequences(
            seq_test_cases, maxlen=max_seq_len, padding="post"
        )

        model = train_model(
            pad_requirements,
            pad_test_cases,
            tokenizer,
            max_seq_len,
            epochs=1,
            batch_size=32,
        )

        new_requirement = "User signs up for a new account"
        generated_test_case = generate_test_case(
            model, tokenizer, new_requirement, max_seq_len
        )

        self.assertIsNotNone(generated_test_case)
        self.assertGreater(len(generated_test_case), 0)

    def test_post_process(self):
        raw_test_case = "the user is on the login page the user enters valid credentials the user should be redirected to the dashboard"
        expected_output = "Given the user is on the login page\nGiven the user enters valid credentials\nGiven the user should be redirected to the dashboard"

        processed_test_case = post_process(raw_test_case)

        self.assertEqual(processed_test_case, expected_output)


if __name__ == "__main__":
    unittest.main()
