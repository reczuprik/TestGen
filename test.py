import pandas as pd
import numpy as np
import re
import logging
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import spacy
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QComboBox,
    QProgressBar,
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
FILE_PATH = "C:/Utils/Bank/penzugy.xlsx"
SHEET_NAME = "tételek"
TO_BE_EVALUATED = ["Főtípus", "Altípus"]
CONFIDENCE_THRESHOLD = 0.70
BATCH_SIZE = 32  # Process embeddings in batches of 32

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")


def load_data(file_path, sheet_name):
    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=sheet_name)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        QMessageBox.critical(None, "Error", f"Error loading data: {e}")
        return None


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def get_bert_embeddings(texts, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[
            i : i + batch_size
        ].tolist()  # Ensure batch_texts is a list of strings
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.extend(batch_embeddings)
        logging.info(
            f"Processed batch {i // batch_size + 1}/{(len(texts) // batch_size) + 1}"
        )
    return np.vstack(embeddings)


def preprocess_data(df, column):
    if column not in df.columns or "Leírás" not in df.columns:
        logging.error(f"Required columns are missing from the data.")
        QMessageBox.critical(
            None, "Error", f"Required columns are missing from the data."
        )
        return None, None

    df["Leírás"] = df["Leírás"].apply(preprocess_text)
    df_cleaned = df[df["Leírás"].notnull() & df[column].notnull()].reset_index(
        drop=True
    )
    df_new_items = df[df["Leírás"].notnull() & df[column].isnull()].reset_index(
        drop=True
    )

    return df_cleaned, df_new_items


def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    param_grid = {
        "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [10, 20, 30]},
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.5],
        },
    }

    for name, model in models.items():
        try:
            grid_search = GridSearchCV(
                model, param_grid[name], cv=5, scoring="accuracy"
            )
            grid_search.fit(X_train, y_train)
            models[name] = grid_search.best_estimator_
            logging.info(
                f"{name} trained successfully with best parameters: {grid_search.best_params_}"
            )
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            models[name] = None

    return models


def evaluate_models(models, X_val, y_val):
    results = {}
    for name, model in models.items():
        if model:
            predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            results[name] = {
                "accuracy": f"{accuracy:.2%}",
                "classification_report": classification_report(y_val, predictions),
            }
        else:
            results[name] = "Training Failed"
    return results


def predict_and_save(
    df_new_items,
    bert_model,
    model,
    label_encoder,
    file_path,
    column,
    accuracy,
    clear_sheet=False,
):
    if df_new_items.empty:
        logging.error("No new items to predict.")
        QMessageBox.critical(None, "Error", "No new items to predict.")
        return

    if model is None:
        logging.error("No trained model available for predictions.")
        QMessageBox.critical(
            None, "Error", "No trained model available for predictions."
        )
        return

    try:
        logging.info(f"Generating BERT embeddings for {len(df_new_items)} new items.")
        X_new = get_bert_embeddings(df_new_items["Leírás"])
        if X_new.shape[0] == 0:
            raise ValueError("No valid samples found for prediction.")
    except ValueError as e:
        logging.error(f"Error transforming data: {e}")
        QMessageBox.critical(None, "Error", f"Error transforming data: {e}")
        return

    try:
        predictions = model.predict(X_new)
        if accuracy < CONFIDENCE_THRESHOLD:
            df_new_items[column] = "Low Confidence"
        else:
            df_new_items[column] = label_encoder.inverse_transform(predictions)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        QMessageBox.critical(None, "Error", f"Error during prediction: {e}")
        return

    try:
        with pd.ExcelWriter(
            file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay"
        ) as writer:
            if clear_sheet:
                df_empty = pd.DataFrame(columns=df_new_items.columns)
                df_empty.to_excel(writer, sheet_name="neew", index=False)

            try:
                existing_df = pd.read_excel(file_path, sheet_name="neew")
                df_combined = (
                    pd.concat([existing_df, df_new_items])
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
            except Exception as e:
                logging.warning(f"No existing data or error loading: {e}")
                df_combined = df_new_items

            df_combined.to_excel(
                writer,
                sheet_name="neew",
                index=False,
                startrow=0,
                header=True,
            )
            logging.info("Predictions saved successfully.")
    except Exception as e:
        logging.error(f"Error saving predictions: {e}")
        QMessageBox.critical(None, "Error", f"Error saving predictions: {e}")


class MainWindow(QMainWindow):
    def __init__(self, model_results, best_model, best_model_accuracy):
        super().__init__()

        self.best_model = best_model
        self.best_model_accuracy = best_model_accuracy

        self.initUI(model_results)

    def initUI(self, model_results):
        self.setWindowTitle("Tipus beallitas")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        results_text = "\n".join(
            [
                f"{name} Accuracy: {metrics['accuracy']}"
                for name, metrics in model_results.items()
            ]
        )
        results_label = QLabel(results_text)
        results_label.setFont(QFont("Arial", 12))
        results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(results_label)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(30, 40, 200, 25)
        self.progress.setMaximum(100)
        layout.addWidget(self.progress)

        model_selection = QComboBox(self)
        model_selection.addItem("Random Forest")
        model_selection.addItem("Gradient Boosting")
        layout.addWidget(model_selection)

        predict_button = QPushButton("Predict")
        predict_button.setFont(QFont("Arial", 16))
        predict_button.setFixedSize(200, 50)
        predict_button.clicked.connect(self.predict)
        layout.addWidget(predict_button, alignment=Qt.AlignCenter)

        central_widget.setLayout(layout)

    def update_progress(self, value):
        self.progress.setValue(value)

    def predict(self):
        self.update_progress(0)
        # Predict for "Főtípus"
        predict_and_save(
            df_new_items_fo,
            bert_model,
            self.best_model,
            le_fo,
            FILE_PATH,
            "Főtípus",
            self.best_model_accuracy,
            clear_sheet=False,
        )
        self.update_progress(50)

        df_updated = load_data(FILE_PATH, SHEET_NAME)
        df_neew = pd.read_excel(FILE_PATH, sheet_name="neew")
        df_updated["Főtípus"] = df_neew["Főtípus"]

        df_cleaned_alt, df_new_items_alt = preprocess_data(df_updated, "Altípus")
        if df_cleaned_alt is None or df_new_items_alt is None:
            sys.exit()

        vectorizer_alt = TfidfVectorizer()
        X_alt = vectorizer_alt.fit_transform(df_cleaned_alt["Leírás"])
        le_alt = LabelEncoder()
        y_alt = le_alt.fit_transform(df_cleaned_alt["Altípus"])

        X_train_alt, X_val_alt, y_train_alt, y_val_alt = train_test_split(
            X_alt, y_alt, test_size=0.2, random_state=42
        )

        models_alt = train_models(X_train_alt, y_train_alt)
        model_results_alt = evaluate_models(models_alt, X_val_alt, y_val_alt)
        best_model_alt = max(
            models_alt.values(),
            key=lambda m: m.score(X_val_alt, y_val_alt) if m else -1,
        )

        best_model_accuracy_alt = max(
            [
                model.score(X_val_alt, y_val_alt)
                for model in models_alt.values()
                if model
            ]
        )

        predict_and_save(
            df_new_items_alt,
            bert_model,
            best_model_alt,
            le_alt,
            FILE_PATH,
            "Altípus",
            best_model_accuracy_alt,
            clear_sheet=False,
        )
        self.update_progress(100)

        QMessageBox.information(self, "Done", "Prediction complete!")
        self.close()


# Main program execution
df = load_data(FILE_PATH, SHEET_NAME)
if df is None:
    sys.exit()

df_cleaned_fo, df_new_items_fo = preprocess_data(df, "Főtípus")
if df_cleaned_fo is None or df_new_items_fo is None:
    sys.exit()

logging.info(
    f"Generating BERT embeddings for training data with {len(df_cleaned_fo)} items."
)
X_fo = get_bert_embeddings(df_cleaned_fo["Leírás"])
le_fo = LabelEncoder()
y_fo = le_fo.fit_transform(df_cleaned_fo["Főtípus"])

X_train_fo, X_val_fo, y_train_fo, y_val_fo = train_test_split(
    X_fo, y_fo, test_size=0.2, random_state=42
)

models_fo = train_models(X_train_fo, y_train_fo)
model_results_fo = evaluate_models(models_fo, X_val_fo, y_val_fo)
best_model_fo = max(
    models_fo.values(), key=lambda m: m.score(X_val_fo, y_val_fo) if m else -1
)

if best_model_fo is None:
    logging.error("No model was successfully trained for 'Főtípus'.")
    QMessageBox.critical(
        None, "Error", "No model was successfully trained for 'Főtípus'."
    )
    sys.exit()

best_model_accuracy_fo = max(
    [model.score(X_val_fo, y_val_fo) for model in models_fo.values() if model]
)

logging.info(f"Predicting 'Főtípus' with accuracy: {best_model_accuracy_fo}")
logging.info(f"Data to be predicted: {df_new_items_fo.head()}")

app = QApplication(sys.argv)
main_window = MainWindow(model_results_fo, best_model_fo, best_model_accuracy_fo)
main_window.show()
sys.exit(app.exec_())
