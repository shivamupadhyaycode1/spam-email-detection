

import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_FILE = Path("SMSSpamCollection")


def load_dataset():
    """Load the SMS Spam Collection dataset, downloading it if needed."""
    if DATA_FILE.exists():
        print(f"Loading dataset from: {DATA_FILE.resolve()}")
        return pd.read_csv(
            DATA_FILE, sep="\t", header=None, names=["label", "message"],
            encoding="utf-8"
        )

    print("Downloading the official UCI SMS Spam Collection dataset...")
    try:
        request = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=40) as response:
            zip_content = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_content)) as archive:
            data_name = next(
                name for name in archive.namelist()
                if name.endswith("SMSSpamCollection")
            )
            data_bytes = archive.read(data_name)

        DATA_FILE.write_bytes(data_bytes)
        print(f"Dataset saved to: {DATA_FILE.resolve()}")
        return pd.read_csv(
            io.BytesIO(data_bytes), sep="\t", header=None,
            names=["label", "message"], encoding="utf-8"
        )
    except Exception as error:
        raise RuntimeError(
            "Dataset download failed. Check your internet connection. "
            "Alternatively, download the UCI SMS Spam Collection ZIP from "
            "https://archive.ics.uci.edu/dataset/228/sms+spam+collection, "
            "extract the file named SMSSpamCollection, and put it beside this script."
        ) from error


def main():
    # 1. Load and inspect data.
    df = load_dataset()
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]

    print("\n========== DATASET OVERVIEW ==========")
    print(df.head())
    print(f"\nNumber of messages: {len(df)}")
    print("\nClass counts:")
    print(df["label"].value_counts())

    # 2. Graph: number of spam and legitimate messages.
    class_counts = df["label"].value_counts().reindex(["ham", "spam"])
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar")
    plt.title("SMS Dataset: Legitimate vs Spam Messages")
    plt.xlabel("Message category")
    plt.ylabel("Number of messages")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.close()
    print("\nSaved graph: class_distribution.png")

    # 3. Separate input text (X) from target labels (y).
    X = df["message"]
    y = df["label"]

    # 4. Split into training and testing sets.
    # Stratify keeps a similar ham/spam proportion in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Build a pipeline:
    # TF-IDF converts text into numeric features; Naive Bayes classifies them.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )),
        ("classifier", MultinomialNB())
    ])

    # 6. Train the model.
    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # 7. Predict labels for messages not used during training.
    y_pred = model.predict(X_test)

    # 8. Evaluate model.
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=["ham", "spam"], zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    print("\n========== MODEL EVALUATION ==========")
    print(f"Test messages: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows = actual; columns = predicted):")
    print(matrix)

    with open("evaluation_report.txt", "w", encoding="utf-8") as file:
        file.write("EMAIL SPAM DETECTION - EVALUATION REPORT\n")
        file.write(f"Dataset messages used: {len(df)}\n")
        file.write(f"Training messages: {len(X_train)}\n")
        file.write(f"Testing messages: {len(X_test)}\n")
        file.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        file.write("Classification report:\n")
        file.write(report + "\n")
        file.write("Confusion matrix (rows = actual; columns = predicted):\n")
        file.write(str(matrix) + "\n")
    print("\nSaved evaluation report: evaluation_report.txt")

    # 9. Graph: confusion matrix.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["Ham", "Spam"]
    )
    display.plot(values_format="d")
    plt.title("Email/SMS Spam Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved graph: confusion_matrix.png")

    # 10. Try some new example messages.
    example_messages = [
        "Congratulations! You have won a free prize. Call now to claim.",
        "Hi, are we still meeting at the library at 4 pm?"
    ]
    example_predictions = model.predict(example_messages)

    print("\n========== EXAMPLE PREDICTIONS ==========")
    for message, prediction in zip(example_messages, example_predictions):
        category = "SPAM" if prediction == "spam" else "NOT SPAM (HAM)"
        print(f"\nMessage: {message}")
        print(f"Prediction: {category}")

    # 11. Save the trained pipeline.
    try:
        import joblib
        joblib.dump(model, "spam_detection_model.joblib")
        print("\nSaved trained model: spam_detection_model.joblib")
    except Exception as error:
        print(f"\nCould not save the model file: {error}")

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
