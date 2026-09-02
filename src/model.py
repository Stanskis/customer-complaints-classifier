"""
model.py — Entraîne (ou charge) le modèle final et l'exporte au
format pickle, pour être attaché comme asset à une release GitHub.
"""
import os
import re
import string
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "model.pkl")

TAG_MAPPING_STEP1 = {
    'Credit reporting': 'Credit reporting, credit repair services, or other personal consumer reports',
    'Credit card': 'Credit card or prepaid card',
    'Prepaid card': 'Credit card or prepaid card',
    'Payday loan': 'Payday loan, title loan, or personal loan',
    'Money transfers': 'Money transfer, virtual currency, or money service',
    'Virtual currency': 'Money transfer, virtual currency, or money service',
}
TAG_MAPPING_STEP2 = {
    'Checking or savings account': 'Bank account or service',
    'Vehicle loan or lease': 'Consumer Loan',
    'Payday loan, title loan, or personal loan': 'Consumer Loan',
}


nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    #  1. Convert to lowercase
    text = text.lower()
    #  2. Remove digits
    text = re.sub(r'\d+', '', text)
    #  3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #  4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    #  5. Remove stopwords
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df['Tag'] = df['Tag'].str.strip()
    df['Tag'] = df['Tag'].replace(TAG_MAPPING_STEP1)
    df['Tag'] = df['Tag'].replace(TAG_MAPPING_STEP2)

    df_train = df[df['Consumer Claim'].notna() & df['Tag'].notna()][
        ['Consumer Claim', 'Tag']
    ].copy()

    return df_train


def build_pipeline():
    return Pipeline(steps=[
        ('tfidf', TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            preprocessor=clean_text
        )),
        ('classifier', CalibratedClassifierCV(
            LinearSVC(random_state=42, max_iter=2000), cv=5
        ))
    ])


def main():
    print("Loading data...")
    df_train = load_and_prepare_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_train['Tag'])

    X = df_train['Consumer Claim']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "label_encoder": label_encoder}, f)

    print(f"Model exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
