from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "G2.csv"
BUNDLE_PATH = BASE_DIR / "ml_preprocessing_and_models_bundle (3).pkl"
FEATURES = ["hp", "attack", "defense", "height_dm", "base_experience"]
FORM_FIELDS = [
    ("hp", "HP", "Pokemon hit points", 1, 500),
    ("attack", "Attack", "Physical attack strength", 1, 500),
    ("defense", "Defense", "Physical defense strength", 1, 500),
    ("height_dm", "Height", "Height in decimeters", 1, 300),
    ("base_experience", "Base experience", "Experience awarded when defeated", 1, 1000),
]

app = Flask(__name__)
app.secret_key = "pokemon-type-predictor"


def load_classifier():
    dataset = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ["type_1"])

    with BUNDLE_PATH.open("rb") as bundle_file:
        bundle = pickle.load(bundle_file)

    label_encoder = bundle["label_encoder"]
    encoded_target = label_encoder.transform(dataset["type_1"])
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    classifier.fit(dataset[FEATURES], encoded_target)
    return classifier, label_encoder


classifier, label_encoder = load_classifier()


@app.get("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            values = {field: float(request.form[field]) for field in FEATURES}
        except (KeyError, TypeError, ValueError):
            flash("Enter a valid number in every field.", "error")
            return render_template("predict.html", fields=FORM_FIELDS, values=request.form)

        prediction = classifier.predict(pd.DataFrame([values], columns=FEATURES))[0]
        predicted_type = str(label_encoder.inverse_transform([prediction])[0]).title()
        return render_template(
            "result.html",
            predicted_type=predicted_type,
            values=values,
        )

    return render_template("predict.html", fields=FORM_FIELDS, values={})


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)