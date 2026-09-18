# DSA-EXIT-EXAM

A Flask app that predicts a Pokemon's primary type from HP, attack, defense, height, and base experience.

A Flask app that predicts a Pokémon's primary type from HP, attack, defense, height, and base experience.

## Run

```powershell
py -m pip install -r requirements.txt
py app.py
```

Open http://127.0.0.1:5000.

The supplied pickle contains a `LabelEncoder` and regression models. Because regression models predict continuous values rather than type classes, the app reuses the saved encoder and trains a classification model from `G2.csv` for the requested primary-type prediction.
