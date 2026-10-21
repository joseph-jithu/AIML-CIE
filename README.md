# AgroSaarthi crop recommendation prototype

A small local website backed by a trained K-nearest-neighbors crop classifier. It uses seven field values (N, P, K, temperature, humidity, pH, rainfall) and returns a top crop plus two alternatives.

## Run it

Requires Python 3.9 or newer. No third-party packages are required.

```powershell
python train_model.py
python app.py
```

Then open <http://127.0.0.1:8000>. Use **Try example** to see a sample recommendation.

The training script reads `Crop_recommendation.csv`, makes a deterministic 80/20 holdout check, then fits the final model on all rows and writes `model.json`. The source is the public Crop Recommendation Dataset by Atharva Ingle on Kaggle, mirrored in this repository for the prototype. The dataset is a demonstration dataset and should not be treated as a locally validated agronomy model.

## Scope

The current prototype only predicts a crop label from the seven dataset features. It does not yet calculate affordability, profit, subsidies, yield, disease, irrigation, market prices, or logistics. Those need separate validated data and models. The displayed vote share is the fraction of distance-weighted neighbor votes and is not a calibrated probability of crop success.
