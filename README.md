# Banking-ML
# Banking Query Intent Classifier

Classifies a customer banking message into one of 77 intents
(e.g. "My card payment was declined" -> `declined_card_payment`).

## Dataset
[Banking77](https://huggingface.co/datasets/PolyAI/banking77):
10,003 training and 3,080 test messages, 77 intents.

## Approach
- Features: TF-IDF on word unigrams and bigrams
- Models compared: Logistic Regression and LinearSVC
- Evaluated on the held-out test set

## Results

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Logistic Regression | 0.8571 | 0.8558 |
| LinearSVC | 0.8938 | 0.8940 |

LinearSVC was selected. Its training accuracy is 0.998 against 0.894 on test,
so it overfits. Next steps: tune regularization (`C`) and use cross-validation.

## Run it

```bash
pip install -r requirements.txt
# Generate the model file: run train notebook, it saves intent_model.joblib
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "my card payment was declined"}'
```

Example response (from my own run):

```json
{"intent": "declined_card_payment"}
```

## Project structure
- `app.py`: FastAPI service
- `Banking77.ipynb`: training and evaluation
- `requirements.txt`: API dependencies

## Limitations
Trained on a single English dataset. Short, clean queries only.
No confidence threshold yet, so it always returns an intent, even for off-topic text.
