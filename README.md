# Early Dropout Prediction System (NexHack 2026)

Flask API + scikit-learn model + single-page frontend. Data is **synthetic**.

## Run locally
    pip install -r requirements.txt
    python train.py        # generates data, trains, saves model.joblib + metrics.json
    python app.py          # open http://localhost:5000
    python -m pytest -q test_api.py   # needs: pip install pytest

## API
| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | status, model name |
| GET | /api/metrics | test-set recall / precision / ROC-AUC |
| GET | /api/students?tier=High&limit=20 | ranked risk list + tier counts |
| GET | /api/students/<id> | one student + top reasons + suggested actions |
| POST | /api/predict | score a new student (JSON with the 7 features) |

## Deploy (free tier, e.g. Render)
1. Push this folder to a GitHub repo (the build step retrains the model on the server, so version mismatches cannot break it).
2. Render -> New Web Service -> connect repo.
3. Build command: `pip install -r requirements.txt && python train.py`   Start command: `gunicorn app:app`
4. Open the URL Render gives you. That is your live site (frontend and API on one origin).

## Notes
- Explanations use occlusion (swap a feature for the population median, measure the drop in log-odds).
- Fairness: fee features are included because the problem statement asks for them. Check flag rates across groups before any real use.
- Human in the loop: scores recommend, counsellors decide.
