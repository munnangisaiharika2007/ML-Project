# NephroScreen: CKD prediction with XGBoost and LightGBM

## Run it in VS Code
1. Open this folder in VS Code (`File > Open Folder`).
2. Install the "Live Server" extension (by Ritwick Dey) from the Extensions tab, if not already installed.
3. Right-click `index.html` in the file explorer and choose **"Open with Live Server"**.
   This opens the site in your browser at a local address like `http://127.0.0.1:5500`.
4. No Live Server? Just double-click `index.html` in your file explorer — it opens directly in the browser (some browser security settings may limit this slightly, so Live Server is recommended).

Demo login: `doctor@demo.test` / `demo1234`

## Files
- `index.html`: portal with login, home, people list, patient profile and clinician profile.
- `screening.html`: screening form with live XGBoost and LightGBM risk bars.
- `train_models.py`: trains both models on the UCI CKD dataset and saves them.
- `api.py`: FastAPI `/predict` endpoint that serves the trained models.

## Use the real models instead of the demo scoring
1. Download `kidney_disease.csv` (UCI CKD dataset) into this folder.
2. `pip install pandas scikit-learn xgboost lightgbm joblib` then `python train_models.py`.
3. `pip install fastapi uvicorn` then `uvicorn api:app --reload`.
4. In `screening.html`, replace the `score()` function with a `fetch("http://localhost:8000/predict", ...)` call.

Not a medical device. Do not use for diagnosis.
