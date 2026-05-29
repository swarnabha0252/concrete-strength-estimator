# Concrete Strength Estimator

A web app that predicts concrete strength instantly in your browser.

**[Try the Live Demo Here](https://concrete-strength-estimator.vercel.app)**

## What It Does

- Predict concrete compressive strength (MPa) in real-time
- Adjust mix design using sliders and inputs
- Auto-calculates water-to-cement ratio

## Setup

### Install

```bash
# Web app
cd web-app
npm install

# Model training (optional)
cd model_training
python -m venv venv
venv\Scripts\activate
pip install pandas numpy xgboost onnxmltools onnxruntime scikit-learn
```

### Run

```bash
cd web-app
npm run dev
```

## How It Works

1. **Frontend:** Next.js app with React sliders and inputs
2. **ML Model:** XGBoost trained on concrete data (188 samples)
3. **Export:** Model saved as ONNX file for browser
4. **Prediction:** Runs in browser using WebAssembly (no server needed)

## Project Files

```
├── web-app/src/pages/index.js      → Main UI
├── web-app/public/concrete_model.onnx  → AI model
├── model_training/train_model.py   → Train the model
└── datasets/iit_bbs_dataset.csv    → Training data
```

## Input Controls

**Mix Design Tab:**
- Cement (kg/m³)
- Total Cementitious Material (kg/m³)
- Water (kg/m³)
- Water–TCM Ratio
- Curing Age (days)

**Advanced Tab:**
- Fly Ash, Slag, Metakaolin (Binders)
- Natural & Recycled Aggregates
- Superplasticizer, Viscosity Agent

## Features

- Water automatically adjusts when you change w/TCM ratio
- If water exceeds 250 kg/m³, TCM adjusts instead
- All values rounded to 2 decimals
- Real-time strength prediction with visual progress bar
- Responsive design (desktop & mobile)

## Train the Model

```bash
cd model_training
python train_model.py
```

Output:
- R² Score: 0.9324
- RMSE: 4.3581 MPa
- Model saved to `web-app/public/concrete_model.onnx`

## Deploy

```bash
cd web-app
npm run build
```

## Tech Stack

- Next.js + React (frontend)
- XGBoost (machine learning)
- ONNX (model format)
- Python (training)

## Troubleshooting

**Model not loading?**
- Check `web-app/public/concrete_model.onnx` exists
- Check browser console for errors

**Wrong predictions?**
- Retrain model: `python train_model.py`
- Check input values are in range

**Values jumping?**
- Normal behavior when water hits 250 limit
- TCM adjusts to maintain ratio

## Version

v1.0 - Initial Commit
