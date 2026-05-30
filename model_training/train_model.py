from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt

base_dir = Path(__file__).resolve().parents[1]
# Assuming dataset is inside a 'datasets' folder in the root directory
dataset_path = base_dir / "datasets" / "iit_bbs_dataset.csv" 
print("Loading dataset...")
df = pd.read_csv(dataset_path)

# preprocessing of the dataset
X = df.drop(columns=['Serial No', 'CS'])
y = df['CS']

# feature names for future reference
feature_names = X.columns.tolist()
print(f"Features used ({len(feature_names)}): {feature_names}")

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training XGBoost model
print("Training XGBoost Model...")
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train.values, y_train)

# evaluation of the model
y_pred = model.predict(X_test.values)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} MPa")


X_train_float32 = X_train.values.astype(np.float32)
initial_types = [("input", FloatTensorType([None, X_train_float32.shape[1]]))]
onnx_model = convert_xgboost(model, initial_types=initial_types)

onnx_path = base_dir / "web-app" / "public" / "concrete_model.onnx"

onnx_path.parent.mkdir(parents=True, exist_ok=True)

with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"\nSuccess! Model saved to: {onnx_path}")


sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

sample = X_test.values[0].astype(np.float32).reshape(1, -1)
onnx_pred = sess.run([label_name], {input_name: sample})[0]

print(f"Predicted Strength: {onnx_pred[0][0]:.2f} MPa")
print(f"Actual Measured Strength: {y_test.iloc[0]:.2f} MPa")