import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
OUT_PATH = "benchmark_result.json"

t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH)
load_time = time.perf_counter() - t0

X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    metric="average_precision",
    min_child_samples=50,
    reg_lambda=5,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)

t0 = time.perf_counter()
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
)
train_time = time.perf_counter() - t0

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

# Latency: mean over 100 single-row predictions after a warm-up
row = X_test.iloc[[0]]
model.predict(row)
n = 100
t0 = time.perf_counter()
for _ in range(n):
    model.predict(row)
latency_ms = (time.perf_counter() - t0) / n * 1000

# Throughput: 1000 rows in one batch, best of 5 runs
batch = X_test.iloc[:1000]
model.predict(batch)
times = []
for _ in range(5):
    t0 = time.perf_counter()
    model.predict(batch)
    times.append(time.perf_counter() - t0)
batch_time = min(times)

result = {
    "load_data_time_s": round(load_time, 4),
    "training_time_s": round(train_time, 4),
    "best_iteration": int(model.best_iteration_ or model.n_estimators),
    "auc_roc": float(roc_auc_score(y_test, proba)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "f1_score": float(f1_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred)),
    "recall": float(recall_score(y_test, pred)),
    "inference_latency_1row_ms": round(latency_ms, 4),
    "inference_1000rows_time_ms": round(batch_time * 1000, 4),
    "inference_throughput_rows_per_s": round(1000 / batch_time, 2),
}

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
