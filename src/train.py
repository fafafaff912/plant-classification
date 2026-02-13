"""
Обучение моделей рекомендации удобрений.
Модели: Linear Regression, Random Forest, XGBoost, MLP (PyTorch).
Данные: FAOSTAT / синтетические на основе агрономических формул.

Запуск:
    python train.py --samples 10000 --epochs 200
"""

import argparse
import json
import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost не установлен, XGBoost будет пропущен")


# ========================== ВОСПРОИЗВОДИМОСТЬ ==========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ========================== ГЕНЕРАЦИЯ ДАННЫХ ==========================

# Параметры культур: (yield_max т/га, оптим N, оптим P, оптим K, чувствительности)
CROP_PARAMS = {
    "wheat":     {"yield_max": 6.0, "opt_N": 150, "opt_P": 60, "opt_K": 50,
                  "c_N": 0.020, "c_P": 0.035, "c_K": 0.040,
                  "temp_opt": 18, "ph_opt": 6.5},
    "rice":      {"yield_max": 7.5, "opt_N": 120, "opt_P": 50, "opt_K": 60,
                  "c_N": 0.022, "c_P": 0.030, "c_K": 0.035,
                  "temp_opt": 28, "ph_opt": 6.0},
    "maize":     {"yield_max": 9.0, "opt_N": 200, "opt_P": 80, "opt_K": 70,
                  "c_N": 0.015, "c_P": 0.025, "c_K": 0.030,
                  "temp_opt": 25, "ph_opt": 6.2},
    "soybean":   {"yield_max": 3.5, "opt_N": 30,  "opt_P": 70, "opt_K": 80,
                  "c_N": 0.050, "c_P": 0.030, "c_K": 0.025,
                  "temp_opt": 26, "ph_opt": 6.3},
    "potato":    {"yield_max": 35.0, "opt_N": 180, "opt_P": 90, "opt_K": 150,
                  "c_N": 0.012, "c_P": 0.020, "c_K": 0.015,
                  "temp_opt": 18, "ph_opt": 5.8},
    "cotton":    {"yield_max": 4.0, "opt_N": 160, "opt_P": 60, "opt_K": 80,
                  "c_N": 0.018, "c_P": 0.030, "c_K": 0.025,
                  "temp_opt": 30, "ph_opt": 6.5},
    "sugarcane": {"yield_max": 70.0, "opt_N": 250, "opt_P": 100, "opt_K": 120,
                  "c_N": 0.010, "c_P": 0.018, "c_K": 0.016,
                  "temp_opt": 30, "ph_opt": 6.0},
    "barley":    {"yield_max": 5.0, "opt_N": 120, "opt_P": 50, "opt_K": 45,
                  "c_N": 0.025, "c_P": 0.040, "c_K": 0.045,
                  "temp_opt": 16, "ph_opt": 6.5},
}


def mitscherlich_yield(N, P, K, crop_params, temperature, soil_ph):
    """
    Урожайность по модели Митчерлиха с поправками на температуру и pH.

    yield = yield_max × prod(1 - exp(-c_i × dose_i)) × f(temp) × f(pH)
    """
    p = crop_params
    y_max = p["yield_max"]

    # Отклик на NPK (закон убывающей отдачи)
    n_response = 1 - np.exp(-p["c_N"] * np.maximum(N, 0))
    p_response = 1 - np.exp(-p["c_P"] * np.maximum(P, 0))
    k_response = 1 - np.exp(-p["c_K"] * np.maximum(K, 0))

    # Температурная поправка (гауссова)
    temp_factor = np.exp(-0.5 * ((temperature - p["temp_opt"]) / 6) ** 2)

    # pH поправка
    ph_factor = np.exp(-0.5 * ((soil_ph - p["ph_opt"]) / 1.2) ** 2)

    y = y_max * n_response * p_response * k_response * temp_factor * ph_factor
    return y


def compute_optimal_npk(soil_N, soil_P, soil_K, crop_params,
                        temperature, soil_ph, rainfall, humidity):
    """
    Вычисление оптимальных доз NPK с учётом почвы и условий.
    Оптимум: доза = max(0, оптимальная_потребность - содержание_в_почве) × корректировки.
    """
    p = crop_params

    # Базовая потребность = оптимум минус то, что уже есть в почве
    base_N = np.maximum(p["opt_N"] - soil_N * 0.8, 0)
    base_P = np.maximum(p["opt_P"] - soil_P * 0.7, 0)
    base_K = np.maximum(p["opt_K"] - soil_K * 0.6, 0)

    # Корректировка по влажности (при засухе — меньше усвоение)
    moisture_factor = np.clip(0.5 + humidity / 100, 0.6, 1.2)

    # Корректировка по осадкам (вымывание при высоких осадках)
    rain_factor = np.where(rainfall > 200, 1.2, np.where(rainfall < 50, 0.8, 1.0))

    # Корректировка по pH (плохая усвояемость при экстремальном pH)
    ph_dev = np.abs(soil_ph - p["ph_opt"])
    ph_factor = 1 + 0.15 * ph_dev

    # Корректировка по температуре
    temp_dev = np.abs(temperature - p["temp_opt"])
    temp_factor = np.where(temp_dev > 10, 0.85, 1.0)

    N_rec = base_N * moisture_factor * rain_factor * ph_factor * temp_factor
    P_rec = base_P * moisture_factor * ph_factor
    K_rec = base_K * moisture_factor * rain_factor * ph_factor

    return N_rec, P_rec, K_rec


def generate_dataset(n_samples=10000, seed=42):
    """
    Генерация реалистичного датасета на основе агрономических моделей.
    Каждая запись — одно поле в один сезон.
    """
    np.random.seed(seed)
    crops = list(CROP_PARAMS.keys())
    records = []

    for _ in range(n_samples):
        crop = np.random.choice(crops)
        p = CROP_PARAMS[crop]

        # Условия среды
        temperature = np.random.normal(p["temp_opt"], 5)
        temperature = np.clip(temperature, 5, 40)

        humidity = np.random.normal(65, 15)
        humidity = np.clip(humidity, 20, 95)

        rainfall = np.random.gamma(3, 40)  # мм/мес, скошено вправо
        rainfall = np.clip(rainfall, 10, 400)

        soil_ph = np.random.normal(p["ph_opt"], 0.8)
        soil_ph = np.clip(soil_ph, 4.0, 8.5)

        soil_moisture = np.random.normal(40, 12)
        soil_moisture = np.clip(soil_moisture, 10, 80)

        organic_carbon = np.random.gamma(2, 0.8)
        organic_carbon = np.clip(organic_carbon, 0.2, 6.0)

        # Текущее содержание NPK в почве (кг/га)
        soil_N = np.random.gamma(3, 15)   # 0-100+
        soil_N = np.clip(soil_N, 5, 120)
        soil_P = np.random.gamma(2, 12)
        soil_P = np.clip(soil_P, 3, 80)
        soil_K = np.random.gamma(3, 12)
        soil_K = np.clip(soil_K, 5, 100)

        area = np.random.lognormal(2, 0.8)
        area = np.clip(area, 0.5, 500)

        # Оптимальные рекомендации
        N_rec, P_rec, K_rec = compute_optimal_npk(
            soil_N, soil_P, soil_K, p,
            temperature, soil_ph, rainfall, humidity
        )

        # Добавляем шум (реальность не идеальна)
        N_rec *= np.random.normal(1.0, 0.08)
        P_rec *= np.random.normal(1.0, 0.08)
        K_rec *= np.random.normal(1.0, 0.08)

        N_rec = max(0, N_rec)
        P_rec = max(0, P_rec)
        K_rec = max(0, K_rec)

        # Урожай прошлого сезона (с некоторыми случайными дозами)
        prev_N = soil_N + np.random.uniform(0, p["opt_N"] * 0.7)
        prev_P = soil_P + np.random.uniform(0, p["opt_P"] * 0.7)
        prev_K = soil_K + np.random.uniform(0, p["opt_K"] * 0.7)
        prev_yield = mitscherlich_yield(
            prev_N, prev_P, prev_K, p, temperature, soil_ph
        )
        prev_yield *= np.random.normal(1.0, 0.1)
        prev_yield = max(0.1, prev_yield)

        records.append({
            "crop": crop,
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "rainfall": round(rainfall, 1),
            "soil_ph": round(soil_ph, 2),
            "soil_moisture": round(soil_moisture, 1),
            "soil_N": round(soil_N, 1),
            "soil_P": round(soil_P, 1),
            "soil_K": round(soil_K, 1),
            "organic_carbon": round(organic_carbon, 2),
            "prev_yield": round(prev_yield, 2),
            "area": round(area, 1),
            "N_recommended": round(N_rec, 1),
            "P_recommended": round(P_rec, 1),
            "K_recommended": round(K_rec, 1),
        })

    df = pd.DataFrame(records)
    print(f"Датасет: {len(df)} записей, {len(crops)} культур")
    print(f"Культуры: {', '.join(crops)}")
    return df


# ========================== ПРЕДОБРАБОТКА ==========================

def preprocess(df):
    """Кодирование, нормализация, разделение на train/val/test."""
    le = LabelEncoder()
    df["crop_encoded"] = le.fit_transform(df["crop"])

    feature_cols = [
        "temperature", "humidity", "rainfall", "soil_ph", "soil_moisture",
        "soil_N", "soil_P", "soil_K", "organic_carbon", "prev_yield",
        "area", "crop_encoded",
    ]
    target_cols = ["N_recommended", "P_recommended", "K_recommended"]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Нормализация
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_s": X_train_s, "X_val_s": X_val_s, "X_test_s": X_test_s,
        "y_train_s": y_train_s, "y_val_s": y_val_s, "y_test_s": y_test_s,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }


# ========================== МОДЕЛИ ==========================

# --- Linear Regression ---
def train_linear(data):
    print("\n--- Linear Regression ---")
    model = MultiOutputRegressor(LinearRegression())
    model.fit(data["X_train_s"], data["y_train"])
    pred = model.predict(data["X_test_s"])
    pred = np.maximum(pred, 0)
    print("  Обучена")
    return pred, model


# --- Random Forest ---
def train_random_forest(data):
    print("\n--- Random Forest ---")
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    )
    model.fit(data["X_train"], data["y_train"])
    pred = model.predict(data["X_test"])
    pred = np.maximum(pred, 0)
    print("  Обучена (200 деревьев)")
    return pred, model


# --- XGBoost ---
def train_xgboost(data):
    print("\n--- XGBoost ---")
    if not HAS_XGB:
        print("  Пропуск (не установлен)")
        return None, None

    model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
        )
    )
    model.fit(data["X_train"], data["y_train"])
    pred = model.predict(data["X_test"])
    pred = np.maximum(pred, 0)
    print("  Обучена (300 деревьев)")
    return pred, model


# --- MLP (PyTorch) ---
class FertilizerMLP(nn.Module):
    """Multi-output MLP для рекомендации NPK."""

    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim),
            nn.ReLU(),  # дозы >= 0
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(data, epochs=200, lr=1e-3, batch_size=256):
    print(f"\n--- MLP ({epochs} epochs) ---")
    device = get_device()
    input_dim = data["X_train_s"].shape[1]

    model = FertilizerMLP(input_dim, 3).to(device)

    X_tr = torch.FloatTensor(data["X_train_s"]).to(device)
    y_tr = torch.FloatTensor(data["y_train_s"]).to(device)
    X_val = torch.FloatTensor(data["X_val_s"]).to(device)
    y_val = torch.FloatTensor(data["y_val_s"]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(X_tr)
        train_losses.append(epoch_loss)
        scheduler.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            vl = criterion(val_pred, y_val).item()
        val_losses.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — train: {epoch_loss:.5f}, val: {vl:.5f}")

    model.load_state_dict(best_state)

    # Прогноз на test
    model.eval()
    with torch.no_grad():
        X_te = torch.FloatTensor(data["X_test_s"]).to(device)
        pred_s = model(X_te).cpu().numpy()

    pred = data["scaler_y"].inverse_transform(pred_s)
    pred = np.maximum(pred, 0)

    print(f"  Best val loss: {best_val_loss:.5f}")
    return pred, model, train_losses, val_losses


# ========================== МЕТРИКИ ==========================

def compute_metrics(y_true, y_pred, target_names=None):
    """MAE, RMSE, R², MAPE для каждого выхода."""
    if target_names is None:
        target_names = ["N", "P", "K"]

    results = {}
    for i, name in enumerate(target_names):
        actual = y_true[:, i]
        predicted = y_pred[:, i]

        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        mask = actual > 1  # avoid division by near-zero
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

        results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2),
                         "R2": round(r2, 4), "MAPE": round(mape, 2)}

    # Общие средние
    avg_mae = np.mean([results[n]["MAE"] for n in target_names])
    avg_r2 = np.mean([results[n]["R2"] for n in target_names])
    avg_mape = np.mean([results[n]["MAPE"] for n in target_names])
    results["AVG"] = {"MAE": round(avg_mae, 2), "R2": round(avg_r2, 4),
                      "MAPE": round(avg_mape, 2)}

    return results


# ========================== MAIN ==========================

def main(args):
    set_seed(args.seed)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Данные
    print("=" * 60)
    print("Генерация / загрузка данных")
    print("=" * 60)

    cache_path = "data/fertilizer_dataset.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        print(f"Загружено из кэша: {len(df)} записей")
    else:
        df = generate_dataset(n_samples=args.samples, seed=args.seed)
        df.to_csv(cache_path, index=False)
        print(f"Сохранено: {cache_path}")

    # 2. Предобработка
    data = preprocess(df)

    # 3. Обучение моделей
    all_preds = {}
    all_models = {}

    # Linear Regression
    pred_lr, model_lr = train_linear(data)
    all_preds["Linear"] = pred_lr
    all_models["Linear"] = model_lr

    # Random Forest
    pred_rf, model_rf = train_random_forest(data)
    all_preds["RandomForest"] = pred_rf
    all_models["RandomForest"] = model_rf

    # XGBoost
    pred_xgb, model_xgb = train_xgboost(data)
    if pred_xgb is not None:
        all_preds["XGBoost"] = pred_xgb
        all_models["XGBoost"] = model_xgb

    # MLP
    pred_mlp, model_mlp, train_losses, val_losses = train_mlp(
        data, epochs=args.epochs
    )
    all_preds["MLP"] = pred_mlp
    all_models["MLP"] = model_mlp

    # 4. Метрики
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ НА ТЕСТЕ")
    print(f"{'='*60}")

    targets = ["N", "P", "K"]
    all_metrics = {}

    for name, pred in all_preds.items():
        metrics = compute_metrics(data["y_test"], pred, targets)
        all_metrics[name] = metrics

        print(f"\n{name}:")
        print(f"  {'':>5} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
        for t in targets:
            m = metrics[t]
            print(f"  {t:>5} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
                  f"{m['R2']:>8.4f} {m['MAPE']:>7.1f}%")
        print(f"  {'AVG':>5} {metrics['AVG']['MAE']:>8.2f} {'':>8s} "
              f"{metrics['AVG']['R2']:>8.4f} {metrics['AVG']['MAPE']:>7.1f}%")

    # 5. Графики loss MLP
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, "b-", alpha=0.7, label="Train Loss")
    ax.plot(val_losses, "r-", alpha=0.7, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("MLP Training", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/mlp_training.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Сохранение
    save_data = {
        "metrics": all_metrics,
        "n_samples": len(df),
        "n_test": data["X_test"].shape[0],
        "features": data["feature_cols"],
        "targets": data["target_cols"],
        "crop_params": {k: {kk: vv for kk, vv in v.items()}
                        for k, v in CROP_PARAMS.items()},
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # Сохранение моделей
    with open("results/sklearn_models.pkl", "wb") as f:
        pickle.dump({
            "linear": model_lr,
            "rf": model_rf,
            "xgb": model_xgb,
            "scaler_X": data["scaler_X"],
            "scaler_y": data["scaler_y"],
            "label_encoder": data["label_encoder"],
            "feature_cols": data["feature_cols"],
        }, f)

    torch.save({
        "model_state": model_mlp.state_dict(),
        "input_dim": data["X_train_s"].shape[1],
        "scaler_X": data["scaler_X"],
        "scaler_y": data["scaler_y"],
        "label_encoder": data["label_encoder"],
    }, "results/mlp_model.pth")

    # Тестовые данные для evaluate
    np.savez("results/test_data.npz",
             X_test=data["X_test"], y_test=data["y_test"],
             X_test_s=data["X_test_s"], y_test_s=data["y_test_s"])

    # Сохранение предсказаний
    preds_save = {k: v.tolist() for k, v in all_preds.items()}
    preds_save["actual"] = data["y_test"].tolist()
    with open("results/predictions.json", "w") as f:
        json.dump(preds_save, f)

    print(f"\nВсё сохранено в results/")
    print("Готово!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение рекомендателя удобрений")
    parser.add_argument("--samples", type=int, default=10000, help="Размер датасета")
    parser.add_argument("--epochs", type=int, default=200, help="Эпохи MLP")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
