

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide",
)

MODEL_PATH = Path("outputs/models")
DATA_PATH = Path("data/processed")

RAW_FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ------------------------------------------------------------------
# Artifact loading (cached so the app doesn't reload pickles on every interaction)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [
        p for p in [
            "knn_imputer.pkl", "robust_scaler.pkl", "kmeans_clusterer.pkl",
            "xgboost_final_model.pkl", "kmeans_rf_final_model.pkl", "selection_metadata.pkl",
        ]
        if not (MODEL_PATH / p).exists()
    ]
    if missing:
        return None, missing

    artifacts = {
        "imputer": joblib.load(MODEL_PATH / "knn_imputer.pkl"),
        "scaler": joblib.load(MODEL_PATH / "robust_scaler.pkl"),
        "kmeans": joblib.load(MODEL_PATH / "kmeans_clusterer.pkl"),
        "xgb_model": joblib.load(MODEL_PATH / "xgboost_final_model.pkl"),
        "kmeans_rf_model": joblib.load(MODEL_PATH / "kmeans_rf_final_model.pkl"),
        "selection_metadata": joblib.load(MODEL_PATH / "selection_metadata.pkl"),
    }
    return artifacts, []


# ------------------------------------------------------------------
# Preprocessing chain
# ------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """All *possible* engineered columns. Only used if a fitted artifact actually asks for them."""
    df = df.copy()
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Age_Glucose"] = df["Age"] * df["Glucose"]
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Glucose_bin"] = pd.cut(df["Glucose"], [0, 100, 126, 300], labels=[0, 1, 2]).astype(int)
    df["BMI_bin"] = pd.cut(df["BMI"], [0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    df["Insulin_Glucose_ratio"] = df["Insulin"] / (df["Glucose"] + 1)
    return df


def _expected_columns(fitted_obj, fallback):
    """Best-effort read of what a fitted sklearn object was actually trained on."""
    cols = getattr(fitted_obj, "feature_names_in_", None)
    return list(cols) if cols is not None else list(fallback)


def impute_raw(raw_input: dict, artifacts: dict) -> pd.DataFrame:
    """Raw patient dict -> single-row DataFrame of imputed values in original clinical units.
    Reused by both preprocess_patient (feeds the model) and the radar chart (real-world display)."""
    row = pd.DataFrame([raw_input])[RAW_FEATURE_ORDER]
    row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)

    imputer = artifacts["imputer"]
    imputer_cols = _expected_columns(imputer, RAW_FEATURE_ORDER)
    imputed = pd.DataFrame(imputer.transform(row[imputer_cols]), columns=imputer_cols)
    return imputed


def preprocess_patient(raw_input: dict, artifacts: dict, model):
    """Raw patient dict -> feature row matching exactly what `model` was fit on."""
    imputed = impute_raw(raw_input, artifacts)

    scaler = artifacts["scaler"]
    scaler_cols = _expected_columns(scaler, imputed.columns)

    candidate_pool = add_features(imputed)  # superset: raw + every engineered column
    missing = [c for c in scaler_cols if c not in candidate_pool.columns]
    if missing:
        raise ValueError(
            f"The fitted scaler expects columns this app doesn't know how to build: {missing}. "
            "The preprocessing pipeline has likely changed — update add_features() to match."
        )
    pre_scale_row = candidate_pool[scaler_cols]
    scaled = pd.DataFrame(scaler.transform(pre_scale_row), columns=scaler_cols)

    # Step 4: attach K-Means distance features only if `model` was actually trained with them
    model_cols = getattr(model, "feature_names_in_", None)
    needs_cluster_dists = model_cols is not None and any(
        str(c).startswith("Cluster_Dist_") for c in model_cols
    )
    if needs_cluster_dists:
        kmeans = artifacts["kmeans"]
        kmeans_cols = _expected_columns(kmeans, scaled.columns)
        dist_cols = [f"Cluster_Dist_{i+1}" for i in range(kmeans.n_clusters)]
        dists = pd.DataFrame(kmeans.transform(scaled[kmeans_cols]), columns=dist_cols)
        scaled = pd.concat([scaled.reset_index(drop=True), dists.reset_index(drop=True)], axis=1)

    # Final: if the model tells us its exact expected columns/order, conform to it exactly
    if model_cols is not None:
        missing_for_model = [c for c in model_cols if c not in scaled.columns]
        if missing_for_model:
            raise ValueError(
                f"Built features are missing columns the model expects: {missing_for_model}"
            )
        scaled = scaled[list(model_cols)]

    return scaled


# ------------------------------------------------------------------
# Radar chart — this patient vs. a typical diabetic-patient profile
# ------------------------------------------------------------------
@st.cache_data
def load_population_reference(_scaler):
    """Reconstruct approximate raw (real-world-unit) training features from the saved,
    already-scaled train_processed.csv, by inverse-transforming through the fitted scaler.
    Returns per-feature min/max (for 0-10 normalization) and per-class means. Returns None
    if train_processed.csv isn't found — the radar chart is skipped gracefully in that case."""
    path = DATA_PATH / "train_processed.csv"
    if not path.exists():
        return None

    train = pd.read_csv(path)
    if "Outcome" not in train.columns:
        return None
    y = train["Outcome"]
    X = train.drop(columns="Outcome")

    scaler_cols = _expected_columns(_scaler, X.columns)
    scaler_cols = [c for c in scaler_cols if c in X.columns]
    if not scaler_cols:
        return None

    X_raw = pd.DataFrame(_scaler.inverse_transform(X[scaler_cols]), columns=scaler_cols)
    X_raw["Outcome"] = y.values

    return {
        "cols": scaler_cols,
        "min": X_raw[scaler_cols].min(),
        "max": X_raw[scaler_cols].max(),
        "diabetic_mean": X_raw.loc[X_raw["Outcome"] == 1, scaler_cols].mean(),
        "nondiabetic_mean": X_raw.loc[X_raw["Outcome"] == 0, scaler_cols].mean(),
    }


def render_patient_radar(imputed_row: pd.DataFrame, pop_ref: dict):
    """Overlaid radar chart: this patient's values vs. the typical diabetic-patient average,
    both normalized to a common 0-10 scale (min-max over the training population) so features
    with very different natural ranges — e.g. Glucose vs DiabetesPedigreeFunction — stay comparable."""
    cols = pop_ref["cols"]
    rng = (pop_ref["max"] - pop_ref["min"]).replace(0, 1)

    def normalize(series):
        return ((series - pop_ref["min"]) / rng * 10).clip(0, 10)

    patient_norm = normalize(imputed_row[cols].iloc[0])
    diabetic_norm = normalize(pop_ref["diabetic_mean"])

    categories = cols + [cols[0]]
    patient_r = patient_norm.tolist() + [patient_norm.tolist()[0]]
    diabetic_r = diabetic_norm.tolist() + [diabetic_norm.tolist()[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=diabetic_r, theta=categories, fill="toself", name="Typical diabetic profile (avg)",
        line=dict(color="crimson"), opacity=0.7,
    ))
    fig.add_trace(go.Scatterpolar(
        r=patient_r, theta=categories, fill="toself", name="This patient",
        line=dict(color="royalblue"), opacity=0.7,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig


# ------------------------------------------------------------------
# Sidebar — patient inputs
# ------------------------------------------------------------------
def patient_input_form():
    st.sidebar.header("Patient Inputs")


    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
    glucose = st.sidebar.slider("Glucose (mg/dL)", 0, 250, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin (mu U/mL)", 0, 850, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 28.0, step=0.1)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.45, step=0.01)
    age = st.sidebar.slider("Age", 18, 100, 33)

    return {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    st.title("Diabetes Risk Prediction")


    artifacts, missing = load_artifacts()

  

    selected_name = artifacts["selection_metadata"]["selected_model"]
    use_kmeans = not selected_name.startswith("XGBoost")
    model = artifacts["kmeans_rf_model"] if use_kmeans else artifacts["xgb_model"]



    raw_input = patient_input_form()

    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.subheader("Patient Summary")
        st.dataframe(pd.DataFrame([raw_input]).T.rename(columns={0: "Value"}), width="stretch")
        predict_clicked = st.button("Predict", type="primary", width="stretch")

    if not predict_clicked:
        st.info("Set the patient's values in the sidebar, then click **Predict**.")
        return

    imputed_row = impute_raw(raw_input, artifacts)
    X_row = preprocess_patient(raw_input, artifacts, model)
    prob = float(model.predict_proba(X_row)[0, 1])
    pred = int(prob >= 0.3)

    with col_right:
        st.subheader("Prediction")
        risk_label = "High risk" if pred == 1 else "Lower risk"
        st.metric("Predicted probability of diabetes", f"{prob:.1%}", risk_label)
        st.progress(min(max(prob, 0.0), 1.0))
        if pred == 1:
            st.warning(
                "Model flags this patient as **likely diabetic** at the default 0.3 threshold. "
                "This is a screening aid, not a diagnosis — clinical follow-up is required."
            )
        else:
            st.success(
                "Model flags this patient as **likely non-diabetic** at the default 0.3 threshold. "
                "This is a screening aid, not a diagnosis."
            )

    st.divider()
    st.subheader("Why did the model say this? (feature-level explanation)")

    CLINICAL_CONTEXT = {
        "Pregnancies": "Number of pregnancies — repeated pregnancies are a modest, non-modifiable risk marker.",
        "Glucose": "Plasma glucose concentration — the single strongest direct clinical marker for diabetes.",
        "BloodPressure": "Diastolic blood pressure — hypertension often co-occurs with metabolic risk.",
        "SkinThickness": "Triceps skinfold thickness — a rough proxy for subcutaneous body fat.",
        "Insulin": "2-hour serum insulin — abnormal levels reflect insulin resistance or deficiency.",
        "BMI": "Body Mass Index — a core, well-established risk factor for type 2 diabetes.",
        "DiabetesPedigreeFunction": "A composite score of family history / genetic predisposition to diabetes.",
        "Age": "Age in years — diabetes risk rises with age, especially past 45.",
        "Cluster_Dist_1": "Distance to patient cluster 1 centroid (K-Means-derived similarity feature).",
        "Cluster_Dist_2": "Distance to patient cluster 2 centroid (K-Means-derived similarity feature).",
    }

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)

        # Handle both possible SHAP output shapes across model/explainer versions
        if isinstance(shap_values, list):
            values_row = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            values_row = shap_values[0]
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[-1]

        explain_df = pd.DataFrame({
            "Feature": X_row.columns,
            "Patient Value": X_row.iloc[0].values,
            "Contribution": np.asarray(values_row, dtype=float),
        })
        explain_df["Direction"] = np.where(
            explain_df["Contribution"] > 0, "↑ Increases risk", "↓ Decreases risk"
        )
        explain_df["Clinical context"] = explain_df["Feature"].map(CLINICAL_CONTEXT).fillna("Engineered feature.")
        explain_df["Abs"] = explain_df["Contribution"].abs()
        explain_df = explain_df.sort_values("Abs", ascending=False).drop(columns="Abs")

        st.dataframe(
            explain_df.style.format({"Patient Value": "{:.2f}", "Contribution": "{:+.4f}"}),
            width="stretch",
            hide_index=True,
        )

        # Plain-language narrative summary of the top drivers
        top = explain_df.head(3)
        increasers = top[top["Contribution"] > 0]["Feature"].tolist()
        decreasers = top[top["Contribution"] < 0]["Feature"].tolist()

        narrative_parts = []
        if increasers:
            narrative_parts.append(
                f"The strongest factors **pushing risk up** for this patient were "
                f"{', '.join(increasers)}."
            )
        if decreasers:
            narrative_parts.append(
                f"The strongest factors **pushing risk down** were {', '.join(decreasers)}."
            )
        narrative_parts.append(
            f"Starting from a baseline risk of {base_value:.1%} across the training population, "
            f"these feature values combined to move the prediction to {prob:.1%} for this patient."
        )
        st.markdown(" ".join(narrative_parts))

        st.caption(
            "Contribution values are SHAP values in probability-of-diabetes units: positive pushes the "
            "prediction toward 'diabetic', negative pushes it toward 'non-diabetic'. Ranked by magnitude "
            "of impact on this specific patient's prediction, not by general importance."
        )
    except Exception as e:
        st.warning(f"Explanation unavailable for this input: {e}")

    st.divider()
    st.subheader("Patient profile vs. typical diabetic profile")

    pop_ref = load_population_reference(artifacts["scaler"])
    if pop_ref is None:
        st.info(
            "Radar comparison needs `data/processed/train_processed.csv` (written by "
            "`03_Preprocessing.ipynb`) alongside `outputs/`. Not found, so this chart is skipped."
        )
    else:
        radar_fig = render_patient_radar(imputed_row, pop_ref)
        st.plotly_chart(radar_fig, width="stretch")
        st.caption(
            "All axes are normalized 0-10 (min-max across the training population) so features on very "
            "different natural scales — e.g. Glucose vs. DiabetesPedigreeFunction — stay visually "
            "comparable. Where the blue (this patient) line sits outside the red (typical diabetic "
            "average) line, that feature is higher than the typical diabetic patient; inside it, lower."
        )


if __name__ == "__main__":
    main()
