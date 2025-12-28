import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Décomposition CAHT – Tx horaire & Coeff achats",
    layout="wide"
)

st.title("CAHT = b + Tx · H(t−tH) + Coeff · A(t−tA)")
st.caption("Régression linéaire par fenêtre glissante avec options : lags, intercept, pénalité ressort")

# =====================================================
# UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Charger un fichier CSV ou Excel", type=["csv", "xls", "xlsx"])
if uploaded_file is None:
    st.stop()

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("Aperçu des données")
st.dataframe(df.head())

cols = list(df.columns)

# =====================================================
# COLONNES
# =====================================================
date_col = st.selectbox("Colonne Date (recommandé)", ["(aucune)"] + cols)
ca_col = st.selectbox("Colonne CAHT", cols)
h_col = st.selectbox("Colonne Heures", cols)
a_col = st.selectbox("Colonne Achats", cols)

# =====================================================
# OPTIONS
# =====================================================
with st.sidebar:
    st.header("Options du modèle")

    use_lags = st.checkbox("Activer décalage temporel", value=True)
    if use_lags:
        tH = st.slider("Lag Heures tH", 0, 12, 0)
        tA = st.slider("Lag Achats tA", 0, 12, 1)
    else:
        tH, tA = 0, 0

    use_intercept = st.checkbox("Inclure intercept b", value=True)

    use_penalty = st.checkbox("Activer pénalité ressort", value=True)
    if use_penalty:
        Tx0 = st.number_input("Tx horaire cible (Tx0)", value=55.0)
        C0 = st.number_input("Coeff cible (C0)", value=1.30)
        lam_Tx = st.number_input("λ Tx", value=50.0)
        lam_C = st.number_input("λ Coeff", value=50.0)
    else:
        Tx0, C0, lam_Tx, lam_C = 0.0, 0.0, 0.0, 0.0

    st.header("Fenêtre glissante")
    window_size = st.number_input("Taille fenêtre m", min_value=4, value=12)

    metric_choice = st.selectbox("Métrique affichée", ["R²", "MAE", "RMSE"])

# =====================================================
# PRÉPARATION DES DONNÉES
# =====================================================
work = df.copy()

if date_col != "(aucune)":
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

for c in [ca_col, h_col, a_col]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

work["H_lag"] = work[h_col].shift(tH)
work["A_lag"] = work[a_col].shift(tA)

work = work.dropna(subset=[ca_col, "H_lag", "A_lag"]).reset_index(drop=True)

if date_col != "(aucune)":
    dates = work[date_col]
else:
    dates = work.index

n = len(work)
m = int(window_size)

if n < m + 1:
    st.error("Pas assez de données après nettoyage.")
    st.stop()

# =====================================================
# RÉGRESSION (OBJECTIF NON RÉSOLU)
# =====================================================
def fit_window(X, y, use_intercept, use_penalty):
    if use_intercept:
        Lam = np.diag([0, lam_Tx, lam_C]) if use_penalty else np.zeros((3, 3))
        beta0 = np.array([0, Tx0, C0])
    else:
        Lam = np.diag([lam_Tx, lam_C]) if use_penalty else np.zeros((2, 2))
        beta0 = np.array([Tx0, C0])

    XtX = X.T @ X
    Xty = X.T @ y

    beta = np.linalg.lstsq(XtX + Lam, Xty + Lam @ beta0, rcond=None)[0]

    y_hat = X @ beta
    resid = y - y_hat

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid ** 2))

    if use_penalty:
        if use_intercept:
            pen = lam_Tx * (beta[1] - Tx0) ** 2 + lam_C * (beta[2] - C0) ** 2
        else:
            pen = lam_Tx * (beta[0] - Tx0) ** 2 + lam_C * (beta[1] - C0) ** 2
    else:
        pen = 0.0

    return beta, r2, mae, rmse, pen

# =====================================================
# FENÊTRE GLISSANTE
# =====================================================
rows = []

for end in range(m - 1, n):
    w = work.iloc[end - m + 1:end + 1]

    y = w[ca_col].to_numpy()
    H = w["H_lag"].to_numpy()
    A = w["A_lag"].to_numpy()

    if use_intercept:
        X = np.column_stack([np.ones(len(w)), H, A])
    else:
        X = np.column_stack([H, A])

    beta, r2, mae, rmse, pen = fit_window(X, y, use_intercept, use_penalty)

    if use_intercept:
        b, tx, c = beta
    else:
        b, tx, c = 0.0, beta[0], beta[1]

    rows.append({
        "Date": dates.iloc[end],
        "Tx_horaire": tx,
        "Coeff": c,
        "Intercept": b,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Penalty": pen
    })

res = pd.DataFrame(rows)

# =====================================================
# AFFICHAGE
# =====================================================
st.subheader("Résultats")
st.dataframe(res)

def plot(y, title, ylabel):
    plt.figure(figsize=(12, 4))
    plt.plot(res["Date"], y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

plot(res["Tx_horaire"], "Évolution du taux horaire estimé", "€/h")
plot(res["Coeff"], "Évolution du coefficient achats", "€/€")

if use_intercept:
    plot(res["Intercept"], "Évolution de l’intercept b", "€")

if metric_choice == "R²":
    plot(res["R2"], "R² par fenêtre", "R²")
elif metric_choice == "MAE":
    plot(res["MAE"], "MAE par fenêtre", "€")
else:
    plot(res["RMSE"], "RMSE par fenêtre", "€")

if use_penalty:
    plot(res["Penalty"], "Valeur de la pénalité ressort", "pénalité")
