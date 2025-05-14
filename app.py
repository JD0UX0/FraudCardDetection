import streamlit as st
import numpy as np
import joblib
import os

# Jellemzők és elérhető modellek
feature_names = ["V1", "V2", "V3", "V4", "Amount"]
models = {
    "Logistic Regression": "logreg_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "MLP (Neurális háló)": "mlp_model.pkl"
}

st.set_page_config(page_title="Csalás előrejelzés", layout="centered")
st.title("💳 Csalás Előrejelzés")

# Bemeneti mezők
user_inputs = []
for name in feature_names:
    val = st.number_input(f"{name} érték:", format="%.5f")
    user_inputs.append(val)

# Modellválasztó
model_choice = st.selectbox("Modell kiválasztása:", list(models.keys()))

# Előrejelzés gomb
if st.button("🔍 Előrejelzés futtatása"):
    try:
        model_path = os.path.join(os.path.dirname(__file__), models[model_choice])
        model = joblib.load(model_path)

        input_array = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        # Próbáljuk megszerezni a valószínűséget
        try:
            probas = model.predict_proba(input_array)[0]
            confidence = max(probas)
            label = f"{'⚠️ Csalás' if prediction == 1 else '✅ Nem csalás'} ({confidence:.2%} bizonyosság)"
        except:
            label = f"{'⚠️ Csalás' if prediction == 1 else '✅ Nem csalás'}"

        if prediction == 1:
            st.error(f"Eredmény: {label}")
        else:
            st.success(f"Eredmény: {label}")

    except Exception as e:
        st.error(f"Hiba történt: {e}")
