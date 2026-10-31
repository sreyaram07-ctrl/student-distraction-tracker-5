import joblib, pandas as pd, streamlit as st

st.title("🎯 Student Distraction Tracker")
model = joblib.load("model.pkl")

study = st.slider("Study hours", 0.0, 10.0, 4.0, 0.5)
social = st.slider("Social media hours", 0.0, 8.0, 2.0, 0.5)
sleep = st.slider("Sleep hours", 4.0, 10.0, 7.0, 0.5)
breaks = st.slider("Number of breaks", 0, 10, 4)

if st.button("Predict"):
    row = pd.DataFrame([{"study_hours": study,
        "social_media_hours": social,
        "sleep_hours": sleep, "breaks": breaks}])
    st.subheader("Predicted productivity: " + model.predict(row)[0])
    if social > 3: st.warning("Too much social media. Try app limits.")
    if sleep < 6: st.warning("Low sleep. Aim for 7-8 hours.")
    if study < 3: st.info("Add a focused 1-2 hour study session.")
    st.bar_chart(pd.Series(model.feature_importances_, index=row.columns))