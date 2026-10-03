import pickle
import streamlit as st

with open("model/phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")
st.title("🛡️ Phishing Email Detection Model")
st.write("Enter email text and classify it as **Phishing** or **Safe**.")

email = st.text_area("Email content", height=220)

if st.button("Analyze Email"):
    if not email.strip():
        st.warning("Please enter email text.")
    else:
        prediction = model.predict([email])[0]
        if prediction == "Phishing":
            st.error("🚨 Phishing")
        else:
            st.success("✅ Safe")

st.caption("Educational model; a Safe prediction does not prove an email is safe.")
