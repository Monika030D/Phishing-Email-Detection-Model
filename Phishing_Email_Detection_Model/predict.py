import pickle

with open("model/phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

print("=== Phishing Email Detector ===")
email = input("\nPaste email text: ")
prediction = model.predict([email])[0]

print("\nPrediction:", prediction)
if prediction == "Phishing":
    print("Warning: phishing-like patterns detected.")
else:
    print("The model classified this email as Safe.")
