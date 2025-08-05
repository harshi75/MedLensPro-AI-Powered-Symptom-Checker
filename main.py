import pickle

with open("medlens_pro_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded")

with open("medlens_pro_symptom_classes.pkl", "rb") as f:
    symptoms = pickle.load(f)
print(f"✅ Symptoms loaded: {len(symptoms)} symptoms")