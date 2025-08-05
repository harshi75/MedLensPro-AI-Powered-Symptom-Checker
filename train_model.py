import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Sample dummy symptoms and diseases for training (expand this with real data)
symptoms = [
    "fever", "cough", "cold", "headache", "fatigue", "anxiety", "depression",
    "vomiting", "nausea", "chest pain", "dizziness", "insomnia", "loss of appetite",
    "back pain", "sore throat", "shortness of breath", "rash", "joint pain",
    "irritability", "confusion", "abdominal pain", "tremors", "panic", "paranoia"
]

# Create synthetic training data
data = []
labels = []

for i in range(1000):
    row = [0] * len(symptoms)
    disease = "Anxiety" if i % 2 == 0 else "Common Cold"
    if disease == "Anxiety":
        for sym in ["anxiety", "insomnia", "fatigue"]:
            row[symptoms.index(sym)] = 1
    else:
        for sym in ["cough", "cold", "fever"]:
            row[symptoms.index(sym)] = 1
    data.append(row)
    labels.append(disease)

# Train the model
model = RandomForestClassifier()
model.fit(data, labels)

# Save the model and symptom list
with open("medlens_pro_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("medlens_pro_symptom_classes.pkl", "wb") as f:
    pickle.dump(symptoms, f)

print("✅ Model and symptom list saved successfully.")



