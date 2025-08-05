# MedLensPro – AI-Powered Symptom Checker & Health Assistant 🩺🤖

MedLensPro is a cutting-edge AI-powered health assistant designed to provide users with disease predictions based on symptoms. Built using Streamlit, it features an interactive UI, ML-based prediction models, and an extended dataset to enhance accuracy and awareness.

---

## 🚀 Features

- 🔍 Symptom-based disease prediction using Machine Learning  
- 📈 Enhanced dataset for improved accuracy  
- 💬 User-friendly Streamlit interface  
- 📄 Real-time outputs with optional insights  
- 🌐 Can be deployed locally or on the cloud (e.g., Streamlit Cloud)  

---

## 🧠 How It Works

1. Users select their symptoms from a dropdown.  
2. The AI model processes the inputs and predicts potential diseases.  
3. The result is displayed instantly with a possible recommendation.  

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python (pandas, scikit-learn)  
- **Model**: Trained ML model using classification algorithm  
- **Other Tools**: CSV datasets, joblib/pickle for model serialization  

---

## 📁 Project Structure

```
MedLensPro/
│
├── main.py               # Main controller file
├── streamlit_app.py      # Streamlit UI
├── model.pkl             # Trained ML model
├── dataset.csv           # Symptom-disease dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/your-username/MedLensPro-AI-Powered-Symptom-Checker.git
cd MedLensPro-AI-Powered-Symptom-Checker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

---

## ✅ Requirements

- Python 3.7+  
- Streamlit  
- scikit-learn  
- pandas  

---

## 🧪 Demo (Optional)

You can test a demo at: [Streamlit Cloud App URL – if deployed]

---

## 📌 TODO (Optional)

- [ ] Add chatbot assistant  
- [ ] Integrate medical tips  
- [ ] Add PDF report generator  

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.

---


## 👩‍💻 Author

**Harshita Singh** – [GitHub](https://github.com/harshi75)
