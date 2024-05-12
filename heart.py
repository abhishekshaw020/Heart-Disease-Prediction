import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import pickle


df = pd.read_csv('dataset.csv')


df.dropna(inplace=True)


X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5]
}
model = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_

# Create the best model with the best parameters
best_model = GradientBoostingClassifier(**best_params, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Save the best model to a file
filename = 'heart_disease.sav'
pickle.dump(best_model, open(filename, 'wb'))

# Streamlit app
st.sidebar.header("Heart Disease Information")
st.sidebar.write("To know more about precautions about this disease, visit this website:")
st.sidebar.write("[Prevention of Heart Disease](https://www.cdc.gov/heartdisease/prevention.htm)")

# Explanation of terms
st.sidebar.header("Meanings of Terms")
st.sidebar.write("**Age**: The person's age in years.")
st.sidebar.write("**Gender**: The person's gender.")
st.sidebar.write("**Chest Pain Type (cp)**: The type of chest pain experienced by the person (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic).")
st.sidebar.write("**Resting Blood Pressure (trestbps)**: The person's resting blood pressure (in mm Hg).")
st.sidebar.write("**Serum Cholesterol (chol)**: The person's serum cholesterol level (in mg/dl).")
st.sidebar.write("**Fasting Blood Sugar > 120 mg/dl (fbs)**: Whether the person's fasting blood sugar is greater than 120 mg/dl (1: yes, 0: no).")
st.sidebar.write("**Resting Electrocardiographic Results (restecg)**: The person's resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy).")
st.sidebar.write("**Maximum Heart Rate Achieved (thalach)**: The person's maximum heart rate achieved during exercise.")
st.sidebar.write("**Exercise Induced Angina (exang)**: Whether the person has exercise-induced angina (1: yes, 0: no).")
st.sidebar.write("**ST Depression Induced By Exercise (oldpeak)**: The person's ST depression induced by exercise relative to rest.")
st.sidebar.write("**Slope Of The Peak Exercise ST Segment (slope)**: The slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping).")
st.sidebar.write("**Number Of Major Vessels Colored By Flourosopy (ca)**: The number of major vessels colored by fluoroscopy (0-3).")
st.sidebar.write("**Thalassemia Type (thal)**: The person's thalassemia type (3: normal, 6: fixed defect, 7: reversible defect).")

st.header("Know If You Are Affected By Heart Disease")
st.write("Please provide the following information:")

age = st.number_input("Age", min_value=0, max_value=150, step=1)
sex = st.radio("Gender", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, step=1)
chol = st.number_input("Serum Cholesterol", min_value=0, max_value=600, step=1)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, step=1)
exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression Induced By Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope Of The Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number Of Major Vessels Colored By Flourosopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])

# Mapping categorical features to numerical values
sex_mapping = {"Male": 1, "Female": 0}
fbs_mapping = {"No": 0, "Yes": 1}
exang_mapping = {"No": 0, "Yes": 1}

sex = sex_mapping[sex]
fbs = fbs_mapping[fbs]
exang = exang_mapping[exang]

# Make prediction
prediction = best_model.predict_proba([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0][1]

if st.button("Predict"):
    if prediction > 0.5:
        st.warning("You Might Be Affected By Heart Disease")
    elif prediction <= 0.5:
        st.success("You Are Safe")
        st.text(f"Probability of being safe: {100 - round(prediction * 100, 2)}%")

# Heart Disease Scale Chart
st.subheader("Heart Disease Risk Scale")
st.write("This chart shows the range of predicted probabilities for heart disease.")

plt.figure(figsize=(8, 6))
x_values = np.linspace(0, 1, 100)
y_values = [best_model.predict_proba([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0][1] for age in x_values]
plt.plot(x_values, y_values, color='blue')
plt.fill_between(x_values, y_values, color='skyblue', alpha=0.3)
plt.xlabel("Predicted Probability")
plt.ylabel("Heart Disease Risk")
plt.title("Heart Disease Risk Scale")
st.pyplot(plt)
