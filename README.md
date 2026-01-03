🫀 Heart Disease Prediction Using Hybrid Machine Learning Models
📌 Project Overview

Heart disease remains one of the leading causes of mortality worldwide. Early detection plays a crucial role in preventing severe complications and reducing death rates. This project presents a machine learning–based heart disease prediction system using a multi-feature, hybrid classification approach that evaluates multiple algorithms and selects the best-performing model for final prediction.

The system integrates data preprocessing, feature extraction, multiple classifiers, performance evaluation, and a user-friendly web interface to provide accurate and interpretable heart disease predictions.

This implementation is based on the methodology and results published in the International Research Journal of Engineering and Technology (IRJET), July 2022 

Heart Disease Prediction

.

🎯 Objectives

Predict the presence or absence of heart disease using clinical attributes

Compare multiple machine learning classifiers

Select the best-performing model based on evaluation metrics

Provide an interactive user interface for real-world usability

Enable early risk awareness and preventive healthcare actions

🧠 Methodology Overview

The system follows a structured machine learning pipeline:

Patient Data Input

Data Preprocessing

Handling missing values

Noise and outlier reduction

Feature normalization

Feature Extraction

Selection of clinically significant attributes

Model Training

Multiple classifiers trained on the same dataset

Hybrid Model Selection

Best model chosen based on performance

Prediction & Result Display

Diet Plan Recommendation (UI Feature)

This hybrid approach ensures robustness and improves prediction accuracy over single-model systems 

Heart Disease Prediction

.

📊 Dataset Description

Source: UCI Machine Learning Repository

Records: 303 patient samples

Features: 14 clinical attributes

Target Variable: Presence of heart disease (binary classification)
| Feature                         | Description                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| Age                             | Age of patient (years)                                          |
| Sex                             | 0 = Female, 1 = Male                                            |
| Chest Pain (cp)                 | Typical angina, atypical angina, non-anginal pain, asymptomatic |
| Resting Blood Pressure          | Measured in mm Hg                                               |
| Serum Cholesterol               | mg/dl                                                           |
| Fasting Blood Sugar (fbs)       | >120 mg/dl (1 = true, 0 = false)                                |
| Resting ECG                     | Normal / ST abnormality / LV hypertrophy                        |
| Max Heart Rate (thalach)        | Maximum heart rate achieved                                     |
| Exercise-Induced Angina (exang) | Yes / No                                                        |
| ST Depression (oldpeak)         | Induced by exercise                                             |
| Slope                           | Slope of peak exercise ST segment                               |
| CA                              | Number of major vessels colored (0–3)                           |
| Thal                            | Normal / Fixed defect / Reversible defect                       |


Heart Disease Prediction

🧹 Data Preprocessing

Missing values handled using statistical normalization

Features standardized using mean and standard deviation

Noise and outliers removed to improve model generalization




	​


This ensures uniform feature scaling across models 

Heart Disease Prediction

🤖 Machine Learning Models Used

The following classifiers were trained and evaluated:

Logistic Regression

Gaussian Naive Bayes

Linear Support Vector Classifier (Linear SVC)

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Support Vector Machine (SVM)

Each model was evaluated using the same training and testing split to ensure fairness 

Heart Disease Prediction


📈 Model Evaluation Metrics

Models were compared using:

Accuracy

Precision

Sensitivity (Recall)

Specificity

F1 Score

ROC Curve

Log Loss

Matthews Correlation Coefficient

🏆 Best Performing Model

Linear SVC achieved the highest overall performance:

Metric	Value
Accuracy	90.78%
Precision	96.87%
Sensitivity	83.78%
F1 Score	89.85%
ROC	90.60%

Hence, Linear SVC was selected as the final prediction model 

Heart Disease Prediction

.

🖥 User Interface Features

The project includes a complete web-based UI, enabling non-technical users to interact with the model:

UI Components

User Registration & Login

Patient Data Input Form

Prediction Result Page

Diet Plan Recommendation Page

Database Storage of User Records

Users enter medical parameters through dropdowns and numeric inputs, and the system returns a clear prediction:

Heart Disease Present

No Heart Disease

Heart Disease Prediction

🛠 Technology Stack

Python

scikit-learn

NumPy / Pandas

Matplotlib / Seaborn

HTML / CSS / Backend Framework

MySQL / phpMyAdmin (for user data storage)

🚀 How to Run the Project (Typical Setup)
git clone https://github.com/SrilakshmiNarayanaMurthy/HeartDiseasePrediction.git
cd HeartDiseasePrediction
pip install -r requirements.txt
python app.py


(Adjust commands based on your actual repo structure.)

⚠️ Disclaimer

This system is intended for educational and research purposes only.
It is not a substitute for professional medical diagnosis or treatment.

📌 Conclusion

This project demonstrates how a hybrid machine learning framework can significantly improve heart disease prediction accuracy. By combining multiple classifiers and selecting the best-performing model, the system provides reliable predictions, early risk awareness, and practical health guidance.

The integration of analytics with a user-friendly interface makes this project suitable for academic research, healthcare analytics portfolios, and ML demonstrations 

Heart Disease Prediction

.

🔮 Future Enhancements

Multi-stage disease severity classification

SMS / alert integration for patients

Real-time data ingestion from healthcare centers

Explainable AI (SHAP/LIME) for clinical interpretability

Cloud deployment with secure APIs
