# Drug Side Effects Classification Project

## 📌 Project Objective
The goal of this project is to build a machine learning classification model that predicts the side effects of a specific drug based on patient demographic details such as:

- Age
- Gender
- Race

This predictive model can help healthcare providers to assess the risk of side effects in different patient profiles before prescribing medication.

---

## 💡 Dataset Description
A synthetic dataset of **400,000** patient records was generated containing:

| Column Name     | Description                         |
|-----------------|-------------------------------------|
| Patient_ID      | Anonymized unique patient ID        |
| Age             | Patient age (0 - 100)               |
| Gender          | Male, Female, or Other              |
| Race            | Patient's racial group              |
| Side_Effect     | None, Mild, Severe, Allergic Reaction |

---

## 🧠 Machine Learning Pipeline

1. **Data Cleaning**  
   - Removed duplicates.
   - Filtered out invalid ages.
   - Handled missing values.

2. **Data Preprocessing**  
   - Encoded categorical data.
   - Scaled numerical features.
   - Partitioned the dataset into training and testing sets.

3. **Model Used**  
   - `Decision Tree Classifier` from Scikit-learn.

4. **Evaluation**  
   - Accuracy Score.
   - Confusion Matrix.
   - Classification Report.

---

## 🚀 How to Run the Project

1. Install the required libraries:

```bash
pip install pandas numpy scikit-learn
