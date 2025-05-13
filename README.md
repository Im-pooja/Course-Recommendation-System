# 🎓 Course Recommendation System

A deep learning-powered Course Recommendation System that suggests the most suitable course or domain for users based on their skills, personality traits, and preferences. Built with a user-friendly Streamlit interface, the system accepts structured Excel data for model training and prediction.

## 🧠 Overview

This project uses a Feedforward Neural Network (FNN) to analyze user responses to various parameters like:

- Coding
- Design
- Communication
- Data Analysis
- Critical Thinking
- Problem Solving
- Extroversion / Introversion
- Risk Appetite
- Curiosity
- Creativity
- Management Orientation

Based on these, the model predicts the most aligned **domain/course** from a predefined set.

## 🚀 Features

- Upload Excel data (training and test sets)
- Automatically trains a deep learning model
- Predicts suitable domains/courses for users
- Displays accuracy metrics and confusion matrix
- Provides clickable course links for user guidance
- Lightweight and easy-to-use Streamlit UI

## 🖥️ Tech Stack

- **Python**
- **TensorFlow / Keras** - Model Building
- **Pandas / NumPy** - Data Processing
- **Matplotlib / Seaborn** - Visualization
- **Streamlit** - Web Interface

## 📊 Sample Output

- Top-1 course prediction
- Accuracy and loss plots
- Confusion matrix
- Suggested course links
## 📌 Usage
- Launch the app using Streamlit.

- Upload your Excel file containing user data.

- Click on "Train Model" or "Predict" based on your workflow.

- View predictions and suggested courses with confidence scores.

## 📚 Example Data Format
| Coding | Design | Communication | Extrovert | Risk Taker | ... | Domain |
| ------ | ------ | ------------- | --------- | ---------- | --- | ------ |
| 8      | 6      | 7             | 1         | 1          | ... | AI     |
## ✅ Future Improvements
Add multilingual support

Integrate resume parsing for auto-profile generation

Connect with real-time course databases (Coursera, edX)

Build REST API for scalability

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your proposed updates.
