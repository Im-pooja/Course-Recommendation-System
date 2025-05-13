import streamlit as st
import pandas as pd
import numpy as np
import wikipedia
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

# Streamlit app
st.title("Course Recommendation System")

# Step 1: Upload Dataset
st.header("Upload User Data Excel File")
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

if uploaded_file:
    # Load the Excel file
    data = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    
    # Display the entire dataset
    st.write("Data Preview:")
    st.dataframe(data)  # Display entire dataframe with scrolling functionality

    # Step 2: Prepare Data
    X = data.drop('DOMAIN', axis=1)  # Features: user responses
    y = data['DOMAIN']  # Labels: domains

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode the target labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # One-hot encode the labels for multi-class classification
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Step 3: Define the Model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Display the model summary
    st.write("Model Summary:")
    st.text(model.summary())

    # Step 4: Train the Model
    if st.button("Train the Model"):
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        st.success("Model Training Completed!")

        # Step 5: Evaluate the Model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        st.write(f'Test Accuracy: {test_accuracy * 100:.2f}%')

        # Save the model
        model.save('model.h5')
        st.success("Model saved as 'model.h5'!")

    # Step 6: Predict Domain for New User Input
    st.header("Predict Domain for New User")

    # Create input fields for the user to fill out
    user_input = []
    for feature in X.columns:
        value = st.slider(f"Rate your {feature} skills (1 to 5):", 1, 5, 3)
        user_input.append(value)

    if st.button("Predict Domain"):
        # Convert the user input to a numpy array and reshape for model input
        user_input = np.array(user_input).reshape(1, -1)

        # Scale the input using the trained scaler
        user_input_scaled = scaler.transform(user_input)

        # Predict the domain using the deep learning model
        predicted_domain = model.predict(user_input_scaled)
        predicted_domain_label = np.argmax(predicted_domain)

        # Convert the predicted label back to the domain name
        predicted_domain_name = label_encoder.inverse_transform([predicted_domain_label])[0]

        # Display the result
        st.success(f"Recommended Domain: {predicted_domain_name}")
        try:
            summary = wikipedia.summary(predicted_domain_name, sentences=3)
            st.write(f"Here's a brief summary of {predicted_domain_name}:")
            st.write(summary)
        except wikipedia.exceptions.DisambiguationError as e:
            st.write("The topic is ambiguous, try refining your search.")
        except wikipedia.exceptions.PageError:
            st.write("Could not find a summary for this course. Please check the course name and try again.")
        except Exception as e:
            st.write("An error occurred while fetching the summary.")
        if predicted_domain_name=="CLOUD COMPUTING":
            st.write("https://www.coursera.org/learn/introduction-to-cloud")
        elif predicted_domain_name=="Data Science":
             st.write("https://www.coursera.org/specializations/introduction-data-science")
        elif predicted_domain_name=="ARTIFICIAL INTELLIGENCE":
             st.write("https://www.coursera.org/learn/ai-for-everyone")
        elif predicted_domain_name=="WEB DEVELOPMENT":
             st.write("https://www.coursera.org/specializations/html-css-javascript-for-web-developers")
        elif predicted_domain_name==" APP DEVELOPMENT":
             st.write("https://www.coursera.org/professional-certificates/meta-android-developer")
        elif predicted_domain_name=="CYBER SECURITY":
             st.write("https://www.coursera.org/learn/cybersecurity-for-everyone")
        elif predicted_domain_name=="BLOCKCHAIN DEVELOPMENT":
             st.write("https://www.coursera.org/specializations/blockchain")
        elif predicted_domain_name=="INTERNET OF THINGS":
             st.write("https://www.coursera.org/specializations/iot")
        elif predicted_domain_name=="GAME DEVELOPMENT":
             st.write("https://www.coursera.org/specializations/game-design-and-development")