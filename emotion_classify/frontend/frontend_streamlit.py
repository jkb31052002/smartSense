import streamlit as st
import requests
import matplotlib.pyplot as plt

st.title("Emotion Classification App")

input_text = st.text_area("Enter your text:", "How are you?")
if st.button("Classify Emotion"):
    if input_text:
        response = requests.post("http://127.0.0.1:8000/classify-emotion/", json={"text": input_text})

        if response.status_code == 200:
            emotion_probs = response.json()["emotion_probs"]

            emotions = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

            st.write("Emotion Mixture:")
            for emotion, prob in zip(emotions, emotion_probs[:5]):
                st.write(f"{emotion}: {prob:.4f}")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Bar Chart")
                fig, ax = plt.subplots()
                ax.bar(emotions, emotion_probs[:5]) 
                ax.set_ylabel('Probability')
                ax.set_title('Emotion Probabilities')
                st.pyplot(fig)

            with col1:
                st.write("Pie Chart")
                fig, ax = plt.subplots()
                ax.pie(emotion_probs[:5], labels=emotions, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                ax.set_title('Emotion Probabilities')
                st.pyplot(fig)

        else:
            st.error(f"Error Encountered! Please try again.: {response.text}")
    else:
        st.warning("Please enter some text for the classification purpose.")

if st.button("Show past classifications"):
    st.header("Past Classifications")
    predictions_response = requests.get(f"http://127.0.0.1:8000/get-predictions/?limit=10")

    if predictions_response.status_code == 200:
        predictions = predictions_response.json()
        if predictions:
            st.write("Recent Predictions:")
            for prediction in predictions:
                st.write(f"Input Text: {prediction['input_text']}")
                st.write(f"Emotion Probabilities: Anger: {prediction['anger_prob']:.4f}, Fear: {prediction['fear_prob']:.4f}, Joy: {prediction['joy_prob']:.4f}, Sadness: {prediction['sadness_prob']:.4f}, Surprise: {prediction['surprise_prob']:.4f}")

                st.write("---")
        else:
            st.info("No predictions available in the database.")
    else:
        st.error(f"Error: {predictions_response.text}")
