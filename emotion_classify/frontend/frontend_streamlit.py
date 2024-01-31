import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np

st.title("Emotion Classification App")

input_text = st.text_area("Enter your text:", "How are you?")

def classify_emotion_michel(input_text):
    response = requests.post("http://127.0.0.1:8000/classify-emotion/", json={"text": input_text})
    return response

def display_emotion_probabilities(emotion_prob_m, emotion_prob_r):
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    emojis = {
        "anger": "😡",
        "disgust": "🤢",
        "fear": "😨",
        "joy": "😄",
        "neutral": "😐",
        "sadness": "😢",
        "surprise": "😲",
    }
    
    st.write("## Emotion Mixture:")
    st.write("### Michel's model")
    for emotion, prob in zip(emotions, emotion_prob_m[:7]):
        st.write(f"{emotion, emojis[emotion]}: {prob:.4f}")

    st.write("### Roberta's model")
    for emotion, prob in zip(emotions, emotion_prob_r[:7]):
        st.write(f"{emotion, emojis[emotion]}: {prob:.4f}")

    bar_width = 0.35

    r1 = np.arange(len(emotions))
    r2 = [x + bar_width for x in r1]

    fig, ax = plt.subplots()
    ax.bar(r1, emotion_prob_m, color='b', width=bar_width, edgecolor='grey', label="Michel's Model")
    ax.bar(r2, emotion_prob_r, color='r', width=bar_width, edgecolor='grey', label="Roberta's Model")

    ax.set_xlabel('Emotions', fontweight='bold', fontsize=15)
    ax.set_xticks([r + bar_width / 2 for r in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel('Probability', fontweight='bold', fontsize=15)
    ax.set_title('Comparative Emotion Probabilities', fontweight='bold', fontsize=18)
    ax.legend()

    st.pyplot(fig)

    # st.write("Bar Chart")
    # fig, ax = plt.subplots()
    # ax.bar(emotions, emotion_prob_m[:7])
    # ax.set_ylabel('Probability')
    # ax.set_title('Emotion Probabilities')
    # st.pyplot(fig)

    # st.write("Pie Chart")
    # fig, ax = plt.subplots()
    # ax.pie(emotion_prob_m[:7], labels=emotions, autopct='%1.1f%%', startangle=90)
    # ax.axis('equal')
    # ax.set_title('Emotion Probabilities')
    # st.pyplot(fig)

def display_past_classifications():
    st.header("Past Classifications")
    predictions_response = requests.get(f"http://127.0.0.1:8000/get-predictions/?limit=10")

    if predictions_response.status_code == 200:
        predictions = predictions_response.json()
        if predictions:
            st.write("Recent Predictions:")
            for prediction in predictions:
                st.write(f"### Input Text: {prediction['input_text']}")
                st.write("### Michel's model")
                st.write(f"Emotion Probabilities: Anger 😡: {prediction['anger_prob_m']:.4f}, Disgust 🤢: {prediction['disgust_prob_m']:.4f}, Fear 😨: {prediction['fear_prob_m']:.4f}, Joy 😄: {prediction['joy_prob_m']:.4f}, Neutral 😐: {prediction['neutral_prob_m']:.4f}, Sadness 😢: {prediction['sadness_prob_m']:.4f}, Surprise 😲: {prediction['surprise_prob_m']:.4f}")
                
                st.write("### Roberta's model")
                st.write(f"Emotion Probabilities: Anger 😡: {prediction['anger_prob_r']:.4f}, Disgust 🤢: {prediction['disgust_prob_r']:.4f}, Fear 😨: {prediction['fear_prob_r']:.4f}, Joy 😄: {prediction['joy_prob_r']:.4f}, Neutral 😐: {prediction['neutral_prob_r']:.4f}, Sadness 😢: {prediction['sadness_prob_r']:.4f}, Surprise 😲: {prediction['surprise_prob_r']:.4f}")

                st.write("---")
        else:
            st.info("No predictions available in the database.")
    else:
        st.error(f"Error: {predictions_response.text}")

if st.button("Classify Emotion of the Text"):
    if input_text:
        response = classify_emotion_michel(input_text)

        if response.status_code == 200:
            emotion_prob_m = response.json()["emotion_prob_m"]
            emotion_prob_r = response.json()["emotion_prob_r"]

            display_emotion_probabilities(emotion_prob_m, emotion_prob_r)
        else:
            st.error(f"Error Encountered! Please try again.: {response.text}")
    else:
        st.warning("Please enter some text for the classification purpose.")

if st.button("Show past classifications"):
    display_past_classifications()