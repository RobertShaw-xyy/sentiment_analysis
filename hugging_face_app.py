import streamlit as st
from transformers import pipeline



# sentiment_pipeline = pipeline('sentiment-analysis',model='distilbert-base-uncased-finetuned-sst-2-english', revision='af0f99b')
sentiment_pipeline = pipeline('sentiment-analysis')

user_input = st.text_area("Input text for sentiment analysis", "Enter text here...")
analyze_button = st.button('Analyzing emotions')


if analyze_button:
    with st.spinner('under analysis...'):
        result = sentiment_pipeline(user_input)
        st.write("Sentiment analysis results:", result[0]['label'])
        st.write("confidence level :{:.2f}%".format(result[0]['score'] * 100))
