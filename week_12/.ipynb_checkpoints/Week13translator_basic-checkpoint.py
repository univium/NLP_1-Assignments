import streamlit as st
from langdetect import detect
from deep_translator import GoogleTranslator

st.title("Basic Translator App")
st.write("Translate to: Chinese, Spanish, German, French")

text = st.text_input("Enter text:", "")

target_options = ["Chinese", "Spanish", "German", "French"]
target_choice = st.sidebar.selectbox("Target Language:", target_options)

target_map = {
    "Chinese": "zh-CN",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
}
 
#history = {"Input Text": [text], "Detected Language": [detect_lang], "Target Language": [target_choice], "Translated Output": [translated]}

target_lang = target_map[target_choice]

if st.button("Translate"):
    cleaned = (text or "").strip()
    if not cleaned:
        st.warning("Please enter text.")
    else:
        try:
            translated = GoogleTranslator(source="auto", target=target_lang).translate(cleaned)
            detect_lang = detect(text) 
            st.subheader("Translated Text")
            st.write(translated)
            st.subheader("Detected Input Langauge")
            st.write(detect_lang)
#            st.dataframe(history)
        except Exception as e:
            st.error("Translation failed. Please try again.")
            st.caption(f"Error: {type(e).__name__}")


