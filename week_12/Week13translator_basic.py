import streamlit as st
import pandas as pd
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import pipeline  # New Import for Extra Credit

# 1. INITIALIZE SESSION STATE & MODEL CACHING
if "history" not in st.session_state:
    st.session_state["history"] = []

# Why: caching the pipeline prevents Streamlit from reloading the 
# heavy model (several hundred MBs) every time you click a button.
@st.cache_resource
def load_hf_pipeline():
    return pipeline("translation_en_to_fr", model="helsinki-nlp/opus-mt-en-fr")

hf_translator = load_hf_pipeline()

st.title("Pro Translator App + AI Engine")
st.write("Translate to: Chinese, Spanish, German, French")

# --- SIDEBAR CONTROLS ---
target_options = ["Chinese", "Spanish", "German", "French"]
target_choice = st.sidebar.selectbox("Target Language:", target_options)
reverse_mode = st.sidebar.checkbox("Reverse translate?")

target_map = {
    "Chinese": "zh-CN",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
}
target_lang_code = target_map[target_choice]

# --- MAIN UI ---
text = st.text_input("Enter text:", "")

if st.button("Translate"):
    cleaned = (text or "").strip()
    
    if not cleaned:
        st.warning("Please enter text.")
    else:
        try:
            # Step 1: Detect Language
            if len(cleaned) < 3:
                st.warning("Input too short for reliable detection. Defaulting to English (en).")
                detect_lang = "en"
            else:
                detect_lang = detect(cleaned)

            # Step 2: Determine Translation Logic (Logic Swap)
            if reverse_mode:
                src, trg = target_lang_code, detect_lang
            else:
                src, trg = "auto", target_lang_code

            # Step 3: EXTRA CREDIT - Hugging Face vs Google Fallback
            # Logic: If source is 'en' and target is 'fr', use local AI.
            # Note: We check 'detect_lang' and 'target_lang_code' specifically for the pair.
            if detect_lang == "en" and target_lang_code == "fr" and not reverse_mode:
                model_used = "Hugging Face (Local AI)"
                result = hf_translator(cleaned)
                translated = result[0]['translation_text']
            else:
                model_used = "Google Translator (Cloud API)"
                translated = GoogleTranslator(source=src, target=trg).translate(cleaned)
            
            # Display Results
            st.subheader("Results")
            st.caption(f"Engine used: {model_used}")
            col1, col2 = st.columns(2)
            col1.metric("Detected Lang", detect_lang)
            col2.metric("Direction", f"{src} -> {trg}")
            st.success(translated)

            # Save to History
            new_record = {
                "Input text": cleaned,
                "Detected language": detect_lang,
                "Target language": target_choice,
                "Translated output": translated
            }
            st.session_state["history"].append(new_record)

        except Exception as e:
            st.error("Translation failed.")
            st.caption(f"Error: {e}")

# --- HISTORY & DOWNLOAD ---
if st.session_state["history"]:
    st.divider()
    st.subheader("Translation History")
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df)

    csv_data = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download History as CSV", csv_data, "history.csv", "text/csv")





#1. Generally they can actually read what things say and stay on the site; but specifcally it removes away the barreir of a user having to spend the time on a site that they can't read to find the method for them getting to read it.  
#2. I like the idea of a use-case for this within a setting like a hosbital: I mean where would understanable communcation matter the most but here? If a hospital were to be in a port-city/town where mutliable language would flow between each other, then it's likely such an application would increase in practaically.

#   For example, emergencey rooms would need attendents that would have to have greater ablites in mutlilanguages, increasing their rareity. A method for recurting a greater amount of staff could be that adapabtilites either on person, a wearable, or located within the enviroment using fast, easily understood translation from inference of the model.    
   
#3. A perfect example is that above, but that still remains the model-based approach change. In general other public translation services will have changes like that of speed or throughput of customers; a larger range of clientile; etc.   
 
