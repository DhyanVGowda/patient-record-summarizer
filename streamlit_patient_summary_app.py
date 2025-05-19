import subprocess
subprocess.run(["python", "postinstall.py"], check=True)

import streamlit as st
from patient_summary_logic import PatientSummaryProcessor
from dotenv import load_dotenv
import os

load_dotenv()

processor = PatientSummaryProcessor()

st.title("🩺 Patient Report Summarizer")

st.markdown("""
<style>
mark {
    background-color: #ffe599;
    padding: 0 4px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

model_choice = st.selectbox("Choose summarization model:", ["DeepSeek R1", "OpenAI GPT-4o"])

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

uploaded_file = st.file_uploader("Upload a Patient PDF Report", type=["pdf"])
if uploaded_file:
    st.session_state.uploaded_files[uploaded_file.name] = uploaded_file
    st.success(f"Uploaded: {uploaded_file.name}")

if st.session_state.uploaded_files:
    st.subheader("📂 Uploaded Files")
    for name in st.session_state.uploaded_files:
        st.markdown(f"- {name}")

    selected_files = st.multiselect(
        "Select which PDFs to summarize:",
        options=list(st.session_state.uploaded_files.keys()),
        default=list(st.session_state.uploaded_files.keys())
    )

    combine = False
    if len(selected_files) > 1:
        combine = st.checkbox("Combine all selected PDFs into one summary")

    if st.button("Start Summarization") and selected_files:
        if not combine:
            for filename in selected_files:
                uploaded_file = st.session_state.uploaded_files[filename]
                result = processor.process_pdf(uploaded_file, model_choice)
                
                st.subheader(f"📄 Extracted Text - {filename}")
                st.text_area(f"Extracted Text - {filename}", result['text'], height=200)

                st.subheader(f"🏷️ Named Medical Entities - {filename}")
                if result['entities']:
                    for word, label in result['entities']:
                        st.write(f"**{word}** → _{label}_")
                else:
                    st.info("No medical entities found.")

                st.subheader(f"🖋️ Highlighted Summary - {filename}")
                st.markdown(result['highlighted_summary'], unsafe_allow_html=True)
                
                with st.expander(f"🔍 Model Reasoning - {filename}"):
                    st.markdown(result['debug'])
        else:
            combined_text = ""
            for filename in selected_files:
                uploaded_file = st.session_state.uploaded_files[filename]
                result = processor.process_pdf(uploaded_file, model_choice)
                combined_text += result['text'] + "\n"

            combined_result = processor.process_pdf(
                type('obj', (object,), {'read': lambda: combined_text.encode(), 'seek': lambda x: None})(),
                model_choice,
                combine=True
            )

            st.subheader("🗂️ Combined Summary from All Selected PDFs")
            st.subheader("🏷️ Named Medical Entities - Combined")
            if combined_result['entities']:
                for word, label in combined_result['entities']:
                    st.write(f"**{word}** → _{label}_")
            else:
                st.info("No medical entities found.")

            st.subheader("🖋️ Highlighted Summary - Combined")
            st.markdown(combined_result['highlighted_summary'], unsafe_allow_html=True)
            with st.expander("🔍 Model Reasoning - Combined"):
                st.markdown(combined_result['debug'])