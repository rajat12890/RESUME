import streamlit as st
from utils.pdf_loader import extract_text_from_pdf
from chains.match_score import get_match_score_chain
from chains.cover_letter import get_cover_letter_chain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# === Load API Key ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# === Streamlit Config ===
st.set_page_config(page_title="Resume Evaluator | AI Recruiter", layout="wide", page_icon="📄")

# === Page Header ===
st.title("📄 Intelligent Resume & JD Matcher")
st.markdown("""
### Evaluate Candidates Like a Tech Giant
Upload a resume and a job description. Our AI, powered by LangChain + Groq (LLaMA3-8B), will:
- Score candidate-job fit
- Suggest improvements
- Generate a custom cover letter
""")

# === UI Layout ===
st.markdown("---")
left, right = st.columns(2)

with left:
    st.subheader("🧾 Upload Resume")
    resume_file = st.file_uploader("Choose a resume (PDF only)", type=["pdf"])

with right:
    st.subheader("📋 Paste Job Description")
    job_description = st.text_area("Enter the JD here", height=300, placeholder="e.g. Backend Engineer at Google with Python, GCP, Kubernetes...")

# === Initialize LLM ===
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)

# === Run Analysis ===
st.markdown("---")
if resume_file and job_description:
    resume_text = extract_text_from_pdf(resume_file)

    if st.button("🚀 Match & Generate", use_container_width=True):
        with st.spinner("Analyzing with LLaMA3-8B on Groq..."):
            score_chain = get_match_score_chain(llm)
            cover_letter_chain = get_cover_letter_chain(llm)

            match_result = score_chain.invoke({
                "resume": resume_text,
                "job_description": job_description
            })

            cover_letter = cover_letter_chain.invoke({
                "resume": resume_text,
                "job_description": job_description
            })

        # === Output ===
        st.markdown("---")
        st.subheader("🎯 Match Score & Analysis")
        st.success(match_result)

        st.subheader("📝 AI-Generated Cover Letter")
        st.code(cover_letter, language="markdown")
else:
    st.info("Upload a resume and paste a job description to begin evaluation.")
