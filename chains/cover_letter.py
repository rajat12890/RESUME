from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_cover_letter_chain(llm):
    """Chain to generate a professional, personalized cover letter."""
    prompt = ChatPromptTemplate.from_template("""
You are an expert career assistant at a leading technology company.
Write a personalized, professional cover letter tailored to the job description below using the resume content.

The tone should be confident, articulate, and aligned with high-performing candidates applying to elite organizations like Google.

--- Resume ---
{resume}

--- Job Description ---
{job_description}

--- Output ---
Cover Letter:
""")

    chain = prompt | llm | StrOutputParser()
    return chain
