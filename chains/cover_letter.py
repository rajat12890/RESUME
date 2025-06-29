from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def get_cover_letter_chain(llm):

    prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""
You are an expert career assistant at a leading technology company.
Write a personalized, professional cover letter tailored to the job description below using the resume content.

The tone should be confident, articulate, and aligned with high-performing candidates applying to elite organizations like Google.

--- Resume ---
{resume}

--- Job Description ---
{job_description}

--- Output ---
Cover Letter:
"""
)

    return LLMChain(prompt=prompt, llm=llm)
