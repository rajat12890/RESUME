from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain_groq import ChatGroq


def get_match_score_chain(llm):


    prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""
You are an AI recruitment analyst for a top-tier tech company (like Google).
Analyze the resume against the provided job description and evaluate how well the candidate matches the role.

Provide the following:
1. A score out of 100 based on qualifications, skills, experience alignment, and role relevance.
2. A concise explanation highlighting key matches and gaps.

--- Resume ---
{resume}

--- Job Description ---
{job_description}

--- Output Format ---
Score: <score out of 100>
Explanation: <brief, bullet-point summary>
"""
)

    return LLMChain(prompt=prompt,llm=llm)
