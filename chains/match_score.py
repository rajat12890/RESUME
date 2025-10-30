from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_match_score_chain(llm):
    """Chain to evaluate how well a resume matches a job description."""
    prompt = ChatPromptTemplate.from_template("""
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
""")

    # Build modern chain using LCEL (LangChain Expression Language)
    chain = prompt | llm | StrOutputParser()
    return chain
