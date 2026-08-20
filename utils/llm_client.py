"""
Google Gemini LLM Integration Engine for Resume Analysis, AI Cover Letter Generator,
Bullet-Point Optimizer, and Mock Interview Questions.
"""
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


def get_gemini_client(api_key: Optional[str] = None):
    """Initializes Google GenAI Client with provided API key or environment variable."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        return client
    except Exception:
        # Fallback to legacy package if installed
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=key)
            return legacy_genai
        except Exception:
            return None


def generate_llm_response(prompt: str, api_key: Optional[str] = None, system_instruction: str = "") -> str:
    """Invokes Google Gemini with structured prompt."""
    client = get_gemini_client(api_key)
    if not client:
        return (
            "⚠️ **Gemini API Key not configured.**\n\n"
            "Please enter your Google Gemini API Key in the left sidebar or create a `.env` file with `GEMINI_API_KEY=your_key`.\n"
            "Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)."
        )

    try:
        # Try google.genai (new SDK)
        if hasattr(client, "models"):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"system_instruction": system_instruction} if system_instruction else None
            )
            return response.text
        else:
            # Legacy SDK
            model = client.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction if system_instruction else None
            )
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ **Error calling Gemini API:** {str(e)}"


def get_ai_resume_critique(
    resume_text: str,
    job_description: str,
    matched_skills: List[str],
    missing_skills: List[str],
    api_key: Optional[str] = None
) -> str:
    """Generates an in-depth ATS recruiter critique and actionable recommendations."""
    prompt = f"""
You are a Senior Technical Recruiter and ATS (Applicant Tracking System) Optimization Expert.
Analyze the following Resume against the Job Description.

### Job Description:
{job_description[:3000]}

### Resume Content:
{resume_text[:3500]}

### Algorithmic Extracted Match Data:
- Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}
- Missing Job Requirements: {', '.join(missing_skills) if missing_skills else 'None'}

Please provide a detailed, well-structured feedback report in Markdown with the following sections:
1. 🎯 **ATS Suitability & First Impression** (Executive summary of strengths and alignment)
2. 🚨 **High-Impact Red Flags & Missing Keywords** (Critical gaps that might trigger auto-rejections)
3. 📝 **Bullet Point Transformation Examples** (Take 2-3 weak or generic points from the resume and rewrite them using the Google X-Y-Z formula: "Accomplished [X] as measured by [Y], by doing [Z]")
4. 📈 **Section-by-Section Optimization Tips** (Header/Summary, Experience, Projects, Skills)
5. 💡 **Final Recommendation & Action Plan**
"""
    system_instruction = "You are an elite career coach and executive tech recruiter. Give constructive, direct, data-backed advice."
    return generate_llm_response(prompt, api_key=api_key, system_instruction=system_instruction)


def generate_tailored_cover_letter(
    resume_text: str,
    job_description: str,
    tone: str = "Professional & Confident",
    api_key: Optional[str] = None
) -> str:
    """Generates a tailored, persuasive cover letter matching candidate experience with job needs."""
    prompt = f"""
Write an outstanding, customized Cover Letter for a candidate applying to the position described in the Job Description using information from their Resume.

### Tone: {tone}

### Job Description:
{job_description[:3000]}

### Candidate's Resume:
{resume_text[:3500]}

### Requirements:
- Do not use generic filler language.
- Highlight 2-3 specific matching accomplishments from the resume that directly solve problems mentioned in the job description.
- Keep it concise (approx. 3-4 impactful paragraphs).
- Include placeholders like [Hiring Manager Name] or [Company Name] where appropriate.
"""
    system_instruction = "You are a professional executive resume and cover letter writer."
    return generate_llm_response(prompt, api_key=api_key, system_instruction=system_instruction)


def generate_interview_prep(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
    api_key: Optional[str] = None
) -> str:
    """Generates targeted technical and behavioral interview questions based on gaps and skills."""
    prompt = f"""
Based on the candidate's resume and target job description (especially identified missing/gap skills: {', '.join(missing_skills)}), generate a tailored Interview Preparation Guide.

### Job Description:
{job_description[:2500]}

### Candidate Resume:
{resume_text[:2500]}

Please generate:
1. 🧠 **5 Technical Deep-Dive Questions** (specific to the tech stack & architecture required)
2. 🔄 **3 Questions Addressing Skill Gaps** (how the candidate can bridge gaps like {', '.join(missing_skills[:5])})
3. 🏆 **3 Behavioral / STAR Method Questions** tailored to their past experiences
4. 💡 **Pro-tips on how to answer & smart questions for the candidate to ask the interviewer**
"""
    system_instruction = "You are a senior engineering manager conducting high-caliber technical interviews."
    return generate_llm_response(prompt, api_key=api_key, system_instruction=system_instruction)
