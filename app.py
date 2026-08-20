"""
AI-Powered Resume Analyzer and Job Match Assistant.
Engineered with Streamlit, Python, NLTK, Pandas, Knuth-Morris-Pratt (KMP) Substring Matcher,
Set Theory Math Engine, and Google Gemini LLM.
"""

import os
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from algorithms.kmp import kmp_search, find_pattern_with_context, find_multiple_keywords_kmp
from algorithms.set_matcher import compute_set_metrics, categorize_skill_matches
from utils.skills_db import SKILLS_TAXONOMY, get_all_skills_flat
from utils.parser import extract_text_from_file, extract_contact_info, extract_skills_with_kmp, analyze_text_statistics
from utils.llm_client import (
    get_ai_resume_critique,
    generate_tailored_cover_letter,
    generate_interview_prep
)

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer & Job Match Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, Vibrant Dark Palette, Polished Metrics)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 1rem;
        color: #e0e7ff;
        margin: 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.3);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .tag-matched {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }
    
    .tag-missing {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }
    
    .tag-bonus {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }
    
    .snippet-box {
        background: #0f172a;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 8px;
    }
    
    .snippet-highlight {
        color: #38bdf8;
        font-weight: bold;
        background: rgba(56, 189, 248, 0.2);
        padding: 2px 4px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/resume.png", width=64)
    st.markdown("### ⚙️ System Configuration")
    
    user_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Optional: Enter key for LLM-powered critiques, cover letters, and interview coaching."
    )
    if user_api_key:
        st.success("✅ API Key active", icon="🔑")
    else:
        st.info("💡 Math algorithms (KMP, Sets, NLP) work without an API key. Add key for AI generation.", icon="ℹ️")
        
    st.markdown("---")
    st.markdown("### 🔬 Matching Engine Settings")
    min_confidence = st.slider("ATS Score Threshold (%)", 0, 100, 70)
    enable_kmp_context = st.checkbox("Show KMP Pattern Snippets", value=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ Built With")
    st.caption("• **KMP Pattern Search** (Exact Substring $O(N+M)$)")
    st.caption("• **Set Theory Metrics** (Jaccard, Overlap, Dice)")
    st.caption("• **NLTK Engine** (Tokenization, Stopwords)")
    st.caption("• **Pandas Data Engine** (Batch Matching)")
    st.caption("• **Google Gemini 2.5 Flash**")


# Top Banner Header
st.markdown("""
<div class="main-header">
    <h1>⚡ AI Resume Analyzer & Job Match Assistant</h1>
    <p>Algorithmic resume analysis with Knuth-Morris-Pratt substring search, Set Theory metrics, NLTK parsing, and Gemini LLM career coach.</p>
</div>
""", unsafe_allow_html=True)


# Sample Resume Text Helper
SAMPLE_RESUME_TEXT = """John Doe
Senior Software Engineer | San Francisco, CA | (555) 123-4567 | john.doe@email.com
linkedin.com/in/johndoe | github.com/johndoe

SUMMARY:
Results-driven Senior Full-Stack Software Engineer with 6+ years of experience designing, developing, and deploying scalable microservices and web applications. Proficient in Python, TypeScript, React, Docker, and AWS cloud architecture.

TECHNICAL SKILLS:
- Languages: Python, TypeScript, JavaScript, SQL, Bash
- Frameworks: React, Next.js, FastAPI, Django, Express, Node.js, Tailwind CSS
- Databases & Storage: PostgreSQL, Redis, MongoDB
- Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes, CI/CD, Git, Linux
- Data & Tools: Pandas, NLTK, Postman, Jira, Agile, System Design, REST API

PROFESSIONAL EXPERIENCE:
Senior Software Engineer | Acme Cloud Solutions | 2022 – Present
- Architected and built high-performance REST APIs using FastAPI and PostgreSQL, serving 15M+ requests monthly.
- Engineered modern web dashboards in React and TypeScript with Tailwind CSS, improving load speed by 35%.
- Implemented Docker containerization and automated CI/CD deployment pipelines on AWS infrastructure.
- Integrated Redis caching layer, decreasing database query response times by 45%.

Software Engineer | DevTech Labs | 2019 – 2022
- Developed full-stack web applications utilizing Python, Django, and React.
- Collaborated in cross-functional Agile teams to deliver key features ahead of schedule.
- Built automated unit testing suites, increasing code coverage to 92%.

EDUCATION:
B.S. in Computer Science | University of California, Berkeley | 2015 – 2019
"""

SAMPLE_JOB_DESC = """We are seeking a Senior Full-Stack Engineer to build scalable financial applications. 
Key Responsibilities:
- Design, build, and deploy reliable microservices using Python and TypeScript.
- Build intuitive user interfaces with React and Next.js.
- Work with PostgreSQL, Redis, and message queues to manage high-volume data streams.
- Deploy and monitor cloud infrastructure on AWS using Docker and Kubernetes.
- Collaborate with cross-functional teams in an Agile environment and participate in system design.

Required Skills:
Python, TypeScript, React, Node.js, PostgreSQL, Docker, AWS, Kubernetes, REST API, System Design, Git, Redis.
"""

# Tabs Setup
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Quick Match & Scorecard",
    "📊 Skill Gap & KMP Inspector",
    "🤖 AI Career Coach & Cover Letter",
    "💼 Batch Job Ranking (Pandas)",
    "📚 Skills Taxonomy & Analytics"
])

# Initialize Session State
if "resume_text" not in st.session_state:
    st.session_state.resume_text = SAMPLE_RESUME_TEXT
if "job_desc" not in st.session_state:
    st.session_state.job_desc = SAMPLE_JOB_DESC


# -------------------------------------------------------------
# TAB 1: QUICK MATCH & SCORECARD
# -------------------------------------------------------------
with tab1:
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        st.subheader("📄 Candidate Resume")
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            key="resume_uploader"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🔄 Load Sample Resume", use_container_width=True):
                st.session_state.resume_text = SAMPLE_RESUME_TEXT
        with col_btn2:
            if st.button("🧹 Clear Resume", use_container_width=True):
                st.session_state.resume_text = ""

        if uploaded_file is not None:
            extracted = extract_text_from_file(uploaded_file.getvalue(), uploaded_file.name)
            st.session_state.resume_text = extracted
            
        resume_input = st.text_area(
            "Resume Content",
            value=st.session_state.resume_text,
            height=280,
            key="resume_text_area"
        )
        st.session_state.resume_text = resume_input

    with col_in2:
        st.subheader("💼 Target Job Description")
        
        # Dropdown to load from preloaded sample jobs
        try:
            sample_df = pd.read_csv("data/sample_jobs.csv")
            job_choices = ["(Custom Input)"] + sample_df["Job Title"].tolist()
            selected_preset = st.selectbox("Load Sample Job Preset:", job_choices)
            
            if selected_preset != "(Custom Input)":
                preset_row = sample_df[sample_df["Job Title"] == selected_preset].iloc[0]
                st.session_state.job_desc = f"Title: {preset_row['Job Title']} at {preset_row['Company']}\n\nDescription:\n{preset_row['Job Description']}\n\nRequired Skills:\n{preset_row['Required Skills']}"
        except Exception:
            pass

        job_input = st.text_area(
            "Job Description Content",
            value=st.session_state.job_desc,
            height=280,
            key="job_text_area"
        )
        st.session_state.job_desc = job_input

    # Parse and Match Operations
    if st.session_state.resume_text.strip() and st.session_state.job_desc.strip():
        resume_skills = extract_skills_with_kmp(st.session_state.resume_text)
        job_skills = extract_skills_with_kmp(st.session_state.job_desc)
        set_results = compute_set_metrics(resume_skills, job_skills)
        contacts = extract_contact_info(st.session_state.resume_text)
        nlp_stats = analyze_text_statistics(st.session_state.resume_text)
        
        st.markdown("---")
        st.subheader("🏆 ATS Compatibility & Algorithmic Scorecard")
        
        # High Level Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            score = set_results["scores"]["composite_ats_score"]
            color = "#10b981" if score >= min_confidence else ("#f59e0b" if score >= 45 else "#ef4444")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {color};">{score}%</div>
                <div class="metric-lbl">Composite ATS Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #38bdf8;">{set_results['scores']['match_rate_pct']}%</div>
                <div class="metric-lbl">Job Skill Recall (R ∩ J / J)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #a855f7;">{set_results['scores']['jaccard_similarity_pct']}%</div>
                <div class="metric-lbl">Jaccard Index (R ∩ J / R ∪ J)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #34d399;">{set_results['counts']['matched_count']} / {set_results['counts']['job_skills_count']}</div>
                <div class="metric-lbl">Matched Skills</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #f87171;">{set_results['counts']['missing_count']}</div>
                <div class="metric-lbl">Missing Skills</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Skill Breakdown Display
        b_col1, b_col2, b_col3 = st.columns(3)
        
        with b_col1:
            st.markdown("#### ✅ Matched Skills ($R \\cap J$)")
            if set_results["matched_skills"]:
                tags_html = "".join([f'<span class="tag-matched">✓ {s.title()}</span>' for s in set_results["matched_skills"]])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.warning("No direct skill matches found.")
                
        with b_col2:
            st.markdown("#### ❌ Missing Requirements ($J \\setminus R$)")
            if set_results["missing_skills"]:
                tags_html = "".join([f'<span class="tag-missing">✗ {s.title()}</span>' for s in set_results["missing_skills"]])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.success("🎉 All target skills found on resume!")
                
        with b_col3:
            st.markdown("#### 🌟 Candidate Bonus Skills ($R \\setminus J$)")
            if set_results["bonus_skills"]:
                tags_html = "".join([f'<span class="tag-bonus">+ {s.title()}</span>' for s in set_results["bonus_skills"][:15]])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info("No additional skills outside job scope.")

        # Candidate Details & NLP Snapshot
        st.markdown("---")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("#### 👤 Extracted Candidate Metadata")
            st.write(f"📧 **Emails:** {', '.join(contacts['emails']) if contacts['emails'] else 'None detected'}")
            st.write(f"📞 **Phone Numbers:** {', '.join(contacts['phones']) if contacts['phones'] else 'None detected'}")
            st.write(f"💼 **LinkedIn:** {', '.join(contacts['linkedin']) if contacts['linkedin'] else 'None detected'}")
            st.write(f"🐙 **GitHub:** {', '.join(contacts['github']) if contacts['github'] else 'None detected'}")
            
        with c_col2:
            st.markdown("#### 📊 Resume Text Statistics (NLTK)")
            st.write(f"📝 **Total Word Count:** {nlp_stats['word_count']} words")
            st.write(f"📜 **Total Sentences:** {nlp_stats['sentence_count']}")
            st.write(f"📏 **Avg Sentence Length:** {nlp_stats['avg_sentence_len']} words/sentence")
            top_kw_str = ", ".join([f"`{k}` ({v})" for k, v in nlp_stats['top_tokens'][:8]])
            st.write(f"🔑 **Top NLTK Non-Stopwords:** {top_kw_str}")
    else:
        st.info("👆 Please enter or upload a resume and job description to see the match analysis.")


# -------------------------------------------------------------
# TAB 2: SKILL GAP & KMP INSPECTOR
# -------------------------------------------------------------
with tab2:
    st.subheader("🔍 Algorithmic Skill Gap & KMP Substring Inspector")
    st.caption("Inspect exact pattern occurrences, prefix-suffix matches, and domain skill taxonomies.")

    if st.session_state.resume_text.strip() and st.session_state.job_desc.strip():
        resume_skills = extract_skills_with_kmp(st.session_state.resume_text)
        job_skills = extract_skills_with_kmp(st.session_state.job_desc)
        set_results = compute_set_metrics(resume_skills, job_skills)
        categorized = categorize_skill_matches(set(set_results["matched_skills"]), set(set_results["missing_skills"]), SKILLS_TAXONOMY)
        
        # Categorized Matrix Chart
        cat_data = []
        for cat, data in categorized.items():
            cat_data.append({
                "Category": cat,
                "Matched": len(data["matched"]),
                "Missing": len(data["missing"]),
                "Total Required": len(data["matched"]) + len(data["missing"])
            })
            
        if cat_data:
            df_cat = pd.DataFrame(cat_data)
            fig = px.bar(
                df_cat,
                x="Category",
                y=["Matched", "Missing"],
                barmode="group",
                title="Skill Domain Coverage Breakdown",
                color_discrete_map={"Matched": "#10b981", "Missing": "#ef4444"}
            )
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("⚡ Knuth-Morris-Pratt (KMP) Context Finder")
        st.write("Search for any custom keyword or multi-word phrase in the resume. The KMP algorithm finds exact occurrences and extracts context windows in $O(N + M)$ time.")
        
        kmp_search_term = st.text_input("Enter pattern/keyword to locate with KMP:", value="FastAPI")
        
        if kmp_search_term:
            occurrences = find_pattern_with_context(
                st.session_state.resume_text,
                kmp_search_term,
                context_window=50,
                case_sensitive=False
            )
            
            if occurrences:
                st.success(f"🎯 KMP Algorithm found **{len(occurrences)}** occurrence(s) of `'{kmp_search_term}'`:")
                for i, occ in enumerate(occurrences, 1):
                    prefix = occ['prefix']
                    matched = occ['matched_text']
                    suffix = occ['suffix']
                    st.markdown(f"""
                    <div class="snippet-box">
                        <strong>Match #{i} (Index {occ['index']}):</strong><br>
                        {prefix}<span class="snippet-highlight">{matched}</span>{suffix}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"No occurrences of `'{kmp_search_term}'` found in the resume text.")
    else:
        st.info("Please provide resume and job description in Tab 1 first.")


# -------------------------------------------------------------
# TAB 3: AI CAREER COACH & COVER LETTER GENERATOR
# -------------------------------------------------------------
with tab3:
    st.subheader("🤖 Google Gemini AI Career Coach & Cover Letter Generator")
    
    if not user_api_key and not os.getenv("GEMINI_API_KEY"):
        st.warning("⚠️ **Gemini API Key Required**: Enter your API key in the left sidebar to unlock AI feedback, cover letters, and interview coaching.", icon="🔑")

    ai_action = st.radio(
        "Select AI Task:",
        ["📋 ATS Resume Critique & Bullet Optimizer", "✉️ Tailored Cover Letter Generator", "🎯 Targeted Mock Interview Prep"],
        horizontal=True
    )
    
    if st.session_state.resume_text.strip() and st.session_state.job_desc.strip():
        resume_skills = extract_skills_with_kmp(st.session_state.resume_text)
        job_skills = extract_skills_with_kmp(st.session_state.job_desc)
        set_results = compute_set_metrics(resume_skills, job_skills)
        
        if ai_action == "📋 ATS Resume Critique & Bullet Optimizer":
            st.markdown("Generate a deep-dive recruiter evaluation with bullet point rewrites using Google's X-Y-Z formula.")
            if st.button("🚀 Generate ATS Critique Report", type="primary"):
                with st.spinner("Analyzing resume against job requirements with Gemini..."):
                    critique = get_ai_resume_critique(
                        st.session_state.resume_text,
                        st.session_state.job_desc,
                        set_results["matched_skills"],
                        set_results["missing_skills"],
                        api_key=user_api_key
                    )
                    st.markdown(critique)
                    st.download_button("📥 Download Critique Report (MD)", critique, file_name="resume_critique.md")

        elif ai_action == "✉️ Tailored Cover Letter Generator":
            st.markdown("Create a compelling, role-specific cover letter that showcases your matching accomplishments.")
            tone_select = st.selectbox("Select Letter Tone:", ["Professional & Confident", "Passionate & Energetic", "Technical & Data-Driven", "Executive & Strategic"])
            
            if st.button("✍️ Generate Custom Cover Letter", type="primary"):
                with st.spinner("Drafting your personalized cover letter with Gemini..."):
                    cover_letter = generate_tailored_cover_letter(
                        st.session_state.resume_text,
                        st.session_state.job_desc,
                        tone=tone_select,
                        api_key=user_api_key
                    )
                    st.markdown(cover_letter)
                    st.download_button("📥 Download Cover Letter (TXT)", cover_letter, file_name="cover_letter.txt")

        elif ai_action == "🎯 Targeted Mock Interview Prep":
            st.markdown("Get interview questions specifically tailored to your resume and skill gap areas.")
            if st.button("🧠 Generate Interview Prep Guide", type="primary"):
                with st.spinner("Crafting tailored technical and behavioral questions with Gemini..."):
                    prep_guide = generate_interview_prep(
                        st.session_state.resume_text,
                        st.session_state.job_desc,
                        set_results["missing_skills"],
                        api_key=user_api_key
                    )
                    st.markdown(prep_guide)
                    st.download_button("📥 Download Interview Guide (MD)", prep_guide, file_name="interview_prep.md")
    else:
        st.info("Please fill in the resume and job description in Tab 1.")


# -------------------------------------------------------------
# TAB 4: BATCH JOB RANKING (PANDAS)
# -------------------------------------------------------------
with tab4:
    st.subheader("💼 Batch Job Matching & Candidate Fit Ranking (Pandas)")
    st.caption("Rank your resume against multiple open job roles simultaneously using high-performance Pandas vectorization.")

    sample_jobs_file = "data/sample_jobs.csv"
    batch_df = None
    
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        uploaded_csv = st.file_uploader("Upload Custom Jobs Dataset (CSV with 'Job Title', 'Job Description', 'Required Skills')", type=["csv"])
        if uploaded_csv:
            batch_df = pd.read_csv(uploaded_csv)
        elif os.path.exists(sample_jobs_file):
            batch_df = pd.read_csv(sample_jobs_file)
            
    with col_b2:
        st.markdown("**Dataset Preview**")
        if batch_df is not None:
            st.write(f"Loaded **{len(batch_df)}** open positions.")
            
    if batch_df is not None and st.session_state.resume_text.strip():
        resume_skills = extract_skills_with_kmp(st.session_state.resume_text)
        
        # Calculate scores across each job
        ranked_rows = []
        for _, row in batch_df.iterrows():
            # Combine title, desc, and skills
            job_full_text = f"{row.get('Job Title', '')} {row.get('Job Description', '')} {row.get('Required Skills', '')}"
            job_skills = extract_skills_with_kmp(job_full_text)
            
            # Compute set metrics
            metrics = compute_set_metrics(resume_skills, job_skills)
            
            ranked_rows.append({
                "Job Title": row.get("Job Title", "Unknown Role"),
                "Company": row.get("Company", "N/A"),
                "Location": row.get("Location", "N/A"),
                "ATS Fit Score (%)": metrics["scores"]["composite_ats_score"],
                "Job Recall (%)": metrics["scores"]["match_rate_pct"],
                "Jaccard Sim (%)": metrics["scores"]["jaccard_similarity_pct"],
                "Matched Count": metrics["counts"]["matched_count"],
                "Missing Count": metrics["counts"]["missing_count"],
                "Matched Skills": ", ".join(metrics["matched_skills"]),
                "Missing Skills": ", ".join(metrics["missing_skills"])
            })
            
        ranking_df = pd.DataFrame(ranked_rows)
        ranking_df = ranking_df.sort_values(by="ATS Fit Score (%)", ascending=False).reset_index(drop=True)
        
        st.markdown("---")
        st.subheader("📊 Ranked Job Opportunities")
        
        # Leaderboard Table
        st.dataframe(
            ranking_df[["Job Title", "Company", "Location", "ATS Fit Score (%)", "Job Recall (%)", "Matched Count", "Missing Count", "Matched Skills"]],
            use_container_width=True,
            column_config={
                "ATS Fit Score (%)": st.column_config.ProgressColumn(
                    "ATS Score",
                    help="Calculated Composite Match Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )
        
        # Visualization
        fig_rank = px.bar(
            ranking_df,
            x="Job Title",
            y="ATS Fit Score (%)",
            color="ATS Fit Score (%)",
            color_continuous_scale="Viridis",
            title="Candidate Match Score by Position",
            text="ATS Fit Score (%)"
        )
        fig_rank.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_rank.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_rank, use_container_width=True)
        
        # Export
        csv_export = ranking_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Ranked Jobs (CSV)", csv_export, "ranked_jobs.csv", "text/csv")


# -------------------------------------------------------------
# TAB 5: SKILLS TAXONOMY & ANALYTICS
# -------------------------------------------------------------
with tab5:
    st.subheader("📚 Skills Taxonomy & Database Browser")
    st.write("Browse through the curated taxonomy of 500+ technical, cloud, AI, and soft skills used by the KMP pattern matcher.")
    
    tax_rows = []
    for category, skills in SKILLS_TAXONOMY.items():
        for skill in skills:
            tax_rows.append({"Category": category, "Skill": skill.title()})
            
    df_tax = pd.DataFrame(tax_rows)
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        cat_filter = st.multiselect("Filter by Category:", list(SKILLS_TAXONOMY.keys()), default=list(SKILLS_TAXONOMY.keys())[:3])
        skill_search = st.text_input("Search skill name:", "")
        
    filtered_tax = df_tax[df_tax["Category"].isin(cat_filter)] if cat_filter else df_tax
    if skill_search:
        filtered_tax = filtered_tax[filtered_tax["Skill"].str.contains(skill_search, case=False)]
        
    with col_t2:
        st.dataframe(filtered_tax, use_container_width=True, height=350)
        st.caption(f"Showing {len(filtered_tax)} of {len(df_tax)} skills.")
