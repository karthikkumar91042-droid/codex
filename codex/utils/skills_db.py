"""
Taxonomy and Skills Dictionary for Software, Data Science, DevOps, AI/ML, Cloud,
Web Development, Product, and Soft Skills.
"""

SKILLS_TAXONOMY = {
    "Programming Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "c", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "dart", "shell", "bash", "powershell", "sql"
    ],
    "Web & Frontend Frameworks": [
        "react", "react.js", "next.js", "vue", "vue.js", "angular", "svelte", "html", "html5", "css",
        "css3", "sass", "tailwind css", "bootstrap", "material-ui", "redux", "graphql", "rest api", "webpack", "vite"
    ],
    "Backend & APIs": [
        "node.js", "express", "express.js", "fastapi", "django", "flask", "spring", "spring boot",
        "asp.net", ".net core", "ruby on rails", "laravel", "nest.js", "grpc", "microservices", "websockets"
    ],
    "Databases & Caching": [
        "postgresql", "mysql", "mongodb", "sqlite", "redis", "elasticsearch", "cassandra", "dynamodb",
        "mariadb", "oracle", "neo4j", "firebase", "supabase", "snowflake", "bigquery"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "terraform", "ansible", "jenkins", "github actions", "gitlab ci", "helm", "linux", "ci/cd",
        "prometheus", "grafana", "nginx", "apache", "serverless", "lambda"
    ],
    "AI, Machine Learning & Data": [
        "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
        "generative ai", "llm", "large language models", "pytorch", "tensorflow", "keras", "scikit-learn",
        "pandas", "numpy", "opencv", "hugging face", "langchain", "llamaindex", "transformers", "spacy",
        "nltk", "bert", "gpt", "rag", "vector databases", "faiss", "pinecone", "chromadb", "data analysis",
        "data engineering", "spark", "apache spark", "hadoop", "kafka", "tableau", "power bi"
    ],
    "Methodologies & Tools": [
        "git", "github", "gitlab", "jira", "confluence", "agile", "scrum", "kanban", "tdd", "unit testing",
        "ci/cd pipelines", "system design", "software architecture", "code review", "postman"
    ],
    "Soft Skills & Management": [
        "leadership", "communication", "problem solving", "critical thinking", "team collaboration",
        "project management", "time management", "mentorship", "stakeholder management", "adaptability",
        "cross-functional collaboration", "analytical skills"
    ]
}


def get_all_skills_flat() -> list[str]:
    """Returns flat list of all unique skills across taxonomy."""
    all_skills = set()
    for cat, skills in SKILLS_TAXONOMY.items():
        for s in skills:
            all_skills.add(s.lower().strip())
    return sorted(list(all_skills))
