"""
Mathematical Set Operations and Similarity Metrics Engine for Resume & Job Description matching.
Calculates Set Intersection, Union, Difference, Symmetric Difference, Jaccard Index,
Dice Coefficient, and Overlap Score.
"""
from typing import Set, Dict, Any, List


def compute_set_metrics(resume_skills: Set[str], job_skills: Set[str]) -> Dict[str, Any]:
    r"""
    Computes rigorous set theory operations and similarity scores between resume skills and job requirements.
    
    Formulas:
    - Matched Skills: R ∩ J (Intersection)
    - Missing Skills: J \ R (Job Difference)
    - Extra/Bonus Skills: R \ J (Resume Difference)
    - All Unique Skills: R ∪ J (Union)
    - Symmetric Difference: (R Δ J) = (R \ J) ∪ (J \ R)
    - Jaccard Similarity: |R ∩ J| / |R ∪ J|
    - Dice / Sorensen Coefficient: 2 * |R ∩ J| / (|R| + |J|)
    - Overlap Coefficient (Job Recall): |R ∩ J| / |J| (if |J| > 0 else 0)
    - Resume Precision: |R ∩ J| / |R| (if |R| > 0 else 0)
    """
    # Normalize skills to lowercase for accurate set comparisons
    r_set = {s.lower().strip() for s in resume_skills if s.strip()}
    j_set = {s.lower().strip() for s in job_skills if s.strip()}

    matched = r_set.intersection(j_set)
    missing = j_set.difference(r_set)
    bonus = r_set.difference(j_set)
    union = r_set.union(j_set)
    symmetric_diff = r_set.symmetric_difference(j_set)

    # Calculate metrics
    jaccard = len(matched) / len(union) if len(union) > 0 else 0.0
    dice = (2.0 * len(matched)) / (len(r_set) + len(j_set)) if (len(r_set) + len(j_set)) > 0 else 0.0
    overlap_job_recall = len(matched) / len(j_set) if len(j_set) > 0 else 0.0
    precision = len(matched) / len(r_set) if len(r_set) > 0 else 0.0

    # Composite ATS Fit Score (0 - 100%)
    # 70% weight on covering required job skills (recall) + 30% Jaccard overall harmony
    composite_score = round(((0.70 * overlap_job_recall) + (0.30 * jaccard)) * 100, 2)

    return {
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "bonus_skills": sorted(list(bonus)),
        "union_skills": sorted(list(union)),
        "symmetric_difference": sorted(list(symmetric_diff)),
        "counts": {
            "resume_skills_count": len(r_set),
            "job_skills_count": len(j_set),
            "matched_count": len(matched),
            "missing_count": len(missing),
            "bonus_count": len(bonus),
            "union_count": len(union),
        },
        "scores": {
            "match_rate_pct": round(overlap_job_recall * 100, 2),
            "jaccard_similarity_pct": round(jaccard * 100, 2),
            "dice_similarity_pct": round(dice * 100, 2),
            "precision_pct": round(precision * 100, 2),
            "composite_ats_score": composite_score
        }
    }


def categorize_skill_matches(
    matched_skills: Set[str],
    missing_skills: Set[str],
    taxonomy: Dict[str, List[str]]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Groups matched and missing skills into domains/categories (e.g. Languages, Cloud, Frameworks, etc.).
    """
    categorized = {}

    for category, skills in taxonomy.items():
        skills_lower = {s.lower().strip() for s in skills}
        cat_matched = sorted(list(matched_skills.intersection(skills_lower)))
        cat_missing = sorted(list(missing_skills.intersection(skills_lower)))

        if cat_matched or cat_missing:
            categorized[category] = {
                "matched": cat_matched,
                "missing": cat_missing
            }

    return categorized
