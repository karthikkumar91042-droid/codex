"""
Unit tests to verify KMP search, Set theory metrics, and text parsing engines.
"""
import unittest
from algorithms.kmp import compute_lps_array, kmp_search, find_pattern_with_context
from algorithms.set_matcher import compute_set_metrics, categorize_skill_matches
from utils.parser import clean_text_for_nlp, extract_contact_info, extract_skills_with_kmp
from utils.skills_db import SKILLS_TAXONOMY


class TestResumeAnalyzer(unittest.TestCase):

    def test_kmp_lps(self):
        pattern = "AAACAAAA"
        lps = compute_lps_array(pattern)
        self.assertEqual(lps, [0, 1, 2, 0, 1, 2, 3, 3])

    def test_kmp_search_exact(self):
        text = "Experienced with React and React Native, building FastAPI backends."
        indices = kmp_search(text, "React", case_sensitive=True)
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices, [17, 27])

    def test_kmp_case_insensitive(self):
        text = "Proficient in Python and pythonic design patterns."
        indices = kmp_search(text, "PYTHON", case_sensitive=False)
        self.assertEqual(len(indices), 2)

    def test_kmp_context_extraction(self):
        text = "We built scalable microservices using FastAPI and Redis for caching."
        matches = find_pattern_with_context(text, "FastAPI", context_window=10)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["pattern"], "FastAPI")
        self.assertIn("FastAPI", matches[0]["context"])

    def test_set_metrics(self):
        resume_skills = {"python", "fastapi", "docker", "postgresql", "git"}
        job_skills = {"python", "fastapi", "kubernetes", "aws", "docker"}

        metrics = compute_set_metrics(resume_skills, job_skills)
        self.assertEqual(set(metrics["matched_skills"]), {"python", "fastapi", "docker"})
        self.assertEqual(set(metrics["missing_skills"]), {"kubernetes", "aws"})
        self.assertEqual(set(metrics["bonus_skills"]), {"postgresql", "git"})
        self.assertEqual(metrics["scores"]["match_rate_pct"], 60.0)

    def test_contact_extractor(self):
        sample = "Contact John at john.doe@techcorp.io or call 555-432-8765. LinkedIn: linkedin.com/in/johndoe."
        contacts = extract_contact_info(sample)
        self.assertIn("john.doe@techcorp.io", contacts["emails"])
        self.assertTrue(len(contacts["phones"]) > 0)
        self.assertTrue(len(contacts["linkedin"]) > 0)

    def test_skill_extractor_with_kmp(self):
        sample = "Expert in machine learning, deep learning, PyTorch, Docker, and PostgreSQL databases."
        skills = extract_skills_with_kmp(sample)
        self.assertIn("machine learning", skills)
        self.assertIn("deep learning", skills)
        self.assertIn("pytorch", skills)
        self.assertIn("docker", skills)
        self.assertIn("postgresql", skills)


if __name__ == "__main__":
    unittest.main()
