"""Tests for grooming plan extraction in JiraAgent."""

import pytest

from jirade.agent import JiraAgent


def _make_comment(body_text: str) -> dict:
    """Create a mock Jira comment with plain text body (ADF format)."""
    return {
        "body": {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": body_text}],
                }
            ],
        }
    }


class TestExtractGroomingPlan:
    """Tests for _extract_grooming_plan static method."""

    def test_no_comments(self):
        assert JiraAgent._extract_grooming_plan([]) is None

    def test_no_plan_comments(self):
        comments = [
            _make_comment("Just a regular comment"),
            _make_comment("[jirade grooming] Questions: What does X mean?"),
        ]
        assert JiraAgent._extract_grooming_plan(comments) is None

    def test_plan_present(self):
        plan_text = "[jirade grooming] Implementation Plan:\n\n## Summary\nAdd new column"
        comments = [
            _make_comment("Some earlier comment"),
            _make_comment(plan_text),
        ]
        result = JiraAgent._extract_grooming_plan(comments)
        assert result == plan_text

    def test_multiple_plans_returns_latest(self):
        plan_v1 = "[jirade grooming] Implementation Plan:\n\n## Summary\nFirst version"
        plan_v2 = "[jirade grooming] Implementation Plan:\n\n## Summary\nRevised version"
        comments = [
            _make_comment(plan_v1),
            _make_comment("Feedback: please change approach"),
            _make_comment(plan_v2),
        ]
        result = JiraAgent._extract_grooming_plan(comments)
        assert result == plan_v2

    def test_plan_with_leading_whitespace(self):
        plan_text = "  [jirade grooming] Implementation Plan:\n\n## Summary\nWith whitespace"
        comments = [_make_comment(plan_text)]
        result = JiraAgent._extract_grooming_plan(comments)
        assert result == plan_text.strip()

    def test_non_plan_grooming_comments_ignored(self):
        comments = [
            _make_comment("[jirade grooming] Questions: What is the scope?"),
            _make_comment("[jirade grooming] Follow-up: See above"),
        ]
        assert JiraAgent._extract_grooming_plan(comments) is None
