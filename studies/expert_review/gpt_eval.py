"""
Minimal helper needed to reproduce two figures: figS4_reviewer_time and
fig06_expert_review (generated from a private notebook, not part of this release).

This is a trimmed extract of the full gpt_eval.py module used elsewhere in the
project: it keeps only `human_judgements`, which pools the first and second expert
rating passes into one judgement-per-row table. Everything about the cross-vendor
verifier and the GPT-as-validator experiments has been removed, since neither target
figure uses it.
"""

from __future__ import annotations

import pandas as pd


def human_judgements(expert: pd.DataFrame, expert2: pd.DataFrame) -> pd.DataFrame:
    """Every human judgement as one row, both passes pooled.

    The study reports human validation as a single body of expert work, not as a
    first-rater result plus a second-rater result: a judgement is a judgement whoever
    made it. `pass_` is kept only so the double-rated subset can still be isolated for
    the inter-rater analysis.
    """
    keep = ["paper", "finding_no", "issue_id", "category", "location", "claim",
            "correctness", "relevance", "correctness_score", "relevance_score",
            "stands", "fully_true", "at_least_minor", "is_relevant", "minutes"]
    a = expert[expert.rated][keep].assign(pass_="first")
    b = expert2[expert2.rated][keep].assign(pass_="second")
    return pd.concat([a, b], ignore_index=True)
