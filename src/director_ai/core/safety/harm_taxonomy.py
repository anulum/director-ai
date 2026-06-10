# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HarmBench harm-category taxonomy

"""The seven HarmBench safety categories and a normaliser onto them.

Each safety detector reports its own free-string category — the input
sanitizer emits pattern names (``instruction_override``, ``bidi_override`` …),
the toxicity detector emits Detoxify labels (``identity_hate``, ``obscene`` …)
and keyword groups (``self_harm_encouragement`` …), the PII detector emits
entity types (``email``, ``ssn`` …). Without a shared vocabulary, policy rules,
threat-intel correlation, and compliance reports cannot align across detectors.

:class:`HarmCategory` is the canonical seven-class HarmBench taxonomy, and
:func:`to_harm_category` maps any detector's category string onto it, so a single
policy/SIEM/report layer can speak one vocabulary regardless of which detector
fired.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["HarmCategory", "to_harm_category"]


class HarmCategory(Enum):
    """The seven standard HarmBench safety categories.

    The value is a stable machine identifier (suitable for SIEM/STIX and
    serialised reports); :attr:`label` is the human-readable HarmBench name.
    """

    ILLICIT_ACTIVITIES = "illicit_activities"
    HATE_AND_ABUSE = "hate_and_abuse"
    PII_AND_IP = "pii_and_ip"
    PROMPT_SECURITY = "prompt_security"
    SEXUAL_CONTENT = "sexual_content"
    MISINFORMATION = "misinformation"
    VIOLENCE_AND_SELF_HARM = "violence_and_self_harm"

    @property
    def label(self) -> str:
        """Human-readable HarmBench category name."""
        return _LABELS[self]


_LABELS: dict[HarmCategory, str] = {
    HarmCategory.ILLICIT_ACTIVITIES: "Illicit Activities",
    HarmCategory.HATE_AND_ABUSE: "Hate & Abuse",
    HarmCategory.PII_AND_IP: "PII & IP",
    HarmCategory.PROMPT_SECURITY: "Prompt Security",
    HarmCategory.SEXUAL_CONTENT: "Sexual Content",
    HarmCategory.MISINFORMATION: "Misinformation",
    HarmCategory.VIOLENCE_AND_SELF_HARM: "Violence & Self-Harm",
}

# Substring keyword → category. Ordered by specificity: the first category whose
# any keyword is a substring of the (lower-cased) detector string wins, so
# narrower signals (``self_harm``, ``identity_hate``) are tested before the
# broader buckets they would otherwise fall into.
_KEYWORD_RULES: tuple[tuple[HarmCategory, tuple[str, ...]], ...] = (
    (
        HarmCategory.PROMPT_SECURITY,
        (
            "injection",
            "instruction_override",
            "system_role",
            "delimiter",
            "jailbreak",
            "prompt",
            "exfiltration",
            "path_traversal",
            "bidi",
            "unicode_escape",
            "control_char",
            "yaml_json",
            "python_code",
            "base64",
            "output_manipulation",
        ),
    ),
    (
        HarmCategory.PII_AND_IP,
        (
            "pii",
            "email",
            "ssn",
            "phone",
            "credit_card",
            "credit card",
            "passport",
            "iban",
            "ip_address",
            "secret",
            "api_key",
        ),
    ),
    (
        HarmCategory.VIOLENCE_AND_SELF_HARM,
        (
            "self_harm",
            "self-harm",
            "suicide",
            "violence",
            "violent",
            "threat",
            "weapon",
            "gore",
        ),
    ),
    (
        HarmCategory.SEXUAL_CONTENT,
        ("sexual", "nsfw", "porn", "explicit"),
    ),
    (
        HarmCategory.ILLICIT_ACTIVITIES,
        (
            "illicit",
            "drug",
            "fraud",
            "hacking",
            "malware",
            "theft",
            "trafficking",
        ),
    ),
    (
        HarmCategory.MISINFORMATION,
        ("misinformation", "disinformation", "fake_news", "fake news", "propaganda"),
    ),
    (
        HarmCategory.HATE_AND_ABUSE,
        (
            "hate",
            "identity",
            "insult",
            "toxic",
            "obscene",
            "harass",
            "abuse",
            "severe",
        ),
    ),
)


def to_harm_category(
    detector_category: str,
    *,
    default: HarmCategory | None = None,
) -> HarmCategory | None:
    """Normalise a detector's free-string category onto the HarmBench taxonomy.

    Case-insensitive substring match against :data:`_KEYWORD_RULES`. Returns
    ``default`` when nothing matches (so callers can choose to drop, bucket, or
    flag an unmapped category rather than guess).
    """
    text = detector_category.strip().lower()
    if not text:
        return default
    for category, keywords in _KEYWORD_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return default
