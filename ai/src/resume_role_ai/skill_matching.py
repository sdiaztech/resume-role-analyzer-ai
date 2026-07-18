from __future__ import annotations

import re


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    ".net": ("dotnet", "dot net"),
    "amazon web services aws software": ("amazon web services",),
    "c#": ("c sharp",),
    "rest apis": ("api", "apis", "rest api", "web api", "web apis"),
}

VENDOR_PREFIXES = {
    "adobe",
    "amazon",
    "atlassian",
    "google",
    "ibm",
    "intuit",
    "microsoft",
    "oracle",
    "salesforce",
    "sap",
}
TRAILING_PRODUCT_WORDS = {"application", "platform", "software", "system", "systems"}


def skill_terms(skill: str) -> set[str]:
    """Generate safe common forms for a canonical product or skill name."""
    canonical = skill.casefold().strip()
    terms = {canonical, *SKILL_ALIASES.get(canonical, ())}

    # O*NET frequently stores expanded names such as "Amazon Web Services AWS software".
    acronym_pattern = r"(?<![\w.])[A-Z][A-Z0-9+#.]{1,}(?!\w)"
    for match in re.finditer(acronym_pattern, skill):
        suffix = skill[match.end() :].casefold().split()
        # AWS identifies the platform, but must not imply AWS CloudFormation or AWS Lambda.
        if not suffix or all(word in TRAILING_PRODUCT_WORDS for word in suffix):
            terms.add(match.group().casefold().rstrip("."))
        else:
            terms.add(" ".join([match.group().casefold().rstrip("."), *suffix]))

    words = canonical.split()
    while len(words) > 1 and words[-1] in TRAILING_PRODUCT_WORDS:
        words.pop()
        terms.add(" ".join(words))
    if len(words) > 1 and words[0] in VENDOR_PREFIXES:
        terms.add(" ".join(words[1:]))

    # Accept punctuation variants commonly used in resumes and applicant systems.
    for term in tuple(terms):
        if "/" in term or "-" in term:
            terms.add(re.sub(r"[/\-]+", " ", term))
    return {term.strip() for term in terms if len(term.strip()) >= 2}


def skill_is_present(skill: str, resume_skills: set[str]) -> bool:
    return bool(skill_terms(skill).intersection(resume_skills))


def normalized_resume_skills(skills: list[str]) -> set[str]:
    return set().union(*(skill_terms(skill) for skill in skills)) if skills else set()
