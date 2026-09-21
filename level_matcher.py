import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LevelMatch:
    level_requirement: str
    has_level_requirement: bool
    source: str
    evidence: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Keep explicit numeric levels ahead of title inference. Title inference is a
# fallback for short JD text such as "Data SM" where level is implied.
EXPLICIT_LEVEL_PATTERNS = [
    r"([PMLT]?\s*\d{1,2})\s*(?:and\s+below|or\s+below|below|及以下|以下)",
    r"([PMLT]?\s*\d{1,2})\s*(?:and\s+above|or\s+above|above|及以上|以上)",
    r"((?:lv|level|l)\s*\d{1,2}(?:\s*[-~至到]\s*(?:lv|level|l)?\s*\d{1,2})?)",
    r"(?:职级|级别|层级|等级|level|Level|LEVEL)\s*(?::|：|为|是|要求|需要|需)?\s*([PMLT]?\s*\d{1,2}(?:\s*[-~至到]\s*[PMLT]?\s*\d{1,2})?)",
    r"([PMLT]\s*\d{1,2}(?:\s*[-~至到]\s*[PMLT]?\s*\d{1,2})?)\s*(?:职级|级别|层级|等级)",
    r"(\d{1,2}(?:\s*[-~至到]\s*\d{1,2})?)\s*(?:级|职级|级别|层级|等级)",
]


TITLE_LEVEL_RULES = [
    {
        "level": 6,
        "confidence": 0.9,
        "patterns": [
            r"(?<![A-Z])SM(?![A-Z])",
            r"\bSENIOR\s+MANAGER\b",
            r"\bSR\.?\s+MANAGER\b",
            r"高级经理",
            r"资深经理",
        ],
    },
    {
        "level": 8,
        "confidence": 0.75,
        "patterns": [
            r"(?<![A-Z])AM(?![A-Z])",
            r"\bASSOCIATE\s+MANAGER\b",
            r"\bASSOC\.?\s+MGR\b",
            r"\bASSOC\.?\s+MANAGER\b",
            r"副经理",
        ],
    },
    {
        "level": 7,
        "confidence": 0.75,
        "patterns": [
            r"(?<![A-Z])M(?![A-Z])",
            r"\bMANAGER\b",
            r"经理",
        ],
    },
    {
        "level": 9,
        "confidence": 0.7,
        "patterns": [
            r"\bTEAM\s+LEAD\b",
            r"团队负责人",
            r"小组负责人",
            r"\bCONSULTANT\b",
            r"顾问",
        ],
    },
    {
        "level": 10,
        "confidence": 0.7,
        "patterns": [
            r"\bSENIOR\s+ANALYST\b",
            r"\bSR\.?\s+ANALYST\b",
            r"高级分析师",
            r"资深分析师",
        ],
    },
    {
        "level": 11,
        "confidence": 0.7,
        "patterns": [
            r"\bANALYST\b",
            r"分析师",
        ],
    },
]


def normalize_level_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def normalize_explicit_level_requirement(value: str, evidence: str) -> str:
    level_text = normalize_level_text(value)
    numbers = [int(item) for item in re.findall(r"\d{1,2}", level_text)]
    number_text = str(numbers[0]) if numbers else level_text
    evidence_lower = evidence.lower()
    if re.search(r"and\s+below|or\s+below|below|及以下|以下", evidence_lower):
        return f"Level {number_text}及以下"
    if re.search(r"and\s+above|or\s+above|above|及以上|以上", evidence_lower):
        return f"Level {number_text}及以上"
    if len(numbers) >= 2:
        return f"Level {min(numbers)}-{max(numbers)}"
    if len(numbers) == 1:
        return f"Level {numbers[0]}"
    return level_text


def extract_explicit_level_requirement(text: str) -> LevelMatch:
    source_text = text.strip()
    if not source_text:
        return LevelMatch("", False, "", "", 0.0)

    for pattern in EXPLICIT_LEVEL_PATTERNS:
        match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if match:
            evidence = match.group(0).strip()
            return LevelMatch(
                level_requirement=normalize_explicit_level_requirement(match.group(1), evidence),
                has_level_requirement=True,
                source="explicit_level_text",
                evidence=evidence,
                confidence=1.0,
            )
    return LevelMatch("", False, "", "", 0.0)


def infer_level_from_title(text: str) -> LevelMatch:
    source_text = text.strip()
    if not source_text:
        return LevelMatch("", False, "", "", 0.0)

    normalized = source_text.upper()
    for rule in TITLE_LEVEL_RULES:
        for pattern in rule["patterns"]:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            level = int(rule["level"])
            return LevelMatch(
                level_requirement=f"Level {level}",
                has_level_requirement=True,
                source="title_level_rule",
                evidence=match.group(0),
                confidence=float(rule["confidence"]),
            )
    return LevelMatch("", False, "", "", 0.0)


def infer_jd_level_requirement(jd_description: str, job_title: str = "") -> LevelMatch:
    explicit_match = extract_explicit_level_requirement(jd_description)
    if explicit_match.has_level_requirement:
        return explicit_match

    title_match = infer_level_from_title(" ".join([job_title, jd_description]).strip())
    if title_match.has_level_requirement:
        return title_match

    return LevelMatch("", False, "", "", 0.0)


def infer_jd_level_requirement_text(jd_description: str, job_title: str = "") -> str:
    return infer_jd_level_requirement(jd_description, job_title).level_requirement


RESUME_SECTION_HEADINGS = {
    "简历",
    "职业背景概述",
    "教育背景与培训",
    "早期职业背景",
    "核心技能",
    "行业经验",
    "项目经验",
    "工作经历",
    "教育经历",
}


def extract_resume_title_candidates(resume_text: str, current_level_text: str = "") -> list[str]:
    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        cleaned = re.sub(r"^[#*\-\s:：/]+|[#*\s]+$", "", value.strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned or cleaned in RESUME_SECTION_HEADINGS:
            return
        if cleaned not in candidates:
            candidates.append(cleaned)

    if current_level_text:
        add_candidate(current_level_text)

    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    for line in lines[:40]:
        heading = re.match(r"^#{1,3}\s+(.+)$", line)
        if heading:
            add_candidate(heading.group(1))
        bracket_title = re.match(r"^【(.+)】$", line)
        if bracket_title:
            add_candidate(bracket_title.group(1))

    for line in lines:
        if re.search(r"职位|职务|岗位", line):
            title_part = line
            title_part = re.sub(r"^\*+\s*时间\s*/\s*职位\s*[:：]\**\s*", "", title_part, flags=re.IGNORECASE)
            title_part = re.sub(r"^.*?(?:职位|职务|岗位)\s*[:：]\s*", "", title_part)
            add_candidate(title_part)
            if len(candidates) >= 8:
                break

    return candidates


def infer_resume_level(resume_text: str, current_level_text: str = "") -> LevelMatch:
    existing_standard_level = re.match(r"^\s*Level\s+\d{1,2}\s*$", current_level_text, flags=re.IGNORECASE)
    explicit_lines = [
        line
        for line in resume_text.splitlines()
        if re.search(r"职级|级别|层级|等级|\blevel\b", line, flags=re.IGNORECASE)
    ]
    explicit_parts = ([] if existing_standard_level else [current_level_text]) + explicit_lines
    explicit_text = "\n".join(explicit_parts).strip()
    explicit_match = extract_explicit_level_requirement(explicit_text)
    if explicit_match.has_level_requirement:
        return LevelMatch(
            level_requirement=explicit_match.level_requirement,
            has_level_requirement=True,
            source="resume_explicit_level_text",
            evidence=explicit_match.evidence,
            confidence=explicit_match.confidence,
        )

    for candidate in extract_resume_title_candidates(resume_text, current_level_text):
        title_match = infer_level_from_title(candidate)
        if title_match.has_level_requirement:
            return LevelMatch(
                level_requirement=title_match.level_requirement,
                has_level_requirement=True,
                source="resume_title_rule",
                evidence=candidate,
                confidence=title_match.confidence,
            )

    return LevelMatch("", False, "", "", 0.0)
