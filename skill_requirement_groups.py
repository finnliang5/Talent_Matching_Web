import re
from typing import Any


ALT_MARKER_PATTERN = re.compile(r"/|\\|\bor\b|或者|或", flags=re.IGNORECASE)
SEGMENT_SPLIT_PATTERN = re.compile(r"\+|,|，|;|；|、|\band\b", flags=re.IGNORECASE)


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def item_id(item: Any, id_key: str = "skill_id") -> str:
    if isinstance(item, dict):
        return str(item.get(id_key, "") or item.get("id", "") or "").strip()
    return str(item or "").strip()


def item_name(item: Any, fallback: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get("name", "") or item.get("label", "") or fallback).strip()
    return str(item or fallback).strip()


def skill_terms(skill: dict[str, Any]) -> list[str]:
    terms = [
        str(skill.get("matched_term", "") or ""),
        str(skill.get("name", "") or ""),
        str(skill.get("skill_id", "") or ""),
    ]
    aliases = skill.get("aliases", [])
    if isinstance(aliases, list):
        terms.extend(str(item or "") for item in aliases)
    return [term for term in terms if term.strip()]


def term_in_segment(term: str, segment: str) -> bool:
    term = term.strip()
    if not term:
        return False
    if re.fullmatch(r"[A-Za-z0-9+#][A-Za-z0-9+#.\-_\s]*", term):
        return bool(re.search(rf"(?<![A-Za-z0-9+#]){re.escape(term)}(?![A-Za-z0-9+#])", segment, re.IGNORECASE))
    return normalized_text(term) in normalized_text(segment)


def skill_in_segment(skill: dict[str, Any], segment: str) -> bool:
    return any(term_in_segment(term, segment) for term in skill_terms(skill))


def group_item(skill: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": item_id(skill, "skill_id"),
        "name": item_name(skill, item_id(skill, "skill_id")),
    }


def build_single_skill_group(skill: dict[str, Any], index: int) -> dict[str, Any]:
    skill_id = item_id(skill, "skill_id")
    name = item_name(skill, skill_id)
    return {
        "group_id": f"TECH_GROUP_{index}",
        "group_name": name,
        "logic": "ALL_OF",
        "items": [group_item(skill)],
        "evidence": str(skill.get("evidence", "") or ""),
        "priority": str(skill.get("priority", "") or "must"),
    }


def build_any_of_group(skills: list[dict[str, Any]], evidence: str, index: int) -> dict[str, Any]:
    names = [item_name(skill, item_id(skill, "skill_id")) for skill in skills]
    priority = "must" if any(str(skill.get("priority", "") or "must") == "must" for skill in skills) else "nice"
    return {
        "group_id": f"TECH_GROUP_{index}",
        "group_name": " 或 ".join(names),
        "logic": "ANY_OF",
        "items": [group_item(skill) for skill in skills],
        "evidence": evidence,
        "priority": priority,
    }


def split_evidence_into_candidate_segments(evidence: str) -> list[str]:
    evidence = str(evidence or "")
    if not evidence.strip():
        return []
    parts = SEGMENT_SPLIT_PATTERN.split(evidence)
    return [part.strip(" ()[]{}.:：-") for part in parts if part.strip(" ()[]{}.:：-")]


def build_technical_skill_requirement_groups(technical_skills: Any) -> list[dict[str, Any]]:
    if not isinstance(technical_skills, list) or not technical_skills:
        return []

    dict_skills = [item for item in technical_skills if isinstance(item, dict) and item_id(item, "skill_id")]
    if not dict_skills:
        return []

    groups: list[dict[str, Any]] = []
    assigned: set[str] = set()
    group_index = 1

    evidence_values = sorted(
        {str(skill.get("evidence", "") or "") for skill in dict_skills if str(skill.get("evidence", "") or "").strip()}
    )
    for evidence in evidence_values:
        evidence_skills = [skill for skill in dict_skills if str(skill.get("evidence", "") or "") == evidence]
        if len(evidence_skills) < 2 or not ALT_MARKER_PATTERN.search(evidence):
            continue

        for segment in split_evidence_into_candidate_segments(evidence):
            if not ALT_MARKER_PATTERN.search(segment):
                continue
            segment_skills = [
                skill
                for skill in evidence_skills
                if item_id(skill, "skill_id") not in assigned and skill_in_segment(skill, segment)
            ]
            if len(segment_skills) < 2:
                continue
            groups.append(build_any_of_group(segment_skills, evidence, group_index))
            assigned.update(item_id(skill, "skill_id") for skill in segment_skills)
            group_index += 1

    for skill in dict_skills:
        if item_id(skill, "skill_id") in assigned:
            continue
        groups.append(build_single_skill_group(skill, group_index))
        group_index += 1

    return groups


def normalize_technical_skill_requirement_groups(groups: Any, technical_skills: Any) -> list[dict[str, Any]]:
    if isinstance(groups, list) and groups:
        normalized_groups: list[dict[str, Any]] = []
        for index, group in enumerate(groups, start=1):
            if not isinstance(group, dict):
                continue
            items = group.get("items", [])
            if not isinstance(items, list) or not items:
                continue
            normalized_items = [
                {"skill_id": item_id(item, "skill_id"), "name": item_name(item, item_id(item, "skill_id"))}
                for item in items
                if item_id(item, "skill_id")
            ]
            if not normalized_items:
                continue
            normalized_groups.append(
                {
                    "group_id": str(group.get("group_id", "") or f"TECH_GROUP_{index}"),
                    "group_name": str(group.get("group_name", "") or " 或 ".join(item["name"] for item in normalized_items)),
                    "logic": str(group.get("logic", "") or "ALL_OF").upper(),
                    "items": normalized_items,
                    "evidence": str(group.get("evidence", "") or ""),
                    "priority": str(group.get("priority", "") or "must"),
                }
            )
        if normalized_groups:
            return normalized_groups
    return build_technical_skill_requirement_groups(technical_skills)
