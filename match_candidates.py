import json
import argparse
import re
from pathlib import Path
from typing import Any

from skill_requirement_groups import normalize_technical_skill_requirement_groups


ROOT = Path(__file__).resolve().parent
TECHNICAL_SKILL_ABOX_PATH = ROOT / "technical_skill_abox.json"
LEGACY_SKILL_ABOX_PATH = ROOT / "skill_abox.json"
INFERENCE_RULE_ABOX_PATH = ROOT / "inference_rule_abox.json"
DEFAULT_IMPLIED_WEIGHT = 0.7
PARSED_RESULTS_DIR = ROOT / "parsed_results"
JSON_MATCH_OUTPUT_PATH = PARSED_RESULTS_DIR / "candidate_match_results.json"


def load_skill_catalog() -> dict[str, dict]:
    skill_path = TECHNICAL_SKILL_ABOX_PATH if TECHNICAL_SKILL_ABOX_PATH.exists() else LEGACY_SKILL_ABOX_PATH
    if not skill_path.exists():
        raise FileNotFoundError(f"未找到技能本体文件：{TECHNICAL_SKILL_ABOX_PATH}")

    data = json.loads(skill_path.read_text(encoding="utf-8"))
    skills = data.get("skills", [])
    if not isinstance(skills, list):
        raise ValueError(f"{skill_path.name} 中的 skills 必须是数组。")

    return {
        str(skill.get("id", "")).strip(): skill
        for skill in skills
        if isinstance(skill, dict) and str(skill.get("id", "")).strip()
    }


def load_inference_rules_by_capability() -> dict[str, list[dict[str, Any]]]:
    if not INFERENCE_RULE_ABOX_PATH.exists():
        return {}

    data = json.loads(INFERENCE_RULE_ABOX_PATH.read_text(encoding="utf-8"))
    rules = data.get("inference_rules", [])
    if not isinstance(rules, list):
        raise ValueError("inference_rule_abox.json 中的 inference_rules 必须是数组。")

    result: dict[str, list[dict[str, Any]]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        source_skill_id = str(rule.get("source_skill_id", "") or "").strip()
        target_capability_id = str(rule.get("target_capability_id", "") or "").strip()
        if not source_skill_id or not target_capability_id:
            continue
        try:
            inference_weight = float(rule.get("inference_weight", DEFAULT_IMPLIED_WEIGHT))
        except (TypeError, ValueError):
            inference_weight = DEFAULT_IMPLIED_WEIGHT
        result.setdefault(target_capability_id, []).append(
            {
                "source_skill_id": source_skill_id,
                "target_capability_id": target_capability_id,
                "inference_weight": max(0.0, min(inference_weight, 1.0)),
                "reason": str(rule.get("reason", "") or "").strip(),
            }
        )
    return result


def skill_name(skill_id: str, catalog: dict[str, dict]) -> str:
    skill = catalog.get(skill_id, {})
    return str(skill.get("name") or skill_id)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_jd_json(results_dir: Path = PARSED_RESULTS_DIR) -> Path:
    jd_files = sorted(
        results_dir.glob("jd_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not jd_files:
        raise FileNotFoundError(f"未在 {results_dir} 找到 jd_*.json，请先解析 JD。")
    return jd_files[0]


def find_resume_json_files(results_dir: Path = PARSED_RESULTS_DIR) -> list[Path]:
    files = [
        path
        for path in results_dir.glob("*_parsed.json")
        if path.is_file() and not path.name.startswith("jd_")
    ]
    return sorted(files, key=lambda item: item.name.lower())


def item_id(item: Any, *keys: str) -> str:
    if isinstance(item, dict):
        for key in keys:
            value = str(item.get(key, "") or "").strip()
            if value:
                return value
        value = str(item.get("id", "") or "").strip()
        if value:
            return value
    return str(item or "").strip()


def item_name(item: Any, fallback: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("名称") or fallback or item_id(item))
    return str(item or fallback)


def get_ids(items: Any, *keys: str) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {value for item in items if (value := item_id(item, *keys))}


def get_id_name_map(items: Any, *keys: str) -> dict[str, str]:
    if not isinstance(items, list):
        return {}
    result: dict[str, str] = {}
    for item in items:
        entity_id = item_id(item, *keys)
        if entity_id:
            result[entity_id] = item_name(item, entity_id)
    return result


def get_id_item_map(items: Any, *keys: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        entity_id = item_id(item, *keys)
        if entity_id:
            result[entity_id] = item
    return result


def find_matching_item(items: Any, entity_id: str, name: str, *keys: str) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if entity_id and item_id(item, *keys) == entity_id:
            return item
        if name and item_name(item) == name:
            return item
    return {}


def item_evidence(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    evidence = item.get("evidence", "") or item.get("证据", "")
    if isinstance(evidence, list):
        return "；".join(str(value) for value in evidence if value)
    return str(evidence or "")


def get_first_list(data: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def normalize_string_items(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    result: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            result.append(item)
            continue
        text = str(item or "").strip()
        if text:
            result.append({"id": text, "name": text})
    return result


def parse_level_number(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"\d{1,2}", text)
    if not match:
        return None
    return int(match.group(0))


def parse_level_range(value: Any) -> tuple[int | None, int | None]:
    text = str(value or "")
    numbers = [int(item) for item in re.findall(r"\d{1,2}", text)]
    if not numbers:
        return None, None
    normalized = text.lower()
    if re.search(r"and\s+below|or\s+below|below|及以下|以下", normalized):
        return numbers[0], None
    if re.search(r"and\s+above|or\s+above|above|及以上|以上", normalized):
        return None, numbers[0]
    return min(numbers), max(numbers)


def format_level_requirement(min_level: int | None, max_level: int | None) -> str:
    if min_level is not None and max_level is not None:
        return f"{min_level}-{max_level}"
    if min_level is not None:
        return f"{min_level}及以下"
    if max_level is not None:
        return f"{max_level}及以上"
    return "未配置"


def infer_json_job_type(jd: dict[str, Any]) -> str:
    job_type = jd.get("岗位类型", {})
    if isinstance(job_type, dict):
        primary = str(job_type.get("primary", "") or "").upper()
    else:
        primary = str(job_type or "").upper()
    if primary and primary != "UNKNOWN":
        return primary

    title = str(jd.get("岗位名称", "") or jd.get("职位名称", "") or jd.get("job_title", "") or "").upper()
    if any(keyword in title for keyword in ["BA", "BUSINESS ANALYST", "业务分析"]):
        return "BA"
    if any(keyword in title for keyword in ["产品", "PRODUCT"]):
        return "PRODUCT"
    if any(keyword in title for keyword in ["项目经理", "PROJECT MANAGER", "PM"]):
        return "PROJECT_MANAGER"
    if any(keyword in title for keyword in ["后端", "前端", "开发", "ENGINEER", "DEVELOPER", "PYTHON", "JAVA"]):
        return "TECHNICAL"

    technical_count = len(get_first_list(jd, "技术技能要求", "技能要求"))
    competency_count = len(jd.get("胜任力要求", []))
    capability_count = len(jd.get("能力要求", []))
    if technical_count >= 4 and competency_count >= 2:
        return "HYBRID"
    if technical_count >= max(2, competency_count):
        return "TECHNICAL"
    if competency_count >= 2 or capability_count >= 4:
        return "BA"
    return "UNKNOWN"


def weights_for_job_type(job_type: str) -> dict[str, float]:
    normalized = job_type.upper()
    if normalized == "TECHNICAL":
        return {
            "technical_skills": 0.80,
            "capabilities": 0.10,
            "competencies": 0.10,
            "domains": 0.00,
            "industries": 0.00,
        }
    if normalized in {"BA", "CONSULTING"}:
        return {
            "technical_skills": 0.00,
            "capabilities": 0.40,
            "competencies": 0.60,
            "domains": 0.00,
            "industries": 0.00,
        }
    if normalized == "HYBRID":
        return {
            "technical_skills": 0.40,
            "capabilities": 0.40,
            "competencies": 0.20,
            "domains": 0.00,
            "industries": 0.00,
        }
    if normalized in {"PRODUCT", "PROJECT_MANAGER"}:
        return {
            "technical_skills": 0.10,
            "capabilities": 0.50,
            "competencies": 0.40,
            "domains": 0.00,
            "industries": 0.00,
        }
    return {
        "technical_skills": 0.30,
        "capabilities": 0.45,
        "competencies": 0.25,
        "domains": 0.00,
        "industries": 0.00,
    }


def rebalance_weights_for_signal(base_weights: dict[str, float], jd: dict[str, Any]) -> dict[str, float]:
    weights = dict(base_weights)
    weights["domains"] = 0.0
    weights["industries"] = 0.0
    return weights


def score_for_partial_matches(matches: list[float]) -> float:
    miss_score = 1.0
    for value in matches:
        miss_score *= 1 - max(0.0, min(value, 1.0))
    return round(1 - miss_score, 4)


def capability_inference_matches(
    required_capability_id: str,
    candidate_skill_ids: set[str],
    inference_rules: dict[str, list[dict[str, Any]]],
    skill_catalog: dict[str, dict],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for rule in inference_rules.get(required_capability_id, []):
        source_skill_id = str(rule.get("source_skill_id", "") or "")
        if source_skill_id not in candidate_skill_ids:
            continue
        matches.append(
            {
                "source_skill_id": source_skill_id,
                "source_skill_name": skill_name(source_skill_id, skill_catalog),
                "weight": float(rule.get("inference_weight", DEFAULT_IMPLIED_WEIGHT)),
                "reason": str(rule.get("reason", "") or "").strip(),
            }
        )
    return matches


def score_capability_dimension(
    required_items: Any,
    candidate_items: Any,
    candidate_skills: Any,
    inference_rules: dict[str, list[dict[str, Any]]],
    skill_catalog: dict[str, dict],
) -> dict[str, Any]:
    if not isinstance(required_items, list) or not required_items:
        return {
            "score": None,
            "required_count": 0,
            "matched_count": 0,
            "matched": [],
            "missing": [],
            "reason": "能力：JD 未配置该维度，评分时不计入权重。",
        }

    candidate_ids = get_ids(candidate_items, "capability_id")
    candidate_names = get_id_name_map(candidate_items, "capability_id")
    candidate_item_by_id = get_id_item_map(candidate_items, "capability_id")
    candidate_skill_by_id = get_id_item_map(candidate_skills, "skill_id")
    candidate_name_values = {name for name in candidate_names.values() if name}
    candidate_skill_ids = get_ids(candidate_skills, "skill_id")

    total_weight = 0.0
    matched_weight = 0.0
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for item in required_items:
        required_id = item_id(item, "capability_id")
        if not required_id:
            continue
        weight = requirement_weight(item)
        total_weight += weight
        required_name = item_name(item, required_id)
        row = {"id": required_id, "name": required_name, "weight": weight, "jd_evidence": item_evidence(item)}
        if required_id in candidate_ids or required_name in candidate_name_values:
            matched_weight += weight
            candidate_item = candidate_item_by_id.get(required_id) or find_matching_item(candidate_items, required_id, required_name, "capability_id")
            row["match_type"] = "direct_capability"
            row["candidate_name"] = candidate_names.get(required_id, item_name(candidate_item, required_name))
            row["candidate_evidence"] = item_evidence(candidate_item)
            row["match_score"] = 1.0
            matched.append(row)
            continue

        inferred = capability_inference_matches(required_id, candidate_skill_ids, inference_rules, skill_catalog)
        if inferred:
            inferred_score = score_for_partial_matches([float(match["weight"]) for match in inferred])
            matched_weight += weight * inferred_score
            row["match_type"] = "inferred_from_technical_skill"
            row["match_score"] = inferred_score
            row["inferred_from"] = [
                {
                    "skill_id": match["source_skill_id"],
                    "skill_name": match["source_skill_name"],
                    "weight": match["weight"],
                    "reason": match["reason"],
                    "evidence": item_evidence(candidate_skill_by_id.get(match["source_skill_id"], {})),
                }
                for match in inferred
            ]
            row["candidate_evidence"] = "；".join(
                item.get("evidence", "") for item in row["inferred_from"] if item.get("evidence")
            )
            matched.append(row)
        else:
            missing.append(row)

    score = round(matched_weight / total_weight * 100, 2) if total_weight else 0.0
    direct_count = sum(1 for item in matched if item.get("match_type") == "direct_capability")
    inferred_count = len(matched) - direct_count
    return {
        "score": score,
        "required_count": len(required_items),
        "matched_count": len(matched),
        "matched": matched,
        "missing": missing,
        "reason": f"能力：直接命中 {direct_count} 项，推导命中 {inferred_count} 项，覆盖 {len(matched)}/{len(required_items)}，得分 {score:.2f}",
    }


def requirement_weight(item: Any) -> float:
    if not isinstance(item, dict):
        return 1.0
    priority = str(item.get("priority", "") or item.get("优先级", "") or "").lower()
    if priority in {"nice", "nice_to_have", "preferred", "加分", "优先"}:
        return 0.5
    return 1.0


def score_dimension(
    label_text: str,
    required_items: Any,
    candidate_items: Any,
    id_key: str,
) -> dict[str, Any]:
    if not isinstance(required_items, list) or not required_items:
        return {
            "score": None,
            "required_count": 0,
            "matched_count": 0,
            "matched": [],
            "missing": [],
            "reason": f"{label_text}：JD 未配置该维度，评分时不计入权重。",
        }

    candidate_ids = get_ids(candidate_items, id_key)
    candidate_names = get_id_name_map(candidate_items, id_key)
    candidate_item_by_id = get_id_item_map(candidate_items, id_key)
    candidate_name_values = {name for name in candidate_names.values() if name}

    total_weight = 0.0
    matched_weight = 0.0
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for item in required_items:
        required_id = item_id(item, id_key)
        if not required_id:
            continue
        weight = requirement_weight(item)
        total_weight += weight
        required_name = item_name(item, required_id)
        row = {"id": required_id, "name": required_name, "weight": weight, "jd_evidence": item_evidence(item)}
        if required_id in candidate_ids or required_name in candidate_name_values:
            matched_weight += weight
            candidate_item = candidate_item_by_id.get(required_id) or find_matching_item(candidate_items, required_id, required_name, id_key)
            row["candidate_name"] = candidate_names.get(required_id, item_name(candidate_item, required_name))
            row["candidate_evidence"] = item_evidence(candidate_item)
            matched.append(row)
        else:
            missing.append(row)

    if total_weight == 0:
        score = 0.0
    else:
        score = round(matched_weight / total_weight * 100, 2)

    return {
        "score": score,
        "required_count": len(required_items),
        "matched_count": len(matched),
        "matched": matched,
        "missing": missing,
        "reason": f"{label_text}：命中 {len(matched)}/{len(required_items)}，得分 {score:.2f}",
    }


def score_technical_skill_groups(required_groups: Any, candidate_items: Any) -> dict[str, Any]:
    if not isinstance(required_groups, list) or not required_groups:
        return {
            "score": None,
            "required_count": 0,
            "matched_count": 0,
            "matched": [],
            "missing": [],
            "reason": "技术技能：JD 未配置该维度，评分时不计入权重。",
        }

    candidate_ids = get_ids(candidate_items, "skill_id")
    candidate_names = get_id_name_map(candidate_items, "skill_id")
    candidate_item_by_id = get_id_item_map(candidate_items, "skill_id")
    candidate_name_values = {name for name in candidate_names.values() if name}

    total_weight = 0.0
    matched_weight = 0.0
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for group in required_groups:
        if not isinstance(group, dict):
            continue
        items = group.get("items", [])
        if not isinstance(items, list) or not items:
            continue
        logic = str(group.get("logic", "ALL_OF") or "ALL_OF").upper()
        weight = requirement_weight(group)
        total_weight += weight

        group_items = []
        matched_items = []
        for item in items:
            required_id = item_id(item, "skill_id")
            required_name = item_name(item, required_id)
            row = {"id": required_id, "name": required_name}
            group_items.append(row)
            if required_id in candidate_ids or required_name in candidate_name_values:
                candidate_item = candidate_item_by_id.get(required_id) or find_matching_item(candidate_items, required_id, required_name, "skill_id")
                row["candidate_name"] = candidate_names.get(required_id, item_name(candidate_item, required_name))
                row["candidate_evidence"] = item_evidence(candidate_item)
                matched_items.append(row)

        is_matched = bool(matched_items) if logic == "ANY_OF" else len(matched_items) == len(group_items)
        row = {
            "group_id": str(group.get("group_id", "") or ""),
            "group_name": str(group.get("group_name", "") or ""),
            "logic": logic,
            "weight": weight,
            "required_items": group_items,
            "matched_items": matched_items,
            "evidence": str(group.get("evidence", "") or ""),
        }
        if is_matched:
            matched_weight += weight
            matched.append(row)
        else:
            missing.append(row)

    score = round(matched_weight / total_weight * 100, 2) if total_weight else 0.0
    any_of_count = sum(1 for group in required_groups if isinstance(group, dict) and str(group.get("logic", "")).upper() == "ANY_OF")
    return {
        "score": score,
        "required_count": len(required_groups),
        "matched_count": len(matched),
        "matched": matched,
        "missing": missing,
        "reason": f"技术技能：命中 {len(matched)}/{len(required_groups)} 个要求组，ANY_OF 组 {any_of_count} 个，得分 {score:.2f}",
    }


def not_scored_dimension(label_text: str) -> dict[str, Any]:
    return {
        "score": None,
        "required_count": 0,
        "matched_count": 0,
        "matched": [],
        "missing": [],
        "reason": f"{label_text}：当前规则不参与匹配评分。",
    }


def score_json_candidate(
    jd: dict[str, Any],
    candidate: dict[str, Any],
    candidate_path: Path,
    skill_catalog: dict[str, dict],
    inference_rules: dict[str, list[dict[str, Any]]],
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    job_type = infer_json_job_type(jd)
    default_weights = rebalance_weights_for_signal(weights_for_job_type(job_type), jd)
    if custom_weights:
        base_weights = dict(default_weights)
        for key in ("technical_skills", "capabilities", "competencies"):
            if key in custom_weights:
                try:
                    base_weights[key] = float(custom_weights[key])
                except (TypeError, ValueError):
                    pass
        base_weights["domains"] = 0.0
        base_weights["industries"] = 0.0
    else:
        base_weights = default_weights

    jd_technical_skills = get_first_list(jd, "技术技能要求", "技能要求")
    jd_technical_skill_groups = normalize_technical_skill_requirement_groups(
        jd.get("技术技能要求组", []),
        jd_technical_skills,
    )

    dimensions = {
        "technical_skills": score_technical_skill_groups(jd_technical_skill_groups, candidate.get("技术技能", [])),
        "capabilities": score_capability_dimension(
            jd.get("能力要求", []),
            candidate.get("能力", []),
            candidate.get("技术技能", []),
            inference_rules,
            skill_catalog,
        ),
        "competencies": score_dimension("胜任力", jd.get("胜任力要求", []), candidate.get("胜任力", []), "competency_id"),
        "domains": not_scored_dimension("领域"),
        "industries": not_scored_dimension("行业"),
    }

    active_weight_sum = sum(
        weight
        for name, weight in base_weights.items()
        if dimensions[name]["score"] is not None
    )
    if active_weight_sum <= 0:
        total_score = 0.0
    else:
        total_score = round(
            sum(
                (dimensions[name]["score"] or 0.0) * (weight / active_weight_sum)
                for name, weight in base_weights.items()
                if dimensions[name]["score"] is not None
            ),
            2,
        )

    candidate_level = parse_level_number(candidate.get("级别"))
    min_required_level, max_required_level = parse_level_range(jd.get("级别要求"))
    has_level_requirement = bool(jd.get("是否有级别要求")) and (
        min_required_level is not None or max_required_level is not None
    )
    eliminated = False
    level_reason = "JD 未配置明确级别要求，未执行级别淘汰。"
    if has_level_requirement:
        if candidate_level is None:
            level_reason = "候选人缺少数字级别，保留参与排名；请结合简历资历做人工复核。"
        elif min_required_level is not None and candidate_level < min_required_level:
            eliminated = True
            level_reason = f"淘汰：候选人级别 {candidate_level} 高于 JD 最高要求 {min_required_level}。"
        elif max_required_level is not None and candidate_level > max_required_level:
            eliminated = True
            level_reason = f"淘汰：候选人级别 {candidate_level} 低于 JD 最低要求 {max_required_level}。"
        else:
            level_reason = (
                f"级别满足：候选人级别 {candidate_level}，JD 要求 "
                f"{format_level_requirement(min_required_level, max_required_level)}。"
            )

    return {
        "candidate_name": str(candidate.get("姓名") or candidate_path.stem.replace("_parsed", "")),
        "candidate_file": str(candidate_path),
        "job_type": job_type,
        "weights": base_weights,
        "active_weight_sum": round(active_weight_sum, 4),
        "eliminated": eliminated,
        "total_score": 0.0 if eliminated else total_score,
        "level_reason": level_reason,
        "dimensions": dimensions,
    }


def match_json_files(jd_path: Path, resume_paths: list[Path], custom_weights: dict[str, float] | None = None) -> dict[str, Any]:
    jd = load_json_file(jd_path)
    skill_catalog = load_skill_catalog()
    inference_rules = load_inference_rules_by_capability()
    job_type = infer_json_job_type(jd)
    default_weights = rebalance_weights_for_signal(weights_for_job_type(job_type), jd)
    if custom_weights:
        active_weights = dict(default_weights)
        for key in ("technical_skills", "capabilities", "competencies"):
            if key in custom_weights:
                try:
                    active_weights[key] = float(custom_weights[key])
                except (TypeError, ValueError):
                    pass
        active_weights["domains"] = 0.0
        active_weights["industries"] = 0.0
    else:
        active_weights = default_weights

    results: list[dict[str, Any]] = []
    for resume_path in resume_paths:
        candidate = load_json_file(resume_path)
        results.append(
            score_json_candidate(
                jd,
                candidate,
                resume_path,
                skill_catalog,
                inference_rules,
                custom_weights=active_weights,
            )
        )

    eliminated_results = [item for item in results if item["eliminated"]]
    visible_results = [item for item in results if not item["eliminated"]]
    visible_results.sort(key=lambda item: (-item["total_score"], item["candidate_name"]))
    return {
        "jd_file": str(jd_path),
        "job_title": str(jd.get("岗位名称", "") or jd_path.stem),
        "job_type": job_type,
        "weights": active_weights,
        "source_candidate_count": len(results),
        "candidate_count": len(visible_results),
        "eliminated_count": len(eliminated_results),
        "results": visible_results,
    }


def print_json_match_results(report: dict[str, Any], top: int = 0) -> None:
    print("=== JSON 人岗匹配结果 ===")
    print(f"JD: {report['job_title']} ({report['job_type']})")
    print(f"候选人数: {report['candidate_count']}（职级过滤 {report.get('eliminated_count', 0)} 人）")
    print()

    results = report["results"][:top] if top and top > 0 else report["results"]
    for index, item in enumerate(results, start=1):
        print(f"{index}. {item['candidate_name']} | 通过 | 总分 {item['total_score']:.2f}")
        print(f"   {item['level_reason']}")
        for dimension_name, detail in item["dimensions"].items():
            if detail["score"] is None:
                continue
            print(f"   - {detail['reason']}")
        print()


def iter_implied_matches(candidate_skill_ids: set[str], required_skill_id: str, catalog: dict[str, dict]) -> list[dict]:
    matches: list[dict] = []
    for source_id in sorted(candidate_skill_ids):
        source_skill = catalog.get(source_id, {})
        implies = source_skill.get("implies", [])
        if not isinstance(implies, list):
            continue
        for item in implies:
            if not isinstance(item, dict):
                continue
            target_id = str(item.get("target_id", "")).strip()
            if target_id != required_skill_id:
                continue
            try:
                weight = float(item.get("weight", DEFAULT_IMPLIED_WEIGHT))
            except (TypeError, ValueError):
                weight = DEFAULT_IMPLIED_WEIGHT
            matches.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "weight": max(0.0, min(weight, 1.0)),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
            # One source skill can contribute only once to the same required skill.
            break
    return matches


def score_skills(required_skill_ids: set[str], candidate_skill_ids: set[str], catalog: dict[str, dict]) -> tuple[float, list[str]]:
    if not required_skill_ids:
        return 100.0, ["岗位未配置技能要求，技能维度按满分处理"]

    total_weight = 0.0
    reasons: list[str] = []

    for required_id in sorted(required_skill_ids):
        required_name = skill_name(required_id, catalog)
        if required_id in candidate_skill_ids:
            total_weight += 1.0
            reasons.append(f"{required_name}：直接匹配，权重 1.00")
            continue

        implied_matches = iter_implied_matches(candidate_skill_ids, required_id, catalog)
        if implied_matches:
            miss_score = 1.0
            for item in implied_matches:
                miss_score *= 1 - item["weight"]
            implied_score = round(1 - miss_score, 2)
            total_weight += implied_score
            source_parts = [
                f"{skill_name(item['source_id'], catalog)}({item['weight']:.2f})"
                for item in implied_matches
            ]
            detail_parts = [
                f"{skill_name(item['source_id'], catalog)}：{item['reason']}"
                for item in implied_matches
                if item["reason"]
            ]
            detail_text = f"，依据：{'；'.join(detail_parts)}" if detail_parts else ""
            reasons.append(
                f"{required_name}：通过 implies 公式匹配，来源 {' + '.join(source_parts)}，"
                f"公式 score = 1 - ∏(1 - weight)，得分 {implied_score:.2f}{detail_text}"
            )
            continue

        reasons.append(f"{required_name}：未匹配，权重 0.00")

    return round(total_weight / len(required_skill_ids) * 100, 2), reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="候选人 JSON 匹配评分。")
    parser.add_argument("--jd", default="", help="JD JSON 文件路径。JSON 模式默认使用最新 jd_*.json。")
    parser.add_argument("--resumes", default="", help="简历 JSON 目录或单个文件。JSON 模式默认 parsed_results。")
    parser.add_argument("--output", default="", help="JSON 模式输出文件，默认 parsed_results/candidate_match_results.json。")
    parser.add_argument("--top", type=int, default=0, help="控制台只展示前 N 名，0 表示全部展示。")
    return parser.parse_args()


def run_json_matching(args: argparse.Namespace) -> None:
    jd_path = Path(args.jd) if args.jd else find_latest_jd_json()
    if not jd_path.is_absolute():
        jd_path = (Path.cwd() / jd_path).resolve()
    if not jd_path.exists():
        raise FileNotFoundError(f"未找到 JD JSON：{jd_path}")

    if args.resumes:
        resume_input = Path(args.resumes)
        if not resume_input.is_absolute():
            resume_input = (Path.cwd() / resume_input).resolve()
        if resume_input.is_file():
            resume_paths = [resume_input]
        else:
            resume_paths = find_resume_json_files(resume_input)
    else:
        resume_paths = find_resume_json_files()

    if not resume_paths:
        raise FileNotFoundError("未找到简历解析 JSON，请先运行简历解析或 batch_parse_resumes.py。")

    report = match_json_files(jd_path, resume_paths)

    output_path = Path(args.output) if args.output else JSON_MATCH_OUTPUT_PATH
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_json_match_results(report, top=args.top)
    print(f"结果文件：{output_path}")


def main() -> None:
    args = parse_args()
    run_json_matching(args)


if __name__ == "__main__":
    main()
