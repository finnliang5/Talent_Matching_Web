import hashlib
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from level_matcher import infer_jd_level_requirement
from skill_requirement_groups import build_technical_skill_requirement_groups
from resume_parser import (
    build_catalog_prompt,
    derive_capabilities,
    derive_domains,
    extract_entities_from_text,
    extract_json_object,
    load_resume_ontology_catalog,
    normalize_entity_items,
    normalize_unknown_skill_items,
    split_items,
)


APP_DIR = Path(__file__).resolve().parent
JD_RESULT_DIR = APP_DIR / "parsed_results"
UNKNOWN_JD_SKILL_LOG_PATH = JD_RESULT_DIR / "out_of_ontology_jd_skills.jsonl"


ROLE_RULES = {
    "DS": {
        "title": "Data Scientist",
        "job_type": "TECHNICAL",
        "patterns": [r"(?<![A-Za-z])DS(?![A-Za-z])", r"\bData\s+Scientist\b", r"数据科学家"],
        "technical_skill_groups": [
            {
                "group_name": "数据科学基础工具",
                "logic": "ANY_OF",
                "skill_ids": ["SKILL_PYTHON_CORE", "SKILL_PANDAS", "SKILL_NUMPY"],
            },
            {
                "group_name": "机器学习建模",
                "logic": "ANY_OF",
                "skill_ids": ["SKILL_MACHINE_LEARNING", "SKILL_SCIKIT_LEARN"],
            },
            {
                "group_name": "特征与模型评估",
                "logic": "ANY_OF",
                "skill_ids": ["SKILL_FEATURE_ENGINEERING", "SKILL_MODEL_EVALUATION"],
            },
        ],
        "capability_ids": [
            "CAP_STATISTICAL_ANALYSIS",
            "CAP_MACHINE_LEARNING_MODELING",
            "CAP_FEATURE_ENGINEERING",
            "CAP_MODEL_EVALUATION",
        ],
    }
}


def ensure_jd_dirs() -> None:
    JD_RESULT_DIR.mkdir(exist_ok=True)


def detect_priority_from_evidence(evidence: str) -> str:
    text = str(evidence or "").strip()
    if re.search(r"【必备】|【必须】|【硬性要求】|必备技能|必须掌握|required|must\s+have|essential|mandatory", text, flags=re.IGNORECASE):
        return "must"
    if re.search(r"【优先】|【加分】|【期望】|优先考虑|加分项|优先条件|preferred|nice\s+to\s+have|is\s+a\s+plus|bonus|advantage", text, flags=re.IGNORECASE):
        return "nice"
    return "must"


def safe_filename_part(value: str, default: str = "jd") -> str:
    text = re.sub(r"\s+", "_", value.strip())
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:40] or default


def jd_content_hash(jd_description: str, result: dict[str, Any]) -> str:
    source_text = re.sub(r"\s+", " ", jd_description.strip())
    if not source_text:
        source_text = json.dumps(result, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:10]


def build_jd_result_path(result: dict[str, Any], jd_description: str = "") -> Path:
    job_title = str(result.get("岗位名称", "") or "jd")
    file_stem = f"jd_{safe_filename_part(job_title)}_{jd_content_hash(jd_description, result)}"
    return JD_RESULT_DIR / f"{file_stem}.json"


def build_jd_prompt(jd_description: str) -> str:
    catalog = load_resume_ontology_catalog()
    competency_catalog = build_catalog_prompt(catalog["competencies"], "competency_id")
    capability_catalog = build_catalog_prompt(catalog["capabilities"], "capability_id")
    return f"""
你是一个招聘 JD 解析 Agent。请从 JD 描述中抽取结构化 JSON。

必须输出合法 JSON，不要输出 Markdown，不要输出解释。

JSON 字段必须为：
{{
  "岗位名称": "岗位名称，无法识别则为空字符串",
  "岗位类型": {{
    "primary": "TECHNICAL、BA、CONSULTING、PRODUCT、PROJECT_MANAGER、HYBRID 或 UNKNOWN",
    "secondary": ["可选的次要岗位类型"],
    "confidence": 0.0
  }},
  "胜任力要求": [
    {{
      "competency_id": "必须来自胜任力本体的 id，如 SKILL_REQUIREMENTS_ANALYSIS",
      "name": "必须来自胜任力本体的 name，如 需求分析",
      "evidence": "JD 中要求该胜任力的原文证据",
      "priority": "must 或 nice，根据JD中的【必备】/【优先】标记判断"
    }}
  ],
  "能力要求": [
    {{
      "capability_id": "必须来自能力本体的 id，如 CAP_REQUIREMENTS_ANALYSIS",
      "name": "必须来自能力本体的 name",
      "evidence": "JD 中要求该能力的原文证据",
      "priority": "must 或 nice"
    }}
  ],
  "本体外技术技能要求": [
    {{
      "skill_id": "",
      "name": "JD 中明确要求但 technical_skill_abox 中不存在的技术、工具、框架、平台",
      "evidence": "JD 中对应原文证据或简短依据"
    }}
  ],
  "行业知识要求": ["行业或业务领域知识要求"],
  "经验要求": ["经验要求1", "经验要求2"],
  "级别要求": "JD 中明确出现的职级/级别要求，如 9、8、7、P7、L8、Level 8；没有则为空字符串",
  "是否有级别要求": true
}}

要求：
1. 技术技能要求由程序使用 technical_skill_abox 的 name 和 aliases 规则匹配，你不要输出“技术技能要求”字段。
2. “胜任力要求”只能从胜任力本体中选择，必须有 JD 原文证据。
3. “能力要求”只能从能力本体中选择，必须有 JD 原文证据；它是最终匹配主轴。
4. 如果 JD 只说具体技术栈，例如 FastAPI、React、PostgreSQL，优先让程序通过技术技能推导能力；你只补充 JD 中语义明确但技术关键词无法直接覆盖的能力。
5. 如果 JD 明确要求的技术名词不在 technical_skill_abox 中，输出到“本体外技术技能要求”。
6. 不要把普通业务词、代码类名、函数名、配置项当成技术技能。
7. 如果 JD 没有明确职级、级别、Level、P/M/T序列、岗位层级等要求，"级别要求" 必须为空字符串，"是否有级别要求" 必须为 false。
8. 不确定的信息留空或返回空数组。
9. 只返回 JSON 对象。

胜任力本体：
{competency_catalog}

能力本体：
{capability_catalog}

JD描述：
{jd_description}
""".strip()


def extract_level_requirement(jd_description: str) -> str:
    return infer_jd_level_requirement(jd_description).level_requirement


def call_jd_llm(jd_description: str, api_key: str, base_url: str, model: str) -> str:
    if not api_key:
        raise ValueError("请先在 .env 文件中配置 OPENAI_API_KEY。")

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    messages = [
        {"role": "system", "content": "你是一个只输出 JSON 的 JD 解析 Agent。"},
        {"role": "user", "content": build_jd_prompt(jd_description)},
    ]
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    try:
        response = client.chat.completions.create(
            **request_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if "response_format" not in str(exc):
            raise
        response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content or "{}"


def merge_entity_items(id_key: str, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            entity_id = str(item.get(id_key, "") or "").strip()
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            result.append(item)
    return result


def merge_unknown_skill_items(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = (item.get("skill_id") or item.get("name") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
    return result


def normalize_capability_requirements(values: Any, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities, _ = normalize_entity_items(
        values,
        catalog["capabilities"],
        "capability_id",
        "capability",
    )
    capability_lookup = catalog["capability_lookup"]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in capabilities:
        capability_id = str(item.get("capability_id", "") or "")
        if not capability_id or capability_id in seen:
            continue
        seen.add(capability_id)
        capability = capability_lookup.get(capability_id, {})
        result.append(
            {
                "capability_id": capability_id,
                "name": str(capability.get("name", "") or item.get("name", "") or capability_id),
                "domain_id": str(capability.get("domain_id", "") or ""),
                "evidence": [str(item.get("evidence", "") or "")] if item.get("evidence") else [],
                "sources": [{"source_type": "jd_direct", "source_id": capability_id, "name": str(item.get("name", "") or capability_id)}],
                "priority": str(item.get("priority", "") or "must"),
            }
        )
    return result


def catalog_item_by_id(items: list[dict[str, Any]], entity_id: str) -> dict[str, Any]:
    for item in items:
        if str(item.get("id", "") or "") == entity_id:
            return item
    return {}


def build_role_skill_item(skill_id: str, catalog: dict[str, Any], evidence: str, role_key: str) -> dict[str, Any]:
    skill = catalog_item_by_id(catalog["technical_skills"], skill_id)
    return {
        "skill_id": skill_id,
        "name": str(skill.get("name", "") or skill_id),
        "type": str(skill.get("type", "") or ""),
        "evidence": evidence,
        "source": "role_rule",
        "source_type": "technical_skill",
        "matched_term": role_key,
    }


def build_role_capability_item(capability_id: str, catalog: dict[str, Any], evidence: str, role_key: str) -> dict[str, Any]:
    capability = catalog["capability_lookup"].get(capability_id, {})
    return {
        "capability_id": capability_id,
        "name": str(capability.get("name", "") or capability_id),
        "domain_id": str(capability.get("domain_id", "") or ""),
        "evidence": [evidence],
        "sources": [{"source_type": "role_rule", "source_id": role_key, "name": role_key}],
        "priority": "must",
    }


def find_role_rule(jd_description: str) -> tuple[str, dict[str, Any]] | tuple[str, None]:
    for role_key, rule in ROLE_RULES.items():
        for pattern in rule.get("patterns", []):
            if re.search(pattern, jd_description, flags=re.IGNORECASE):
                return role_key, rule
    return "", None


def build_role_skill_groups(role_key: str, role_rule: dict[str, Any], catalog: dict[str, Any], evidence: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(role_rule.get("technical_skill_groups", []), start=1):
        items = [
            {
                "skill_id": skill_id,
                "name": str(catalog_item_by_id(catalog["technical_skills"], skill_id).get("name", "") or skill_id),
            }
            for skill_id in group.get("skill_ids", [])
        ]
        if not items:
            continue
        groups.append(
            {
                "group_id": f"{role_key}_GROUP_{index}",
                "group_name": str(group.get("group_name", "") or " 或 ".join(item["name"] for item in items)),
                "logic": str(group.get("logic", "ANY_OF") or "ANY_OF").upper(),
                "items": items,
                "evidence": evidence,
                "source": "role_rule",
            }
        )
    return groups


def extract_project_constraints(jd_description: str) -> dict[str, str]:
    constraints: dict[str, str] = {}
    location_match = re.search(r"(上海|北京|深圳|广州|杭州|成都|南京|苏州|武汉|西安|大连|远程)", jd_description)
    if location_match:
        constraints["工作地点"] = location_match.group(1)

    availability_match = re.search(
        r"(\d{1,2}\s*月\s*(?:上旬|中旬|下旬|月初|月底|月末|初|中|底|末)?)\s*(?:可上项目|可入场|可到岗|到岗|入场|可上)?",
        jd_description,
    )
    if availability_match and re.search(r"可上项目|可入场|可到岗|到岗|入场|可上", jd_description):
        constraints["可上项目时间"] = re.sub(r"\s+", "", availability_match.group(1))

    if constraints:
        constraints["原文"] = jd_description
    return constraints


def merge_capability_requirements(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result_by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            capability_id = str(item.get("capability_id", "") or "")
            if not capability_id:
                continue
            entry = result_by_id.setdefault(
                capability_id,
                {
                    "capability_id": capability_id,
                    "name": str(item.get("name", "") or capability_id),
                    "domain_id": str(item.get("domain_id", "") or ""),
                    "evidence": [],
                    "sources": [],
                    "priority": str(item.get("priority", "") or "must"),
                },
            )
            if not entry.get("domain_id") and item.get("domain_id"):
                entry["domain_id"] = str(item.get("domain_id", ""))
            for evidence in item.get("evidence", []) or []:
                evidence_text = str(evidence or "")
                if evidence_text and evidence_text not in entry["evidence"]:
                    entry["evidence"].append(evidence_text)
            for source in item.get("sources", []) or []:
                if source not in entry["sources"]:
                    entry["sources"].append(source)
            if item.get("priority") == "must":
                entry["priority"] = "must"
    return sorted(result_by_id.values(), key=lambda item: item["capability_id"])


def infer_job_type(data: dict[str, Any], technical_skills: list[dict[str, Any]], competencies: list[dict[str, Any]]) -> dict[str, Any]:
    raw_type = data.get("岗位类型", data.get("job_type", {}))
    if isinstance(raw_type, dict):
        primary = str(raw_type.get("primary", "") or "").upper()
        secondary = raw_type.get("secondary", [])
        confidence = raw_type.get("confidence", 0)
    else:
        primary = str(raw_type or "").upper()
        secondary = []
        confidence = 0

    technical_count = len(technical_skills)
    competency_count = len(competencies)
    if not primary or primary == "UNKNOWN":
        if technical_count >= 4 and competency_count >= 3:
            primary = "HYBRID"
        elif technical_count >= max(2, competency_count):
            primary = "TECHNICAL"
        elif competency_count >= 2:
            primary = "BA"
        else:
            primary = "UNKNOWN"
        confidence = 0.6 if primary != "UNKNOWN" else 0.0

    if isinstance(secondary, str):
        secondary = split_items(secondary)
    elif not isinstance(secondary, list):
        secondary = []

    return {
        "primary": primary,
        "secondary": [str(item).upper() for item in secondary if item],
        "confidence": confidence,
        "signals": {
            "technical_skill_requirement_count": technical_count,
            "competency_requirement_count": competency_count,
        },
    }


def build_jd_evidence_items(
    technical_skills: list[dict[str, Any]],
    competencies: list[dict[str, Any]],
    capabilities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in technical_skills:
        evidence.append(
            {
                "evidence_type": "technical_skill_requirement",
                "source_text": str(item.get("evidence", "")),
                "supports_entity_id": str(item.get("skill_id", "")),
                "supports_entity_name": str(item.get("name", "")),
                "derived_capability_ids": [
                    capability["capability_id"]
                    for capability in capabilities
                    for source in capability.get("sources", [])
                    if source.get("source_type") == "technical_skill" and source.get("source_id") == item.get("skill_id")
                ],
            }
        )
    for item in competencies:
        evidence.append(
            {
                "evidence_type": "competency_requirement",
                "source_text": str(item.get("evidence", "")),
                "supports_entity_id": str(item.get("competency_id", "")),
                "supports_entity_name": str(item.get("name", "")),
                "derived_capability_ids": [
                    capability["capability_id"]
                    for capability in capabilities
                    for source in capability.get("sources", [])
                    if source.get("source_type") == "competency" and source.get("source_id") == item.get("competency_id")
                ],
            }
        )
    return evidence


def normalize_jd_result(data: dict[str, Any], jd_description: str = "") -> dict[str, Any]:
    catalog = load_resume_ontology_catalog()
    technical_skills_catalog = catalog["technical_skills"]
    competency_catalog = catalog["competencies"]
    capability_catalog = catalog["capabilities"]
    role_key, role_rule = find_role_rule(jd_description)
    role_evidence = jd_description.strip()

    rule_technical_skills = (
        extract_entities_from_text(jd_description, technical_skills_catalog, "skill_id", "technical_skill")
        if jd_description
        else []
    )
    rule_competencies = (
        extract_entities_from_text(jd_description, competency_catalog, "competency_id", "competency")
        if jd_description
        else []
    )
    rule_capabilities = (
        extract_entities_from_text(jd_description, capability_catalog, "capability_id", "capability")
        if jd_description
        else []
    )

    llm_competencies, _ = normalize_entity_items(
        data.get("胜任力要求", data.get("competency_requirements", [])),
        competency_catalog,
        "competency_id",
        "competency",
    )
    known_from_unknown, explicit_unknown_skills = normalize_unknown_skill_items(
        data.get("本体外技术技能要求", []),
        technical_skills_catalog,
    )
    direct_capabilities = normalize_capability_requirements(data.get("能力要求", []), catalog)
    role_technical_skills = (
        [
            build_role_skill_item(skill_id, catalog, role_evidence, role_key)
            for group in role_rule.get("technical_skill_groups", [])
            for skill_id in group.get("skill_ids", [])
        ]
        if role_rule
        else []
    )
    role_capabilities = (
        [build_role_capability_item(capability_id, catalog, role_evidence, role_key) for capability_id in role_rule.get("capability_ids", [])]
        if role_rule
        else []
    )

    technical_skills = merge_entity_items("skill_id", rule_technical_skills, known_from_unknown, role_technical_skills)
    for skill in technical_skills:
        skill["priority"] = detect_priority_from_evidence(skill.get("evidence", ""))
    competencies = merge_entity_items("competency_id", rule_competencies, llm_competencies)
    for competency in competencies:
        if not competency.get("priority"):
            competency["priority"] = detect_priority_from_evidence(competency.get("evidence", ""))
    derived_capabilities = derive_capabilities(technical_skills, competencies, catalog)
    normalized_rule_capabilities = normalize_capability_requirements(rule_capabilities, catalog)
    capabilities = merge_capability_requirements(derived_capabilities, normalized_rule_capabilities, direct_capabilities, role_capabilities)
    domains = derive_domains(capabilities, catalog)
    unknown_skills = merge_unknown_skill_items(explicit_unknown_skills)

    job_title = str(data.get("岗位名称", "") or data.get("职位名称", "") or data.get("job_title", "") or "")
    if role_rule and (not job_title or job_title.upper() == role_key):
        job_title = str(role_rule.get("title", "") or role_key)
    level_requirement = str(data.get("级别要求", "") or data.get("职级要求", "") or data.get("level_requirement", "") or "").strip()
    inferred_level = infer_jd_level_requirement(jd_description, job_title)
    extracted_level = ""
    if not level_requirement:
        extracted_level = inferred_level.level_requirement
        level_requirement = extracted_level
    has_level = bool(level_requirement)
    if isinstance(data.get("是否有级别要求"), bool) and not extracted_level:
        has_level = bool(data.get("是否有级别要求")) and bool(level_requirement)

    job_type_data = data
    if role_rule and (not data.get("岗位类型") or str(data.get("岗位类型", "")).upper() == "UNKNOWN"):
        job_type_data = {
            **data,
            "岗位类型": {
                "primary": str(role_rule.get("job_type", "TECHNICAL")),
                "secondary": [],
                "confidence": 0.9,
            },
        }
    role_skill_groups = build_role_skill_groups(role_key, role_rule, catalog, role_evidence) if role_rule else []
    technical_skill_groups = role_skill_groups or build_technical_skill_requirement_groups(technical_skills)

    result = {
        "岗位名称": job_title,
        "岗位类型": infer_job_type(job_type_data, technical_skills, competencies),
        "技术技能要求": technical_skills,
        "技术技能要求组": technical_skill_groups,
        "胜任力要求": competencies,
        "能力要求": capabilities,
        "领域要求": domains,
        "本体外技术技能要求": unknown_skills,
        "经验要求": split_items(data.get("经验要求", [])),
        "级别要求": level_requirement,
        "是否有级别要求": has_level,
        "证据": build_jd_evidence_items(technical_skills, competencies, capabilities),
    }
    if inferred_level.has_level_requirement and extracted_level:
        result["级别推断"] = inferred_level.to_dict()
    industry_knowledge = split_items(data.get("行业知识要求", data.get("知识要求", [])))
    if industry_knowledge:
        result["行业知识要求"] = industry_knowledge
    project_constraints = extract_project_constraints(jd_description)
    if project_constraints:
        result["项目约束"] = project_constraints
    return result


def parse_jd_to_json(jd_description: str, api_key: str, base_url: str, model: str) -> dict[str, Any]:
    raw_result = call_jd_llm(jd_description, api_key, base_url, model)
    parsed = extract_json_object(raw_result)
    return normalize_jd_result(parsed, jd_description)


def save_jd_result(result: dict[str, Any], jd_description: str = "") -> Path:
    JD_RESULT_DIR.mkdir(exist_ok=True)
    result_path = build_jd_result_path(result, jd_description)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_unknown_jd_skills_log(result, result_path.name)
    return result_path


def append_unknown_jd_skills_log(result: dict[str, Any], source: str) -> None:
    unknown_skills = result.get("本体外技术技能要求", [])
    if not unknown_skills:
        return

    JD_RESULT_DIR.mkdir(exist_ok=True)
    log_item = {
        "source": source,
        "job_title": result.get("岗位名称", ""),
        "unknown_skills": unknown_skills,
    }
    with UNKNOWN_JD_SKILL_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_item, ensure_ascii=False) + "\n")
