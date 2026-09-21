import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


APP_DIR = Path(__file__).resolve().parent
RESUME_DIR = APP_DIR / "resumes"
RESULT_DIR = APP_DIR / "parsed_results"
SKILL_ABOX_PATH = APP_DIR / "skill_abox.json"
TECHNICAL_SKILL_ABOX_PATH = APP_DIR / "technical_skill_abox.json"
COMPETENCY_ABOX_PATH = APP_DIR / "competency_abox.json"
CAPABILITY_ABOX_PATH = APP_DIR / "capability_abox.json"
DOMAIN_ABOX_PATH = APP_DIR / "domain_abox.json"
INFERENCE_RULE_ABOX_PATH = APP_DIR / "inference_rule_abox.json"
INDUSTRY_ABOX_PATH = APP_DIR / "industry_abox.json"
UNKNOWN_SKILL_LOG_PATH = RESULT_DIR / "out_of_ontology_skills.jsonl"
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def ensure_resume_dirs() -> None:
    RESUME_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)


def list_resume_files() -> list[Path]:
    files: list[Path] = []
    if not RESUME_DIR.exists():
        return files
    for path in RESUME_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(set(files), key=lambda p: p.name.lower())


def extract_text_from_pdf(path: Path) -> str:
    import pdfplumber

    chunks: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)
    return "\n\n".join(chunks)


def extract_text_from_docx(path: Path) -> str:
    from docx import Document

    document = Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def read_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"暂不支持该文件类型：{suffix}")


def load_abox_items(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到本体文件：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"{path.name} 中的 {key} 必须是数组。")
    return [item for item in values if isinstance(item, dict) and item.get("id") and item.get("name")]


def load_technical_skills() -> list[dict[str, Any]]:
    if TECHNICAL_SKILL_ABOX_PATH.exists():
        return load_abox_items(TECHNICAL_SKILL_ABOX_PATH, "skills")
    return load_skill_abox_legacy()


def load_competencies() -> list[dict[str, Any]]:
    return load_abox_items(COMPETENCY_ABOX_PATH, "competencies")


def load_capabilities() -> list[dict[str, Any]]:
    return load_abox_items(CAPABILITY_ABOX_PATH, "capabilities")


def load_domains() -> list[dict[str, Any]]:
    return load_abox_items(DOMAIN_ABOX_PATH, "domains")


def load_inference_rules() -> list[dict[str, Any]]:
    if not INFERENCE_RULE_ABOX_PATH.exists():
        return []
    return load_abox_items(INFERENCE_RULE_ABOX_PATH, "inference_rules")


def load_industries() -> list[dict[str, Any]]:
    if not INDUSTRY_ABOX_PATH.exists():
        return []
    return load_abox_items(INDUSTRY_ABOX_PATH, "industries")


def load_industry_policy() -> dict[str, Any]:
    if not INDUSTRY_ABOX_PATH.exists():
        return {}
    data = json.loads(INDUSTRY_ABOX_PATH.read_text(encoding="utf-8"))
    policy = data.get("meta", {}).get("extraction_policy", {})
    return policy if isinstance(policy, dict) else {}


def load_skill_abox_legacy() -> list[dict[str, Any]]:
    if not SKILL_ABOX_PATH.exists():
        raise FileNotFoundError(f"未找到技能本体文件：{SKILL_ABOX_PATH}")
    data = json.loads(SKILL_ABOX_PATH.read_text(encoding="utf-8"))
    skills = data.get("skills", [])
    if not isinstance(skills, list):
        raise ValueError("skill_abox.json 中的 skills 必须是数组。")
    return [item for item in skills if isinstance(item, dict) and item.get("id") and item.get("name")]


def load_skill_abox() -> list[dict[str, Any]]:
    """兼容 JD 解析器的旧入口；新架构下返回技术技能 ABox。"""
    return load_technical_skills()


def load_resume_ontology_catalog() -> dict[str, Any]:
    technical_skills = load_technical_skills()
    competencies = load_competencies()
    capabilities = load_capabilities()
    domains = load_domains()
    inference_rules = load_inference_rules()
    industries = load_industries()
    industry_policy = load_industry_policy()
    return {
        "technical_skills": technical_skills,
        "competencies": competencies,
        "capabilities": capabilities,
        "domains": domains,
        "inference_rules": inference_rules,
        "industries": industries,
        "industry_policy": industry_policy,
        "technical_skill_lookup": build_entity_lookup(technical_skills),
        "competency_lookup": build_entity_lookup(competencies),
        "capability_lookup": {str(item.get("id", "")): item for item in capabilities},
        "domain_lookup": {str(item.get("id", "")): item for item in domains},
        "industry_lookup": build_entity_lookup(industries),
        "inference_rule_lookup": {str(item.get("id", "")): item for item in inference_rules},
    }


def build_entity_lookup(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        entity_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        if entity_id:
            lookup[entity_id.lower()] = item
            lookup[normalize_match_key(entity_id)] = item
        if name:
            lookup[name.lower()] = item
            lookup[normalize_match_key(name)] = item
        for alias in item.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if alias_text:
                lookup[alias_text.lower()] = item
                lookup[normalize_match_key(alias_text)] = item
    return lookup


def build_skill_lookup(skills: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return build_entity_lookup(skills)


def normalize_match_key(value: str) -> str:
    return re.sub(r"[\s_\-\.]+", "", value.lower())


def build_normalized_skill_lookup(skills: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for skill in skills:
        for term in entity_terms(skill, include_id=True):
            key = normalize_match_key(term)
            if key:
                lookup[key] = skill
    return lookup


def entity_terms(entity: dict[str, Any], include_id: bool = False) -> list[str]:
    terms = [str(entity.get("name", "") or "")]
    if include_id:
        terms.append(str(entity.get("id", "") or ""))
    terms.extend(str(alias or "") for alias in entity.get("aliases", []) or [])

    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        value = term.strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def list_value(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_entity_result(
    entity: dict[str, Any],
    id_key: str,
    source: str,
    source_type: str,
    evidence: str = "",
    matched_term: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        id_key: str(entity.get("id", "") or ""),
        "name": str(entity.get("name", "") or matched_term),
        "type": str(entity.get("type", "") or ""),
        "evidence": evidence,
        "source": source,
        "source_type": source_type,
    }

    aliases = list_value(entity.get("aliases", []))
    languages = list_value(entity.get("languages", []))
    belongs_to = list_value(entity.get("belongs_to", []))
    if aliases:
        result["aliases"] = aliases
    if languages:
        result["languages"] = languages
    if belongs_to:
        result["belongs_to"] = belongs_to
    if matched_term:
        result["matched_term"] = matched_term
    return result


def build_industry_result(
    industry: dict[str, Any],
    source: str,
    evidence: str = "",
    matched_term: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "industry_id": str(industry.get("id", "") or ""),
        "name": str(industry.get("name", "") or matched_term),
        "evidence": evidence,
        "source": source,
    }
    if matched_term:
        result["matched_term"] = matched_term
    return result


def skill_terms(skill: dict[str, Any]) -> list[str]:
    return entity_terms(skill)


def is_ascii_token(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9+#][A-Za-z0-9+#.\-_\s]*", value))


def find_term_match(text: str, term: str) -> re.Match[str] | None:
    if not term.strip():
        return None

    if is_ascii_token(term):
        compact = re.sub(r"[^A-Za-z0-9+#]+", "", term)
        if len(compact) <= 2:
            return None
        pattern = rf"(?<![A-Za-z0-9+#]){re.escape(term)}(?![A-Za-z0-9+#])"
        return re.search(pattern, text, flags=re.IGNORECASE)

    return re.search(re.escape(term), text, flags=re.IGNORECASE)


def term_matches_text(text: str, term: str) -> bool:
    if find_term_match(text, term):
        return True

    normalized_term = normalize_match_key(term)
    if len(normalized_term) <= 2:
        return False
    if is_ascii_token(term) and not re.search(r"[\s_\-\.]+", term.strip()):
        return False
    return normalized_term in normalize_match_key(text)


def split_evidence_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[。！？.!?])\s+|[\r\n]+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def find_evidence(text: str, term: str) -> str:
    normalized_term = normalize_match_key(term)
    for sentence in split_evidence_sentences(text):
        if find_term_match(sentence, term) or term_matches_text(sentence, term):
            return sentence[:240]
    return f"简历文本中命中：{term}"


def resolve_entity(entity_id: str, name: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    lookup = build_entity_lookup(items)
    if entity_id:
        entity = lookup.get(entity_id.lower()) or lookup.get(normalize_match_key(entity_id))
        if entity:
            return entity
    if name:
        return lookup.get(name.lower()) or lookup.get(normalize_match_key(name))
    return None


def resolve_skill(skill_id: str, name: str, skills: list[dict[str, Any]]) -> dict[str, Any] | None:
    return resolve_entity(skill_id, name, skills)


def extract_entities_from_text(
    text: str,
    items: list[dict[str, Any]],
    id_key: str,
    source_type: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        entity_id = str(item.get("id", "") or "").strip()
        if not entity_id or entity_id in seen:
            continue

        for term in entity_terms(item):
            if not term_matches_text(text, term):
                continue
            seen.add(entity_id)
            result.append(
                build_entity_result(
                    item,
                    id_key,
                    "rule",
                    source_type,
                    evidence=find_evidence(text, term),
                    matched_term=term,
                )
            )
            break
    return result


def extract_ontology_skills_from_text(resume_text: str, skills: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "skill_id": str(item.get("skill_id", "")),
            "name": str(item.get("name", "")),
            "evidence": str(item.get("evidence", "")),
        }
        for item in extract_entities_from_text(resume_text, skills, "skill_id", "technical_skill")
    ]


def build_catalog_prompt(items: list[dict[str, Any]], id_key: str) -> str:
    catalog = []
    for item in items:
        catalog.append(
            {
                id_key: item.get("id", ""),
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "description": item.get("description", ""),
                "aliases": item.get("aliases", []),
                "related_capabilities": item.get("related_capabilities", []),
            }
        )
    return json.dumps(catalog, ensure_ascii=False, indent=2)


def build_skill_catalog_prompt(skills: list[dict[str, Any]]) -> str:
    return build_catalog_prompt(skills, "skill_id")


def build_resume_prompt(resume_text: str, jd_description: str = "") -> str:
    catalog = load_resume_ontology_catalog()
    competency_catalog = build_catalog_prompt(catalog["competencies"], "competency_id")
    industry_catalog = build_catalog_prompt(catalog["industries"], "industry_id")
    jd_context = jd_description.strip()
    return f"""
你是一个人才管理简历解析 Agent。请从简历文本中抽取结构化 JSON。

必须输出合法 JSON，不要输出 Markdown，不要输出解释。

JSON 字段必须为：
{{
  "姓名": "候选人姓名，无法识别则为空字符串",
  "胜任力": [
    {{
      "competency_id": "必须来自胜任力本体的 id，如 SKILL_REQUIREMENTS_ANALYSIS",
      "name": "必须来自胜任力本体的 name，如 需求分析",
      "evidence": "简历中能证明该胜任力的具体原文证据"
    }}
  ],
  "本体外技术技能": [
    {{
      "skill_id": "",
      "name": "简历中明确出现但 technical_skill_abox 中不存在的技术、工具、框架、平台",
      "evidence": "简历中对应原文证据或简短依据"
    }}
  ],
  "级别": "候选人级别，无法识别则为空字符串",
  "经验": [
    {{
      "公司": "公司或组织名称，无法识别则为空字符串",
      "职位": "职位名称，无法识别则为空字符串",
      "时间": "经历时间，无法识别则为空字符串",
      "描述": "该段经历的核心职责、项目或成果"
    }}
  ],
  "行业知识": [
    {{
      "industry_id": "必须来自行业本体的 id",
      "name": "必须来自行业本体的 name",
      "evidence": "简历中能证明该行业经验的具体原文证据"
    }}
  ],
  "profile_type": {{
    "primary": "TECHNICAL、BA、CONSULTING、PRODUCT、PROJECT_MANAGER、HYBRID 或 UNKNOWN",
    "secondary": ["可选的次要画像类型"],
    "confidence": 0.0
  }}
}}

要求：
1. 技术技能由程序使用 technical_skill_abox 的 name 和 aliases 规则匹配，你不要输出“技术技能”字段。
2. “胜任力”只能从下方胜任力本体中选择，并且必须有具体经历证据；不要只因为简历写了“沟通能力强”就认定胜任力。
3. 如果技术名词无法明确映射到 technical_skill_abox，输出到“本体外技术技能”；不要把普通业务词、代码类名、函数名、配置项当成技术技能。
4. 不要输出“能力”字段；能力由程序根据 技术技能/胜任力 的 related_capabilities 确定性推导。
5. “行业知识”只能从下方行业本体中选择，必须有简历原文证据；不能仅凭技术工具、平台或公司名称推断行业。
6. JD 描述仅作为辅助语境；不能编造简历中不存在的信息。
7. 不确定的信息留空或返回空数组。
8. 只返回 JSON 对象。

胜任力本体：
{competency_catalog}

行业本体：
{industry_catalog}

JD描述：
{jd_context}

简历文本：
{resume_text}
""".strip()


def call_resume_llm(resume_text: str, api_key: str, base_url: str, model: str, jd_description: str = "") -> str:
    if not api_key:
        raise ValueError("请先在 .env 文件中配置 OPENAI_API_KEY。")

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    messages = [
        {"role": "system", "content": "你是一个只输出 JSON 的简历解析 Agent。"},
        {"role": "user", "content": build_resume_prompt(resume_text, jd_description)},
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


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def split_items(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_items = re.split(r"[,，、;/；\n]+", values)
    elif isinstance(values, list):
        raw_items = []
        for item in values:
            if isinstance(item, str):
                raw_items.extend(re.split(r"[,，、;/；\n]+", item))
            elif item:
                raw_items.append(str(item))
    else:
        raw_items = [str(values)]

    seen: set[str] = set()
    result: list[str] = []
    for item in raw_items:
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_entity_items(
    values: Any,
    items: list[dict[str, Any]],
    id_key: str,
    source_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_items = values if isinstance(values, list) else ([] if values in (None, "") else [values])
    result: list[dict[str, Any]] = []
    unknown: list[dict[str, str]] = []
    seen: set[str] = set()
    unknown_seen: set[str] = set()

    for item in raw_items:
        entity_id = ""
        name = ""
        evidence = ""
        if isinstance(item, dict):
            entity_id = str(item.get(id_key, "") or item.get("skill_id", "") or item.get("id", "") or "").strip()
            name = str(item.get("name", "") or item.get("技能", "") or item.get("胜任力", "") or "").strip()
            evidence = str(item.get("evidence", "") or item.get("依据", "") or item.get("证据", "") or "").strip()
        elif isinstance(item, str):
            name = item.strip()
        elif item:
            name = str(item).strip()

        entity = resolve_entity(entity_id, name, items)
        if not entity:
            key = (entity_id or name).lower()
            if key and key not in unknown_seen:
                unknown_seen.add(key)
                unknown.append({id_key: entity_id, "name": name, "evidence": evidence})
            continue

        resolved_id = str(entity.get("id", "")).strip()
        if not resolved_id or resolved_id in seen:
            continue
        seen.add(resolved_id)
        result.append(build_entity_result(entity, id_key, "llm", source_type, evidence=evidence, matched_term=name))
    return result, unknown


def normalize_industry_items(
    values: Any,
    industries: list[dict[str, Any]],
    resume_text: str = "",
) -> list[dict[str, Any]]:
    raw_items = values if isinstance(values, list) else ([] if values in (None, "") else [values])
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_industry(industry: dict[str, Any], source: str, evidence: str = "", matched_term: str = "") -> None:
        industry_id = str(industry.get("id", "") or "").strip()
        if not industry_id or industry_id in seen:
            return
        seen.add(industry_id)
        result.append(build_industry_result(industry, source, evidence=evidence, matched_term=matched_term))

    for industry in industries:
        for term in entity_terms(industry):
            if resume_text and term_matches_text(resume_text, term):
                add_industry(industry, "rule", evidence=find_evidence(resume_text, term), matched_term=term)
                break

    for item in raw_items:
        industry_id = ""
        name = ""
        evidence = ""
        if isinstance(item, dict):
            industry_id = str(item.get("industry_id", "") or item.get("id", "") or "").strip()
            name = str(item.get("name", "") or item.get("行业", "") or item.get("行业知识", "") or "").strip()
            evidence = str(item.get("evidence", "") or item.get("依据", "") or item.get("证据", "") or "").strip()
        elif isinstance(item, str):
            name = item.strip()
        elif item:
            name = str(item).strip()

        industry = resolve_entity(industry_id, name, industries)
        if not industry:
            continue
        add_industry(industry, "llm", evidence=evidence, matched_term=name)

    return result


def normalize_skill_items(values: Any, skills: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    known, unknown = normalize_entity_items(values, skills, "skill_id", "technical_skill")
    return [
        {
            "skill_id": str(item.get("skill_id", "")),
            "name": str(item.get("name", "")),
            "evidence": str(item.get("evidence", "")),
        }
        for item in known
    ], unknown


def normalize_unknown_skill_items(values: Any, skills: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw_items = values if isinstance(values, list) else ([] if values in (None, "") else [values])
    known: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    known_seen: set[str] = set()
    unknown_seen: set[str] = set()

    for item in raw_items:
        skill_id = ""
        name = ""
        evidence = ""
        if isinstance(item, dict):
            skill_id = str(item.get("skill_id", "") or item.get("id", "") or "").strip()
            name = str(item.get("name", "") or item.get("技能", "") or "").strip()
            evidence = str(item.get("evidence", "") or item.get("依据", "") or item.get("证据", "") or "").strip()
        elif isinstance(item, str):
            name = item.strip()
        elif item:
            name = str(item).strip()

        skill = resolve_skill(skill_id, name, skills)
        if skill:
            resolved_id = str(skill.get("id", "") or "").strip()
            if resolved_id and resolved_id not in known_seen:
                known_seen.add(resolved_id)
                known.append(
                    build_entity_result(
                        skill,
                        "skill_id",
                        "llm",
                        "technical_skill",
                        evidence=evidence,
                        matched_term=name,
                    )
                )
            continue

        key = (skill_id or name).lower()
        if not key or key in unknown_seen:
            continue
        unknown_seen.add(key)
        unknown.append({"skill_id": skill_id, "name": name, "evidence": evidence})
    return known, unknown


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


def merge_skill_items(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_entity_items("skill_id", *groups)


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


def normalize_experience(values: Any) -> list[dict[str, str]]:
    if isinstance(values, str):
        values = [{"公司": "", "职位": "", "时间": "", "描述": values}]
    elif not isinstance(values, list):
        values = []

    result: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            result.append(
                {
                    "公司": str(item.get("公司", "") or ""),
                    "职位": str(item.get("职位", "") or ""),
                    "时间": str(item.get("时间", "") or ""),
                    "描述": str(item.get("描述", "") or item.get("职责", "") or ""),
                }
            )
        elif item:
            result.append({"公司": "", "职位": "", "时间": "", "描述": str(item)})
    return result


def derive_capabilities(
    technical_skills: list[dict[str, Any]],
    competencies: list[dict[str, Any]],
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    skill_lookup = {str(item.get("id", "")): item for item in catalog["technical_skills"]}
    competency_lookup = {str(item.get("id", "")): item for item in catalog["competencies"]}
    capability_lookup = catalog["capability_lookup"]
    inference_rule_lookup = catalog.get("inference_rule_lookup", {})

    result_by_id: dict[str, dict[str, Any]] = {}

    def add_capability(
        capability_id: str,
        source: dict[str, Any],
        source_id_key: str,
        source_type: str,
        inference_rule: dict[str, Any] | None = None,
    ) -> None:
        capability = capability_lookup.get(capability_id)
        if not capability:
            return

        entry = result_by_id.setdefault(
            capability_id,
            {
                "capability_id": capability_id,
                "name": str(capability.get("name", "") or capability_id),
                "domain_id": str(capability.get("domain_id", "") or ""),
                "evidence": [],
                "sources": [],
                "priority": "",
            },
        )
        source_id = str(source.get(source_id_key, "") or "")
        source_name = str(source.get("name", "") or source_id)
        source_evidence = str(source.get("evidence", "") or "")
        source_priority = str(source.get("priority", "") or "must")
        if source_priority == "must":
            entry["priority"] = "must"
        elif not entry["priority"]:
            entry["priority"] = source_priority
        source_item = {"source_type": source_type, "source_id": source_id, "name": source_name}
        if inference_rule:
            source_item["rule_id"] = str(inference_rule.get("id", "") or "")
            source_item["rule_name"] = str(inference_rule.get("name", "") or "")
            source_item["inference_reason"] = str(inference_rule.get("reason", "") or "")
            if inference_rule.get("inference_weight") is not None:
                source_item["inference_weight"] = inference_rule.get("inference_weight")
        if source_item not in entry["sources"]:
            entry["sources"].append(source_item)
        if source_evidence and source_evidence not in entry["evidence"]:
            entry["evidence"].append(source_evidence)

    for skill_item in technical_skills:
        skill = skill_lookup.get(str(skill_item.get("skill_id", "") or ""))
        if not skill:
            continue
        for capability_id in skill.get("related_capabilities", []) or []:
            add_capability(str(capability_id), skill_item, "skill_id", "technical_skill")
        for rule_id in skill.get("inference_rule_ids", []) or []:
            rule = inference_rule_lookup.get(str(rule_id))
            if not rule:
                continue
            if str(rule.get("source_skill_id", "") or "") != str(skill.get("id", "") or ""):
                continue
            capability_id = str(rule.get("target_capability_id", "") or "")
            if capability_id:
                add_capability(capability_id, skill_item, "skill_id", "technical_skill", rule)

    for competency_item in competencies:
        competency = competency_lookup.get(str(competency_item.get("competency_id", "") or ""))
        if not competency:
            continue
        for capability_id in competency.get("related_capabilities", []) or []:
            add_capability(str(capability_id), competency_item, "competency_id", "competency")

    return sorted(result_by_id.values(), key=lambda item: item["capability_id"])


def derive_domains(capabilities: list[dict[str, Any]], catalog: dict[str, Any], max_domains: int = 3) -> list[dict[str, Any]]:
    domain_lookup = catalog["domain_lookup"]
    grouped: dict[str, set[str]] = {}
    for capability in capabilities:
        domain_id = str(capability.get("domain_id", "") or "")
        capability_id = str(capability.get("capability_id", "") or "")
        if not domain_id:
            continue
        domain = domain_lookup.get(domain_id)
        if not domain:
            continue
        grouped.setdefault(domain_id, set())
        if capability_id:
            grouped[domain_id].add(capability_id)

    ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    result: list[dict[str, Any]] = []
    for domain_id, capability_ids in ranked[:max_domains]:
        domain = domain_lookup[domain_id]
        result.append(
            {
                "domain_id": domain_id,
                "name": str(domain.get("name", "") or domain_id),
                "capability_count": len(capability_ids),
            }
        )
    return result


def build_evidence_items(
    technical_skills: list[dict[str, Any]],
    competencies: list[dict[str, Any]],
    capabilities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in technical_skills:
        key = ("technical_skill", str(item.get("skill_id", "")), str(item.get("evidence", "")))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "evidence_type": "technical_skill",
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
        key = ("competency", str(item.get("competency_id", "")), str(item.get("evidence", "")))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "evidence_type": "competency",
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


def infer_profile_type(data: dict[str, Any], technical_skills: list[dict[str, Any]], competencies: list[dict[str, Any]]) -> dict[str, Any]:
    raw_profile = data.get("profile_type", {})
    if isinstance(raw_profile, dict):
        primary = str(raw_profile.get("primary", "") or "").upper()
        secondary = raw_profile.get("secondary", [])
        confidence = raw_profile.get("confidence", 0)
    else:
        primary = str(raw_profile or "").upper()
        secondary = []
        confidence = 0

    technical_count = len(technical_skills)
    competency_count = len(competencies)
    if not primary or primary == "UNKNOWN":
        if technical_count >= 8 and competency_count >= 4:
            primary = "HYBRID"
        elif technical_count >= max(3, competency_count):
            primary = "TECHNICAL"
        elif competency_count >= 3:
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
            "technical_skill_count": technical_count,
            "competency_count": competency_count,
        },
    }


def normalize_resume_result(data: dict[str, Any], resume_text: str = "") -> dict[str, Any]:
    catalog = load_resume_ontology_catalog()
    technical_skills_catalog = catalog["technical_skills"]
    competency_catalog = catalog["competencies"]
    industry_catalog = catalog["industries"]
    industry_policy = catalog.get("industry_policy", {})
    max_industries = int(industry_policy.get("resume_max_industries", 3) or 3)

    rule_technical_skills = (
        extract_entities_from_text(resume_text, technical_skills_catalog, "skill_id", "technical_skill")
        if resume_text
        else []
    )
    rule_competencies = (
        extract_entities_from_text(resume_text, competency_catalog, "competency_id", "competency")
        if resume_text
        else []
    )
    llm_competencies, _ = normalize_entity_items(
        data.get("胜任力", data.get("competencies", [])),
        competency_catalog,
        "competency_id",
        "competency",
    )
    known_from_unknown, explicit_unknown_skills = normalize_unknown_skill_items(
        data.get("本体外技术技能", []),
        technical_skills_catalog,
    )

    technical_skills = merge_skill_items(rule_technical_skills, known_from_unknown)
    competencies = merge_entity_items("competency_id", rule_competencies, llm_competencies)
    unknown_skills = merge_unknown_skill_items(explicit_unknown_skills)
    capabilities = derive_capabilities(technical_skills, competencies, catalog)
    domains = derive_domains(capabilities, catalog)
    experience = normalize_experience(data.get("经验", []))
    evidence = build_evidence_items(technical_skills, competencies, capabilities)
    industries = normalize_industry_items(
        data.get("行业知识", data.get("知识", [])),
        industry_catalog,
        resume_text=resume_text,
    )[:max_industries]

    return {
        "姓名": str(data.get("姓名", "") or data.get("name", "") or ""),
        "级别": str(data.get("级别", "") or data.get("level", "") or ""),
        "profile_type": infer_profile_type(data, technical_skills, competencies),
        "技术技能": technical_skills,
        "胜任力": competencies,
        "能力": capabilities,
        "领域": domains,
        "证据": evidence,
        "本体外技术技能": unknown_skills,
        "经验": experience,
        "行业知识": industries,
    }


def parse_resume_to_json(resume_text: str, api_key: str, base_url: str, model: str, jd_description: str = "") -> dict[str, Any]:
    raw_result = call_resume_llm(resume_text, api_key, base_url, model, jd_description)
    parsed = extract_json_object(raw_result)
    return normalize_resume_result(parsed, resume_text)


def save_resume_result(source_file: Path, result: dict[str, Any]) -> Path:
    RESULT_DIR.mkdir(exist_ok=True)
    result_path = RESULT_DIR / f"{source_file.stem}_parsed.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_unknown_skills_log(source_file, result)
    return result_path


def append_unknown_skills_log(source_file: Path, result: dict[str, Any]) -> None:
    unknown_skills = result.get("本体外技术技能", [])
    RESULT_DIR.mkdir(exist_ok=True)

    existing_items: list[dict[str, Any]] = []
    if UNKNOWN_SKILL_LOG_PATH.exists():
        for line in UNKNOWN_SKILL_LOG_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("source_file") != source_file.name:
                existing_items.append(item)

    if unknown_skills:
        existing_items.append(
            {
                "source_file": source_file.name,
                "unknown_skills": unknown_skills,
            }
        )

    if existing_items:
        with UNKNOWN_SKILL_LOG_PATH.open("w", encoding="utf-8") as file:
            for item in existing_items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif UNKNOWN_SKILL_LOG_PATH.exists():
        UNKNOWN_SKILL_LOG_PATH.unlink()
