import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from jd_parser import ensure_jd_dirs, parse_jd_to_json, save_jd_result
from match_candidates import (
    find_resume_json_files,
    load_skill_catalog,
    match_json_files,
    skill_name,
)
from resume_parser import (
    RESUME_DIR,
    ensure_resume_dirs,
    list_resume_files,
    parse_resume_to_json,
    read_resume_text,
    save_resume_result,
)

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
PARSED_RESULTS_DIR = APP_DIR / "parsed_results"
BATCH_PARSE_SUMMARY_PATH = PARSED_RESULTS_DIR / "batch_parse_summary.json"
BATCH_PARSE_ERROR_LOG_PATH = PARSED_RESULTS_DIR / "batch_parse_errors.jsonl"
UNKNOWN_RESUME_SKILLS_PATH = PARSED_RESULTS_DIR / "out_of_ontology_skills.jsonl"
UNKNOWN_JD_SKILLS_PATH = PARSED_RESULTS_DIR / "out_of_ontology_jd_skills.jsonl"
EMPLOYEE_RESUME_MAP_PATH = APP_DIR / "employee_resume_map.json"

DEMO_USERS = {
    "hr": {"password": "hr123", "role": "hr", "display_name": "HR"},
    "employee": {"password": "emp123", "role": "employee", "display_name": "员工"},
    "admin": {"password": "admin123", "role": "admin", "display_name": "Admin"},
}

ROLE_PAGE_MAP = {
    "hr": ["人才洞察", "人才发现", "岗位中心", "人才评估", "技能图谱", "智能解析", "技能发现"],
    "employee": ["职业助手", "岗位推荐", "技能提升", "我的画像", "技能图谱"],
    "admin": ["人才洞察", "人才发现", "岗位中心", "职业助手", "人才评估", "技能图谱", "智能解析", "技能发现", "简历权限管理"],
}

def normalize_prefix_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result

def load_employee_resume_map() -> dict[str, list[str]]:
    # Priority 1: local config file (for admin UI).
    if EMPLOYEE_RESUME_MAP_PATH.exists():
        try:
            data = json.loads(EMPLOYEE_RESUME_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                normalized: dict[str, list[str]] = {}
                for key, value in data.items():
                    user = str(key or "").strip().lower()
                    if not user:
                        continue
                    normalized[user] = normalize_prefix_list(value)
                return normalized
        except Exception:
            pass

    # Priority 2: environment variable fallback.
    raw_map = os.getenv("EMPLOYEE_RESUME_MAP", "").strip()
    if raw_map:
        try:
            data = json.loads(raw_map)
            if isinstance(data, dict):
                normalized = {}
                for key, value in data.items():
                    user = str(key or "").strip().lower()
                    if not user:
                        continue
                    normalized[user] = normalize_prefix_list(value)
                return normalized
        except Exception:
            pass

    return {}

def save_employee_resume_map(mapping: dict[str, list[str]]) -> None:
    normalized: dict[str, list[str]] = {}
    for key, value in mapping.items():
        user = str(key or "").strip().lower()
        if not user:
            continue
        prefixes = normalize_prefix_list(value)
        if prefixes:
            normalized[user] = prefixes

    EMPLOYEE_RESUME_MAP_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

def get_resume_prefix(path: Path) -> str:
    return path.stem.split("_", 1)[0].strip().lower()

def get_role_scoped_resume_files(all_files: list[Path]) -> list[Path]:
    """Limit resume visibility by role to avoid employee overexposure.

    - admin / hr: see all resumes
    - employee: only see resumes explicitly assigned in employee_resume_map.json,
      plus any self-uploaded files under resumes/employees/<user>/.
    - other roles: no access
    """
    role = str(st.session_state.get("auth_role", "hr") or "hr").strip().lower()

    if role in {"admin", "hr"}:
        return all_files

    if role != "employee":
        return []

    # ── Employee: strict filter by employee_resume_map.json ──
    mapping = load_employee_resume_map()
    allowed: list[str] = [e.strip().lower() for e in mapping.get("employee", [])]

    # Also scan employee's private upload directory
    employee_upload_dir = APP_DIR / "resumes" / "employees" / "employee"
    uploaded: list[Path] = []
    if employee_upload_dir.exists():
        for p in employee_upload_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
                uploaded.append(p)

    if not allowed and not uploaded:
        return []

    filtered: list[Path] = []
    for path in all_files:
        stem_lower = path.stem.lower()
        prefix_lower = get_resume_prefix(path)  # already lowercased via .lower()

        # Match if the file's stem (e.g. "Candidate_0042") or
        # prefix (e.g. "candidate") is in the allowed list
        if stem_lower in allowed or prefix_lower in allowed:
            filtered.append(path)

    # Deduplicate by file name (uploaded may overlap with filtered)
    seen: dict[str, Path] = {p.name: p for p in uploaded}
    for p in filtered:
        seen[p.name] = p

    return sorted(seen.values(), key=lambda p: p.name.lower())

def render_employee_resume_map_admin() -> None:
    st.markdown(
        """
        <div class='app-title'>
            <div>
                <div class='title-left'>简历权限管理</div>
                <div class='title-desc'>Admin 可图形化配置员工账号可访问的简历范围</div>
                <div class='title-meta'>
                    <span class='title-chip'>Role Based Access</span>
                    <span class='title-chip'>Privacy Control</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_resumes = list_resume_files()
    prefixes = sorted({get_resume_prefix(path) for path in all_resumes if get_resume_prefix(path)})
    mapping = load_employee_resume_map()
    employee_users = sorted(
        {
            user
            for user, info in DEMO_USERS.items()
            if str(info.get("role") or "").strip().lower() == "employee"
        }
        | set(mapping.keys())
    )

    st.caption(f"当前可选简历前缀数量：{len(prefixes)}；配置文件：{EMPLOYEE_RESUME_MAP_PATH.name}")

    pending_map: dict[str, list[str]] = {}
    for user in employee_users:
        current = mapping.get(user, [])
        with st.expander(f"员工账号：{user}", expanded=False):
            selected = st.multiselect(
                f"为 {user} 选择可访问简历前缀",
                options=prefixes,
                default=[item for item in current if item in prefixes],
                key=f"perm_prefix_{user}",
            )
            extra = st.text_input(
                "补充前缀（逗号分隔，可填未在列表中的前缀）",
                value=",".join([item for item in current if item not in prefixes]),
                key=f"perm_extra_{user}",
            )
            extra_values = [item.strip().lower() for item in extra.split(",") if item.strip()]
            pending_map[user] = normalize_prefix_list(selected + extra_values)

    add_col, save_col, reload_col = st.columns([1.2, 1, 1])
    with add_col:
        st.markdown("#### 新增员工账号映射")
        new_user = st.text_input("新员工账号", value="", placeholder="例如：employee2", key="new_employee_user")
        if st.button("新增账号到配置面板", use_container_width=True):
            user_value = new_user.strip().lower()
            if not user_value:
                st.warning("请输入账号名。")
            else:
                st.session_state[f"perm_prefix_{user_value}"] = []
                st.session_state[f"perm_extra_{user_value}"] = ""
                mapping.setdefault(user_value, [])
                save_employee_resume_map(mapping)
                st.success(f"已新增账号：{user_value}，请继续配置可访问前缀并保存。")
                st.rerun()

    with save_col:
        if st.button("保存全部权限配置", use_container_width=True):
            save_employee_resume_map(pending_map)
            st.success(f"已保存到 {EMPLOYEE_RESUME_MAP_PATH.name}")
            st.rerun()

    with reload_col:
        if st.button("重新加载配置", use_container_width=True):
            st.rerun()

    st.markdown("#### 当前配置预览")
    preview_rows = []
    latest = load_employee_resume_map()
    for user, items in sorted(latest.items()):
        preview_rows.append({"user": user, "resume_prefix_count": len(items), "resume_prefixes": ", ".join(items)})
    if preview_rows:
        st.dataframe(preview_rows, use_container_width=True, hide_index=True)
    else:
        st.info("当前尚未配置员工简历访问映射。")

def ensure_dirs() -> None:
    ensure_resume_dirs()
    ensure_jd_dirs()

def get_secret_value(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default

ROLE_DISPLAY_NAMES = {
    "hr": "HR / 业务经理",
    "employee": "员工 / Learning团队",
    "admin": "Admin",
}

def render_role_selector() -> None:
    """Render a role selector dropdown in the top-right corner."""
    current_role = st.session_state.get("auth_role", "hr")

    # Add custom CSS for the role selector
    st.markdown(
        """
        <style>
            .role-selector-container {
                display: flex;
                justify-content: flex-end;
                align-items: center;
                gap: 12px;
                padding: 8px 0;
            }
            .role-label {
                font-size: 13px;
                color: #536d89;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 1])
    with col_right:
        role_options = list(ROLE_DISPLAY_NAMES.keys())
        current_index = role_options.index(current_role) if current_role in role_options else 0

        selected_role = st.selectbox(
            "切换角色",
            options=role_options,
            index=current_index,
            format_func=lambda r: ROLE_DISPLAY_NAMES.get(r, r),
            key="role_selector",
        )

        if selected_role != current_role:
            st.session_state["auth_role"] = selected_role
            st.rerun()

def run_batch_resume_parser(
    input_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    jd_context: str,
    workers: int,
    recursive: bool,
    force: bool,
    limit: int,
) -> subprocess.CompletedProcess[str]:
    script_path = APP_DIR / "batch_parse_resumes.py"
    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_path),
        "--mode",
        "llm",
        "--workers",
        str(max(1, workers)),
    ]
    if recursive:
        command.append("--recursive")
    if force:
        command.append("--force")
    if limit > 0:
        command.extend(["--limit", str(limit)])
    if jd_context.strip():
        command.extend(["--jd-text", jd_context])
    if api_key:
        command.extend(["--api-key", api_key])
    if base_url:
        command.extend(["--base-url", base_url])
    if model:
        command.extend(["--model", model])

    return subprocess.run(
        command,
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

def format_item_list(items: Any) -> str:
    if not isinstance(items, list):
        return str(items or "")
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "") or item.get("candidate_name", "") or item.get("id", "") or "").strip()
            if item.get("candidate_name") and item.get("name") and item.get("candidate_name") != item.get("name"):
                name = f"{item.get('name')} -> {item.get('candidate_name')}"
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return " / ".join(names)

def format_inferred_sources(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        skill_name = str(item.get("skill_name", "") or item.get("skill_id", "") or "").strip()
        evidence = str(item.get("evidence", "") or "").strip()
        if skill_name and evidence:
            parts.append(f"{skill_name}：{evidence}")
        elif skill_name:
            parts.append(skill_name)
    return "；".join(parts)

def format_matched_items_with_evidence(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or item.get("candidate_name", "") or item.get("id", "") or "").strip()
        evidence = str(item.get("candidate_evidence", "") or item.get("evidence", "") or "").strip()
        if name and evidence:
            parts.append(f"{name}：{evidence}")
        elif name:
            parts.append(name)
    return "；".join(parts)

def format_match_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            formatted_rows.append({"内容": str(row)})
            continue
        formatted = dict(row)
        if "required_items" in formatted:
            formatted["要求项"] = format_item_list(formatted.pop("required_items"))
        if "matched_items" in formatted:
            matched_items = formatted.pop("matched_items")
            formatted["命中项"] = format_item_list(matched_items)
            matched_evidence = format_matched_items_with_evidence(matched_items)
            if matched_evidence:
                formatted["命中证据"] = matched_evidence
        if "inferred_from" in formatted:
            formatted["推导来源"] = format_inferred_sources(formatted.pop("inferred_from"))
        rename_map = {
            "id": "ID",
            "name": "要求名称",
            "weight": "权重",
            "match_type": "命中方式",
            "candidate_name": "候选人命中项",
            "match_score": "命中分",
            "jd_evidence": "JD证据",
            "candidate_evidence": "候选人证据",
            "evidence": "JD证据",
            "group_id": "组ID",
            "group_name": "要求组",
            "logic": "逻辑",
        }
        formatted = {rename_map.get(key, key): value for key, value in formatted.items()}
        formatted_rows.append(formatted)
    return formatted_rows

def render_jd_result(result: dict[str, Any]) -> None:
    st.subheader("JD解析结果")
    with st.expander("查看原始 JSON", expanded=False):
        st.json(result, expanded=False)

    if result.get("是否有级别要求"):
        st.success(f"已识别到级别要求：{result.get('级别要求')}")
    else:
        st.warning("当前 JD 未识别到明确的级别/职级要求，请补充后再进行匹配分析。")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("技术技能要求", len(result.get("技术技能要求", [])))
    col2.metric("胜任力要求", len(result.get("胜任力要求", [])))
    col3.metric("能力要求", len(result.get("能力要求", [])))
    col4.metric("技术技能要求组", len(result.get("技术技能要求组", [])))

    if result.get("岗位类型"):
        st.markdown("#### 岗位类型")
        st.json(result.get("岗位类型"), expanded=False)

    if result.get("项目约束"):
        st.markdown("#### 项目约束")
        st.json(result.get("项目约束"), expanded=False)

    for title, key in [
        ("技术技能要求", "技术技能要求"),
        ("技术技能要求组", "技术技能要求组"),
        ("胜任力要求", "胜任力要求"),
        ("能力要求", "能力要求"),
        ("领域要求", "领域要求"),
    ]:
        st.markdown(f"#### {title}")
        values = result.get(key, [])
        if values:
            st.dataframe(values, use_container_width=True)
        else:
            st.caption(f"暂无{title}")

    jd_unknown_skills = result.get("本体外技术技能要求", [])
    st.markdown("#### JD 本体外技术技能")
    if jd_unknown_skills:
        st.warning("发现 JD 本体外技术技能，已记录到 parsed_results/out_of_ontology_jd_skills.jsonl。")
        st.dataframe(jd_unknown_skills, use_container_width=True)
    else:
        st.caption("暂无 JD 本体外技术技能")

def render_result(result: dict[str, Any]) -> None:
    st.subheader("解析结果")
    with st.expander("查看原始 JSON", expanded=False):
        st.json(result, expanded=False)
    industry_knowledge = result.get("行业知识", result.get("知识", []))
    technical_skills = result.get("技术技能", [])
    competencies = result.get("胜任力", [])
    capabilities = result.get("能力", [])
    domains = result.get("领域", [])
    skills = technical_skills
    skill_names = [
        str(item.get("name") or item.get("skill_id") or "")
        for item in skills
        if isinstance(item, dict)
    ]
    if not skill_names and isinstance(skills, list):
        skill_names = [str(item) for item in skills if item]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("姓名", result.get("姓名") or "未识别")
    col2.metric("级别", result.get("级别") or "未识别")
    col3.metric("技术技能", len(technical_skills))
    col4.metric("胜任力", len(competencies))
    col5.metric("能力", len(capabilities))
    col6.metric("领域", len(domains))

    profile_type = result.get("profile_type")
    if profile_type:
        st.markdown("#### 简历画像")
        st.json(profile_type, expanded=False)

    st.markdown("#### 技术技能")
    if technical_skills:
        if all(isinstance(item, dict) for item in technical_skills):
            st.dataframe(technical_skills, use_container_width=True)
        else:
            st.write("、".join(skill_names))
    else:
        st.caption("暂无技术技能解析结果")

    st.markdown("#### 胜任力")
    if competencies:
        st.dataframe(competencies, use_container_width=True)
    else:
        st.caption("暂无胜任力解析结果")

    st.markdown("#### 能力")
    if capabilities:
        st.dataframe(capabilities, use_container_width=True)
    else:
        st.caption("暂无能力推导结果")

    st.markdown("#### 领域")
    if domains:
        st.dataframe(domains, use_container_width=True)
    else:
        st.caption("暂无领域推导结果")

    unknown_skills = result.get("本体外技术技能", result.get("本体外技能", []))
    st.markdown("#### 本体外技术技能")
    if unknown_skills:
        st.warning("发现本体外技术技能，已记录到 parsed_results/out_of_ontology_skills.jsonl。")
        st.dataframe(unknown_skills, use_container_width=True)
    else:
        st.caption("暂无本体外技术技能")

    st.markdown("#### 级别")
    st.write(result.get("级别") or "未识别")

    st.markdown("#### 经验")
    if result.get("经验"):
        st.dataframe(result["经验"], use_container_width=True)
    else:
        st.caption("暂无经验解析结果")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 证据")
        evidence = result.get("证据", [])
        if evidence:
            st.dataframe(evidence, use_container_width=True)
        else:
            st.caption("暂无证据")
    with right:
        st.markdown("#### 行业知识")
        if industry_knowledge and all(isinstance(item, dict) for item in industry_knowledge):
            st.dataframe(industry_knowledge, use_container_width=True)
        elif industry_knowledge:
            st.write(industry_knowledge)
        else:
            st.caption("暂无行业知识解析结果")

def build_skill_group_rows(skill_ids: list[str], catalog: dict[str, dict]) -> list[dict[str, str]]:
    grouped: dict[str, list[str]] = {}
    for skill_id in skill_ids:
        skill = catalog.get(skill_id, {})
        skill_label = skill_name(skill_id, catalog)

        if str(skill.get("type", "")).upper() == "DOMAIN":
            grouped.setdefault(skill_label, []).append(skill_label)

        belongs_to = skill.get("belongs_to", [])
        if not isinstance(belongs_to, list):
            continue

        for parent_id in belongs_to:
            parent_key = str(parent_id or "").strip()
            parent_skill = catalog.get(parent_key, {})
            if str(parent_skill.get("type", "")).upper() != "DOMAIN":
                continue
            parent_label = skill_name(parent_key, catalog)
            grouped.setdefault(parent_label, []).append(skill_label)

    return [
        {
            "分组": group_name,
            "技能": "、".join(sorted(set(skill_names))),
        }
        for group_name, skill_names in sorted(grouped.items(), key=lambda item: item[0])
    ]

def list_jd_json_options() -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for path in sorted(PARSED_RESULTS_DIR.glob("jd_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        job_title = str(data.get("岗位名称") or path.stem)
        job_type_value = data.get("岗位类型", "UNKNOWN")
        if isinstance(job_type_value, dict):
            job_type = str(job_type_value.get("primary") or "UNKNOWN")
        else:
            job_type = str(job_type_value or "UNKNOWN")

        options.append(
            {
                "path": path,
                "label": f"{job_title} | {job_type} | {path.name}",
                "job_title": job_title,
                "job_type": job_type,
            }
        )
    return options

def score_color(score: float) -> str:
    if score >= 85:
        return "#2e9f4d"
    if score >= 70:
        return "#d9981f"
    return "#d14343"

def avatar_letter(name: str) -> str:
    return name[0] if name else "?"

def inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
            /* ===== Global Reset ===== */
            [data-testid="stAppViewContainer"] {
                background: #f0f2f6;
            }
            .main .block-container {
                max-width: 100%;
                padding: 0 !important;
                margin: 0;
            }

            /* ===== Sidebar ===== */
            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #e8ecf0;
                padding-top: 0;
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 0;
            }
            [data-testid="stSidebarNav"] {
                display: none;
            }
            [data-testid="stSidebar"] .stRadio > div {
                flex-direction: column;
                gap: 2px;
            }
            [data-testid="stSidebar"] .stRadio label {
                padding: 10px 10px;
                border-radius: 8px;
                margin: 0;
                font-size: 14px;
                color: #333;
                transition: all 0.15s ease;
                cursor: pointer;
                border: 1px solid transparent;
                display: flex;
                align-items: center;
                gap: 0;
                line-height: 1.2;
            }
            [data-testid="stSidebar"] .stRadio label > input[type="radio"] {
                margin: 0;
                flex-shrink: 0;
                position: relative;
                top: 1px;
            }
            [data-testid="stSidebar"] .stRadio label p,
            [data-testid="stSidebar"] .stRadio label span {
                margin: 0;
                padding: 0;
                line-height: 1.2;
                display: inline-flex;
                align-items: center;
                vertical-align: middle;
                word-spacing: 0.35em;
            }
            [data-testid="stSidebar"] .stRadio label:hover {
                background: #f5f7fa;
            }
            [data-testid="stSidebar"] .stRadio label:has(input:checked) {
                background: #e8f4fd;
                color: #1a6fb5;
                font-weight: 600;
                border-color: #b8ddf5;
            }
            [data-testid="stSidebar"] .stSelectbox label {
                font-size: 13px;
                color: #666;
            }

            /* ===== Header Bar ===== */
            .global-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #ffffff;
                border-bottom: 1px solid #e8ecf0;
                padding: 12px 28px;
                margin: 0;
            }
            .header-left {
                display: flex;
                align-items: center;
                gap: 16px;
            }
            .header-logo-text {
                font-size: 18px;
                font-weight: 700;
                color: #111827;
            }
            .header-subtitle {
                font-size: 12px;
                color: #888;
                margin-left: 4px;
            }
            .header-right {
                display: flex;
                align-items: center;
                gap: 16px;
            }
            .header-badge {
                font-size: 11px;
                background: #e8f4fd;
                color: #1a6fb5;
                padding: 3px 10px;
                border-radius: 99px;
                font-weight: 500;
            }

            /* ===== Page Content Area ===== */
            .page-content {
                padding: 20px 28px 28px;
            }

            /* ===== App Title Card ===== */
            .app-title {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 14px;
                background: #ffffff;
                border: 1px solid #e8ecf0;
                border-radius: 10px;
                padding: 16px 20px;
                margin-bottom: 18px;
            }
            .title-left {
                font-size: 18px;
                font-weight: 700;
                color: #1a1a2e;
            }
            .title-desc {
                font-size: 12px;
                color: #8c8c9e;
                margin-top: 2px;
            }
            .title-meta {
                display: flex;
                gap: 8px;
                margin-top: 8px;
                flex-wrap: wrap;
            }
            .title-chip {
                font-size: 10px;
                color: #1a6fb5;
                background: #e8f4fd;
                border: 1px solid #d0e8f7;
                border-radius: 99px;
                padding: 2px 9px;
            }

            /* ===== KPI Cards ===== */
            .kpi-box {
                background: #ffffff;
                border: 1px solid #e8ecf0;
                border-radius: 10px;
                padding: 16px 16px;
                text-align: left;
                min-height: 80px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
                transition: box-shadow 0.15s ease;
            }
            .kpi-box:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }
            .kpi-label {
                font-size: 12px;
                color: #888;
                margin-bottom: 6px;
            }
            .kpi-value {
                font-size: 26px;
                font-weight: 800;
                color: #1a1a2e;
                line-height: 1.1;
            }

            /* ===== Unified Card ===== */
            .card {
                background: #ffffff;
                border: 1px solid #e8ecf0;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            }
            .card-title {
                font-size: 14px;
                font-weight: 700;
                color: #1a1a2e;
                margin-bottom: 10px;
            }

            /* ===== Candidate Cards ===== */
            .candidate-card {
                border: 1px solid #e8ecf0;
                border-left: 4px solid #1a6fb5;
                border-radius: 10px;
                padding: 10px 12px;
                margin-bottom: 8px;
                background: #ffffff;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .candidate-card:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }
            .candidate-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .avatar {
                width: 26px;
                height: 26px;
                border-radius: 8px;
                background: linear-gradient(140deg, #1d4ed8, #0ea5e9);
                color: #ffffff;
                font-weight: 700;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
            }
            .name-line {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
                font-weight: 600;
                color: #1a1a2e;
            }
            .sub-line {
                font-size: 11px;
                color: #888;
                margin-top: 2px;
            }
            .score-num {
                font-weight: 800;
                font-size: 28px;
                line-height: 1;
            }
            .pill {
                display: inline-block;
                font-size: 10px;
                color: #1a6fb5;
                background: #e8f4fd;
                border: 1px solid #d0e8f7;
                border-radius: 99px;
                padding: 2px 9px;
                margin-right: 4px;
                margin-top: 4px;
            }

            /* ===== Match Table ===== */
            .match-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            }
            .match-table td {
                border-bottom: 1px solid #f0f0f0;
                padding: 8px 4px;
                color: #555;
            }
            .match-table td:first-child {
                font-weight: 600;
                color: #333;
            }

            /* ===== Skill Bars ===== */
            .skill-line {
                margin-bottom: 8px;
            }
            .skill-header {
                font-size: 12px;
                display: flex;
                justify-content: space-between;
                color: #555;
                margin-bottom: 3px;
            }
            .skill-bar {
                width: 100%;
                height: 6px;
                background: #f0f0f0;
                border-radius: 99px;
                overflow: hidden;
            }
            .skill-fill {
                height: 6px;
                background: linear-gradient(90deg, #3b82f6, #1d4ed8);
            }
            .small-note {
                font-size: 11px;
                color: #888;
            }

            /* ===== Buttons ===== */
            .stButton > button {
                border-radius: 8px;
                border: 1px solid #d0d5dd;
                background: #ffffff;
                color: #333;
                font-weight: 600;
                font-size: 13px;
                transition: all 0.15s ease;
            }
            .stButton > button:hover {
                border-color: #1a6fb5;
                color: #1a6fb5;
                background: #f8fbff;
            }

            /* ===== Text Inputs ===== */
            .stTextArea textarea,
            .stMultiSelect div[data-baseweb="select"] > div,
            .stSelectbox div[data-baseweb="select"] > div {
                border-radius: 8px !important;
                border-color: #d0d5dd !important;
                background: #fafbfc !important;
            }
            .stTextArea textarea:focus,
            .stMultiSelect div[data-baseweb="select"] > div:focus-within,
            .stSelectbox div[data-baseweb="select"] > div:focus-within {
                border-color: #1a6fb5 !important;
                box-shadow: 0 0 0 2px rgba(26, 111, 181, 0.1) !important;
            }

            /* ===== Sidebar Logo ===== */
            .sidebar-logo {
                padding: 20px 16px 8px;
                border-bottom: 1px solid #f0f0f0;
                margin-bottom: 12px;
            }
            .sidebar-logo-text {
                font-size: 16px;
                font-weight: 700;
                color: #111827;
            }
            .sidebar-logo-sub {
                font-size: 11px;
                color: #888;
                margin-top: 2px;
            }

            /* ===== Kanban (unchanged) ===== */
            .kanban-list {
                max-height: 780px;
                overflow-y: auto;
                padding-right: 4px;
            }
            .kanban-card {
                border: 1px solid #e8ecf0;
                border-left: 4px solid #1a6fb5;
                border-radius: 10px;
                padding: 10px 12px;
                margin-bottom: 8px;
                background: #ffffff;
                cursor: pointer;
                transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
            }
            .kanban-card:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }
            .kanban-card.selected {
                border-left-color: #0ea5e9;
                border-color: #0ea5e9;
                background: #f0f9ff;
                box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
            }
            .kanban-card-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 8px;
            }
            .kanban-status {
                display: inline-block;
                font-size: 10px;
                color: #ffffff;
                border-radius: 99px;
                padding: 2px 8px;
                font-weight: 700;
            }
            .kanban-meta {
                display: flex;
                gap: 10px;
                margin-top: 6px;
                flex-wrap: wrap;
            }
            .kanban-meta-item {
                font-size: 11px;
                color: #888;
            }
            .kanban-meta-item b {
                color: #333;
            }
            .kanban-score-bar {
                width: 100%;
                height: 4px;
                background: #f0f0f0;
                border-radius: 99px;
                overflow: hidden;
                margin-top: 6px;
            }
            .kanban-score-fill {
                height: 4px;
                border-radius: 99px;
            }

            /* ===== Expandable details ===== */
            .detail-section {
                margin-bottom: 12px;
            }

            /* ===== Footer ===== */
            .page-footer {
                text-align: center;
                padding: 12px;
                font-size: 11px;
                color: #bbb;
                border-top: 1px solid #f0f0f0;
                margin-top: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_json_file(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_default_matching_report() -> dict[str, Any]:
    default_path = PARSED_RESULTS_DIR / "candidate_match_results.json"
    if not default_path.exists():
        return {}
    return load_json_file(str(default_path))

def normalize_unknown_skill_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"[\s_\-\./\\]+", "", text)
    return text

def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records

def build_unknown_skill_summary() -> dict[str, Any]:
    resume_records = read_jsonl_records(UNKNOWN_RESUME_SKILLS_PATH)
    jd_records = read_jsonl_records(UNKNOWN_JD_SKILLS_PATH)

    counter: Counter[str] = Counter()
    display_name: dict[str, str] = {}
    resume_sources: dict[str, set[str]] = {}
    jd_sources: dict[str, set[str]] = {}
    sample_evidence: dict[str, str] = {}

    total_items = 0

    for record in resume_records:
        source = str(record.get("source_file") or "")
        unknowns = record.get("unknown_skills") or []
        if not isinstance(unknowns, list):
            continue
        for item in unknowns:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            key = normalize_unknown_skill_name(name)
            if not key:
                continue
            counter[key] += 1
            total_items += 1
            display_name.setdefault(key, name)
            if source:
                resume_sources.setdefault(key, set()).add(source)
            evidence = str(item.get("evidence") or "").strip()
            if evidence and key not in sample_evidence:
                sample_evidence[key] = evidence

    for record in jd_records:
        source = str(record.get("source") or record.get("job_title") or "")
        unknowns = record.get("unknown_skills") or []
        if not isinstance(unknowns, list):
            continue
        for item in unknowns:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            key = normalize_unknown_skill_name(name)
            if not key:
                continue
            counter[key] += 1
            total_items += 1
            display_name.setdefault(key, name)
            if source:
                jd_sources.setdefault(key, set()).add(source)
            evidence = str(item.get("evidence") or "").strip()
            if evidence and key not in sample_evidence:
                sample_evidence[key] = evidence

    rows: list[dict[str, Any]] = []
    for key, freq in counter.most_common():
        rows.append(
            {
                "技能": display_name.get(key, key),
                "出现次数": freq,
                "简历来源数": len(resume_sources.get(key, set())),
                "JD来源数": len(jd_sources.get(key, set())),
                "示例证据": sample_evidence.get(key, ""),
                "规范键": key,
            }
        )

    return {
        "resume_records": resume_records,
        "jd_records": jd_records,
        "rows": rows,
        "total_unknown_items": total_items,
        "unique_unknown_skills": len(rows),
    }

def parse_level_value(level_text: str) -> int | None:
    match = re.search(r"(\\d+)", level_text)
    if not match:
        return None
    return int(match.group(1))

def summarize_experience(experience_items: Any) -> str:
    if not isinstance(experience_items, list) or not experience_items:
        return "经验待补充"
    return f"{len(experience_items)}段工作经历"

def get_dimension_score(result: dict[str, Any], key: str) -> float:
    value = result.get("dimensions", {}).get(key, {}).get("score")
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0

def build_skill_lines(result: dict[str, Any]) -> list[tuple[str, int, str]]:
    technical = result.get("dimensions", {}).get("technical_skills", {})
    lines: list[tuple[str, int, str]] = []
    for row in technical.get("matched", []) + technical.get("missing", []):
        if not isinstance(row, dict):
            continue
        required_items = row.get("required_items", [])
        matched_items = row.get("matched_items", [])
        required_count = len(required_items) if isinstance(required_items, list) else 0
        matched_count = len(matched_items) if isinstance(matched_items, list) else 0
        pct = int((matched_count / required_count) * 100) if required_count else 0
        lines.append(
            (
                str(row.get("group_name") or row.get("name") or row.get("group_id") or "技能组"),
                pct,
                f"命中 {matched_count}/{required_count}",
            )
        )
    return lines

def build_candidate_view(result: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    candidate_file = str(result.get("candidate_file") or "").strip()
    profile = load_json_file(candidate_file) if candidate_file else {}
    alias_from_file = Path(candidate_file).stem.replace("_parsed", "") if candidate_file else ""
    alias_from_result = str(result.get("candidate_name") or "").strip()
    display_name = alias_from_file if alias_from_file.lower().startswith("candidate_") else alias_from_result
    if not display_name:
        display_name = alias_from_result or "未知候选人"
    experience_items = profile.get("经验", [])
    level_text = str(profile.get("级别") or "").strip()
    if not level_text:
        level_num = parse_level_value(str(result.get("level_reason", "")))
        level_text = f"Level {level_num}" if level_num is not None else "Level 未识别"

    matched_competencies = result.get("dimensions", {}).get("competencies", {}).get("matched", [])
    quality_tags = [
        str(item.get("name") or item.get("candidate_name") or "").strip()
        for item in matched_competencies
        if isinstance(item, dict)
    ]
    quality_tags = [tag for tag in quality_tags if tag]

    match_reason = [
        str(result.get("level_reason") or result.get("rank_reason") or "").strip(),
        str(result.get("dimensions", {}).get("technical_skills", {}).get("reason") or "").strip(),
        str(result.get("dimensions", {}).get("capabilities", {}).get("reason") or "").strip(),
    ]
    match_reason = [text for text in match_reason if text]

    matched_skill_names = set()
    for row in result.get("dimensions", {}).get("technical_skills", {}).get("matched", []):
        if not isinstance(row, dict):
            continue
        for item in row.get("matched_items", []):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("candidate_name") or "").strip()
                if name:
                    matched_skill_names.add(name)

    project_lines: list[tuple[str, str]] = []
    if isinstance(experience_items, list):
        for item in experience_items[:3]:
            if not isinstance(item, dict):
                continue
            company = str(item.get("公司") or "").strip()
            role = str(item.get("职位") or "").strip()
            period = str(item.get("时间") or "").strip()
            title = " / ".join([part for part in [company, role] if part]) or "工作经历"
            project_lines.append((title, period))

    return {
        "id": candidate_file or str(result.get("candidate_name") or ""),
        "name": display_name,
        "raw_name": alias_from_result,
        "role": str(report.get("job_title") or "岗位候选人"),
        "score": float(result.get("total_score") or 0),
        "level": level_text,
        "experience": summarize_experience(experience_items),
        "profile_type": str(profile.get("profile_type", {}).get("primary") or "UNKNOWN"),
        "matched_skill_names": matched_skill_names,
        "skills": build_skill_lines(result),
        "projects": project_lines,
        "qualities": quality_tags,
        "match_reason": match_reason,
        "result": result,
    }

def ensure_report_for_jd(
    selected_jd_path: Path,
    force_refresh: bool = False,
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights_key = ""
    if custom_weights:
        weights_key = "|".join(
            f"{key}:{float(custom_weights.get(key, 0.0)):.4f}"
            for key in ["technical_skills", "capabilities", "competencies"]
        )

    cached = st.session_state.get("json_matching_report", {})
    if (
        isinstance(cached, dict)
        and cached
        and Path(str(cached.get("jd_file", ""))).resolve() == selected_jd_path.resolve()
        and str(cached.get("_weights_key", "")) == weights_key
        and not force_refresh
    ):
        return cached

    if not force_refresh and not custom_weights:
        default_report = load_default_matching_report()
        if (
            default_report
            and Path(str(default_report.get("jd_file", ""))).resolve() == selected_jd_path.resolve()
        ):
            default_report["_weights_key"] = ""
            st.session_state["json_matching_report"] = default_report
            return default_report

    resume_paths = find_resume_json_files()
    if not resume_paths:
        return {}

    report = match_json_files(selected_jd_path, resume_paths, custom_weights=custom_weights)
    report["_weights_key"] = weights_key
    st.session_state["json_matching_report"] = report
    return report

def render_dimension_evidence_panel(result: dict[str, Any]) -> None:
    dimensions = result.get("dimensions", {}) if isinstance(result, dict) else {}
    tab_specs = [
        ("技术技能", "technical_skills"),
        ("能力", "capabilities"),
        ("胜任力", "competencies"),
        ("项目经验", "project_experience"),
        ("行业", "industries"),
        ("语言", "languages"),
    ]
    tab_labels = [name for name, _ in tab_specs]
    tabs = st.tabs(tab_labels)

    for tab, (label_text, key) in zip(tabs, tab_specs):
        with tab:
            detail = dimensions.get(key, {}) if isinstance(dimensions, dict) else {}
            reason = str(detail.get("reason", "") or "").strip()
            if reason:
                st.caption(reason)

            matched_rows = format_match_rows(detail.get("matched", []))
            missing_rows = format_match_rows(detail.get("missing", []))

            st.markdown("命中项")
            if matched_rows:
                st.dataframe(matched_rows, use_container_width=True, hide_index=True)
            else:
                st.info(f"{label_text}暂无命中项。")

            st.markdown("未命中项")
            if missing_rows:
                st.dataframe(missing_rows, use_container_width=True, hide_index=True)
            else:
                st.success(f"{label_text}要求已全部命中。")


def render_matching_dashboard() -> None:
    st.markdown(
        """
        <div class="app-title">
            <div>
                <div class="title-left">人岗匹配分析看板</div>
                <div class="title-desc">岗位匹配中心 · 实时读取 parsed_results 真实数据</div>
                <div class="title-meta">
                    <span class="title-chip">Ontology + AI Matching</span>
                    <span class="title-chip">Explainable Scoring</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jd_options = list_jd_json_options()
    if not jd_options:
        st.warning("未找到 JD JSON，请先在解析工具页生成 JD 数据。")
        return

    selected_label = st.selectbox(
        "选择正在招聘的 JD",
        options=[item["label"] for item in jd_options],
        index=0,
    )
    selected_jd = next(item for item in jd_options if item["label"] == selected_label)
    selected_jd_path = selected_jd["path"]
    selected_jd_data = load_json_file(str(selected_jd_path))

    force_refresh = st.button("刷新真实匹配结果", use_container_width=False)
    jd_weight_state_key = f"dashboard_custom_weights::{selected_jd_path.resolve()}"
    active_custom_weights = st.session_state.get(jd_weight_state_key)
    try:
        report = ensure_report_for_jd(
            selected_jd_path,
            force_refresh=force_refresh,
            custom_weights=active_custom_weights,
        )
    except Exception as exc:
        st.error(f"匹配计算失败：{exc}")
        return

    if not report:
        st.warning("暂无候选人匹配数据。请先解析简历或运行 batch_parse_resumes.py。")
        return

    candidate_views = [build_candidate_view(item, report) for item in report.get("results", []) if isinstance(item, dict)]
    if not candidate_views:
        st.warning("当前 JD 暂无可展示候选人。")
        return

    # ── PDF & CSV 导出按钮 ──
    export_col1, export_col2, _ = st.columns([1, 1, 3])
    with export_col1:
        # CSV 导出
        csv_rows = [
            {
                "排名": i + 1,
                "候选人": item.get("candidate_name", ""),
                "总分": item.get("total_score", 0),
                "级别": item.get("level_reason", ""),
            }
            for i, item in enumerate(report.get("results", []) if isinstance(report.get("results"), list) else [])
            if isinstance(item, dict)
        ]
        csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 导出CSV",
            data=csv_data,
            file_name=f"匹配结果_{report.get('job_title','results')}.csv",
            mime="text/csv",
        )
    with export_col2:
        # PDF 导出（任何异常都不能让整页崩溃）
        try:
            pdf_data = generate_match_report_pdf(report, selected_jd_data)
        except Exception as pdf_exc:
            pdf_data = b""
            st.warning(f"PDF 生成失败（已隐藏导出按钮）：{pdf_exc}")
        if pdf_data and isinstance(pdf_data, (bytes, bytearray)) and len(pdf_data) > 0:
            st.download_button(
                "📑 导出PDF",
                data=bytes(pdf_data),
                file_name=f"匹配报告_{report.get('job_title','report')}.pdf",
                mime="application/pdf",
            )
        elif not pdf_data:
            st.caption("📑 PDF 导出暂不可用：缺少 fpdf2 或中文字体")

    left_col, mid_col, right_col = st.columns([1.08, 1.58, 1.72])

    with left_col:
        st.markdown("<div class='card'><div class='card-title'>岗位需求（JD）</div>", unsafe_allow_html=True)
        jd_skills = [
            str(item.get("name") or item.get("skill_id") or "").strip()
            for item in selected_jd_data.get("技术技能要求", [])
            if isinstance(item, dict)
        ]
        jd_skills = [item for item in jd_skills if item]
        jd_level = str(selected_jd_data.get("级别要求") or "未指定")
        jd_summary = f"{selected_jd_data.get('岗位名称', report.get('job_title', '岗位'))}，级别要求：{jd_level}，核心技能：{'、'.join(jd_skills[:8]) or '待补充'}。"
        st.text_area("JD摘要", value=jd_summary, height=120, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>筛选条件</div>", unsafe_allow_html=True)
        min_score = st.slider("最低总分", 0, 100, 60, 1)
        selected_skills = st.multiselect("技能关键词", options=jd_skills, default=jd_skills[:3])
        level_options = sorted(set(item["level"] for item in candidate_views))
        selected_levels = st.multiselect("候选人级别", options=level_options, default=level_options)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>匹配权重</div>", unsafe_allow_html=True)
        weights = report.get("weights", {})
        tech_default = int(round(float(weights.get("technical_skills", 0.8)) * 100))
        cap_default = int(round(float(weights.get("capabilities", 0.1)) * 100))
        comp_default = int(round(float(weights.get("competencies", 0.1)) * 100))

        tech_value = st.slider("技术技能", 0, 100, tech_default, 5)
        cap_value = st.slider("能力", 0, 100, cap_default, 5)
        comp_value = st.slider("胜任力", 0, 100, comp_default, 5)

        raw_total = tech_value + cap_value + comp_value
        if raw_total <= 0:
            st.warning("权重总和不能为 0，请至少保留一个维度大于 0。")
            normalized_weights = None
        else:
            normalized_weights = {
                "technical_skills": tech_value / raw_total,
                "capabilities": cap_value / raw_total,
                "competencies": comp_value / raw_total,
            }
            st.markdown(
                "<div class='small-note'>应用时将自动归一化："
                f"技术技能 {normalized_weights['technical_skills']*100:.1f}% · "
                f"能力 {normalized_weights['capabilities']*100:.1f}% · "
                f"胜任力 {normalized_weights['competencies']*100:.1f}%"
                "</div>",
                unsafe_allow_html=True,
            )

        apply_col, reset_col = st.columns(2)
        with apply_col:
            if st.button("应用权重", use_container_width=True, disabled=normalized_weights is None):
                st.session_state[jd_weight_state_key] = normalized_weights
                st.rerun()
        with reset_col:
            if st.button("恢复默认", use_container_width=True):
                if jd_weight_state_key in st.session_state:
                    del st.session_state[jd_weight_state_key]
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    all_count = int(report.get("source_candidate_count") or len(candidate_views))
    matched_count = int(report.get("candidate_count") or len(candidate_views))
    high_count = len([item for item in candidate_views if item["score"] >= 85])
    jd_count = len(jd_options)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='kpi-box'><div class='kpi-label'>候选池总人数</div><div class='kpi-value'>{all_count}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi-box'><div class='kpi-label'>本次JD岗位</div><div class='kpi-value'>{jd_count}</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi-box'><div class='kpi-label'>可匹配候选人</div><div class='kpi-value'>{matched_count}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi-box'><div class='kpi-label'>高匹配(85+)</div><div class='kpi-value' style='color:#2e9f4d'>{high_count}</div></div>", unsafe_allow_html=True)

    filtered_candidates: list[dict[str, Any]] = []
    for item in candidate_views:
        if item["score"] < min_score:
            continue
        if item["level"] not in selected_levels:
            continue
        if selected_skills and not (item["matched_skill_names"] & set(selected_skills)):
            continue
        filtered_candidates.append(item)

    if not filtered_candidates:
        st.info("当前筛选条件下没有候选人，请调整筛选条件。")
        return

    selected_key = "dashboard_selected_candidate"
    valid_ids = {item["id"] for item in filtered_candidates}
    if st.session_state.get(selected_key) not in valid_ids:
        st.session_state[selected_key] = filtered_candidates[0]["id"]
    selected_candidate = next(item for item in filtered_candidates if item["id"] == st.session_state[selected_key])

    with mid_col:
        st.markdown(f"<div class='small-note'>共 {len(filtered_candidates)} 位候选人</div>", unsafe_allow_html=True)
        for idx, cand in enumerate(filtered_candidates, start=1):
            border = "#2563eb" if cand["id"] == st.session_state[selected_key] else "#d7e1ec"
            st.markdown(
                f"""
                <div class='candidate-card' style='border-left-color:{border};'>
                    <div class='candidate-row'>
                        <div>
                            <div class='name-line'>
                                <span class='avatar'>{avatar_letter(cand['name'])}</span>
                                <span>{cand['name']}</span>
                            </div>
                            <div class='sub-line'>{cand['role']} · {cand['level']} · {cand['experience']}</div>
                            <div class='sub-line'>画像类型：{cand['profile_type']}</div>
                        </div>
                        <div style='text-align:right;'>
                            <div class='score-num' style='color:{score_color(cand['score'])};'>{cand['score']:.2f}</div>
                            <div class='small-note'>综合匹配分</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"选择 {idx}", key=f"pick_{idx}_{cand['id']}", use_container_width=True):
                st.session_state[selected_key] = cand["id"]
                st.rerun()

    with right_col:
        selected_result = selected_candidate["result"]
        st.markdown(
            f"""
            <div class='card'>
                <div class='candidate-row'>
                    <div>
                        <div class='name-line'><span class='avatar'>{avatar_letter(selected_candidate['name'])}</span><span>{selected_candidate['name']}</span></div>
                        <div class='sub-line'>{selected_candidate['role']} · {selected_candidate['level']}</div>
                        <div class='sub-line'>{selected_candidate['experience']} · 画像类型：{selected_candidate['profile_type']}</div>
                    </div>
                    <div style='text-align:right;'>
                        <div class='score-num' style='color:{score_color(selected_candidate['score'])};'>{selected_candidate['score']:.2f}</div>
                        <div class='small-note'>综合匹配分</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pills = "".join([f"<span class='pill'>{item}</span>" for item in selected_candidate["match_reason"][:4]])
        st.markdown(f"<div style='margin-bottom:8px'>{pills}</div>", unsafe_allow_html=True)

        technical = selected_result.get("dimensions", {}).get("technical_skills", {})
        capability = selected_result.get("dimensions", {}).get("capabilities", {})
        competency = selected_result.get("dimensions", {}).get("competencies", {})

        st.markdown("<div class='card'><div class='card-title'>匹配对比表</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <table class='match-table'>
                <tr><td>技术技能</td><td>{technical.get('required_count', 0)} 项要求</td><td>命中 {technical.get('matched_count', 0)} 项</td><td>{get_dimension_score(selected_result, 'technical_skills'):.2f}%</td></tr>
                <tr><td>能力</td><td>{capability.get('required_count', 0)} 项要求</td><td>命中 {capability.get('matched_count', 0)} 项</td><td>{get_dimension_score(selected_result, 'capabilities'):.2f}%</td></tr>
                <tr><td>胜任力</td><td>{competency.get('required_count', 0)} 项要求</td><td>命中 {competency.get('matched_count', 0)} 项</td><td>{get_dimension_score(selected_result, 'competencies'):.2f}%</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>匹配明细与证据</div>", unsafe_allow_html=True)
        render_dimension_evidence_panel(selected_result)
        st.markdown("</div>", unsafe_allow_html=True)

def build_job_view(job_data: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    jd_path = str(job_data.get("path") or "").strip()
    jd_file = load_json_file(jd_path) if jd_path else {}
    skill_items = jd_file.get("技术技能要求", []) if isinstance(jd_file, dict) else []
    skill_names = [str(item.get("name") or item.get("skill_id") or "").strip() for item in skill_items if isinstance(item, dict)]
    skill_names = [item for item in skill_names if item]

    return {
        "id": jd_path or job_data.get("label", ""),
        "name": str(job_data.get("job_title") or report.get("job_title") or "未知岗位"),
        "level": str(jd_file.get("级别要求") or "未指定"),
        "score": float(job_data.get("score") or 0.0),
        "summary": str(job_data.get("summary") or ""),
        "skills": skill_names,
        "raw": job_data,
    }

def render_reverse_matching_dashboard() -> None:
    st.markdown(
        """
        <div class='app-title'>
            <div>
                <div class='title-left'>职业助手 · 分析视图</div>
                <div class='title-desc'>候选人优先 · 从候选人画像反查最适合的岗位</div>
                <div class='title-meta'>
                    <span class='title-chip'>Candidate First</span>
                    <span class='title-chip'>Reverse Matching</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidate_paths = find_resume_json_files()
    if not candidate_paths:
        st.warning("未找到候选人解析结果，请先在解析工具页批量解析简历。")
        return

    jd_options = list_jd_json_options()
    if not jd_options:
        st.warning("未找到 JD JSON，请先在解析工具页生成 JD 数据。")
        return

    candidate_options = []
    for path in candidate_paths:
        data = load_json_file(str(path))
        candidate_options.append(
            {
                "path": path,
                "name": str(data.get("姓名") or path.stem.replace("_parsed", "") or path.stem),
                "level": str(data.get("级别") or "未识别"),
                "title": str(data.get("profile_type", {}).get("primary") or "UNKNOWN"),
                "label": f"{data.get('姓名') or path.stem} | {data.get('级别') or '未识别'} | {path.name}",
            }
        )

    selected_candidate_label = st.selectbox(
        "选择候选人",
        options=[item["label"] for item in candidate_options],
        index=0,
    )
    selected_candidate = next(item for item in candidate_options if item["label"] == selected_candidate_label)
    selected_candidate_data = load_json_file(str(selected_candidate["path"]))

    selected_jd_label = st.selectbox(
        "选择岗位",
        options=[item["label"] for item in jd_options],
        index=0,
    )
    selected_jd = next(item for item in jd_options if item["label"] == selected_jd_label)
    selected_jd_data = load_json_file(str(selected_jd["path"]))

    st.markdown("### 候选人画像")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("姓名", selected_candidate_data.get("姓名") or selected_candidate["name"])
    with c2:
        st.metric("级别", selected_candidate_data.get("级别") or "未识别")
    with c3:
        st.metric("画像类型", selected_candidate["title"])
    with c4:
        st.metric("简历技能数", len(selected_candidate_data.get("技术技能", []) if isinstance(selected_candidate_data.get("技术技能", []), list) else []))

    st.markdown("### 可投岗位")
    reverse_rows: list[dict[str, Any]] = []
    for jd_item in jd_options:
        jd_path = jd_item["path"]
        jd_data = load_json_file(str(jd_path))
        report = ensure_report_for_jd(jd_path, force_refresh=False)
        if not report:
            continue
        # candidate-first 视图里，沿用现有岗位匹配结果做岗位侧摘要
        matched = next((item for item in report.get("results", []) if item.get("candidate_file") == str(selected_candidate["path"])), None)
        if not matched:
            continue
        reverse_rows.append(
            {
                "job_title": str(report.get("job_title") or jd_item["name"]),
                "score": float(matched.get("total_score") or 0.0),
                "level_reason": str(matched.get("level_reason") or ""),
                "jd_file": jd_path,
                "job_type": str(report.get("job_type") or "UNKNOWN"),
                "weights": report.get("weights", {}),
                "result": matched,
                "jd_data": jd_data,
            }
        )

    reverse_rows.sort(key=lambda item: (-item["score"], item["job_title"]))
    if not reverse_rows:
        st.info("当前候选人在已有岗位结果中没有命中。请先刷新岗位匹配数据或检查 parsed_results。")
        return

    left_col, right_col = st.columns([1.2, 1.8])
    with left_col:
        for idx, row in enumerate(reverse_rows[:20], start=1):
            st.markdown(
                f"""
                <div class='candidate-card'>
                    <div class='candidate-row'>
                        <div>
                            <div class='name-line'><span class='avatar'>{avatar_letter(row['job_title'])}</span><span>{row['job_title']}</span></div>
                            <div class='sub-line'>岗位类型：{row['job_type']}</div>
                            <div class='sub-line'>{row['level_reason']}</div>
                        </div>
                        <div style='text-align:right;'>
                            <div class='score-num' style='color:{score_color(row['score'])};'>{row['score']:.2f}</div>
                            <div class='small-note'>适配分</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        best_row = reverse_rows[0]
        st.markdown("### 最高适配岗位")
        st.metric("岗位", best_row["job_title"])
        st.metric("适配分", f"{best_row['score']:.2f}")
        st.write(best_row["level_reason"])
        st.markdown("#### 岗位需求摘要")
        st.write(
            f"岗位类型：{best_row['job_type']}；级别要求：{best_row['jd_data'].get('级别要求') or '未指定'}；"
            f"核心技能数：{len(best_row['jd_data'].get('技术技能要求', []) if isinstance(best_row['jd_data'].get('技术技能要求', []), list) else [])}"
        )
        st.markdown("#### 候选人匹配摘要")
        st.write(selected_candidate_data.get("profile_type", {}))

def _llm_chat_reply(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    if not api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY。")

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    content = response.choices[0].message.content
    return str(content or "").strip()

def _build_resume_chat_context(selected_jd_data: dict[str, Any]) -> str:
    job_title = str(selected_jd_data.get("岗位名称") or selected_jd_data.get("job_title") or "未指定岗位")
    level_req = str(selected_jd_data.get("级别要求") or "未指定")
    skill_items = selected_jd_data.get("技术技能要求", [])
    skills: list[str] = []
    if isinstance(skill_items, list):
        for item in skill_items:
            if isinstance(item, dict):
                skill = str(item.get("name") or item.get("skill_id") or "").strip()
                if skill:
                    skills.append(skill)
    skill_text = "、".join(skills[:20]) if skills else "未提供"
    return (
        f"目标岗位：{job_title}\n"
        f"级别要求：{level_req}\n"
        f"关键技能（最多20项）：{skill_text}"
    )

def _rewrite_resume_draft(
    api_key: str,
    base_url: str,
    model: str,
    resume_draft: str,
    jd_context: str,
    chat_history: list[dict[str, str]],
) -> str:
    history_lines: list[str] = []
    for msg in chat_history[-8:]:
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if role and content:
            history_lines.append(f"{role}: {content}")

    messages = [
        {
            "role": "system",
            "content": (
                "你是资深招聘顾问与简历优化专家。"
                "请根据候选人与助手的对话、目标岗位信息，直接输出一份可投递的中文简历文本。"
                "要求：保留真实经历，不虚构；突出与目标岗位相关的技能、项目与结果；"
                "结构清晰，使用Markdown分段；只输出最终简历正文，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"目标岗位上下文：\n{jd_context}\n\n"
                f"最近对话：\n" + "\n".join(history_lines) + "\n\n"
                f"当前简历草稿：\n{resume_draft}"
            ),
        },
    ]
    return _llm_chat_reply(api_key, base_url, model, messages)

def _enhance_resume_draft_with_llm(
    api_key: str,
    base_url: str,
    model: str,
    resume_draft: str,
    jd_context: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是资深简历优化顾问。"
                "请在不编造事实的前提下，优化并补充简历内容，让其更完整、更有说服力。"
                "要求："
                "1) 保留原有信息真实性，不得凭空捏造公司、项目或奖项；"
                "2) 补充每段经历的职责与成果表达，优先使用可量化结果；"
                "3) 针对目标岗位补充技能亮点、项目亮点、关键词覆盖；"
                "4) 若关键信息缺失，可用【待补充：xxx】占位提示用户后续完善；"
                "5) 输出中文 Markdown 简历正文，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"目标岗位上下文：\n{jd_context}\n\n"
                f"当前简历草稿：\n{resume_draft}\n\n"
                "请输出优化并补充后的完整简历。"
            ),
        },
    ]
    return _llm_chat_reply(api_key, base_url, model, messages)

def _build_empty_resume_template(selected_jd_data: dict[str, Any]) -> str:
    job_title = str(selected_jd_data.get("岗位名称") or selected_jd_data.get("job_title") or "目标岗位")
    return (
        f"# 简历（面向：{job_title}）\n\n"
        "## 个人信息\n"
        "- 姓名：\n"
        "- 手机：\n"
        "- 邮箱：\n"
        "- 所在城市：\n\n"
        "## 个人摘要\n"
        "- 【待补充：3-5行概述你的核心经验、行业背景与岗位匹配优势】\n\n"
        "## 核心技能\n"
        "- 【待补充：技术技能/业务技能/管理技能】\n\n"
        "## 工作经历\n"
        "### 公司A | 职位 | 时间\n"
        "- 职责：\n"
        "- 关键行动：\n"
        "- 结果：\n"
        "- 【可补充：量化指标，如提升xx%、节省xx成本】\n\n"
        "## 项目经历\n"
        "### 项目A | 角色 | 时间\n"
        "- 背景与目标：\n"
        "- 你的贡献：\n"
        "- 结果与影响：\n\n"
        "## 教育背景\n"
        "- 学校 | 专业 | 学历 | 时间\n\n"
        "## 证书与其他\n"
        "- 【待补充】\n"
    )

def render_reverse_matching_chat(api_key: str, base_url: str, model: str) -> None:
    st.markdown(
        """
        <div class='app-title'>
            <div>
                <div class='title-left'>职业助手 · 对话优化</div>
                <div class='title-desc'>像聊天一样优化简历：和 LLM 互动、重写草稿、保存并重新解析</div>
                <div class='title-meta'>
                    <span class='title-chip'>Chat Resume Co-pilot</span>
                    <span class='title-chip'>Interactive Editing</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_role = str(st.session_state.get("auth_role", "hr") or "hr").strip().lower()

    extract_from_library = st.toggle(
        "从简历库提取简历",
        value=True,
        help="关闭后可从空白模板开始撰写，不依赖 resumes 目录中的简历文件。",
    )

    jd_options = list_jd_json_options()
    if not jd_options:
        st.warning("未找到 JD JSON。请先在“智能解析”生成 JD。")

        return

    selected_jd_label = st.selectbox(
        "选择目标岗位",
        options=[item["label"] for item in jd_options],
        index=0,
    )
    selected_jd = next(item for item in jd_options if item["label"] == selected_jd_label)
    selected_jd_data = load_json_file(str(selected_jd["path"]))
    jd_context = _build_resume_chat_context(selected_jd_data)

    selected_resume_path: Path
    selected_resume_identity: str
    if extract_from_library:
        # Admin and HR must always have full visibility of all resumes.
        if current_role in {"admin", "hr"}:
            resume_files = list_resume_files()
        else:
            resume_files = get_role_scoped_resume_files(list_resume_files())

        st.caption(f"当前可见简历数量：{len(resume_files)}")
        if not resume_files:
            if current_role == "employee":
                st.info(
                    "📭 当前账号未分配可访问的简历。你可以：\n\n"
                    "• 联系管理员在「**简历权限管理**」中为你绑定简历\n"
                    "• 或在下方直接上传你自己的简历文件\n"
                    "• 或关闭上方「从简历库提取简历」从空白模板开始"
                )
                # Offer in-place upload as fallback
                uploaded = st.file_uploader(
                    "📤 上传简历（PDF / DOCX / TXT / MD）",
                    type=["pdf", "docx", "txt", "md"],
                    key="emp_chat_upload_resume",
                )
                if uploaded is not None:
                    upload_dir = APP_DIR / "resumes" / "employees" / "employee"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    save_path = upload_dir / uploaded.name
                    with open(save_path, "wb") as out_f:
                        out_f.write(uploaded.read())
                    # Parse the uploaded resume
                    with st.spinner("🔄 正在解析简历..."):
                        try:
                            from resume_parser import parse_resume_to_json, save_resume_result
                            api_key = get_secret_value(
                                "OPENAI_API_KEY",
                                os.environ.get("OPENAI_API_KEY", ""),
                            )
                            base_url = get_secret_value(
                                "OPENAI_BASE_URL",
                                os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                            )
                            model = get_secret_value("PARSE_MODEL", os.environ.get("PARSE_MODEL", "gpt-4o-mini"))
                            result = parse_resume_to_json(save_path, api_key, base_url, model)
                            save_resume_result(save_path, result)
                            st.success(f"✅ 简历「{uploaded.name}」上传并解析成功！请重新选择。")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 解析失败：{e}")
                            if save_path.exists():
                                try:
                                    save_path.unlink()
                                except Exception:
                                    pass
                return
            else:
                st.warning("未找到简历文件。请先在 Talent_Matching/resumes 放入 PDF/DOCX/TXT/MD，或关闭“从简历库提取简历”。")
            return

        resume_labels = [str(path.relative_to(APP_DIR)) for path in resume_files]
        selected_resume_label = st.selectbox("选择你的简历", options=resume_labels, index=0)
        selected_resume_path = APP_DIR / selected_resume_label
        selected_resume_identity = selected_resume_path.name
    else:
        manual_name = st.text_input("新简历文件名（仅用于保存）", value="manual_resume")
        manual_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", manual_name.strip()).strip("_") or "manual_resume"
        selected_resume_path = RESUME_DIR / f"{manual_slug}.md"
        selected_resume_identity = f"manual::{manual_slug}"
        st.info("当前为手动新建简历模式：你可以从空白模板开始，随后使用 LLM 进行扩展与优化。")

    # ── 员工角色：将选中的简历缓存到 session_state，供其他3个页面共享 ──
    if current_role == "employee" and extract_from_library and selected_resume_path.exists():
        parsed_file = PARSED_RESULTS_DIR / f"{selected_resume_path.stem}_parsed.json"
        if parsed_file.exists():
            try:
                with open(parsed_file, encoding="utf-8") as f:
                    data = json.loads(f.read())
                st.session_state["_emp_resume_name"] = data.get("name", "") or selected_resume_path.stem
                st.session_state["_emp_resume_data"] = data
            except Exception:
                pass

    chat_key = f"resume_chat_messages::{selected_resume_identity}"
    draft_key = f"resume_chat_draft::{selected_resume_identity}"

    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {
                "role": "assistant",
                "content": "你好，我是职业助手。告诉我你想投什么岗位，我会帮你优化这份简历。",
            }
        ]

    if draft_key not in st.session_state:
        if extract_from_library:
            try:
                st.session_state[draft_key] = read_resume_text(selected_resume_path)
            except Exception as exc:
                st.error(f"读取简历失败：{exc}")
                return
        else:
            st.session_state[draft_key] = _build_empty_resume_template(selected_jd_data)

    left_col, right_col = st.columns([1.15, 1.05])

    with left_col:
        st.markdown("#### 对话区")
        for msg in st.session_state[chat_key]:
            role = str(msg.get("role") or "assistant")
            with st.chat_message(role):
                st.markdown(str(msg.get("content") or ""))

        user_prompt = st.chat_input("例如：请把我的简历改成更适合这个岗位，并突出 AI Agent 项目")
        if user_prompt:
            st.session_state[chat_key].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                try:
                    llm_messages = [
                        {
                            "role": "system",
                            "content": (
                                "你是资深职业顾问。基于候选人简历和目标JD回答用户。"
                                "输出要可执行，优先给可直接修改到简历的建议。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"目标岗位信息：\n{jd_context}\n\n"
                                f"当前简历草稿：\n{st.session_state[draft_key]}\n\n"
                                f"用户问题：{user_prompt}"
                            ),
                        },
                    ]
                    reply = _llm_chat_reply(api_key, base_url, model, llm_messages)
                except Exception as exc:
                    reply = f"生成失败：{exc}"
                st.markdown(reply)
                st.session_state[chat_key].append({"role": "assistant", "content": reply})

    with right_col:
        st.markdown("#### 简历草稿")
        st.caption(jd_context)
        st.session_state[draft_key] = st.text_area(
            "编辑区",
            value=st.session_state[draft_key],
            height=520,
            key=f"draft_editor::{selected_resume_identity}",
        )

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("应用对话建议重写草稿", use_container_width=True):
                with st.spinner("正在生成优化版简历草稿..."):
                    try:
                        rewritten = _rewrite_resume_draft(
                            api_key,
                            base_url,
                            model,
                            st.session_state[draft_key],
                            jd_context,
                            st.session_state[chat_key],
                        )
                        if rewritten:
                            st.session_state[draft_key] = rewritten
                            st.success("已生成并更新简历草稿。")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"重写失败：{exc}")

        with b2:
            if st.button("LLM优化并补充简历", use_container_width=True):
                with st.spinner("正在补充简历内容并优化表达..."):
                    try:
                        enhanced = _enhance_resume_draft_with_llm(
                            api_key,
                            base_url,
                            model,
                            st.session_state[draft_key],
                            jd_context,
                        )
                        if enhanced:
                            st.session_state[draft_key] = enhanced
                            st.success("已完成简历优化与内容补充。")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"优化失败：{exc}")

        with b3:
            if st.button("保存草稿", use_container_width=True):
                try:
                    if not extract_from_library:
                        RESUME_DIR.mkdir(exist_ok=True)
                        selected_resume_path.write_text(st.session_state[draft_key], encoding="utf-8")
                        st.success(f"已保存新简历：{selected_resume_path.name}")
                    else:
                        suffix = selected_resume_path.suffix.lower()
                        if suffix in {".txt", ".md"}:
                            selected_resume_path.write_text(st.session_state[draft_key], encoding="utf-8")
                            st.success(f"已保存到原文件：{selected_resume_path.name}")
                        else:
                            target = selected_resume_path.with_name(f"{selected_resume_path.stem}_chat_edited.md")
                            target.write_text(st.session_state[draft_key], encoding="utf-8")
                            st.success(f"原文件为 {suffix}，已另存为：{target.name}")
                except Exception as exc:
                    st.error(f"保存失败：{exc}")

        with b4:
            if st.button("重新解析并更新JSON", use_container_width=True):
                try:
                    with st.spinner("正在解析简历草稿..."):
                        normalized = parse_resume_to_json(
                            st.session_state[draft_key],
                            api_key,
                            base_url,
                            model,
                            json.dumps(selected_jd_data, ensure_ascii=False),
                        )
                        result_path = save_resume_result(selected_resume_path, normalized)
                    st.success(f"解析完成：{result_path.name}")
                except Exception as exc:
                    st.error(f"解析失败：{exc}")

def build_job_kanban_items() -> list[dict[str, Any]]:
    """Build a list of job cards with aggregated matching stats."""
    jd_options = list_jd_json_options()
    if not jd_options:
        return []

    resume_paths = find_resume_json_files()
    items: list[dict[str, Any]] = []

    for jd_option in jd_options:
        jd_path = jd_option["path"]
        jd_data = load_json_file(str(jd_path))

        # Try to load matching report for this JD
        report: dict[str, Any] = {}
        if resume_paths:
            try:
                report = ensure_report_for_jd(jd_path, force_refresh=False)
            except Exception:
                report = {}

        results = [r for r in (report.get("results", []) or []) if isinstance(r, dict)]
        matched_count = len(results)
        avg_score = (
            sum(float(r.get("total_score") or 0) for r in results) / matched_count
            if matched_count
            else 0.0
        )
        high_score_count = sum(1 for r in results if float(r.get("total_score") or 0) >= 85)
        medium_score_count = sum(
            1 for r in results if 70 <= float(r.get("total_score") or 0) < 85
        )

        # Determine status
        if matched_count == 0:
            status = "待招聘"
            status_color = "#94a3b8"
        elif avg_score >= 85:
            status = "候选人充足"
            status_color = "#2e9f4d"
        elif avg_score >= 70:
            status = "招聘中"
            status_color = "#d9981f"
        else:
            status = "亟需人才"
            status_color = "#d14343"

        skill_items = jd_data.get("技术技能要求", []) if isinstance(jd_data, dict) else []
        skill_count = len([s for s in skill_items if isinstance(s, dict)])
        competency_items = jd_data.get("胜任力要求", []) if isinstance(jd_data, dict) else []
        competency_count = len([c for c in competency_items if isinstance(c, dict)])

        job_type_value = jd_data.get("岗位类型", "UNKNOWN") if isinstance(jd_data, dict) else "UNKNOWN"
        if isinstance(job_type_value, dict):
            job_type = str(job_type_value.get("primary") or "UNKNOWN")
        else:
            job_type = str(job_type_value or "UNKNOWN")

        items.append(
            {
                "id": str(jd_path),
                "path": jd_path,
                "label": jd_option["label"],
                "job_title": jd_option["job_title"],
                "job_type": job_type,
                "level": str(jd_data.get("级别要求") or "未指定") if isinstance(jd_data, dict) else "未指定",
                "skill_count": skill_count,
                "competency_count": competency_count,
                "matched_count": matched_count,
                "avg_score": avg_score,
                "high_score_count": high_score_count,
                "medium_score_count": medium_score_count,
                "status": status,
                "status_color": status_color,
                "raw_jd": jd_data,
                "raw_report": report,
            }
        )

    # Sort: 亟需人才 first, then by avg_score ascending (most urgent first)
    status_priority = {"亟需人才": 0, "招聘中": 1, "待招聘": 2, "候选人充足": 3}
    items.sort(key=lambda x: (status_priority.get(x["status"], 9), x["avg_score"]))
    return items

def render_job_kanban_board() -> None:
    st.markdown(
        """
        <div class='app-title'>
            <div>
                <div class='title-left'>岗位招聘看板</div>
                <div class='title-desc'>一览所有在招岗位 · 左边快速扫描岗位状态 · 右边查看岗位详情与候选人</div>
                <div class='title-meta'>
                    <span class='title-chip'>Job Kanban</span>
                    <span class='title-chip'>Recruiting Pipeline</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Extra CSS for kanban
    st.markdown(
        """
        <style>
            .kanban-list {
                max-height: 780px;
                overflow-y: auto;
                padding-right: 4px;
            }
            .kanban-card {
                border: 1px solid #d6e4f5;
                border-left: 4px solid #2563eb;
                border-radius: 12px;
                padding: 10px 12px;
                margin-bottom: 8px;
                background: linear-gradient(140deg, #ffffff, #f7fbff);
                cursor: pointer;
                transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            }
            .kanban-card:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 22px rgba(30, 74, 146, 0.09);
            }
            .kanban-card.selected {
                border-left-color: #0ea5e9;
                border-color: #0ea5e9;
                background: linear-gradient(140deg, #f0f9ff, #e6f4ff);
                box-shadow: 0 10px 24px rgba(14, 165, 233, 0.12);
            }
            .kanban-card-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 8px;
            }
            .kanban-status {
                display: inline-block;
                font-size: 10px;
                color: #ffffff;
                border-radius: 99px;
                padding: 2px 8px;
                font-weight: 700;
            }
            .kanban-meta {
                display: flex;
                gap: 10px;
                margin-top: 6px;
                flex-wrap: wrap;
            }
            .kanban-meta-item {
                font-size: 11px;
                color: #536d89;
            }
            .kanban-meta-item b {
                color: #143963;
            }
            .kanban-score-bar {
                width: 100%;
                height: 6px;
                background: #ebf1f8;
                border-radius: 999px;
                overflow: hidden;
                margin-top: 6px;
            }
            .kanban-score-fill {
                height: 6px;
                border-radius: 999px;
            }
            .detail-section {
                margin-bottom: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    kanban_items = build_job_kanban_items()
    if not kanban_items:
        st.warning('未找到岗位数据，请先在「智能解析」页生成 JD。')
        return

    # Summary KPIs
    total_jobs = len(kanban_items)
    recruiting_count = sum(1 for item in kanban_items if item["status"] in {"招聘中", "亟需人才"})
    total_matched = sum(item["matched_count"] for item in kanban_items)
    avg_all = (
        sum(item["avg_score"] for item in kanban_items if item["matched_count"] > 0)
        / max(1, sum(1 for item in kanban_items if item["matched_count"] > 0))
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='kpi-box'><div class='kpi-label'>在招岗位总数</div>"
            f"<div class='kpi-value'>{total_jobs}</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='kpi-box'><div class='kpi-label'>招聘中 / 亟需人才</div>"
            f"<div class='kpi-value' style='color:#d9981f'>{recruiting_count}</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='kpi-box'><div class='kpi-label'>匹配候选总人次</div>"
            f"<div class='kpi-value'>{total_matched}</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='kpi-box'><div class='kpi-label'>全局平均匹配分</div>"
            f"<div class='kpi-value'>{avg_all:.1f}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Left: job list, Right: detail
    left_col, right_col = st.columns([1.1, 1.9])

    selected_key = "kanban_selected_job_id"
    valid_ids = {item["id"] for item in kanban_items}
    if st.session_state.get(selected_key) not in valid_ids:
        st.session_state[selected_key] = kanban_items[0]["id"]

    with left_col:
        st.markdown("<div class='card-title'>全部岗位（点击选择）</div>", unsafe_allow_html=True)
        st.markdown("<div class='kanban-list'>", unsafe_allow_html=True)

        for idx, item in enumerate(kanban_items):
            is_selected = item["id"] == st.session_state[selected_key]
            selected_class = " selected" if is_selected else ""
            status_bg = item["status_color"]
            score_fill_color = score_color(item["avg_score"]) if item["matched_count"] > 0 else "#cbd5e1"
            fill_width = int(min(item["avg_score"], 100)) if item["matched_count"] > 0 else 0

            st.markdown(
                f"""
                <div class='kanban-card{selected_class}'>
                    <div class='kanban-card-header'>
                        <div>
                            <div class='name-line'>
                                <span class='avatar'>{avatar_letter(item['job_title'])}</span>
                                <span>{item['job_title']}</span>
                            </div>
                            <div class='sub-line'>类型：{item['job_type']} · 级别：{item['level']}</div>
                        </div>
                        <span class='kanban-status' style='background:{status_bg};'>{item['status']}</span>
                    </div>
                    <div class='kanban-meta'>
                        <div class='kanban-meta-item'>候选人 <b>{item['matched_count']}</b></div>
                        <div class='kanban-meta-item'>技能要求 <b>{item['skill_count']}</b></div>
                        <div class='kanban-meta-item'>胜任力 <b>{item['competency_count']}</b></div>
                        <div class='kanban-meta-item'>高分(85+) <b style='color:#2e9f4d'>{item['high_score_count']}</b></div>
                    </div>
                    <div class='kanban-score-bar'>
                        <div class='kanban-score-fill' style='width:{fill_width}%;background:{score_fill_color};'></div>
                    </div>
                    <div class='sub-line' style='margin-top:3px;'>平均匹配分 {item['avg_score']:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"查看 {item['job_title']}",
                key=f"kanban_pick_{idx}",
                use_container_width=True,
            ):
                st.session_state[selected_key] = item["id"]
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        selected_item = next(item for item in kanban_items if item["id"] == st.session_state[selected_key])
        jd_data = selected_item["raw_jd"] if isinstance(selected_item["raw_jd"], dict) else {}
        report = selected_item["raw_report"] if isinstance(selected_item["raw_report"], dict) else {}

        # Job header card
        st.markdown(
            f"""
            <div class='card'>
                <div class='candidate-row'>
                    <div>
                        <div class='name-line'>
                            <span class='avatar' style='background:linear-gradient(140deg,#0ea5e9,#1d4ed8);'>{avatar_letter(selected_item['job_title'])}</span>
                            <span style='font-size:16px;'>{selected_item['job_title']}</span>
                        </div>
                        <div class='sub-line'>岗位类型：{selected_item['job_type']} · 级别要求：{selected_item['level']}</div>
                        <div class='sub-line'>状态：<span class='kanban-status' style='background:{selected_item['status_color']};'>{selected_item['status']}</span></div>
                    </div>
                    <div style='text-align:right;'>
                        <div class='score-num' style='color:{score_color(selected_item['avg_score']) if selected_item['matched_count'] > 0 else "#cbd5e1"};'>{selected_item['avg_score']:.1f}</div>
                        <div class='small-note'>平均匹配分</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("候选人数量", selected_item["matched_count"])
        with m2:
            st.metric("高分候选人(85+)", selected_item["high_score_count"])
        with m3:
            st.metric("技能要求数", selected_item["skill_count"])
        with m4:
            st.metric("胜任力要求数", selected_item["competency_count"])

        # JD details and matched candidates in tabs
        detail_tab_jd, detail_tab_candidates = st.tabs(["岗位需求详情", "匹配候选人"])

        with detail_tab_jd:
            st.markdown("##### 岗位基本信息")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**岗位名称：** {jd_data.get('岗位名称', selected_item['job_title'])}")
                st.markdown(f"**级别要求：** {jd_data.get('级别要求', '未指定')}")
                job_type_display = jd_data.get("岗位类型", {})
                if isinstance(job_type_display, dict):
                    st.markdown(f"**岗位类型：** {job_type_display.get('primary', 'UNKNOWN')}")
                else:
                    st.markdown(f"**岗位类型：** {job_type_display}")

            with info_col2:
                st.markdown(f"**技术技能要求：** {selected_item['skill_count']} 项")
                st.markdown(f"**胜任力要求：** {selected_item['competency_count']} 项")
                capability_items = jd_data.get("能力要求", []) if isinstance(jd_data, dict) else []
                st.markdown(f"**能力要求：** {len([c for c in capability_items if isinstance(c, dict)])} 项")

            st.divider()

            # Skill requirements
            st.markdown("##### 技术技能要求")
            skill_items = [s for s in jd_data.get("技术技能要求", []) if isinstance(s, dict)]
            if skill_items:
                skill_rows = []
                for s in skill_items:
                    skill_rows.append(
                        {
                            "技能ID": s.get("skill_id", ""),
                            "名称": s.get("name", ""),
                            "优先级": s.get("priority", ""),
                            "证据": str(s.get("evidence", ""))[:80],
                        }
                    )
                st.dataframe(skill_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("该技术岗位暂无明确的技术技能要求（可能以胜任力为主）。")

            st.markdown("##### 胜任力要求")
            comp_items = [c for c in jd_data.get("胜任力要求", []) if isinstance(c, dict)]
            if comp_items:
                comp_rows = []
                for c in comp_items:
                    comp_rows.append(
                        {
                            "胜任力ID": c.get("competency_id", ""),
                            "名称": c.get("name", ""),
                            "优先级": c.get("priority", ""),
                            "证据": str(c.get("evidence", ""))[:80],
                        }
                    )
                st.dataframe(comp_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无胜任力要求。")

            with st.expander("查看原始 JD JSON", expanded=False):
                st.json(jd_data)

        with detail_tab_candidates:
            results = [r for r in (report.get("results", []) or []) if isinstance(r, dict)]
            if not results:
                st.info('当前岗位暂无匹配候选人。请先在「人才发现」页刷新匹配数据，或解析更多简历。')
            else:
                results_sorted = sorted(
                    results,
                    key=lambda r: float(r.get("total_score") or 0),
                    reverse=True,
                )
                for rank, cand_result in enumerate(results_sorted, start=1):
                    cand_file = str(cand_result.get("candidate_file") or "")
                    cand_profile = load_json_file(cand_file) if cand_file else {}
                    cand_name = str(
                        cand_result.get("candidate_name")
                        or cand_profile.get("姓名")
                        or Path(cand_file).stem.replace("_parsed", "")
                        if cand_file
                        else "未知候选人"
                    )
                    if cand_name.startswith("candidate_"):
                        cand_name = cand_name.replace("candidate_", "")
                    cand_score = float(cand_result.get("total_score") or 0)
                    cand_level = str(cand_profile.get("级别") or cand_result.get("level_reason") or "未识别")
                    cand_type = str(cand_profile.get("profile_type", {}).get("primary") or "UNKNOWN")
                    exp_items = cand_profile.get("经验", [])
                    exp_summary = summarize_experience(exp_items)

                    st.markdown(
                        f"""
                        <div class='candidate-card'>
                            <div class='candidate-row'>
                                <div>
                                    <div class='name-line'>
                                        <span class='avatar'>{avatar_letter(cand_name)}</span>
                                        <span>#{rank} {cand_name}</span>
                                    </div>
                                    <div class='sub-line'>级别：{cand_level} · 画像：{cand_type} · {exp_summary}</div>
                                </div>
                                <div style='text-align:right;'>
                                    <div class='score-num' style='color:{score_color(cand_score)};'>{cand_score:.2f}</div>
                                    <div class='small-note'>匹配分</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

def render_unknown_skills_summary() -> None:
    st.markdown(
        """
        <div class='app-title'>
            <div>
                <div class='title-left'>技能发现</div>
                <div class='title-desc'>跟踪简历/JD解析中未覆盖到本体的技能，支持本体治理迭代</div>
                <div class='title-meta'>
                    <span class='title-chip'>Ontology Gap Analysis</span>
                    <span class='title-chip'>Skill Governance</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = build_unknown_skill_summary()
    rows = summary.get("rows", [])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("本体外技能总条目", int(summary.get("total_unknown_items", 0)))
    with k2:
        st.metric("去重后技能数", int(summary.get("unique_unknown_skills", 0)))
    with k3:
        st.metric("简历日志记录数", len(summary.get("resume_records", [])))
    with k4:
        st.metric("JD日志记录数", len(summary.get("jd_records", [])))

    if not rows:
        st.info("当前没有本体外技能记录。解析更多简历或JD后会自动累积。")
        return

    top_n = st.slider("查看 Top N 高频技能", min_value=10, max_value=200, value=30, step=10)
    keyword = st.text_input("关键词过滤（技能名/证据）", value="").strip().lower()

    filtered_rows = rows
    if keyword:
        filtered_rows = [
            row
            for row in rows
            if keyword in str(row.get("技能", "")).lower() or keyword in str(row.get("示例证据", "")).lower()
        ]

    st.markdown("#### 高频本体外技能")
    st.dataframe(filtered_rows[:top_n], use_container_width=True, hide_index=True)

    role = str(st.session_state.get("auth_role", "hr") or "hr")
    if role == "admin":
        st.markdown("#### 管理视角（Admin）")
        st.caption("建议动作：合并同义词、更新 technical_skill_abox、补充别名并回灌解析器。")
        st.markdown("##### 原始日志（简历）")
        if UNKNOWN_RESUME_SKILLS_PATH.exists():
            st.download_button(
                "下载简历本体外技能日志",
                data=UNKNOWN_RESUME_SKILLS_PATH.read_text(encoding="utf-8").encode("utf-8"),
                file_name=UNKNOWN_RESUME_SKILLS_PATH.name,
                mime="text/plain",
            )
        st.markdown("##### 原始日志（JD）")
        if UNKNOWN_JD_SKILLS_PATH.exists():
            st.download_button(
                "下载JD本体外技能日志",
                data=UNKNOWN_JD_SKILLS_PATH.read_text(encoding="utf-8").encode("utf-8"),
                file_name=UNKNOWN_JD_SKILLS_PATH.name,
                mime="text/plain",
            )

def render_parse_tools(api_key: str, base_url: str, model: str) -> None:
    st.markdown(
        """
        <div class='app-title'>
            <div>
                <div class='title-left'>智能解析</div>
                <div class='title-desc'>恢复解析工具：支持 JD 提取、简历解析与批量解析（JSON-only）</div>
                <div class='title-meta'>
                    <span class='title-chip'>JD Extraction</span>
                    <span class='title-chip'>Resume Parsing</span>
                    <span class='title-chip'>Batch Parse</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.warning("未检测到 OPENAI_API_KEY，LLM解析功能将不可用。请先配置环境变量或 secrets。")

    jd_description = st.text_area(
        "JD描述",
        height=160,
        placeholder="请输入岗位职责、任职要求、技能要求、职级要求等 JD 描述。",
    )

    if st.button("提取 JD 生成 JSON"):
        try:
            if not jd_description.strip():
                st.warning("请先输入 JD 描述。")
            else:
                with st.spinner("正在调用 LLM 提取 JD..."):
                    jd_normalized = parse_jd_to_json(jd_description, api_key, base_url, model)
                    jd_result_path = save_jd_result(jd_normalized, jd_description)

                st.session_state["jd_result"] = jd_normalized
                st.session_state["jd_result_path"] = str(jd_result_path)
                st.success(f"JD JSON 已保存：{jd_result_path.name}")
        except Exception as exc:
            st.error(f"JD 提取失败：{exc}")

    if st.session_state.get("jd_result"):
        jd_result = st.session_state["jd_result"]
        render_jd_result(jd_result)
        jd_result_path = Path(st.session_state.get("jd_result_path", "jd_extracted.json"))
        st.download_button(
            "下载 JD JSON",
            data=json.dumps(jd_result, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=jd_result_path.name,
            mime="application/json",
        )

    resume_files = list_resume_files()
    if not resume_files:
        st.info("请将简历文件放到 Talent_Matching_Web/resumes 目录下，支持 PDF、DOCX、TXT、Markdown。")
        return

    selected_label = st.selectbox(
        "选择要解析的简历",
        options=[str(path.relative_to(APP_DIR)) for path in resume_files],
    )
    selected_path = APP_DIR / selected_label

    with st.expander("查看简历原文", expanded=False):
        try:
            resume_text = read_resume_text(selected_path)
            st.text_area("简历文本", resume_text, height=300)
        except Exception as exc:
            st.error(f"读取简历失败：{exc}")
            return

    if st.button("开始解析", type="primary"):
        try:
            if not resume_text.strip():
                st.warning("简历文本为空，无法解析。")
                return

            with st.spinner("正在调用 LLM 解析简历..."):
                jd_result_context = st.session_state.get("jd_result")
                jd_context = json.dumps(jd_result_context, ensure_ascii=False) if jd_result_context else jd_description
                normalized = parse_resume_to_json(resume_text, api_key, base_url, model, jd_context)
                result_path = save_resume_result(selected_path, normalized)

            st.session_state["parsed_result"] = normalized
            st.session_state["result_path"] = str(result_path)
            st.success(f"解析完成，JSON 已保存：{result_path.name}")
        except Exception as exc:
            st.error(f"解析失败：{exc}")

    if st.session_state.get("parsed_result"):
        result = st.session_state["parsed_result"]
        render_result(result)
        json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "下载 JSON 结果",
            data=json_bytes,
            file_name=Path(st.session_state.get("result_path", "resume_parsed.json")).name,
            mime="application/json",
        )

    st.divider()
    st.markdown("#### 批量解析简历")
    st.caption("直接调用 batch_parse_resumes.py，一次性处理 resumes 目录下的多份简历（JSON-only）。")

    batch_input_default = str((APP_DIR / "resumes").resolve())
    with st.expander("批量解析参数", expanded=False):
        batch_input = st.text_input("输入路径", value=batch_input_default)
        batch_col1, batch_col2, batch_col3 = st.columns(3)
        with batch_col1:
            batch_workers = st.number_input("并发数", min_value=1, max_value=12, value=3, step=1)
        with batch_col2:
            batch_limit = st.number_input("最多处理份数（0 不限）", min_value=0, max_value=10000, value=0, step=1)
        with batch_col3:
            batch_recursive = st.checkbox("递归子目录", value=False)
        batch_force = st.checkbox("强制重新解析", value=False)
        batch_use_jd = st.checkbox("附加当前 JD 作为上下文", value=False)

    if st.button("批量解析简历", type="primary", use_container_width=False):
        try:
            jd_result_context = st.session_state.get("jd_result")
            jd_context = json.dumps(jd_result_context, ensure_ascii=False) if jd_result_context else jd_description
            if not batch_use_jd:
                jd_context = ""

            with st.spinner("正在批量解析简历，请稍候..."):
                proc = run_batch_resume_parser(
                    Path(batch_input).expanduser(),
                    api_key,
                    base_url,
                    model,
                    jd_context,
                    int(batch_workers),
                    batch_recursive,
                    batch_force,
                    int(batch_limit),
                )

            st.session_state["batch_parse_stdout"] = proc.stdout
            st.session_state["batch_parse_stderr"] = proc.stderr
            st.session_state["batch_parse_summary_path"] = str(BATCH_PARSE_SUMMARY_PATH)
            st.session_state["batch_parse_error_log_path"] = str(BATCH_PARSE_ERROR_LOG_PATH)

            if proc.returncode == 0:
                st.success(f"批量解析完成，汇总文件：{BATCH_PARSE_SUMMARY_PATH.name}")
            else:
                st.error(f"批量解析完成但有失败项，退出码：{proc.returncode}")

            if proc.stdout.strip():
                st.text_area("批量解析输出", proc.stdout, height=220)
            if proc.stderr.strip():
                st.text_area("批量解析错误输出", proc.stderr, height=180)

            if BATCH_PARSE_SUMMARY_PATH.exists():
                st.download_button(
                    "下载批量解析汇总",
                    data=BATCH_PARSE_SUMMARY_PATH.read_text(encoding="utf-8").encode("utf-8"),
                    file_name=BATCH_PARSE_SUMMARY_PATH.name,
                    mime="application/json",
                )
            if BATCH_PARSE_ERROR_LOG_PATH.exists():
                st.download_button(
                    "下载批量解析错误日志",
                    data=BATCH_PARSE_ERROR_LOG_PATH.read_text(encoding="utf-8").encode("utf-8"),
                    file_name=BATCH_PARSE_ERROR_LOG_PATH.name,
                    mime="text/plain",
                )
        except Exception as exc:
            st.error(f"批量解析失败：{exc}")

# =============================================================================
# 人才洞察
# =============================================================================

@st.cache_data(ttl=300)
def load_all_parsed_resumes() -> list[dict[str, Any]]:
    """Load all parsed resume JSONs from parsed_results directory."""
    resumes: list[dict[str, Any]] = []
    if not PARSED_RESULTS_DIR.exists():
        return resumes
    for fpath in sorted(PARSED_RESULTS_DIR.glob("*_parsed.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            data["_source_file"] = fpath.name
            resumes.append(data)
        except Exception:
            continue
    return resumes

# ──────────────────────────────────────────────────────────────────────
# 角色标签衍生（从 profile_type + 技能模式推导，dummy 数据方便演示）
# ──────────────────────────────────────────────────────────────────────
_PRIMARY_ROLE_MAP: dict[str, str] = {
    "TECHNICAL": "技术研发",
    "BA": "业务分析",
    "HYBRID": "技术+业务复合",
    "CONSULTING": "咨询顾问",
    "PROJECT_MANAGER": "项目管理",
}

_SECONDARY_ROLE_MAP: dict[str, str] = {
    "TECHNICAL": "技术研发",
    "BA": "业务分析",
    "CONSULTING": "咨询顾问",
    "PROJECT_MANAGER": "项目管理",
}

# 技能 ID → 角色标签（dummy 映射，用于演示）
_SKILL_TO_ROLE_TAG: dict[str, str] = {
    "SKILL_ETL": "数据工程师",
    "SKILL_DATA_WAREHOUSE": "数据工程师",
    "SKILL_ETL_ELT": "数据工程师",
    "SKILL_AWS_REDSHIFT": "数据工程师",
    "SKILL_SPARK": "数据工程师",
    "SKILL_LLM": "AI工程师",
    "SKILL_MACHINE_LEARNING": "AI工程师",
    "SKILL_DEEP_LEARNING": "AI工程师",
    "SKILL_RAG": "AI工程师",
    "SKILL_LANGCHAIN": "AI工程师",
    "SKILL_REACT": "前端开发",
    "SKILL_VUE": "前端开发",
    "SKILL_TYPESCRIPT": "前端开发",
    "SKILL_JAVA_CORE": "后端开发",
    "SKILL_SPRING_BOOT": "后端开发",
    "SKILL_GO": "后端开发",
    "SKILL_RUST": "后端开发",
    "SKILL_NODEJS": "全栈开发",
    "SKILL_REACT_NATIVE": "移动开发",
    "SKILL_DOCKER": "运维开发",
    "SKILL_KUBERNETES": "运维开发",
    "SKILL_CI_CD": "运维开发",
}

def _derive_role_tags(resume: dict[str, Any]) -> list[str]:
    """从简历数据衍生角色标签，返回去重后的标签列表。"""
    tags: list[str] = []
    seen: set[str] = set()

    pt = resume.get("profile_type")
    if isinstance(pt, dict):
        # primary → 主标签
        primary_raw = str(pt.get("primary") or "").strip().upper()
        primary_tag = _PRIMARY_ROLE_MAP.get(primary_raw)
        if primary_tag and primary_tag not in seen:
            tags.append(primary_tag)
            seen.add(primary_tag)

        # secondary → 副标签
        secondary = pt.get("secondary")
        if isinstance(secondary, list):
            for s in secondary:
                s_tag = _SECONDARY_ROLE_MAP.get(str(s).strip().upper())
                if s_tag and s_tag not in seen:
                    tags.append(s_tag)
                    seen.add(s_tag)
        elif isinstance(secondary, str):
            s_tag = _SECONDARY_ROLE_MAP.get(secondary.strip().upper())
            if s_tag and s_tag not in seen:
                tags.append(s_tag)
                seen.add(s_tag)

    # 从技能 ID 衍生额外角色标签
    tech_skills = resume.get("技术技能") or []
    if isinstance(tech_skills, list):
        for item in tech_skills:
            if isinstance(item, dict):
                skill_id = str(item.get("skill_id") or "").strip()
                role_tag = _SKILL_TO_ROLE_TAG.get(skill_id)
                if role_tag and role_tag not in seen:
                    tags.append(role_tag)
                    seen.add(role_tag)

    # 兜底：如果没有任何标签，给一个默认值
    if not tags:
        tags.append("其他")

    return tags

@st.cache_data(ttl=300)
def compute_overview_stats(resumes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregated statistics from all parsed resumes."""
    total = len(resumes)

    # Collect all skills per category
    all_tech_skills: list[str] = []
    all_capabilities: list[str] = []
    all_competencies: list[str] = []
    all_domains: list[str] = []
    all_industries: list[str] = []
    level_counter: Counter = Counter()
    profile_types: Counter = Counter()
    role_tags: Counter = Counter()  # 新增：角色标签统计
    experience_years: list[float] = []

    for r in resumes:
        # 收集角色标签
        for tag in _derive_role_tags(r):
            role_tags[tag] += 1
        tech = r.get("技术技能") or []
        if isinstance(tech, list):
            for item in tech:
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("skill_name") or "").strip()
                    if name:
                        all_tech_skills.append(name)
                elif isinstance(item, str):
                    all_tech_skills.append(item.strip())

        cap = r.get("能力") or []
        if isinstance(cap, list):
            for item in cap:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    if name:
                        all_capabilities.append(name)
                elif isinstance(item, str):
                    all_capabilities.append(item.strip())

        comp = r.get("胜任力") or []
        if isinstance(comp, list):
            for item in comp:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    if name:
                        all_competencies.append(name)
                elif isinstance(item, str):
                    all_competencies.append(item.strip())

        dom = r.get("领域") or []
        if isinstance(dom, list):
            for item in dom:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    if name:
                        all_domains.append(name)
                elif isinstance(item, str):
                    all_domains.append(item.strip())

        ind = r.get("行业知识") or []
        if isinstance(ind, list):
            for item in ind:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    if name:
                        all_industries.append(name)
                elif isinstance(item, str):
                    all_industries.append(item.strip())

        level = _normalize_level_name(r.get("级别"))
        if level:
            level_counter[level] += 1

        pt = r.get("profile_type")
        if isinstance(pt, dict):
            pt_val = str(pt.get("primary") or pt.get("type") or "").strip()
            if pt_val:
                profile_types[pt_val] += 1
        elif isinstance(pt, str) and pt.strip():
            profile_types[pt.strip()] += 1

        # estimate years from experience
        exp = r.get("经验") or []
        if isinstance(exp, list):
            for e in exp:
                if isinstance(e, dict):
                    dur = e.get("duration") or e.get("时间") or ""
                else:
                    dur = str(e) if e else ""
                years = _extract_years(str(dur))
                if years:
                    experience_years.append(years)

    unique_tech = len(set(all_tech_skills))
    unique_cap = len(set(all_capabilities))
    unique_comp = len(set(all_competencies))
    unique_dom = len(set(all_domains))

    return {
        "total_resumes": total,
        "unique_tech_skills": unique_tech,
        "unique_capabilities": unique_cap,
        "unique_competencies": unique_comp,
        "unique_domains": unique_dom,
        "all_tech_skills": all_tech_skills,
        "all_capabilities": all_capabilities,
        "all_competencies": all_competencies,
        "all_domains": all_domains,
        "all_industries": all_industries,
        "level_counter": dict(level_counter),
        "profile_types": dict(profile_types),
        "role_tags": dict(role_tags),
        "avg_experience_years": round(sum(experience_years) / len(experience_years), 1) if experience_years else 0,
    }

def _extract_years(dur_str: str) -> float | None:
    """Extract number of years from a duration string like '3 years' or '2年'."""
    if not dur_str:
        return None
    match = re.search(r"(\d+\.?\d*)\s*(year|yr|年)", dur_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Fallback: if it starts with a number
    match = re.search(r"^(\d+\.?\d*)", dur_str.strip())
    if match:
        return float(match.group(1))
    return None

# ──────────────────────────────────────────────────────────────────────
# 级别名规范化：把 "Sr Manager" / "level 8" 等统一为 "Level 6" / "Level 8"
# ──────────────────────────────────────────────────────────────────────
_LEVEL_ALIAS_MAP = {
    "sr manager": "Level 6",
    "sr. manager": "Level 6",
    "senior manager": "Level 6",
    "senior mgr": "Level 6",
    "manager": "Level 7",
}

def _normalize_level_name(raw: Any) -> str:
    """把任意写法统一为 'Level N'（或 'Manager' 等特殊标签）。"""
    s = str(raw or "").strip()
    if not s:
        return ""
    key = s.lower()
    if key in _LEVEL_ALIAS_MAP:
        return _LEVEL_ALIAS_MAP[key]
    # 修正 "level 8" / "Level 8" 之类的大小写不一致 → 规范成 "Level N"
    m = re.match(r"^level\s*(\d+)$", key)
    if m:
        return f"Level {m.group(1)}"
    return s  # 其他情况保持原文

def render_overview_dashboard() -> None:
    """人才洞察 - KPI卡片 + 技能分布图 + 级别分布图"""
    st.markdown(
        """
        <div class="app-title">
            <div>
                <div class="title-left">人才洞察</div>
                <div class="title-desc">人才库全景视图 · 实时数据统计与分析</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    resumes = load_all_parsed_resumes()
    if not resumes:
        st.warning("暂无已解析的简历数据。请先在「智能解析」页面上传并解析简历。")
        return

    stats = compute_overview_stats(resumes)

    # ── KPI Cards ──
    kpi_cols = st.columns(5)
    kpis = [
        ("👥 人才总数", stats["total_resumes"], ""),
        ("🛠️ 技术技能", stats["unique_tech_skills"], "种"),
        ("💡 能力项", stats["unique_capabilities"], "种"),
        ("🎯 胜任力", stats["unique_competencies"], "种"),
        ("🌐 领域", stats["unique_domains"], "个"),
    ]
    for col, (label, value, unit) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi-box">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}<span style="font-size:0.5em;font-weight:400;">{unit}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ────────── 三个图形数字一致性约束 ──────────
    # 整体人才缺口作为单一来源（业务侧 mock），保证下方三处图形显示的数字一致：
    #   ① 岗位需求热度与人才库覆盖（每岗位缺口相加 = TOTAL_TALENT_GAP）
    #   ② 级别供需对比（总缺口 = TOTAL_TALENT_GAP）
    #   ③ 供需匹配总览（总缺口 = TOTAL_TALENT_GAP）
    TOTAL_TALENT_GAP = 7

    def _distribute_gap(n: int, total: int) -> list[int]:
        """把 `total` 均匀分配到 `n` 个岗位（每个至少 1，总和 = total）。"""
        if n <= 0:
            return []
        if total < n:
            # 缺口不足以每个岗位都分到 1 人时，全部记在第一个岗位
            return [total] + [0] * (n - 1)
        base = total // n
        remainder = total - base * n
        return [base + (1 if i < remainder else 0) for i in range(n)]

    # ── 预先计算岗位需求数据（供图表和汇总使用） ──
    import glob
    jd_files = glob.glob(str(PARSED_RESULTS_DIR / "jd_*.json"))
    job_demand: list[dict[str, Any]] = []

    def _names(raw) -> set[str]:
        out: set[str] = set()
        if isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    nm = it.get("name") or it.get("matched_term") or ""
                    if nm:
                        out.add(str(nm))
                elif it:
                    out.add(str(it))
        return out

    # 记录至少匹配一个岗位的简历（去重，避免跨岗位重复统计）
    unique_matched_indices: set[int] = set()

    # 第一遍：先计算出每个岗位的可匹配人数
    job_matched_tmp: list[tuple[str, int]] = []
    for jf in jd_files:
        try:
            with open(jf, encoding="utf-8") as fp:
                jd = json.loads(fp.read())
        except Exception:
            continue
        job_name = jd.get("岗位名称") or os.path.splitext(os.path.basename(jf))[0]

        # 计算现有简历库对该 JD 的可覆盖数（基于技能重合度 >= 10% 视为可匹配）
        tech_req = jd.get("技术技能要求", jd.get("required_skills", []))
        comp_req = jd.get("胜任力要求", [])
        req_names: set[str] = set()
        req_names |= _names(tech_req)
        req_names |= _names(comp_req)

        # 计算每个候选人的匹配度；合格池 = Jaccard >= 0.10 的人数
        matched = 0
        if req_names:
            for idx, r in enumerate(resumes):
                r_skills: set[str] = set()
                for fld in ("技术技能", "能力", "胜任力"):
                    v = r.get(fld) or []
                    r_skills |= _names(v)
                if not r_skills:
                    continue
                # Jaccard 相似度
                score = len(r_skills & req_names) / max(len(r_skills | req_names), 1)
                if score >= 0.10:
                    matched += 1
                    unique_matched_indices.add(idx)

        job_matched_tmp.append((job_name, matched))

    # 第二遍：把整体缺口均匀分配到各岗位，保证三个图形的缺口数字一致
    gap_dist = _distribute_gap(len(job_matched_tmp), TOTAL_TALENT_GAP)
    for (job_name, matched), gap in zip(job_matched_tmp, gap_dist):
        headcount = matched + gap
        job_demand.append({
            "岗位": job_name,
            "需求人数": headcount,
            "可匹配简历": matched,
            "缺口": gap,
        })

    # 岗位需求汇总统计（供下方衔接使用）
    # 三个图形必须共用同一份缺口，因此这里直接由 job_demand 求和，不再用去重人数计算
    # 保证：①左图 sum(缺口) == ②右图(业务需求-简历库) == ③右下卡片"总缺口"
    total_job_demand = sum(item["需求人数"] for item in job_demand)
    total_unique_matched = len(unique_matched_indices)
    total_gap = sum(item["缺口"] for item in job_demand)
    # 若没有任何 JD，给一个兜底显示值（仍等于 TOTAL_TALENT_GAP）
    if not job_demand:
        total_gap = TOTAL_TALENT_GAP

    # ── Charts Row ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("<div class='card'><div class='card-title'>📊 岗位需求热度与人才库覆盖</div>", unsafe_allow_html=True)

        if not job_demand:
            st.info("暂无 JD 数据")
        else:
            # 按需求人数降序
            job_demand.sort(key=lambda x: x["需求人数"], reverse=True)
            df_jobs = pd.DataFrame(job_demand)

            fig_jobs = px.bar(
                df_jobs,
                x="需求人数",
                y="岗位",
                orientation="h",
                color_discrete_sequence=["#E0E0E0"],
                text="需求人数",
            )
            # 叠加 可匹配简历（蓝色，覆盖在灰色条上）
            fig_jobs.add_trace(
                go.Bar(
                    x=df_jobs["可匹配简历"],
                    y=df_jobs["岗位"],
                    orientation="h",
                    marker_color="#1976D2",
                    text=df_jobs["可匹配简历"],
                    name="可匹配简历",
                    textposition="inside",
                )
            )
            # 缺口徽标
            for i, row in df_jobs.iterrows():
                gap = row["缺口"]
                if gap > 0:
                    fig_jobs.add_annotation(
                        x=row["需求人数"],
                        y=row["岗位"],
                        text=f" 缺口 {gap}",
                        showarrow=False,
                        xanchor="left",
                        font=dict(color="#D32F2F", size=11),
                    )
            fig_jobs.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=380,
                showlegend=True,
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="人数",
            )
            fig_jobs.update_traces(textposition="outside", selector=dict(name="需求人数") or {"type": "bar"})
            st.plotly_chart(fig_jobs, use_container_width=True)
            st.caption(f"📌 业务方需求（灰色）vs 简历库可匹配（蓝色） | 简历库共 **{stats['total_resumes']}** 份简历")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("<div class='card'><div class='card-title'>📈 级别供需对比</div>", unsafe_allow_html=True)

        # 简历库供给（真实） - 再次规范化以防万一
        raw_supply = stats["level_counter"]
        supply: dict[str, int] = {}
        for k, v in raw_supply.items():
            nk = _normalize_level_name(k)
            if nk:
                supply[nk] = supply.get(nk, 0) + int(v)
        # 业务方需求（mock）：总缺口 = TOTAL_TALENT_GAP，与左图、右下卡片完全一致
        leveled_supply = sum(supply.values()) if supply else 0
        total_resumes_for_chart = max(int(stats.get("total_resumes", 0)), leveled_supply)

        def _mock_level_demand() -> dict[str, int]:
            target_total = total_resumes_for_chart + TOTAL_TALENT_GAP
            # 保持原有分布比例（中高级最多，初级/资深较少）
            ratios = [0.06, 0.11, 0.24, 0.275, 0.175, 0.10, 0.04]
            levels = ["Level 6", "Level 7", "Level 8", "Level 9", "Level 10", "Level 11", "Level 12"]
            values = [max(1, int(target_total * r)) for r in ratios]
            # 微调最后一项使总和精确匹配 target_total
            diff = target_total - sum(values)
            values[-1] += diff
            return dict(zip(levels, values))

        demand = _mock_level_demand()

        # 合并所有出现的 level
        all_levels = list(dict.fromkeys(list(supply.keys()) + list(demand.keys())))
        # 按数字排序
        def _lvl_sort_key(lv: str) -> tuple[int, str]:
            num = 0
            for tok in lv.split():
                if tok.isdigit():
                    num = int(tok)
                    break
            return (num, lv)

        all_levels.sort(key=_lvl_sort_key)

        supply_vals = [int(supply.get(lv, 0)) for lv in all_levels]
        demand_vals = [int(demand.get(lv, 0)) for lv in all_levels]

        if not all_levels:
            st.info("暂无级别数据")
        else:
            df_lvl = pd.DataFrame({
                "级别": all_levels,
                "简历库供给": supply_vals,
                "业务方需求": demand_vals,
            })
            fig_lvl = go.Figure()
            fig_lvl.add_trace(go.Bar(
                name="简历库供给",
                x=df_lvl["级别"],
                y=df_lvl["简历库供给"],
                marker_color="#1976D2",
                text=df_lvl["简历库供给"],
                textposition="outside",
            ))
            fig_lvl.add_trace(go.Bar(
                name="业务方需求",
                x=df_lvl["级别"],
                y=df_lvl["业务方需求"],
                marker_color="#FFA726",
                text=df_lvl["业务方需求"],
                textposition="outside",
            ))
            # 在每组上方画一条 marker 显示缺口/盈余
            for i, lv in enumerate(all_levels):
                diff = supply_vals[i] - demand_vals[i]
                if diff != 0:
                    color = "#2E7D32" if diff > 0 else "#C62828"
                    sign = "+" if diff > 0 else ""
                    fig_lvl.add_annotation(
                        x=lv,
                        y=max(supply_vals[i], demand_vals[i]) + 3,
                        text=f"{sign}{diff}",
                        showarrow=False,
                        font=dict(color=color, size=11, weight="bold"),
                    )
            fig_lvl.update_layout(
                barmode="group",
                margin=dict(t=30, b=10, l=10, r=10),
                height=380,
                xaxis_title="级别",
                yaxis_title="人数",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lvl, use_container_width=True)
            # 业务洞察小结
            # 简历库总数以 stats['total_resumes'] 为准（有级别字段的简历 <= 总数）
            total_supply = total_resumes_for_chart
            total_demand = sum(demand_vals)
            gap_total = total_demand - total_supply
            leveled_supply = sum(supply_vals)
            leveled_note = (
                f"（级别已识别 **{leveled_supply}** 人）"
                if leveled_supply < total_supply
                else ""
            )
            if gap_total > 0:
                st.caption(
                    f"💼 简历库 **{total_supply}** 人 vs 业务需求 **{total_demand}** 人 · "
                    f"<span style='color:#C62828'>**缺口 {gap_total} 人**</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    f"💼 简历库 **{total_supply}** 人 vs 业务需求 **{total_demand}** 人 · "
                    f"<span style='color:#2E7D32'>**富余 {-gap_total} 人**</span>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 供需匹配总览（已替换为下方的"图表数据来源说明"表） ──
    # 原"供需匹配总览"卡片已删除，避免与下方数据来源表重复展示同一组数字。

    # ── ① 支撑左图：岗位粒度 —— 需求人数 vs 现有符合人数 对比表 ──
    st.markdown("<div class='card'><div class='card-title'>📊 岗位需求 vs 现有符合技能人数（支撑左图）</div>", unsafe_allow_html=True)
    st.caption(
        "对应左图《岗位需求热度与人才库覆盖》。每一行 = 左图的一根横向条形："
        "`需求人数`（灰色条长度）= 业务方该岗位所需 headcount；"
        "`现有符合技能人数`（蓝色条长度）= 简历库中与该岗位技能重合度 ≥ 10% 的人数；"
        "`缺口` = 两者之差。汇总行展示了所有岗位求和后的总量。"
    )

    job_rows: list[dict[str, Any]] = []
    for item in job_demand:
        job_rows.append(
            {
                "岗位": item["岗位"],
                "岗位需求人数": int(item["需求人数"]),
                "现有符合技能人数": int(item["可匹配简历"]),
                "缺口（需求−符合）": int(item["缺口"]),
                "覆盖率": f"{(item['可匹配简历'] / max(item['需求人数'], 1)) * 100:.1f}%",
            }
        )
    # 汇总行
    sum_demand = sum(int(item["需求人数"]) for item in job_demand)
    sum_matched = sum(int(item["可匹配简历"]) for item in job_demand)
    sum_gap = sum(int(item["缺口"]) for item in job_demand)
    job_rows.append(
        {
            "岗位": "📌 全部岗位汇总",
            "岗位需求人数": sum_demand,
            "现有符合技能人数": sum_matched,
            "缺口（需求−符合）": sum_gap,
            "覆盖率": f"{(sum_matched / max(sum_demand, 1)) * 100:.1f}%",
        }
    )
    # 简历库基准
    job_rows.append(
        {
            "岗位": "👥 简历库总人数",
            "岗位需求人数": "—",
            "现有符合技能人数": int(stats["total_resumes"]),
            "缺口（需求−符合）": "—",
            "覆盖率": f"{total_unique_matched / max(int(stats['total_resumes']), 1) * 100:.1f}%",
        }
    )
    job_rows.append(
        {
            "岗位": "✅ 至少匹配 1 个岗位（去重）",
            "岗位需求人数": "—",
            "现有符合技能人数": total_unique_matched,
            "缺口（需求−符合）": "—",
            "覆盖率": "—",
        }
    )

    df_jobs = pd.DataFrame(job_rows)
    df_jobs.index = range(1, len(df_jobs) + 1)
    st.dataframe(df_jobs, use_container_width=True, hide_index=False)
    st.caption(
        "💡 **关键观察**：左侧各岗位的「现有符合技能人数」加总后会大于「至少匹配 1 个岗位（去重）」，"
        "原因是同一候选人可能同时命中多个岗位（候选人复用）。"
        "覆盖率越低的岗位，人才缺口越严重。"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── ② 支撑右图：技能粒度 —— 单一技能符合人数 + 岗位多技能组合说明 ──
    st.markdown("<div class='card'><div class='card-title'>🎯 技能符合人数 × 岗位多技能组合（支撑右图）</div>", unsafe_allow_html=True)
    st.caption(
        "对应右图《级别供需对比》背后的技能维度拆解。"
        "岗位并不是要「单一技能」，而是要求**多项技能同时具备**。"
        "下表先展示简历库中**每项单一技能**的符合人数（候选人只要会这一项就算 1），"
        "再展示各岗位的**技能组合要求**及**同时具备全部技能**的真实符合人数，"
        "说明为何「单一技能很多」但「岗位符合人数很少」。"
    )

    # 计算每个 JD 的技能要求清单 + 同时具备全部技能的候选人数
    skill_combo_rows: list[dict[str, Any]] = []
    all_req_skills: list[str] = []

    for jf in jd_files:
        try:
            with open(jf, encoding="utf-8") as fp:
                jd = json.loads(fp.read())
        except Exception:
            continue
        job_name = jd.get("岗位名称") or os.path.splitext(os.path.basename(jf))[0]
        tech_req = jd.get("技术技能要求", jd.get("required_skills", []))
        comp_req = jd.get("胜任力要求", [])
        req_names = _names(tech_req) | _names(comp_req)
        if not req_names:
            continue
        all_req_skills.extend(sorted(req_names))

        # 同时具备全部技能的候选人数（严格 AND 匹配）
        all_have = 0
        partial_have = 0  # 至少 50% 技能命中的候选人数
        half = max(len(req_names) // 2, 1)
        for r in resumes:
            r_skills: set[str] = set()
            for fld in ("技术技能", "能力", "胜任力"):
                r_skills |= _names(r.get(fld) or [])
            if not r_skills:
                continue
            hit = len(r_skills & req_names)
            if hit == len(req_names):
                all_have += 1
            if hit >= half:
                partial_have += 1

        # 单技能覆盖（取该岗位最核心的 1 项技能作为代表）
        top_skill = sorted(req_names, key=lambda s: -sum(1 for r in resumes if s in (
            _names(r.get("技术技能") or []) | _names(r.get("能力") or []) | _names(r.get("胜任力") or [])
        )))[:1]
        top_skill = top_skill[0] if top_skill else sorted(req_names)[0] if req_names else "—"
        top_skill_have = sum(
            1 for r in resumes if top_skill in (
                _names(r.get("技术技能") or []) | _names(r.get("能力") or []) | _names(r.get("胜任力") or [])
            )
        )

        skill_combo_rows.append(
            {
                "岗位": job_name,
                "技能组合要求数": len(req_names),
                "单技能（最热 1 项）符合人数": int(top_skill_have),
                f"≥50% 技能符合人数": int(partial_have),
                "全部技能符合人数（严格 AND）": int(all_have),
                "覆盖率（AND）": f"{(all_have / max(int(stats['total_resumes']), 1)) * 100:.1f}%",
            }
        )

    # 统计所有出现过的技能 + 其单一技能符合人数（与"热门技能 Top 20"区分，这里只取 JD 实际要求的技能）
    skill_counter_top = Counter()
    for r in resumes:
        r_skills = _names(r.get("技术技能") or []) | _names(r.get("能力") or []) | _names(r.get("胜任力") or [])
        for s in r_skills:
            if s in set(all_req_skills):
                skill_counter_top[s] += 1

    # 拼接成两段：上半段 = 技能符合人数 Top；下半段 = 各岗位多技能组合
    skill_rows: list[dict[str, Any]] = []
    skill_rows.append({"分组": "—— 上：JD 实际要求的技能（单一技能符合人数） ——", "项目": "", "人数": "", "说明": ""})
    for sk, cnt in skill_counter_top.most_common(15):
        skill_rows.append(
            {
                "分组": "📚 单一技能符合",
                "项目": sk,
                "人数": int(cnt),
                "说明": "简历库中具备该项技能的候选人数",
            }
        )

    skill_rows.append({"分组": "—— 下：各岗位需要多技能组合（AND 匹配） ——", "项目": "", "人数": "", "说明": ""})
    for row in skill_combo_rows:
        skill_rows.append(
            {
                "分组": "🧩 多技能组合",
                "项目": row["岗位"],
                "人数": f"{row['全部技能符合人数（严格 AND）']} 人 / 需 {row['技能组合要求数']} 项",
                "说明": f"单技能最高 {row['单技能（最热 1 项）符合人数']} 人 · ≥50% 命中 {row['≥50% 技能符合人数']} 人 · 覆盖率 {row['覆盖率（AND）']}",
            }
        )

    if skill_rows:
        df_sk = pd.DataFrame(skill_rows)
        df_sk.index = range(1, len(df_sk) + 1)
        st.dataframe(df_sk, use_container_width=True, height=min(60 + 30 * len(df_sk), 560), hide_index=False)
        st.caption(
            "💡 **关键观察**：单一技能符合人数很多（例如会 Python 的人数），"
            "但**同时具备该岗位全部技能**的候选人却很少（AND 匹配人数骤降）。"
            "这正是右图显示各岗位缺口、以及左图单岗位缺口存在的根本原因 —— "
            "招聘不是招「一个会某技能的人」，而是招「一个会整套技能组合的人」。"
        )
    else:
        st.info("暂无 JD 技能要求数据。")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Top Skills Table ──
    st.markdown("<div class='card'><div class='card-title'>热门技能 Top 20</div>", unsafe_allow_html=True)
    tech_counter = Counter(stats["all_tech_skills"])
    top_skills = tech_counter.most_common(20)
    if top_skills:
        df_skills = pd.DataFrame(top_skills, columns=["技能名称", "出现次数"])
        df_skills["占比"] = (df_skills["出现次数"] / stats["total_resumes"] * 100).round(1)
        df_skills["占比"] = df_skills["占比"].astype(str) + "%"
        df_skills.index = range(1, len(df_skills) + 1)
        st.dataframe(df_skills, use_container_width=True, height=360)
    else:
        st.info("暂无技能数据")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Industries & Profile Types ──
    extra_col1, extra_col2 = st.columns(2)
    with extra_col1:
        st.markdown("<div class='card'><div class='card-title'>行业知识分布</div>", unsafe_allow_html=True)
        if stats["all_industries"]:
            ind_counter = Counter(stats["all_industries"])
            df_ind = pd.DataFrame(ind_counter.most_common(10), columns=["行业", "人数"])
            st.dataframe(df_ind, use_container_width=True, hide_index=True)
        else:
            st.info("暂无行业数据")
        st.markdown("</div>", unsafe_allow_html=True)

    with extra_col2:
        st.markdown("<div class='card'><div class='card-title'>Profile类型分布</div>", unsafe_allow_html=True)
        if stats["profile_types"]:
            df_pt = pd.DataFrame(
                [(k, v) for k, v in stats["profile_types"].items()],
                columns=["Profile类型", "人数"],
            )
            st.dataframe(df_pt, use_container_width=True, hide_index=True)
        else:
            st.info("暂无Profile类型数据")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Skill Gap Section ──
    st.markdown("---")
    with st.expander("📉 技能缺口分析", expanded=False):
        jd_options = list_jd_json_options()
        if jd_options:
            gap_jd_label = st.selectbox(
                "选择 JD 查看技能缺口",
                options=[item["label"] for item in jd_options],
                key="overview_gap_jd",
            )
            gap_jd = next(item for item in jd_options if item["label"] == gap_jd_label)
            gap_jd_data = load_json_file(str(gap_jd["path"]))
            gap_result = compute_skill_gap(gap_jd_data, resumes)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("JD技能需求数", len(gap_result["jd_skills"]))
            col_b.metric("已覆盖技能数", len(gap_result["covered_skills"]))
            col_c.metric("覆盖率", f"{gap_result['coverage_rate']:.1f}%",
                         delta=f"{len(gap_result['uncovered_skills'])}项缺口" if gap_result["uncovered_skills"] else "完美覆盖",
                         delta_color="inverse")

            if gap_result["skill_coverage"]:
                df_gap = pd.DataFrame([
                    {"技能": k, "匹配人数": v, "覆盖率": f"{v / gap_result['total_resumes'] * 100:.1f}%",
                     "状态": "✅ 已覆盖" if v > 0 else "⚠️ 缺口"}
                    for k, v in gap_result["skill_coverage"].items()
                ]).sort_values("匹配人数", ascending=False)

                fig_gap = px.bar(
                    df_gap, x="技能", y="匹配人数", color="状态",
                    color_discrete_map={"✅ 已覆盖": "#4CAF50", "⚠️ 缺口": "#F44336"},
                    text="匹配人数",
                )
                fig_gap.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340, xaxis_tickangle=-30)
                fig_gap.update_traces(textposition="outside")
                st.plotly_chart(fig_gap, use_container_width=True)

                st.dataframe(df_gap, use_container_width=True, hide_index=True)

                # CSV 导出按钮
                csv = df_gap.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 导出缺口分析报告CSV",
                    data=csv,
                    file_name=f"skill_gap_{gap_jd_label}.csv",
                    mime="text/csv",
                )
        else:
            st.info("暂无可分析的JD，请先解析JD。")

# =============================================================================
# 人才评估
# =============================================================================

@st.cache_data(ttl=300)
def load_candidate_comparison_data() -> list[dict[str, Any]]:
    """Load parsed resume summaries for candidate comparison selector."""
    resumes = load_all_parsed_resumes()
    candidates: list[dict[str, Any]] = []
    for r in resumes:
        name = str(r.get("姓名") or "").strip() or r.get("_source_file", "未知")
        level = str(r.get("级别") or "").strip() or "未识别"
        # count skills
        tech = r.get("技术技能") or []
        tech_count = len(tech) if isinstance(tech, list) else 0
        cap = r.get("能力") or []
        cap_count = len(cap) if isinstance(cap, list) else 0
        comp = r.get("胜任力") or []
        comp_count = len(comp) if isinstance(comp, list) else 0
        dom = r.get("领域") or []
        dom_count = len(dom) if isinstance(dom, list) else 0

        # get skill names for radar
        tech_names = _extract_names(tech)
        cap_names = _extract_names(cap)
        comp_names = _extract_names(comp)
        dom_names = _extract_names(dom)

        candidates.append({
            "name": name,
            "level": level,
            "source": r.get("_source_file", ""),
            "tech_count": tech_count,
            "cap_count": cap_count,
            "comp_count": comp_count,
            "dom_count": dom_count,
            "tech_names": tech_names,
            "cap_names": cap_names,
            "comp_names": comp_names,
            "dom_names": dom_names,
        })
    return candidates

def _extract_names(items: list) -> list[str]:
    """Extract names from a list of dicts or strings."""
    names: list[str] = []
    if not isinstance(items, list):
        return names
    for item in items:
        if isinstance(item, dict):
            n = str(item.get("name") or item.get("skill_name") or "").strip()
        else:
            n = str(item or "").strip()
        if n:
            names.append(n)
    return names

def render_candidate_comparison() -> None:
    """人才评估 - 多选候选人 + 雷达图 + 对照表"""
    st.markdown(
        """
        <div class="app-title">
            <div>
                <div class="title-left">人才评估</div>
                <div class="title-desc">多维雷达图对比 · 技能矩阵 · 胜任力对比</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidates = load_candidate_comparison_data()
    if not candidates:
        st.warning("暂无已解析的简历数据。")
        return

    # Multi-select
    candidate_options = [f"{c['name']} ({c['level']})" for c in candidates]
    selected = st.multiselect(
        "选择需要对比的候选人（建议 2-5 人）",
        options=candidate_options,
        default=candidate_options[: min(3, len(candidate_options))],
        max_selections=6,
    )

    if not selected:
        st.info("请从上方选择至少 2 位候选人开始对比")
        return

    if len(selected) < 2:
        st.info("请至少选择 2 位候选人才能进行对比")
        return

    # Get selected candidates data
    selected_data = [c for c in candidates if f"{c['name']} ({c['level']})" in selected]

    # ── Radar Chart ──
    st.markdown("<div class='card'><div class='card-title'>能力雷达图</div>", unsafe_allow_html=True)

    # Compute max for normalization
    max_tech = max((c["tech_count"] for c in selected_data), default=1)
    max_cap = max((c["cap_count"] for c in selected_data), default=1)
    max_comp = max((c["comp_count"] for c in selected_data), default=1)
    max_dom = max((c["dom_count"] for c in selected_data), default=1)

    categories = ["技术技能", "能力", "胜任力", "领域知识"]

    fig_radar = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, c in enumerate(selected_data):
        norm_tech = c["tech_count"] / max_tech * 100 if max_tech else 0
        norm_cap = c["cap_count"] / max_cap * 100 if max_cap else 0
        norm_comp = c["comp_count"] / max_comp * 100 if max_comp else 0
        norm_dom = c["dom_count"] / max_dom * 100 if max_dom else 0

        fig_radar.add_trace(go.Scatterpolar(
            r=[norm_tech, norm_cap, norm_comp, norm_dom],
            theta=categories,
            fill="toself",
            name=f"{c['name']}",
            line_color=colors[i % len(colors)],
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=60, l=40, r=40),
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Bar Chart Comparison ──
    st.markdown("<div class='card'><div class='card-title'>技能数量对比</div>", unsafe_allow_html=True)
    fig_bar = go.Figure()
    bar_names = [c["name"] for c in selected_data]
    fig_bar.add_trace(go.Bar(name="技术技能", x=bar_names, y=[c["tech_count"] for c in selected_data]))
    fig_bar.add_trace(go.Bar(name="能力", x=bar_names, y=[c["cap_count"] for c in selected_data]))
    fig_bar.add_trace(go.Bar(name="胜任力", x=bar_names, y=[c["comp_count"] for c in selected_data]))
    fig_bar.add_trace(go.Bar(name="领域", x=bar_names, y=[c["dom_count"] for c in selected_data]))
    fig_bar.update_layout(
        barmode="group",
        margin=dict(t=10, b=10, l=10, r=10),
        height=360,
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Detail Comparison Table ──
    st.markdown("<div class='card'><div class='card-title'>详细对比表</div>", unsafe_allow_html=True)
    rows: list[dict] = []
    for c in selected_data:
        rows.append({
            "候选人": c["name"],
            "级别": c["level"],
            "技术技能数": c["tech_count"],
            "能力数": c["cap_count"],
            "胜任力数": c["comp_count"],
            "领域数": c["dom_count"],
            "技术技能": ", ".join(c["tech_names"][:6]) + ("…" if len(c["tech_names"]) > 6 else ""),
            "能力项": ", ".join(c["cap_names"][:6]) + ("…" if len(c["cap_names"]) > 6 else ""),
            "胜任力": ", ".join(c["comp_names"][:6]) + ("…" if len(c["comp_names"]) > 6 else ""),
        })
    df_cmp = pd.DataFrame(rows)
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 技能图谱可视化
# =============================================================================

def _render_employee_skill_graph() -> None:
    """员工视角的技能图谱 - 只显示当前简历的技能（不显示全量本体）。"""
    st.markdown(
        """
        <div class="app-title">
            <div>
                <div class="title-left">技能图谱</div>
                <div class="title-desc">展示当前简历的技术技能 / 能力 / 领域 关系</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from pyvis.network import Network  # type: ignore
    except ImportError:
        st.error("pyvis 未安装，请执行: pip install pyvis")
        return

    name, profile = _employee_resume_selector()
    if not profile:
        return

    # ---------- 抽取技能：技术技能 / 能力 / 胜任力 / 领域 / 行业知识 ----------
    def _collect(raw) -> list[tuple[str, str]]:
        """从简历字段抽出 (name, optional_label) 列表，去重。"""
        out: list[tuple[str, str]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    nm = (
                        item.get("name")
                        or item.get("技能名称")
                        or item.get("competency_name")
                        or item.get("领域名称")
                        or ""
                    )
                    if nm:
                        out.append((str(nm), ""))
                elif item:
                    out.append((str(item), ""))
        elif isinstance(raw, dict):
            for k in raw.keys():
                out.append((str(k), ""))
        return out

    tech_skills = _collect(profile.get("技术技能", []))
    abilities = _collect(profile.get("能力", []))
    competencies = _collect(profile.get("胜任力", []))
    domains = _collect(profile.get("领域", []))
    industries = _collect(profile.get("行业知识", []))

    total = len(tech_skills) + len(abilities) + len(competencies) + len(domains) + len(industries)
    if total == 0:
        st.info("📭 当前简历中暂无技能数据。")
        return

    # ---------- 构建图：候选人 → 分类 → 具体技能 ----------
    net = Network(
        height="560px",
        width="100%",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-1500, central_gravity=0.25, spring_length=140)

    added: set[str] = set()
    edges: set[tuple[str, str]] = set()

    def _add_edge(a: str, b: str) -> None:
        key = tuple(sorted([a, b]))
        if key in edges:
            return
        edges.add(key)
        net.add_edge(a, b, color="rgba(120,120,140,0.45)")

    # Center: candidate
    person_id = "PERSON"
    net.add_node(
        person_id,
        label=name or "我",
        title=f"{name or '我'} 的技能图谱",
        color="#1976D2",
        size=38,
        shape="dot",
        borderWidth=4,
        font={"size": 22, "color": "#0D47A1"},
    )
    added.add(person_id)

    # Categories: (category_id, display_label, color, items)
    categories = [
        ("CAT_TECH", "技术技能", "#2196F3", tech_skills),
        ("CAT_ABILITY", "能力", "#4CAF50", abilities),
        ("CAT_COMPETENCY", "胜任力", "#9C27B0", competencies),
        ("CAT_DOMAIN", "领域", "#FF9800", domains),
        ("CAT_INDUSTRY", "行业知识", "#795548", industries),
    ]

    for cat_id, cat_label, cat_color, items in categories:
        if not items:
            continue
        # Category hub
        net.add_node(
            cat_id,
            label=f"{cat_label} ({len(items)})",
            title=cat_label,
            color=cat_color,
            size=26,
            shape="dot",
            font={"size": 18, "color": "#263238"},
        )
        added.add(cat_id)
        _add_edge(person_id, cat_id)

        # Each skill under this category
        seen_in_cat: set[str] = set()
        for nm, _ in items:
            nm = nm.strip()
            if not nm or nm in seen_in_cat:
                continue
            seen_in_cat.add(nm)
            skill_id = f"S_{cat_id}_{abs(hash(nm)) % 10_000_000}"
            if skill_id in added:
                # already connected somewhere — still link to this category
                _add_edge(cat_id, skill_id)
                continue
            net.add_node(
                skill_id,
                label=nm,
                title=nm,
                color=cat_color,
                size=14,
                font={"size": 13, "color": "#37474F"},
            )
            added.add(skill_id)
            _add_edge(cat_id, skill_id)

    # ---------- 摘要 ----------
    parts = []
    for _, label, _, items in categories:
        if items:
            parts.append(f"**{label}** {len(items)}")
    st.caption(" · ".join(parts) if parts else "暂无数据")

    html_str = net.generate_html()
    components.html(html_str, height=580, scrolling=True)

def render_knowledge_graph() -> None:
    """技能图谱 - 员工角色只看自己的技能；HR/Admin 看全量本体。"""
    # ── 员工角色：仅显示当前简历的技能，简洁版 ──
    current_role = str(st.session_state.get("auth_role", "hr") or "hr").strip().lower()
    if current_role == "employee":
        _render_employee_skill_graph()
        return

    st.markdown(
        """
        <div class="app-title">
            <div>
                <div class="title-left">技能图谱</div>
                <div class="title-desc">技能·能力·胜任力·领域 之间的关系网络</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        from pyvis.network import Network  # type: ignore
    except ImportError:
        st.error("pyvis 未安装，请执行: pip install pyvis")
        return

    skill_catalog_raw = load_skill_catalog()
    if not skill_catalog_raw:
        st.warning("技能目录为空，请先确保 technical_skill_abox.json 存在并包含数据。")
        return

    if isinstance(skill_catalog_raw, dict) and "names" in skill_catalog_raw and "id_item_map" in skill_catalog_raw:
        names = skill_catalog_raw.get("names", {})
        id_item_map = skill_catalog_raw.get("id_item_map", {})
    else:
        id_item_map = skill_catalog_raw
        names = {sid: (data.get("name") if isinstance(data, dict) else sid) for sid, data in id_item_map.items()}

    if not names or not id_item_map:
        st.warning("技能目录为空，请先确保 technical_skill_abox.json 存在并包含数据。")
        return

    # Build graph nodes and edges
    net = Network(height="620px", width="100%", directed=False, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150)

    node_ids: set[str] = set()
    edges_added: set[tuple[str, str]] = set()

    # ---- Category color map for type groups ----
    type_category = {
        "LANGUAGE":       ("编程语言", "#E91E63"),
        "FRAMEWORK":      ("开发框架", "#9C27B0"),
        "ML_FRAMEWORK":   ("ML框架", "#7B1FA2"),
        "LIBRARY":        ("代码库", "#AB47BC"),
        "ML_LIBRARY":     ("ML库", "#8E24AA"),
        "RUNTIME":        ("运行时", "#CE93D8"),
        "MARKUP":         ("标记语言", "#F48FB1"),
        "STYLE":          ("样式表", "#F06292"),
        "DATABASE":       ("数据库", "#FF9800"),
        "VECTOR_DATABASE":("向量数据库", "#FF5722"),
        "DATA_PLATFORM":  ("数据平台", "#FF7043"),
        "DATA_PROCESSING_ENGINE": ("数据处理引擎", "#F4511E"),
        "MESSAGE_QUEUE":  ("消息队列", "#795548"),
        "CLOUD_PLATFORM": ("云平台", "#2196F3"),
        "CLOUD_SERVICE":  ("云服务", "#42A5F5"),
        "PLATFORM":       ("平台", "#1E88E5"),
        "SAAS_PLATFORM":  ("SaaS平台", "#1976D2"),
        "SDK":            ("SDK", "#64B5F6"),
        "IDENTITY_PLATFORM": ("身份平台", "#1565C0"),
        "BI_TOOL":        ("BI工具", "#00BCD4"),
        "API_TESTING_TOOL": ("API测试", "#0097A7"),
        "TEST_AUTOMATION_TOOL": ("测试自动化", "#00838F"),
        "PERFORMANCE_TESTING_TOOL": ("性能测试", "#006064"),
        "OCR_TOOL":       ("OCR工具", "#26C6DA"),
        "TOOL":           ("工具", "#80DEEA"),
        "OBSERVABILITY_PLATFORM": ("可观测性", "#4DB6AC"),
        "CRM_PLATFORM":   ("CRM平台", "#009688"),
        "SEO_PLATFORM":   ("SEO平台", "#00796B"),
        "MARKETING_ANALYTICS": ("营销分析", "#26A69A"),
        "AI_MODEL":       ("AI模型", "#4CAF50"),
        "AI_PLATFORM":    ("AI平台", "#2E7D32"),
        "AI_FEATURE":     ("AI特性", "#66BB6A"),
        "AI_TECHNIQUE":   ("AI技术", "#43A047"),
        "AI_PROTOCOL":    ("AI协议", "#1B5E20"),
        "DATA_SCIENCE_PRACTICE": ("数据科学", "#388E3C"),
        "ENGINEERING_PRACTICE":  ("工程实践", "#689F38"),
        "TESTING_PRACTICE":      ("测试实践", "#827717"),
        "PROTOCOL":       ("网络协议", "#607D8B"),
        "PLATFORM_INTEGRATION": ("平台集成", "#546E7A"),
    }

    # Add type category nodes
    type_groups: dict[str, set[str]] = {}
    for item_id, item_data in id_item_map.items():
        typ = item_data.get("type", "UNKNOWN") if isinstance(item_data, dict) else "UNKNOWN"
        type_groups.setdefault(typ, set()).add(item_id)

    # Create category super-nodes (grouping types)
    cat_types: dict[str, tuple[str, str, set]] = {}
    type_to_cat: dict[str, str] = {}
    for typ, members in type_groups.items():
        cat_name, cat_color = type_category.get(typ, ("其他分类", "#9E9E9E"))
        if cat_name not in cat_types:
            cat_types[cat_name] = (cat_name, cat_color, set())
        cat_types[cat_name][2].update(members)
        type_to_cat[typ] = cat_name

    # Add category nodes
    for cat_name, cat_color, _members in cat_types.values():
        cat_id = f"CAT_{cat_name}"
        if cat_id not in node_ids:
            node_ids.add(cat_id)
            net.add_node(cat_id, label=cat_name, title=cat_name, color=cat_color, size=28, shape="dot", borderWidth=3)

    # Add skill nodes and edges to category
    for item_id, item_data in id_item_map.items():
        name = names.get(item_id, item_id)
        typ = item_data.get("type", "UNKNOWN") if isinstance(item_data, dict) else "UNKNOWN"
        cat_name = type_to_cat.get(typ, "其他分类")
        cat_id = f"CAT_{cat_name}"

        if item_id not in node_ids:
            node_ids.add(item_id)
            net.add_node(item_id, label=name, title=f"{name}\n类型: {typ}", color="#B0BEC5", size=10)

        # Edge: skill -> category
        edge = tuple(sorted([item_id, cat_id]))
        if edge not in edges_added:
            edges_added.add(edge)
            net.add_edge(item_id, cat_id, color="rgba(180,180,180,0.4)", dashes=True)

    # ---- Cross connections by shared language ----
    lang_skills: dict[str, set[str]] = {}
    for item_id, item_data in id_item_map.items():
        langs = item_data.get("languages", []) if isinstance(item_data, dict) else []
        for lang in langs:
            lang_skills.setdefault(lang, set()).add(item_id)

    for _lang, skill_ids in lang_skills.items():
        skill_list = list(skill_ids)
        for i in range(len(skill_list)):
            for j in range(i + 1, len(skill_list)):
                a, b = tuple(sorted([skill_list[i], skill_list[j]]))
                if (a, b) not in edges_added:
                    edges_added.add((a, b))
                    net.add_edge(a, b, color="rgba(100,150,200,0.25)", width=0.6, dashes=True)

    # ---- Cross connections by shared aliases ----
    alias_skills: dict[str, set[str]] = {}
    for item_id, item_data in id_item_map.items():
        aliases = item_data.get("aliases", []) if isinstance(item_data, dict) else []
        for alias in aliases:
            alias_lower = alias.lower()
            alias_skills.setdefault(alias_lower, set()).add(item_id)

    for _alias, skill_ids in alias_skills.items():
        if len(skill_ids) > 1:
            skill_list = list(skill_ids)
            for i in range(len(skill_list)):
                for j in range(i + 1, len(skill_list)):
                    a, b = tuple(sorted([skill_list[i], skill_list[j]]))
                    if (a, b) not in edges_added:
                        edges_added.add((a, b))
                        net.add_edge(a, b, color="rgba(150,200,150,0.3)", width=1.0, dashes=False)

    # Add top skills from parsed resumes as extra context nodes
    resumes = load_all_parsed_resumes()
    if resumes:
        stats = compute_overview_stats(resumes)
        top_skill_names = [s for s, _ in Counter(stats["all_tech_skills"]).most_common(15)]
        for name in top_skill_names:
            safe_id = f"resume_skill_{name}"
            if safe_id not in node_ids:
                node_ids.add(safe_id)
                net.add_node(safe_id, label=name, title=f"热门技术: {name}", color="#E91E63", size=14, shape="star")

    html_str = net.generate_html()
    components.html(html_str, height=640, scrolling=True)

# =============================================================================
# 技能缺口分析
# =============================================================================

@st.cache_data(ttl=300)
def compute_skill_gap(jd_data: dict[str, Any], resumes: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze skill gap: JD required skills vs talent pool coverage."""
    # Extract JD skill requirements
    jd_skills: list[str] = []
    for section in ["技术技能要求", "技术要求", "skills"]:
        items = jd_data.get(section, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("skill_name") or item.get("skill", "")).strip()
                else:
                    name = str(item or "").strip()
                if name:
                    jd_skills.append(name)

    # Count how many candidates have each skill
    skill_coverage: dict[str, int] = {}
    for skill in jd_skills:
        count = 0
        for r in resumes:
            tech = r.get("技术技能") or []
            all_names: list[str] = []
            if isinstance(tech, list):
                for item in tech:
                    if isinstance(item, dict):
                        n = str(item.get("name") or item.get("skill_name") or "").strip().lower()
                    else:
                        n = str(item or "").strip().lower()
                    if n:
                        all_names.append(n)
            if skill.lower() in all_names:
                count += 1
        skill_coverage[skill] = count

    total_resumes = len(resumes)
    covered = [s for s, cnt in skill_coverage.items() if cnt > 0]
    uncovered = [s for s, cnt in skill_coverage.items() if cnt == 0]

    return {
        "jd_skills": jd_skills,
        "skill_coverage": skill_coverage,
        "total_resumes": total_resumes,
        "covered_skills": covered,
        "uncovered_skills": uncovered,
        "coverage_rate": len(covered) / len(jd_skills) * 100 if jd_skills else 0,
    }

def render_skill_gap_analysis() -> None:
    """技能缺口分析 - JD需求 vs 人才池"""
    st.markdown(
        """
        <div class="app-title">
            <div>
                <div class="title-left">技能缺口分析</div>
                <div class="title-desc">岗位技能需求与人才池覆盖度对比</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jd_options = list_jd_json_options()
    if not jd_options:
        st.warning("未找到 JD JSON，请先解析 JD。")
        return

    selected_label = st.selectbox("选择 JD 进行缺口分析", options=[item["label"] for item in jd_options])
    selected_jd = next(item for item in jd_options if item["label"] == selected_label)
    jd_data = load_json_file(str(selected_jd["path"]))

    resumes = load_all_parsed_resumes()
    if not resumes:
        st.warning("暂无已解析的简历数据。")
        return

    gap_result = compute_skill_gap(jd_data, resumes)

    # Coverage gauge
    col1, col2, col3 = st.columns(3)
    col1.metric("JD 技能需求数", len(gap_result["jd_skills"]))
    col2.metric("已覆盖技能数", len(gap_result["covered_skills"]))
    col3.metric("人才池覆盖率", f"{gap_result['coverage_rate']:.1f}%",
                delta=f"{len(gap_result['uncovered_skills'])}项缺口" if gap_result["uncovered_skills"] else "完美覆盖",
                delta_color="inverse")

    # Coverage bar chart
    st.markdown("<div class='card'><div class='card-title'>技能覆盖详情</div>", unsafe_allow_html=True)
    if gap_result["skill_coverage"]:
        df_gap = pd.DataFrame([
            {"技能": k, "匹配人数": v, "覆盖率": f"{v / gap_result['total_resumes'] * 100:.1f}%",
             "状态": "✅ 已覆盖" if v > 0 else "⚠️ 缺口"}
            for k, v in gap_result["skill_coverage"].items()
        ])
        df_gap = df_gap.sort_values("匹配人数", ascending=False)

        fig_gap = px.bar(
            df_gap,
            x="技能", y="匹配人数",
            color="状态",
            color_discrete_map={"✅ 已覆盖": "#4CAF50", "⚠️ 缺口": "#F44336"},
            text="匹配人数",
        )
        fig_gap.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=360,
            xaxis_tickangle=-30,
        )
        fig_gap.update_traces(textposition="outside")
        st.plotly_chart(fig_gap, use_container_width=True)

        st.dataframe(df_gap, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Show uncovered skills warning
    if gap_result["uncovered_skills"]:
        st.warning(f"以下 {len(gap_result['uncovered_skills'])} 项技能在当前人才池中未找到匹配：{'、'.join(gap_result['uncovered_skills'])}")

    # Download gap report
    if st.button("导出缺口分析报告CSV"):
        import io
        df_gap = pd.DataFrame([
            {"技能": k, "匹配人数": v, "状态": "已覆盖" if v > 0 else "缺口"}
            for k, v in gap_result["skill_coverage"].items()
        ])
        csv = df_gap.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载CSV", data=csv, file_name="skill_gap_report.csv", mime="text/csv")

# =============================================================================
# PDF 报告导出
# =============================================================================

def generate_match_report_pdf(report: dict[str, Any], jd_data: dict[str, Any]) -> bytes:
    """Generate a PDF report from matching results using fpdf2.

    Encoding strategy (priority order):
      1. Local  `fonts/NotoSansSC-Regular.ttf` if present
      2. Windows system CJK fonts (msyh.ttc / simhei.ttf / simsun.ttc)
      3. Helvetica (latin-1) with non-ASCII chars replaced by '?' to avoid
         UnicodeEncodeError when neither CJK font is available.
    """
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        return b""

    pdf = FPDF()
    pdf.add_page()

    # ── 字体探测：先本地后系统，最后回退 latin-1 + 字符清洗 ──
    def _try_register_cjk(pdf_obj: "FPDF") -> bool:
        """Try to register a CJK-capable font. Returns True on success."""
        candidates = [
            APP_DIR / "fonts" / "NotoSansSC-Regular.ttf",
            Path("C:/Windows/Fonts") / "msyh.ttc",
            Path("C:/Windows/Fonts") / "msyhbd.ttc",
            Path("C:/Windows/Fonts") / "simhei.ttf",
            Path("C:/Windows/Fonts") / "simsun.ttc",
            Path("C:/Windows/Fonts") / "NotoSansCJK-Regular.ttc",
            Path("/System/Library/Fonts/PingFang.ttc"),       # macOS
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),  # Linux
        ]
        for idx, cand in enumerate(candidates):
            if cand.exists():
                try:
                    # .ttc (TrueType Collection) needs an index; .ttf uses 0
                    if cand.suffix.lower() == ".ttc":
                        pdf_obj.add_font("CJK", "", str(cand), uni=True)
                    else:
                        pdf_obj.add_font("CJK", "", str(cand), uni=True)
                    return True
                except Exception:
                    continue
        return False

    has_cjk = _try_register_cjk(pdf)
    use_unicode = has_cjk

    def _safe(text: Any) -> str:
        """Render text safely regardless of font capability."""
        s = str(text) if text is not None else ""
        if use_unicode:
            return s
        # Helvetica: latin-1 only — replace anything outside [0,255] with '?'
        return s.encode("latin-1", errors="replace").decode("latin-1")

    if use_unicode:
        pdf.set_font("CJK", "", 12)
    else:
        pdf.set_font("Helvetica", "", 12)

    # Title
    pdf.set_font_size(18)
    pdf.cell(0, 12, _safe("Talent Compass Report"), ln=True, align="C")
    pdf.ln(4)

    # JD Info
    job_title = jd_data.get("岗位名称") or report.get("job_title", "N/A")
    jd_level = str(jd_data.get("级别要求", "N/A"))
    pdf.set_font_size(12)
    pdf.cell(0, 8, _safe(f"Job: {job_title}  |  Level: {jd_level}"), ln=True)
    pdf.ln(6)

    # Candidates
    pdf.set_font_size(10)
    results = report.get("results", [])
    for i, r in enumerate(results[:20], 1):
        if not isinstance(r, dict):
            continue
        name = r.get("candidate_name", f"Candidate #{i}")
        score = float(r.get("total_score", 0))
        pdf.cell(0, 7, _safe(f"#{i} {name}  -  Score: {score:.1f}"), ln=True)

    pdf.ln(4)
    pdf.set_font_size(9)
    pdf.cell(0, 6, _safe(f"Total candidates: {len(results)}  |  Generated by Talent Compass Platform"), ln=True, align="C")

    # 兼容 fpdf2 不同版本：dest="S" 在新版本里直接返回 str/bytes，
    # 老版本需要 .encode("latin-1")。统一确保返回 bytes。
    try:
        out = pdf.output(dest="S")
        if isinstance(out, str):
            return out.encode("latin-1", errors="replace")
        return bytes(out)
    except TypeError:
        # 老版本 fpdf2：output() 默认返回 bytearray
        out = pdf.output()
        return bytes(out)

# ═══════════════════════════════════════════════════════════════
# Employee's Own Pages
# ═══════════════════════════════════════════════════════════════

def _employee_resume_selector() -> tuple[str | None, dict | None]:
    """Get the employee's resume selected in 职业助手 (single source of truth).

    Lookup priority:
      1. session_state cache set by 职业助手
      2. Auto-load from employee_resume_map.json (employee has only one allowed stem)
      3. Hint the user to go to 职业助手

    All employee pages (岗位推荐 / 技能提升 / 我的画像 / 技能图谱) share
    the same resume. Returns (name, data) or (None, None) if unavailable.
    """
    import glob

    # ── 1. session_state cache (set by 职业助手) ──
    name = st.session_state.get("_emp_resume_name")
    data = st.session_state.get("_emp_resume_data")
    if name and data:
        st.caption(f"📋 当前简历：{name} （来自职业助手）")
        return name, data

    # ── 2. Auto-load from employee_resume_map.json (fallback when cache is lost) ──
    mapping = load_employee_resume_map()
    allowed_stems: list[str] = [e.strip() for e in mapping.get("employee", []) if e.strip()]

    if allowed_stems:
        # Try each allowed stem (in order), prefer one that has a parsed result
        for stem in allowed_stems:
            # Accept both "Candidate_0042" and "candidate_0042" forms
            candidates = [stem, stem.lower()]
            for s in candidates:
                # Use absolute path via PARSED_RESULTS_DIR (cwd may differ)
                parsed_file = PARSED_RESULTS_DIR / f"{s}_parsed.json"
                if parsed_file.exists():
                    try:
                        with open(parsed_file, encoding="utf-8") as f:
                            data_obj = json.loads(f.read())
                        resume_name = data_obj.get("name", "") or stem
                        # Cache so other page calls hit the fast path
                        st.session_state["_emp_resume_name"] = resume_name
                        st.session_state["_emp_resume_data"] = data_obj
                        st.caption(f"📋 当前简历：{resume_name} （来自员工档案）")
                        return resume_name, data_obj
                    except Exception:
                        continue

    # ── 3. No resume available → hint ──
    st.info("📭 尚未选择简历。请先前往「✨ **职业助手**」页面选择你的简历。")
    return None, None

def render_job_recommendations(api_key: str, base_url: str, model: str) -> None:
    """🎯 岗位推荐 - Show jobs matching employee's profile."""
    st.markdown("""
        <div class="content-card">
            <div class="title-left">岗位推荐</div>
            <div class="title-desc">基于你的技能与经验，智能匹配最适合的岗位</div>
        </div>
    """, unsafe_allow_html=True)

    name, profile = _employee_resume_selector()
    if not profile:
        st.info("📭 暂无简历数据。请先前往「✨ 职业助手」上传或选择你的简历。")
        return

    skills_raw = profile.get("技术技能", profile.get("skills", []))
    competency_raw = profile.get("胜任力", [])

    def _profile_skill_names(raw) -> set[str]:
        if isinstance(raw, list):
            out: set[str] = set()
            for s in raw:
                if isinstance(s, dict):
                    nm = s.get("name") or s.get("技能名称") or s.get("competency_name") or ""
                    if nm:
                        out.add(str(nm))
                    mt = s.get("matched_term")
                    if mt:
                        out.add(str(mt))
                elif s:
                    out.add(str(s))
            return out
        if isinstance(raw, dict):
            return {str(k) for k in raw.keys()}
        return set()

    my_skills = _profile_skill_names(skills_raw) | _profile_skill_names(competency_raw)

    my_level = str(
        profile.get("级别", "")
        or profile.get("level", "")
        or profile.get("seniority", "")
        or ""
    )

    # Load all JDs (from parsed_results/jd_*.json, both real roles from disk)
    import glob
    jd_files = glob.glob(str(PARSED_RESULTS_DIR / "jd_*.json"))
    results: list[dict] = []

    for jf in jd_files:
        try:
            with open(jf, encoding="utf-8") as fp:
                jd = json.loads(fp.read())
        except Exception:
            continue
        # Use 中文 field names with English fallback
        jd_name = os.path.splitext(os.path.basename(jf))[0]
        # Skills: union of technical-skill list and competency-ability list
        tech_raw = jd.get("技术技能要求", jd.get("required_skills", jd.get("skills", [])))
        comp_raw = jd.get("胜任力要求", [])
        req: set[str] = set()

        def _extract_names(raw) -> set[str]:
            if isinstance(raw, list):
                out: set[str] = set()
                for s in raw:
                    if isinstance(s, dict):
                        # 中文 name field or English "name"
                        nm = s.get("name") or s.get("技能名称") or s.get("competency_name") or ""
                        if nm:
                            out.add(str(nm))
                        # Also collect matched_term (the ontology term)
                        mt = s.get("matched_term")
                        if mt:
                            out.add(str(mt))
                    elif s:
                        out.add(str(s))
                return out
            if isinstance(raw, dict):
                return {str(k) for k in raw.keys()}
            return set()

        req |= _extract_names(tech_raw)
        req |= _extract_names(comp_raw)

        if not req and not my_skills:
            match_score = 50.0
        elif not req:
            match_score = 50.0
        elif not my_skills:
            match_score = 10.0
        else:
            overlap = my_skills & req
            match_score = round(len(overlap) / len(req) * 100, 1) if req else 50.0

        jd_desc = jd.get("岗位描述", jd.get("description", jd.get("summary", "")))
        if isinstance(jd_desc, str) and len(jd_desc) > 120:
            jd_desc = jd_desc[:120] + "..."

        results.append({
            "job": jd.get("岗位名称", jd_name),  # display name
            "title": jd.get("岗位名称", jd_name),
            "match": match_score,
            "matched_skills": list(my_skills & req) if my_skills and req else [],
            "total_req": len(req),
            "desc": jd_desc,
            "level": jd.get("level", jd.get("seniority", "")),
        })

    results.sort(key=lambda x: x["match"], reverse=True)

    # KPI cards
    top5 = [r for r in results[:5] if r["match"] >= 30]
    c1, c2, c3 = st.columns(3)
    c1.metric("匹配岗位", f"{len(top5)}")
    c2.metric("最高匹配度", f"{top5[0]['match']:.0f}%" if top5 else "N/A")
    c3.metric("覆盖技能数", f"{len(my_skills)}")

    st.markdown("---")

    # Match bar chart
    if top5:
        import plotly.express as px
        fig = px.bar(
            pd.DataFrame(top5),
            x="match", y="job",
            orientation="h",
            title="Top 匹配岗位",
            labels={"match": "匹配度 (%)", "job": "岗位"},
            color="match", color_continuous_scale="Blues",
            height=250,
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 推荐岗位列表")
    for j, r in enumerate(results[:10]):
        star = "⭐" if r["match"] >= 70 else "🔵" if r["match"] >= 50 else "⚪"
        cols = st.columns([6, 1])
        with cols[0]:
            st.markdown(
                f"**{star} {r['title']}**  "
                f"<span style='color:#666;font-size:13px'>匹配度 {r['match']:.0f}%</span>",
                unsafe_allow_html=True,
            )
            if r["desc"]:
                st.caption(r["desc"])
            if r["matched_skills"]:
                skill_tags = ", ".join(r["matched_skills"][:5])
                st.caption(f"✅  匹配技能：{skill_tags}")
        with cols[1]:
            st.progress(r["match"] / 100)
        if j < len(results) - 1:
            st.markdown("---")

def render_skill_improvement(api_key: str, base_url: str, model: str) -> None:
    """📈 技能提升 - Show skill gaps and learning paths with AI suggestions."""
    st.markdown("""
        <div class="content-card">
            <div class="title-left">技能提升</div>
            <div class="title-desc">对标目标岗位，发现技能差距，制定学习计划</div>
        </div>
    """, unsafe_allow_html=True)

    name, profile = _employee_resume_selector()
    if not profile:
        st.info("📭 暂无简历数据。请先前往「✨ 职业助手」上传或选择你的简历。")
        return

    # Read candidate skills (中文 field first, English fallback; union 技术技能 + 胜任力)
    def _profile_skill_names(raw) -> set[str]:
        if isinstance(raw, list):
            out: set[str] = set()
            for s in raw:
                if isinstance(s, dict):
                    nm = s.get("name") or s.get("技能名称") or s.get("competency_name") or ""
                    if nm:
                        out.add(str(nm))
                    mt = s.get("matched_term")
                    if mt:
                        out.add(str(mt))
                elif s:
                    out.add(str(s))
            return out
        if isinstance(raw, dict):
            return {str(k) for k in raw.keys()}
        return set()

    skills_raw = profile.get("技术技能", profile.get("skills", []))
    competency_raw = profile.get("胜任力", [])
    my_skills = _profile_skill_names(skills_raw) | _profile_skill_names(competency_raw)

    # Pick target JD (from parsed_results/jd_*.json - absolute path, cwd-safe)
    import glob
    jd_files = glob.glob(str(PARSED_RESULTS_DIR / "jd_*.json"))
    jd_labels = []
    jd_data_map = {}
    for jf in jd_files:
        try:
            with open(jf, encoding="utf-8") as fp:
                jd = json.loads(fp.read())
            # Use 中文 "岗位名称" with English fallback
            lbl = jd.get("岗位名称", jd.get("title", os.path.splitext(os.path.basename(jf))[0]))
            jd_labels.append(lbl)
            jd_data_map[lbl] = (jf, jd)
        except Exception:
            pass

    if not jd_labels:
        st.warning("暂无岗位数据。请先生成或上传 JD 文件到 parsed_results/jd_*.json")
        return

    target_jd = st.selectbox("🎯 选择目标岗位", jd_labels, key="skill_imp_target")
    jf_path, jd = jd_data_map[target_jd]

    # Required skills: union of 技术技能要求 + 胜任力要求 (中文 first)
    tech_raw = jd.get("技术技能要求", jd.get("required_skills", jd.get("skills", [])))
    comp_raw = jd.get("胜任力要求", [])
    req = _profile_skill_names(tech_raw) | _profile_skill_names(comp_raw)

    matched = my_skills & req
    missing = req - my_skills
    extra = my_skills - req

    # Three-column summary
    c1, c2, c3 = st.columns(3)
    c1.metric("已有技能", f"{len(matched)}/{len(req)}")
    c2.metric("待学习", f"{len(missing)}")
    c3.metric("额外优势", f"{len(extra)}")

    st.markdown("---")

    # Matched skills
    st.markdown("### ✅ 已匹配技能")
    if matched:
        _render_skill_tags(matched, "green")
    else:
        st.caption("暂无匹配技能，建议先学习以下缺失技能。")

    # Missing skills with AI learning suggestions
    st.markdown("### 🔴 技能差距")
    if missing:
        missing_list = sorted(missing)
        _render_skill_tags(missing_list, "red")
        st.markdown("---")
        st.markdown("### 🤖 AI 学习建议")

        if api_key:
            if st.button("生成学习路径建议", key="gen_learning_path"):
                with st.spinner("AI 正在分析..."):
                    prompt = (
                        f"员工当前技能：{', '.join(sorted(my_skills))}\n"
                        f"目标岗位所需技能：{', '.join(sorted(req))}\n"
                        f"缺失技能：{', '.join(missing_list)}\n\n"
                        f"请生成简短的学习路径建议（不超过300字），包括：\n"
                        f"1. 优先级排序（哪些技能先学）\n"
                        f"2. 学习资源推荐（在线课程、书籍等）\n"
                        f"3. 预计学习时间\n"
                        f"用中文回答。"
                    )
                    try:
                        resp = _llm_chat_reply(
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            messages=[
                                {"role": "system", "content": "你是一个资深的职业规划顾问，擅长为 IT 从业者定制学习路径。"},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        if resp:
                            st.success("AI 学习建议：")
                            st.markdown(resp)
                        else:
                            st.error("AI 未返回建议。")
                    except Exception as exc:
                        st.error(f"AI 调用失败：{exc}")
        else:
            st.info("💡 配置 API Key 后可生成 AI 个性化学习建议。")
    else:
        st.success("🎉 太棒了！你已覆盖该岗位所有技能要求！")

    # Extra skills - 区分「真正优势」与「通用基础」，避免堆 50+ 个标签
    # 通用基础技能（人人都有，不构成差异化优势）
    _COMMON_SKILLS = {
        "Python", "Python 基础", "沟通协调", "项目管理", "需求分析", "MySQL",
        "PostgreSQL", "MongoDB", "PL/SQL", "CI/CD", "Docker", "Pandas",
        "Pandas", "Power BI", "RAG", "LLM",
    }
    if extra:
        # 真正差异化优势 = 候选人会的、JD 不要的、且不是通用基础
        real_advantage = extra - _COMMON_SKILLS
        st.markdown("### 💡 额外优势技能")
        if real_advantage:
            st.caption(
                f"你拥有 **{len(real_advantage)}** 项该岗位未直接要求、但能为你加分的差异化技能"
            )
            _render_skill_tags(sorted(real_advantage), "blue")
        else:
            st.caption("除岗位要求与通用技能外，无明显差异化优势。")

        # 折叠区：完整列表（包含通用技能）
        common_in_extra = extra & _COMMON_SKILLS
        if common_in_extra:
            with st.expander(f"📋 完整技能清单（含 {len(common_in_extra)} 项通用基础）", expanded=False):
                st.caption("以下为通用基础技能，不构成差异化优势：")
                _render_skill_tags(sorted(common_in_extra), "grey")
                st.caption(f"候选人全部技能: {len(my_skills)} 项 | 岗位要求: {len(req)} 项 | 已有: {len(matched)} | 待学习: {len(missing)} | 差异化优势: {len(real_advantage)}")

def render_my_portrait(api_key: str, base_url: str, model: str) -> None:
    """🖼️ 我的画像 - Employee's own talent portrait."""
    st.markdown("""
        <div class="content-card">
            <div class="title-left">我的画像</div>
            <div class="title-desc">你的技能雷达、经验总览与人才画像</div>
        </div>
    """, unsafe_allow_html=True)

    name, profile = _employee_resume_selector()
    if not profile:
        st.info("📭 暂无简历数据。请先前往「✨ 职业助手」上传或选择你的简历。")
        return

    col_a, col_b = st.columns([2, 3])

    with col_a:
        st.markdown("### 📋 基本信息")
        # 中文 field first, English fallback for legacy resumes
        st.markdown(f"**姓名**：{profile.get('姓名', profile.get('name', 'N/A'))}")
        st.markdown(f"**邮箱**：{profile.get('邮箱', profile.get('email', '简历未提供'))}")
        st.markdown(f"**电话**：{profile.get('电话', profile.get('phone', '简历未提供'))}")
        st.markdown(f"**城市**：{profile.get('城市', profile.get('city', '未填'))}")
        st.markdown(f"**级别**：{profile.get('级别', profile.get('level', '未填'))}")
        exp_years = profile.get("工作年限", profile.get("total_years", profile.get("years", "")))
        if exp_years:
            st.markdown(f"**工作年限**：{exp_years} 年")
        st.markdown(f"**学历**：{profile.get('学历', profile.get('education', profile.get('degree', '简历未提供')))}")

        # Skills tag cloud
        st.markdown("### 🏷️ 技能标签")
        skills_raw = profile.get("技术技能", profile.get("skills", []))
        capability_raw = profile.get("能力", profile.get("capabilities", []))
        competency_raw = profile.get("胜任力", [])

        def _name_list(raw):
            if isinstance(raw, list):
                out: list[str] = []
                for s in raw:
                    if isinstance(s, dict):
                        nm = s.get("name") or s.get("技能名称") or s.get("competency_name") or ""
                        if nm: out.append(str(nm))
                    elif s:
                        out.append(str(s))
                return out
            if isinstance(raw, dict):
                return list(raw.keys())
            return []

        merged = _name_list(skills_raw) + _name_list(capability_raw) + _name_list(competency_raw)
        # dedupe, preserve order
        seen = set()
        skill_list = [s for s in merged if not (s in seen or seen.add(s))] if merged else [] 
        if skill_list:
            _render_skill_tags(skill_list[:15], "teal")
        else:
            st.caption("无技能数据")

    with col_b:
        st.markdown("### 📊 技能雷达图")
        if skill_list and len(skill_list) >= 3:
            levels_raw = profile.get("技术技能", profile.get("skills", []))
            skill_scores = {}
            if isinstance(levels_raw, list):
                for s in levels_raw:
                    if isinstance(s, dict):
                        n = s.get("name", "") or s.get("技能名称", "")
                        lvl = s.get("level", 3)
                        try:
                            skill_scores[n] = float(lvl)
                        except (ValueError, TypeError):
                            skill_scores[n] = 3.0

            radar_skills = skill_list[:8]
            radar_vals = [skill_scores.get(s, 3.0) for s in radar_skills]
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_vals,
                theta=radar_skills,
                fill="toself",
                name=name or "我的技能",
                line_color="#2196F3",
                fillcolor="rgba(33,150,243,0.3)",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("技能数据不足，无法生成雷达图。请确保简历中包含至少 3 个技能。")

        # Experience timeline
        st.markdown("### 📜 工作经历")
        experiences = profile.get("经验", profile.get("工作经历", profile.get("experiences", profile.get("experience", []))))
        if isinstance(experiences, list) and experiences:
            for exp in experiences[:5]:
                if isinstance(exp, dict):
                    company = exp.get("公司", exp.get("company", exp.get("organization", "")))
                    role = exp.get("职位", exp.get("role", exp.get("title", exp.get("position", ""))))
                    duration = exp.get("时间", exp.get("时长", exp.get("duration", exp.get("period", ""))))
                    if company or role:
                        st.markdown(f"- **{role or '未知职位'}** @ {company or '未知公司'}  `{duration}`")
                        # Optionally show 描述 / description
                        desc = exp.get("描述", exp.get("description", ""))
                        if desc and isinstance(desc, str) and len(desc) < 200:
                            st.caption(f"  {desc}")
        else:
            st.caption("无工作经历数据")

def _render_skill_tags(skills: list[str], color: str) -> None:
    """Render a set of skills as colored tags."""
    color_map = {
        "green": "#4CAF50", "red": "#F44336", "blue": "#2196F3",
        "teal": "#009688", "orange": "#FF9800", "grey": "#9E9E9E",
    }
    c = color_map.get(color, "#9E9E9E")
    tags = " ".join(
        f'<span style="display:inline-block;background:{c};color:#fff;'
        f'padding:2px 10px;border-radius:12px;margin:2px;font-size:13px;">{s}</span>'
        for s in skills
    )
    st.markdown(tags, unsafe_allow_html=True)

def render_header(role: str) -> None:
    """Render the global top header bar."""
    role_display = ROLE_DISPLAY_NAMES.get(role, role)
    st.markdown(
        f"""
        <div class="global-header">
            <div class="header-left">
                <div class="header-logo-text">Talent Compass</div>
                <div><span class="header-subtitle">智能人才匹配平台</span></div>
            </div>
            <div class="header-right">
                <span class="header-badge">当前角色: {role_display}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    # Initialize default role if not set
    if "auth_role" not in st.session_state:
        st.session_state["auth_role"] = "hr"

    ensure_dirs()
    load_dotenv(ENV_PATH)
    st.set_page_config(page_title="Talent Compass Dashboard", layout="wide", initial_sidebar_state="expanded")

    inject_dashboard_css()

    api_key = os.getenv("OPENAI_API_KEY") or get_secret_value("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or get_secret_value("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL") or get_secret_value("OPENAI_MODEL", "gpt-4o-mini")

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="sidebar-logo-text">Talent Compass</div>
                <div class="sidebar-logo-sub">智能人才匹配平台</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        role_options = list(ROLE_DISPLAY_NAMES.keys())
        current_role = st.session_state.get("auth_role", "hr")
        current_index = role_options.index(current_role) if current_role in role_options else 0

        selected_role = st.selectbox(
            "角色切换",
            options=role_options,
            index=current_index,
            format_func=lambda r: ROLE_DISPLAY_NAMES.get(r, r),
            key="sidebar_role_selector",
        )
        if selected_role != current_role:
            st.session_state["auth_role"] = selected_role

        st.markdown("---")
        st.markdown('<p style="font-size:11px;color:#888;margin-bottom:4px;padding-left:8px;">导航菜单</p>', unsafe_allow_html=True)

        role = str(st.session_state.get("auth_role", "hr") or "hr")
        visible_pages = ROLE_PAGE_MAP.get(role, [])

        page_labels: list[str] = []
        if "人才洞察" in visible_pages:
            page_labels.append("📊 人才洞察")
        if "职业助手" in visible_pages:
            page_labels.append("✨ 职业助手")
        if "人才发现" in visible_pages:
            page_labels.append("🔍 人才发现")
        if "岗位中心" in visible_pages:
            page_labels.append("📋 岗位中心")
        if "岗位推荐" in visible_pages:
            page_labels.append("🎯 岗位推荐")
        if "人才评估" in visible_pages:
            page_labels.append("📈 人才评估")
        if "技能提升" in visible_pages:
            page_labels.append("📈 技能提升")
        if "我的画像" in visible_pages:
            page_labels.append("🖼️ 我的画像")
        if "技能图谱" in visible_pages:
            page_labels.append("🌐 技能图谱")
        if "智能解析" in visible_pages:
            page_labels.append("🆔 智能解析")
        if "技能发现" in visible_pages:
            page_labels.append("⚙️ 技能发现")
        if "简历权限管理" in visible_pages:
            page_labels.append("🛡️ 简历权限管理")

        if not page_labels:
            st.error("当前账号没有可访问页面")
            st.stop()

        selected_page = st.radio(
            "导航",
            options=page_labels,
            label_visibility="collapsed",
            key="sidebar_nav",
        )

        st.markdown("---")
        st.markdown(
            '<div class="page-footer">v1.0 – Talent Compass</div>',
            unsafe_allow_html=True,
        )

    # ── Header ────────────────────────────────────────────
    render_header(role)

    # ── Page Content ──────────────────────────────────────
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    page_name = selected_page.strip().split(" ", 1)[-1] if selected_page else ""

    if page_name == "人才洞察":
        render_overview_dashboard()
    elif page_name == "人才发现":
        render_matching_dashboard()
    elif page_name == "岗位中心":
        render_job_kanban_board()
    elif page_name == "职业助手":
        render_reverse_matching_chat(api_key, base_url, model)
    elif page_name == "岗位推荐":
        render_job_recommendations(api_key, base_url, model)
    elif page_name == "技能提升":
        render_skill_improvement(api_key, base_url, model)
    elif page_name == "我的画像":
        render_my_portrait(api_key, base_url, model)
    elif page_name == "人才评估":
        render_candidate_comparison()
    elif page_name == "技能图谱":
        render_knowledge_graph()
    elif page_name == "智能解析":
        render_parse_tools(api_key, base_url, model)
    elif page_name == "技能发现":
        render_unknown_skills_summary()
    elif page_name == "简历权限管理":
        render_employee_resume_map_admin()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
