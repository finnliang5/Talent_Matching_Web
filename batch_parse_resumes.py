import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from resume_parser import (
    APP_DIR,
    RESULT_DIR,
    SUPPORTED_SUFFIXES,
    normalize_resume_result,
    parse_resume_to_json,
    read_resume_text,
    save_resume_result,
)


SUMMARY_PATH = RESULT_DIR / "batch_parse_summary.json"
ERROR_LOG_PATH = RESULT_DIR / "batch_parse_errors.jsonl"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_resume_files(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_SUFFIXES else []

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda item: str(item).lower())


def result_path_for(source_file: Path) -> Path:
    return RESULT_DIR / f"{source_file.stem}_parsed.json"


def cached_result_is_current(source_file: Path, source_hash: str, mode: str) -> bool:
    result_path = result_path_for(source_file)
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    batch_meta = result.get("_batch", {})
    return (
        isinstance(batch_meta, dict)
        and batch_meta.get("source_hash") == source_hash
        and batch_meta.get("parser_mode") == mode
    )


def attach_batch_metadata(result: dict[str, Any], source_file: Path, source_hash: str, mode: str) -> dict[str, Any]:
    result["_batch"] = {
        "source_file": source_file.name,
        "source_path": str(source_file),
        "source_hash": source_hash,
        "parser_mode": mode,
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
    }
    return result


def parse_one_resume(
    source_file: Path,
    mode: str,
    api_key: str,
    base_url: str,
    model: str,
    jd_description: str,
) -> dict[str, Any]:
    source_hash = sha256_file(source_file)
    resume_text = read_resume_text(source_file)
    if not resume_text.strip():
        raise ValueError("简历文本为空，可能是 PDF/DOCX 无法提取文本。")

    if mode == "rule":
        result = normalize_resume_result({"姓名": source_file.stem}, resume_text)
    elif mode == "llm":
        result = parse_resume_to_json(resume_text, api_key, base_url, model, jd_description)
        if not result.get("姓名"):
            result["姓名"] = source_file.stem
    else:
        raise ValueError(f"未知解析模式：{mode}")

    return attach_batch_metadata(result, source_file, source_hash, mode)


def append_error(item: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_summary(summary: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量解析简历并输出到 parsed_results。")
    parser.add_argument(
        "--input",
        default=str(APP_DIR / "resumes"),
        help="简历文件或目录路径，默认 Talent_Matching/resumes。",
    )
    parser.add_argument(
        "--mode",
        choices=["rule", "llm"],
        default="rule",
        help="rule 只跑本体规则，llm 调用大模型补充解析。默认 rule。",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发解析数量。LLM 模式建议 3-5。")
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重新解析。")
    parser.add_argument("--recursive", action="store_true", help="递归读取输入目录。")
    parser.add_argument("--limit", type=int, default=0, help="最多解析多少份，0 表示不限制。")
    parser.add_argument("--jd-file", default="", help="可选：读取 JD 文件内容作为简历解析上下文。")
    parser.add_argument("--jd-text", default="", help="可选：直接传入 JD 文本作为简历解析上下文。")
    parser.add_argument("--api-key", default="", help="LLM 模式可显式传入 API key，默认读取 OPENAI_API_KEY。")
    parser.add_argument("--base-url", default="", help="LLM 模式可显式传入 base URL，默认读取 OPENAI_BASE_URL。")
    parser.add_argument("--model", default="", help="LLM 模式可显式传入模型名，默认读取 OPENAI_MODEL。")
    return parser.parse_args()


def main() -> int:
    load_dotenv(APP_DIR / ".env")
    args = parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    files = collect_resume_files(input_path, recursive=args.recursive)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if args.mode == "llm" and not api_key:
        raise ValueError("LLM 模式需要 API key，请设置 OPENAI_API_KEY 或传入 --api-key。")

    jd_description = args.jd_text
    if args.jd_file:
        jd_description = Path(args.jd_file).read_text(encoding="utf-8", errors="ignore")

    RESULT_DIR.mkdir(exist_ok=True)
    if ERROR_LOG_PATH.exists():
        ERROR_LOG_PATH.unlink()

    started_at = time.time()
    summary: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "mode": args.mode,
        "workers": max(1, args.workers),
        "total_files": len(files),
        "parsed": 0,
        "skipped": 0,
        "failed": 0,
        "outputs": [],
        "errors": [],
    }

    pending: list[Path] = []
    for source_file in files:
        try:
            source_hash = sha256_file(source_file)
            if not args.force and cached_result_is_current(source_file, source_hash, args.mode):
                summary["skipped"] += 1
                summary["outputs"].append(
                    {
                        "source_file": source_file.name,
                        "status": "skipped",
                        "result_path": str(result_path_for(source_file)),
                    }
                )
                print(f"[SKIP] {source_file.name}")
                continue
            pending.append(source_file)
        except Exception as exc:
            summary["failed"] += 1
            error_item = {"source_file": source_file.name, "error": str(exc)}
            summary["errors"].append(error_item)
            append_error(error_item)
            print(f"[FAIL] {source_file.name}: {exc}")

    def worker(path: Path) -> tuple[Path, dict[str, Any]]:
        result = parse_one_resume(path, args.mode, api_key, base_url, model, jd_description)
        return path, result

    max_workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, path): path for path in pending}
        for future in as_completed(futures):
            source_file = futures[future]
            try:
                parsed_source, result = future.result()
                result_path = save_resume_result(parsed_source, result)
                summary["parsed"] += 1
                summary["outputs"].append(
                    {
                        "source_file": parsed_source.name,
                        "status": "parsed",
                        "result_path": str(result_path),
                    }
                )
                print(f"[OK] {parsed_source.name} -> {result_path.name}")
            except Exception as exc:
                summary["failed"] += 1
                error_item = {"source_file": source_file.name, "error": str(exc)}
                summary["errors"].append(error_item)
                append_error(error_item)
                print(f"[FAIL] {source_file.name}: {exc}")

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["elapsed_seconds"] = round(time.time() - started_at, 2)
    write_summary(summary)

    print(
        "完成："
        f"总数 {summary['total_files']}，"
        f"解析 {summary['parsed']}，"
        f"跳过 {summary['skipped']}，"
        f"失败 {summary['failed']}。"
    )
    print(f"汇总文件：{SUMMARY_PATH}")
    if summary["failed"]:
        print(f"错误日志：{ERROR_LOG_PATH}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
