#!/usr/bin/env python3
"""
Archive external web retrieval evidence into the local knowledge base as Markdown docs.

Goal:
- Keep existing pipeline code untouched.
- Persist web-retrieved evidence locally for future KB expansion.
- Explicitly mark each saved item as external/web and keep source provenance (engine + URL).

Modes:
1) Live retrieval mode (calls current Search Agent):
   python scripts/archive_web_context_to_kb.py \
	   --question "..." \
	   --target-info "missing fact phrase" \
	   --q-type MC

2) Import mode from JSON / JSONL:
   python scripts/archive_web_context_to_kb.py --input-json path/to/evidence.jsonl

Output (default under knowledge_base/raw_files/web_external):
- articles/*.md                     (ingest-ready docs)
- index.json                        (dedup index)
- ingest_events.jsonl               (append-only event log)

These Markdown files can be ingested by your existing KB ingestion pipeline later.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_text(x: Any) -> str:
	return str(x or "").strip()


def _slug(s: str, limit: int = 80) -> str:
	s = _safe_text(s).lower()
	s = re.sub(r"\s+", "-", s)
	s = re.sub(r"[^a-z0-9._-]", "-", s)
	s = re.sub(r"-+", "-", s).strip("-._")
	if not s:
		return "item"
	return s[:limit]


def _sha256(s: str) -> str:
	return hashlib.sha256(_safe_text(s).encode("utf-8")).hexdigest()


def _normalize_url(url: str) -> str:
	u = _safe_text(url)
	if not u:
		return ""
	# normalize trivial trailing slash; keep query string to preserve exact source URL
	if u.endswith("/"):
		return u[:-1]
	return u


def _domain(url: str) -> str:
	try:
		return (urlparse(url).netloc or "").lower()
	except Exception:
		return ""


def _ensure_dirs(base_dir: Path) -> Tuple[Path, Path, Path]:
	base_dir.mkdir(parents=True, exist_ok=True)
	articles_dir = base_dir / "articles"
	articles_dir.mkdir(parents=True, exist_ok=True)
	index_path = base_dir / "index.json"
	events_path = base_dir / "ingest_events.jsonl"
	return articles_dir, index_path, events_path


def _load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _load_jsonl(path: Path) -> List[Any]:
	out: List[Any] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			out.append(json.loads(line))
	return out


def _load_records(input_path: Path) -> List[Dict[str, Any]]:
	ext = input_path.suffix.lower()
	raw: Any
	if ext == ".jsonl":
		raw = _load_jsonl(input_path)
	else:
		raw = _load_json(input_path)

	# Normalize into a list of "record" dicts:
	# {
	#   question_id, question, retrieval_query, q_type,
	#   evidence: [{source,title,url,text}, ...]
	# }
	records: List[Dict[str, Any]] = []

	if isinstance(raw, dict):
		if "evidence" in raw and isinstance(raw["evidence"], list):
			records.append(raw)
		elif "records" in raw and isinstance(raw["records"], list):
			for r in raw["records"]:
				if isinstance(r, dict):
					records.append(r)
		elif "items" in raw and isinstance(raw["items"], list):
			records.append({"evidence": raw["items"]})
		else:
			records.append({"evidence": [raw]})
		return records

	if isinstance(raw, list):
		# If every item already looks like evidence item => wrap once.
		if all(isinstance(x, dict) and "text" in x for x in raw):
			records.append({"evidence": raw})
		else:
			for x in raw:
				if isinstance(x, dict):
					if "evidence" in x and isinstance(x["evidence"], list):
						records.append(x)
					elif "items" in x and isinstance(x["items"], list):
						records.append({**x, "evidence": x["items"]})
					elif "text" in x:
						records.append({"evidence": [x]})
		return records

	return [{"evidence": []}]


def _build_markdown(
	item: Dict[str, Any],
	*,
	retrieved_at: str,
	question_id: str,
	question: str,
	retrieval_query: str,
	q_type: str,
) -> str:
	source = _safe_text(item.get("source") or "unknown")
	title = _safe_text(item.get("title") or "Untitled")
	url = _normalize_url(item.get("url") or "")
	dom = _domain(url)
	text = _safe_text(item.get("text"))

	# 1. 将所有溯源信息（包括测试题目）全部沉降到 YAML 元数据中
	meta = {
		"source_type": "external_web_retrieval",
		"retrieval_source": source,
		"source_url": url,
		"source_domain": dom,
		"retrieved_at": retrieved_at,
		"question_id": question_id,
		"q_type": q_type,
		"retrieval_query": retrieval_query,
		"original_question": question, # 题干存入 metadata，不参与向量嵌入
		"text_sha256": _sha256(text),
	}

	front_matter = ["---"]
	for k, v in meta.items():
		vv = str(v).replace("\n", " ").strip()
		front_matter.append(f"{k}: {json.dumps(vv, ensure_ascii=False)}")
	front_matter.append("---")

	# 2. 正文 (page_content) 只保留纯粹的临床医学事实
	body = [
		f"# {title}",
		"",
		text,
		"",
	]

	return "\n".join(front_matter + [""] + body).strip() + "\n"

def _dedup_key(item: Dict[str, Any]) -> str:
	url = _normalize_url(item.get("url") or "")
	txt = _safe_text(item.get("text"))
	if url:
		return f"url:{_sha256(url)}"
	return f"text:{_sha256(txt[:3000])}"


def _load_index(index_path: Path) -> Dict[str, Any]:
	if not index_path.exists():
		return {}
	try:
		with index_path.open("r", encoding="utf-8") as f:
			obj = json.load(f)
			if isinstance(obj, dict):
				return obj
	except Exception:
		pass
	return {}


def _save_index(index_path: Path, index_obj: Dict[str, Any]) -> None:
	with index_path.open("w", encoding="utf-8") as f:
		json.dump(index_obj, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def archive_evidence_records(records: List[Dict[str, Any]], base_dir: Path) -> Dict[str, int]:
	articles_dir, index_path, events_path = _ensure_dirs(base_dir)
	index_obj = _load_index(index_path)

	stats = {
		"records": 0,
		"items_total": 0,
		"items_saved": 0,
		"items_duplicate": 0,
		"items_skipped": 0,
	}

	for ridx, rec in enumerate(records, start=1):
		stats["records"] += 1
		qid = _safe_text(rec.get("question_id") or rec.get("qid") or f"record_{ridx}")
		q = _safe_text(rec.get("question"))
		rq = _safe_text(rec.get("retrieval_query") or rec.get("query") or rec.get("target_info"))
		qt = _safe_text(rec.get("q_type") or "MC")
		evidence = rec.get("evidence") or rec.get("items") or []
		if not isinstance(evidence, list):
			evidence = []

		for item in evidence:
			if not isinstance(item, dict):
				stats["items_skipped"] += 1
				continue
			text = _safe_text(item.get("text"))
			if not text:
				stats["items_skipped"] += 1
				continue

			stats["items_total"] += 1
			key = _dedup_key(item)
			retrieved_at = _now_iso()

			if key in index_obj:
				stats["items_duplicate"] += 1
				_append_jsonl(
					events_path,
					{
						"ts": retrieved_at,
						"event": "duplicate",
						"key": key,
						"existing_file": index_obj[key].get("file"),
						"question_id": qid,
						"retrieval_query": rq,
					},
				)
				continue

			source = _slug(item.get("source") or "unknown", limit=40)
			url = _normalize_url(item.get("url") or "")
			title = _safe_text(item.get("title") or "untitled")
			short = _sha256(url or text[:1200])[:12]
			fname = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source}_{short}.md"
			fpath = articles_dir / fname

			md = _build_markdown(
				item,
				retrieved_at=retrieved_at,
				question_id=qid,
				question=q,
				retrieval_query=rq,
				q_type=qt,
			)
			fpath.write_text(md, encoding="utf-8")

			index_obj[key] = {
				"file": str(fpath.relative_to(base_dir)),
				"saved_at": retrieved_at,
				"source": _safe_text(item.get("source") or "unknown"),
				"title": title,
				"url": url,
				"domain": _domain(url),
				"question_id": qid,
				"retrieval_query": rq,
				"text_sha256": _sha256(text),
			}
			_append_jsonl(
				events_path,
				{
					"ts": retrieved_at,
					"event": "saved",
					"key": key,
					"file": str(fpath.relative_to(base_dir)),
					"question_id": qid,
					"retrieval_query": rq,
				},
			)
			stats["items_saved"] += 1

	_save_index(index_path, index_obj)
	return stats


def _live_research_record(
	*,
	question: str,
	target_info: str,
	q_type: str,
	question_id: str,
	model_name: str,
	use_local_fallback: bool,
) -> Dict[str, Any]:
	from core.llm_engine import LocalLLMClient
	from core.search_agent import research

	retriever = None
	if use_local_fallback:
		from core.advanced_retriever import AdvancedRetriever
		retriever = AdvancedRetriever()

	client = LocalLLMClient()
	evidence, query, _ = research(
		client=client,
		target_info=target_info,
		question=question,
		retriever=retriever,
		q_type=q_type,
		model_name=model_name,
	)

	return {
		"question_id": question_id,
		"question": question,
		"retrieval_query": query,
		"q_type": q_type,
		"evidence": evidence,
	}


def _build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Archive external web retrieval evidence into local KB markdown files."
	)
	p.add_argument(
		"--base-dir",
		default=str(PROJECT_ROOT / "knowledge_base" / "raw_files" / "web_external"),
		help="Target KB expansion directory (default: knowledge_base/raw_files/web_external).",
	)

	# Import mode
	p.add_argument("--input-json", help="Path to JSON/JSONL evidence file to archive.")

	# Live retrieval mode
	p.add_argument("--question", help="Original question for live retrieval mode.")
	p.add_argument("--target-info", help="Missing fact / target info for live retrieval mode.")
	p.add_argument("--q-type", default="MC", choices=["MC", "TF", "QA"], help="Question type for live retrieval mode.")
	p.add_argument("--question-id", default="", help="Optional question ID for provenance.")
	p.add_argument("--model", default=os.environ.get("GLOBAL_AGENT_MODEL", "gpt-4o"), help="Model for query generation/refinement in live retrieval mode.")
	p.add_argument("--disable-local-fallback", action="store_true", help="Disable local retriever fallback during live web research.")
	return p


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	base_dir = Path(args.base_dir).resolve()
	records: List[Dict[str, Any]] = []

	if args.input_json:
		input_path = Path(args.input_json).resolve()
		if not input_path.exists():
			raise FileNotFoundError(f"Input file not found: {input_path}")
		records = _load_records(input_path)

	else:
		if not args.question or not args.target_info:
			raise ValueError("Live mode requires --question and --target-info.")
		rec = _live_research_record(
			question=args.question,
			target_info=args.target_info,
			q_type=args.q_type,
			question_id=args.question_id,
			model_name=args.model,
			use_local_fallback=(not args.disable_local_fallback),
		)
		records = [rec]

	stats = archive_evidence_records(records, base_dir=base_dir)

	print("✅ Web evidence archive completed.")
	print(f"- Base dir: {base_dir}")
	print(f"- Records: {stats['records']}")
	print(f"- Evidence items total: {stats['items_total']}")
	print(f"- Saved new docs: {stats['items_saved']}")
	print(f"- Duplicates skipped: {stats['items_duplicate']}")
	print(f"- Invalid/empty skipped: {stats['items_skipped']}")


if __name__ == "__main__":
	main()