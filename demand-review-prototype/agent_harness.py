# -*- coding: utf-8 -*-
"""
에이전트 하네스: LLM을 '정해진 일'에만 쓰고, 모든 출력을 검증하고, 권한과 호출 상한을 지키게 하는 틀

  - 도구 등록부와 권한 등급: 읽기, 계산, LLM, 사내발송, 제한쓰기(변경 이력 필수), 제안. 승인값 기록은 사람 전용
  - LLM 호출: JSON 스키마 강제, 실패하면 오류를 알려주고 1회 재시도, 그래도 실패하면 규칙으로 대체하고 '사람 확인' 표시
  - 검증기: 출력의 모든 숫자는 입력에 있어야 함(숫자 근거), 인용문은 원문에 있어야 함, 금지 표현, 다음 행동과 내용의 일치
  - 호출 상한과 비용 기록, 모든 호출을 기록 파일로 저장하고 같은 입력이면 기록을 재생(재현성, LLM 없는 PC에서도 실행)

LLM 백엔드: Claude Code CLI(`claude -p`, 구조화 출력). 환경변수 DEMAND_AGENT_LLM
  auto(기본): 기록이 있으면 재생, 없으면 CLI 호출, CLI가 없으면 규칙 대체
  replay: 기록만 사용 / claude: 기록을 무시하고 항상 새로 호출 / off: LLM 없이 규칙만
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "llm_records")
os.makedirs(REC_DIR, exist_ok=True)
MODE = os.environ.get("DEMAND_AGENT_LLM", "auto")
LLM_MODEL = os.environ.get("DEMAND_AGENT_MODEL", "haiku")
CLAUDE = shutil.which("claude")


class PermissionDenied(Exception):
    pass


class Harness:
    LEVELS_ALLOWED = {"읽기", "계산", "LLM", "사내발송", "제한쓰기", "제안"}  # 에이전트에게 허용된 등급

    def __init__(self, max_llm_calls=80):
        self.tools = {}
        self.log = []
        self.llm_calls = 0
        self.max_llm_calls = max_llm_calls
        self.cost = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.llm_seconds = 0.0
        self.lock = threading.Lock()
        self.used_keys = set()

    # ------------------------------------------------------------------ 도구
    def register(self, name, level, fn, desc=""):
        self.tools[name] = {"level": level, "fn": fn, "desc": desc}

    def call(self, step, name, **kw):
        t = self.tools[name]
        if t["level"] not in self.LEVELS_ALLOWED:
            self._log(step, name, t["level"], "차단", f"권한 등급 '{t['level']}'은 에이전트에게 허용되지 않음")
            raise PermissionDenied(name)
        t0 = time.time()
        out = t["fn"](**kw)
        self._log(step, name, t["level"], "실행", "", time.time() - t0)
        return out

    def _log(self, step, name, level, result, note="", sec=0.0, **extra):
        with self.lock:
            self.log.append({"단계": step, "도구": name, "권한": level, "결과": result, "비고": note,
                             "초": round(sec, 2), **extra})

    # ------------------------------------------------------------------ LLM
    def _key(self, task, system, user, schema):
        h = hashlib.sha256(json.dumps([task, system, user, schema, LLM_MODEL], ensure_ascii=False).encode()).hexdigest()
        return f"{task}_{h[:16]}"

    def _raw_llm(self, task, system, user, schema):
        key = self._key(task, system, user, schema)
        path = os.path.join(REC_DIR, key + ".json")
        with self.lock:
            self.used_keys.add(key)
        if MODE in ("auto", "replay") and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            return rec["output"], {"source": "기록 재생", "cost": rec.get("cost", 0), "sec": rec.get("sec", 0),
                                   "tin": rec.get("tokens_in", 0), "tout": rec.get("tokens_out", 0)}
        if MODE in ("replay", "off") or CLAUDE is None:
            return None, {"source": "LLM 없음"}
        with self.lock:
            if self.llm_calls >= self.max_llm_calls:
                return None, {"source": "호출 상한 도달"}
            self.llm_calls += 1
        t0 = time.time()
        cmd = [CLAUDE, "-p", "--model", LLM_MODEL, "--system-prompt", system, "--tools", "",
               "--no-session-persistence", "--output-format", "json", "--json-schema", json.dumps(schema, ensure_ascii=False)]
        try:
            p = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", timeout=180)
            d = json.loads(p.stdout)
            out = d.get("structured_output")
            if out is None:
                out = json.loads(d.get("result", "null"))
        except Exception as e:  # 실패는 검증 실패와 같게 취급
            return None, {"source": f"호출 실패: {type(e).__name__}"}
        sec = time.time() - t0
        meta = {"source": "LLM 호출", "cost": d.get("total_cost_usd", 0.0), "sec": sec,
                "tin": d.get("usage", {}).get("input_tokens", 0) + d.get("usage", {}).get("cache_read_input_tokens", 0)
                + d.get("usage", {}).get("cache_creation_input_tokens", 0),
                "tout": d.get("usage", {}).get("output_tokens", 0)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"task": task, "model": LLM_MODEL, "system": system, "user": user, "schema": schema,
                       "output": out, "cost": meta["cost"], "sec": round(sec, 2), "tokens_in": meta["tin"],
                       "tokens_out": meta["tout"]}, f, ensure_ascii=False, indent=1)
        return out, meta

    def llm_task(self, step, task, system, user, schema, validators, fallback, item=""):
        """LLM 1회 + 검증 실패 시 오류를 알려주고 1회 재시도 + 그래도 실패면 규칙 대체"""
        errs_all = []
        prompt = user
        for attempt in (1, 2):
            out, meta = self._raw_llm(task, system, prompt, schema)
            if out is not None:
                errs = []
                for v in validators:
                    errs += v(out)
                with self.lock:
                    self.cost += meta.get("cost", 0) or 0
                    self.tokens_in += meta.get("tin", 0) or 0
                    self.tokens_out += meta.get("tout", 0) or 0
                    self.llm_seconds += meta.get("sec", 0) or 0
                self._log(step, task, "LLM", "통과" if not errs else "검증 실패", "; ".join(errs)[:200], meta.get("sec", 0),
                          대상=item, 출처=meta["source"], 시도=attempt)
                if not errs:
                    return out, {"source": meta["source"], "attempts": attempt, "fallback": False, "errors": errs_all}
                errs_all += errs
                prompt = user + "\n\n[이전 출력의 검증 오류. 고쳐서 다시 출력]\n" + "\n".join(errs)
            else:
                errs_all.append(meta["source"])
                self._log(step, task, "LLM", "응답 없음", meta["source"], 0, 대상=item, 출처=meta["source"], 시도=attempt)
                break
        out = fallback()
        self._log(step, task, "규칙", "규칙 대체", "사람 확인 필요", 0, 대상=item, 출처="규칙 대체", 시도=0)
        return out, {"source": "규칙 대체", "attempts": 0, "fallback": True, "errors": errs_all}

    def prune_records(self):
        """이번 실행에 쓰지 않은 기록 파일 삭제 (프롬프트를 바꾼 뒤 남은 옛 기록 정리)"""
        n = 0
        for f in os.listdir(REC_DIR):
            if f.endswith(".json") and f[:-5] not in self.used_keys:
                os.remove(os.path.join(REC_DIR, f))
                n += 1
        return n

    def run_parallel(self, fn, args, workers=4):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda a: fn(*a), args))


# ---------------------------------------------------------------------- 검증기
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text):
    out = set()
    for m in NUM.findall(str(text)):
        v = m.replace(",", "")
        try:
            f = float(v)
        except ValueError:
            continue
        out.add(f)
        out.add(round(f))
    return out


def grounded_numbers(source_text, fields):
    """출력 필드의 모든 숫자가 입력에 있어야 한다 (LLM이 숫자를 만들어내지 못하게)"""
    allowed = numbers(source_text)

    def v(out):
        errs = []
        for f in fields:
            val = out.get(f)
            vals = val if isinstance(val, list) else [val]
            for x in vals:
                if x is None:
                    continue
                for n in (numbers(json.dumps(x, ensure_ascii=False)) if isinstance(x, (dict, list)) else numbers(x)):
                    if n not in allowed and abs(n) not in allowed:
                        errs.append(f"'{f}'의 숫자 {n:g}은 입력에 없음")
        return sorted(set(errs))
    return v


def quote_in(source_text, field):
    norm = lambda s: re.sub(r"\s+", "", str(s))

    def v(out):
        q = out.get(field) or ""
        return [] if q and norm(q) in norm(source_text) else [f"인용문 '{q[:30]}'이 원문에 없음"]
    return v


def forbidden(field, phrases):
    def v(out):
        t = str(out.get(field, ""))
        return [f"금지 표현 '{p}'" for p in phrases if p in t]
    return v


def month_fmt(*fields):
    def v(out):
        errs = []
        for f in fields:
            x = out.get(f)
            if x is not None and not re.fullmatch(r"20\d\d-(0[1-9]|1[0-2])", str(x)):
                errs.append(f"'{f}' 형식 오류: {x}")
        return errs
    return v
