# -*- coding: utf-8 -*-
"""
월간 수요검토 에이전트 프로토타입 v7 (가상 데이터)

역할 분담
  ML(숫자)      : 독립 예측, 80% 구간, AI 보정 FCST, 수요 패턴별 모델 자동 선택, 변수 기여도
  LLM(말과 연결): 뉴스 구조화, 예측 설명, 확인 질문 작성, 답변 해석과 다음 행동 선택, 월간 브리핑
  하네스        : 순서 실행, 권한, 스키마와 숫자 근거 검증, 재시도와 규칙 대체, 호출 상한, 비용과 기록
  사람          : 승인 (승인값 기록 도구는 에이전트 권한 밖)

실행: python run_monthly_review.py   (LLM 설정은 agent_harness.py 상단 설명 참고)
"""
import json
import os
import time

import numpy as np
import pandas as pd

import virtual_data as V
import forecast_engine as F
import agent_harness as H

T_START = time.time()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

REVIEW_ORIGIN = pd.Period("2026-10", "M")
FORECAST_DATE = pd.Timestamp("2026-10-01")
REVIEW_DATE = pd.Timestamp("2026-10-07")     # 검토와 질문 발송일
ANSWER_DUE = pd.Timestamp("2026-10-09")      # 1차 답변 마감
REMIND_DATE = pd.Timestamp("2026-10-10")     # 재질문, 에스컬레이션 판단일
ESCALATE_DAYS = 5                             # 판단 기한까지 이 일수 이하로 남았는데 답이 없으면 팀장에게
TARGETS = [REVIEW_ORIGIN + h for h in (1, 2, 3)]
SPARSE_MA6 = 5.0
PREBUY_SIG_CHG = 0.03   # 생산지수 전년 동월 대비 변화가 이보다 작은데 주문이 많으면 선구매 여부 질문
SURGE_RATIO = 1.3
QUESTION_CAP = 4       # 담당자 1명에게 한 회차에 묻는 품목 상한 (영업 부담 관리)
MATERIAL_TONS = 10.0   # 구간 이탈이라도 차이 물량이 이 값 미만이면 검토 대신 알림
SPARSE_PATTERNS = ("간헐", "덩어리", "신규")  # 월별 구간 비교가 의미 없는 패턴은 구간 참고
EC_NAME = V.END_CUSTOMERS.set_index("code").name.to_dict()
EC_LIST = list(EC_NAME)

hx = H.Harness(max_llm_calls=150)
NO_DOT = "가운뎃점(·)은 쓰지 말고 쉼표를 쓴다."


def eq(a, b):
    na = a is None or (isinstance(a, float) and np.isnan(a))
    nb = b is None or (isinstance(b, float) and np.isnan(b))
    return (na and nb) or (not na and not nb and a == b)


# ===========================================================================
# 0) 백테스트와 모델 선택 (ML)
# ===========================================================================
BT = F.run_backtest()
CHAMP, CHAMP_TABLE = F.select_champions(BT)
BT = F.apply_champion(BT, CHAMP)
F.CALIB_LEVEL, CALIB_TRIALS = F.choose_level(BT)   # 모델 선택 구간에서만 고름
BT = F.add_intervals(BT, BT)
BT_SUMMARY = F.backtest_summary(BT)
FLAG_EVAL = F.eval_flags(BT)
START_CHOICE, START_TABLE = F.choose_start(BT)
INV_SIM = F.inventory_sim(BT, START_CHOICE)
PLAN_K, PLAN_COST = F.choose_plan_factor(BT, START_CHOICE)
INV_SIM2 = F.inventory_sim2(BT, START_CHOICE, PLAN_K)
PAT_SUMMARY = F.pattern_summary(BT)
HIER = F.hierarchy_summary(BT)
HIER = HIER[HIER.계층 != "계열"]  # 이 가상 데이터는 계열당 그레이드가 하나라 그레이드와 같음
MES_G, MES_LOTS = F.mes_lead_and_inventory()
LEAD80 = MES_G.set_index("그레이드").리드타임_80분위.to_dict()   # 판단 기한 계산에 쓰는 MES 리드타임
COLOR_MIX = F.color_mix(MES_LOTS)

_ok = BT[(BT.y > 0) & (BT.target < REVIEW_ORIGIN)]
TYP_SHARE = {h: float(np.median(_ok[_ok.h == h].accepted / _ok[_ok.h == h].y)) for h in (1, 2, 3)}

# 거래처별 FCST 신뢰도 이력: 검토 기준월에 마감된 납기 월, 최근 6개 회차, 1개월 앞
_closed = BT[(BT.h == 1) & (BT.target < REVIEW_ORIGIN)]
_last6 = sorted(_closed.origin.unique())[-6:]
_rec = _closed[_closed.origin.isin(_last6)].copy()
_rec["거래처"] = _rec.item.str.split("-").str[0]
REL = (_rec.groupby("거래처").apply(lambda g: pd.Series({"회차": g.origin.nunique(), "건수": len(g),
                                                          "합계편향": (g.fcst.sum() - g.y.sum()) / g.y.sum(),
                                                          "WAPE": F.wape(g.y, g.fcst)}), include_groups=False)
       .reset_index().sort_values("합계편향", ascending=False))
REL_MAP = REL.set_index("거래처").합계편향.to_dict()


def rel_text(d):
    b = REL_MAP.get(d, 0.0)
    return f"최근 6회 합계 {b:+.0%} ({'과대' if b > 0.05 else '과소' if b < -0.05 else '중립'})"


# ===========================================================================
# 1) 도구 정의
# ===========================================================================
def fit_lgb(train, cols, alpha):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=10, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                          random_state=F.SEED, verbose=-1)
    m.fit(train[cols], train.y)
    return m


DRIVER_GROUPS = {"최근 주문 흐름": ["lag1", "lag2", "lag3", "ma3", "ma6", "ma12", "yoy", "zero_share12", "months_active"],
                 "접수 주문": ["accepted", "acc_ratio"], "최종 고객 생산지수": ["sig_last", "sig_chg3"],
                 "계절": ["target_month"], "품목 특성": ["item_code", "dtype_code", "fam_code", "pattern_code", "h"]}


def tool_data_check():
    rows = []
    for it in V.ITEMS.item:
        hist = F.history(it, REVIEW_ORIGIN, 24)
        valid = hist[~np.isnan(hist)]
        med = np.median(valid[valid > 0]) if np.any(valid > 0) else 0
        spikes = int(np.sum(valid > 3 * med)) if med > 0 else 0
        rows.append({"품목": it, "이력개월": len(valid), "0인달": int(np.sum(valid == 0)), "급증달(중앙값 3배 초과)": spikes,
                     "패턴": F.classify_pattern(it, REVIEW_ORIGIN)})
    return pd.DataFrame(rows)


def tool_forecast(link_table):
    """기준월 예측: 패턴별 챔피언 점예측, 보정 구간, AI 보정 FCST, 변수 기여도"""
    tr = F.build(pd.period_range("2024-01", REVIEW_ORIGIN - 1, freq="M"), link_table)
    tr = tr[tr.target < REVIEW_ORIGIN]
    te = F.build([REVIEW_ORIGIN], link_table, with_target=False)
    m50 = fit_lgb(tr, F.SIG_COLS, 0.5)
    te["p_lgb"] = np.maximum(m50.predict(te[F.SIG_COLS]), 0)
    te["lgb_q10"] = np.minimum(np.maximum(fit_lgb(tr, F.SIG_COLS, 0.1).predict(te[F.SIG_COLS]), 0), te.p_lgb)
    te["lgb_q90"] = np.maximum(fit_lgb(tr, F.SIG_COLS, 0.9).predict(te[F.SIG_COLS]), te.p_lgb)
    te["p_base"] = te.ma3.fillna(0)
    te["p_sba"] = [F.sba(i, REVIEW_ORIGIN) for i in te.item]
    te["p_corr"] = np.maximum(fit_lgb(tr, F.CORR_COLS, 0.5).predict(te[F.CORR_COLS]), 0)
    te["scale"] = np.maximum(te.ma12.fillna(te.ma3).fillna(1), 1)
    te = F.apply_champion(te, CHAMP)
    te = F.add_intervals(te, BT)
    contrib = m50.predict(te[F.SIG_COLS], pred_contrib=True)
    cdf = pd.DataFrame(contrib[:, :-1], columns=F.SIG_COLS)
    for g, cols in DRIVER_GROUPS.items():
        te["기여_" + g] = cdf[cols].sum(axis=1).values
    te["기여_기준값"] = contrib[:, -1]
    te["고객FCST"] = te.fcst
    return te


def tool_link_events(events, link_table):
    """구조화된 근거를 연결표와 대조: 채택, 참고 보관, 시차 불명확, 연결 부족"""
    known = {v for v in link_table.values() if v}
    out = []
    for ev in events:
        ec, eff = ev["end_customer"], ev["effect_from"]
        effp = pd.Period(eff, "M") if eff else None
        if ec == "UNKNOWN" or ec not in known:
            out.append({**ev, "품목": "-", "분류": "참고 보관", "사유": "연결표에 없는 기업 또는 기업 미특정"})
            continue
        for it in V.ITEMS.item:
            lk = link_table.get(it)
            if lk is None:
                out.append({**ev, "품목": it, "분류": "연결 부족", "사유": "품목의 최종 고객 미확인"})
            elif lk == ec:
                if effp is None:
                    c, r = "시차 불명확", "영향 시작 월 없음"
                elif effp > TARGETS[-1]:
                    c, r = "참고 보관", "예측 기간 밖"
                else:
                    c, r = "채택", "최종 고객, 거래처, 그레이드, 기간 연결 확인"
                out.append({**ev, "품목": it, "분류": c, "사유": r})
    df = pd.DataFrame(out)
    keep = df[df.분류 != "참고 보관"]
    ref = df[df.분류 == "참고 보관"].drop_duplicates("id")
    return pd.concat([keep, ref], ignore_index=True)


def tool_review(fc, ev_link):
    rows = []
    FLAG, _thr = F.review_flag(fc)  # 검토 예산: AI 보정 차이 상위 (모델 선택 구간에서 고른 규칙)
    for _, r in fc.iterrows():
        it, d, g = r["item"], r["item"].split("-")[0], r["item"].split("-", 1)[1]
        deadline = r.target.start_time - pd.Timedelta(days=int(np.ceil(LEAD80[g])))
        fixed = REVIEW_DATE > deadline
        low_conf = r.pattern in SPARSE_PATTERNS or (not np.isnan(r.ma6) and r.ma6 < SPARSE_MA6) or r.calib_n < F.MIN_CALIB_N
        reasons, topics = [], []
        if not low_conf and FLAG[r.name]:
            side = "구간 상단 초과" if r.고객FCST > r.q90 else "구간 하단 미만" if r.고객FCST < r.q10 else "구간 안"
            reasons.append(f"고객 FCST와 AI 보정값 차이 상위 {F.REVIEW_SHARE:.0%} ({side})")
            topics.append("FCST 상향 근거(신규 프로젝트, 재고 확충)" if r.고객FCST > r.p_corr else "FCST 하향 사유(감산, 공급사 변경)")
        surge = r.accepted / (TYP_SHARE[int(r.h)] * r.ma6) if r.ma6 and r.ma6 > 0 else np.nan
        driver = r.sig_yoy if not np.isnan(r.sig_yoy) else r.market_yoy  # 최종 고객(없으면 시장) 생산지수 전년 동월 대비
        if r.pattern not in SPARSE_PATTERNS and not np.isnan(surge) and surge > SURGE_RATIO:
            reasons.append("접수 주문이 평소보다 많음")
            topics.append("접수 주문 중 취소나 이연 가능 물량")
            if abs(driver) < PREBUY_SIG_CHG:
                topics.append("생산 변화 없이 주문이 많은 이유(가격 기대에 따른 선구매 여부)")
        e_it = ev_link[ev_link.품목 == it]
        for _, e in e_it[e_it.분류 == "채택"].iterrows():
            if pd.Period(e.effect_from, "M") <= r.target:
                reasons.append(f"채택 근거 {e.id}({e.direction})")
                topics.append(f"{e.id}의 변화가 이 그레이드 적용 제품에 해당하는지와 물량 영향")
        link_need = bool(len(e_it[e_it.분류 == "연결 부족"]))
        if link_need:
            topics.append("이 품목의 최종 고객과 적용 제품")
        for _, e in e_it[e_it.분류 == "시차 불명확"].iterrows():
            topics.append(f"{e.id}의 영향 시작 시점")
        gap = abs(r.고객FCST - r.p_champ)
        width = r.q90 - r.q10
        status = "정상" if not reasons else ("확정 구간" if fixed and gap <= width else "검토")
        rows.append({"품목": it, "거래처": d, "그레이드": g, "담당": V.ITEMS.set_index("item").loc[it, "담당"],
                     "납기월": str(r.target), "패턴": r.pattern, "챔피언": r.champion, "상태": status,
                     "판단기한": deadline.date(), "연결확인": "필요" if link_need else "-",
                     "구간신뢰": "구간 참고" if low_conf else "보정 구간", "검토사유": "; ".join(reasons) or "-",
                     "확인주제": " / ".join(dict.fromkeys(topics)) or "-",
                     "고객FCST": round(r.고객FCST, 1), "독립예측": round(r.p_champ, 1),
                     "구간하단": round(r.q10, 1), "구간상단": round(r.q90, 1), "AI보정FCST": round(r.p_corr, 1),
                     "접수주문": round(r.accepted, 1), "FCST신뢰도이력": rel_text(d),
                     **{k: round(r[k], 1) for k in r.index if str(k).startswith("기여_")},
                     "_기한": deadline, "_차이": gap * (1 + abs(REL_MAP.get(d, 0))),
                     "_그룹": {"검토": 0, "확정 구간": 1}.get(status, 2)})
    df = pd.DataFrame(rows).sort_values(["_그룹", "_기한", "_차이"], ascending=[True, True, False])
    return df.drop(columns=["_그룹"])


def tool_quantity(base, accepted, target, fact):
    """확인된 사실을 수량 제안으로 바꾸는 결정적 계산 (LLM은 이 계산을 하지 않는다). 접수 주문 아래로는 내리지 않음"""
    v, note = base, "AI 보정 FCST"
    if fact:
        ft = fact.get("fact_type")
        if ft in ("감산", "증산") and fact.get("rate_pct") is not None and fact.get("start_month"):
            if pd.Period(target, "M") >= pd.Period(fact["start_month"], "M"):
                v = base * (1 + fact["rate_pct"] / 100)
                note = f"AI 보정 FCST {base:.1f} x (1{fact['rate_pct']:+g}%) [담당 확인]"
        if ft == "선구매":
            red = {x["month"]: abs(x["tons"]) for x in fact.get("reduce_by_month") or []}
            if str(target) in red:
                v = base - red[str(target)]
                note = f"AI 보정 FCST {base:.1f} - 앞당긴 주문 {red[str(target)]:g}톤 [담당 확인]"
    if v < accepted:
        v, note = accepted, note + f" -> 접수 주문 {accepted:.1f} 하한"
    return round(max(v, 0.0), 1), note


def tool_approve(*a, **k):  # 사람 전용
    raise RuntimeError("사람만 실행")


hx.register("데이터점검", "읽기", tool_data_check)
hx.register("예측실행", "계산", tool_forecast)
hx.register("근거연결", "계산", tool_link_events)
hx.register("검토표작성", "계산", tool_review)
hx.register("수량환산", "제안", tool_quantity)
hx.register("승인값기록", "사람전용", tool_approve)

# ===========================================================================
# 2) LLM 작업 정의 (프롬프트, 스키마, 검증기, 규칙 대체)
# ===========================================================================
EC_TEXT = ", ".join(f"{k}={v}" for k, v in EC_NAME.items())

EVENT_SYS = ("너는 소재 회사 수요계획팀의 근거 정리 담당이다. 기사 한 건에서 사실만 뽑아 JSON으로 낸다. "
             "추측하지 말고, 기사에 없는 값은 null로 둔다. 기업이 목록에 없거나 특정 기업이 아니면 UNKNOWN. " + NO_DOT)
EVENT_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "end_customer": {"type": "string", "enum": EC_LIST + ["UNKNOWN"]},
    "direction": {"type": "string", "enum": ["상향", "하향", "중립"]},
    "effect_from": {"type": ["string", "null"], "description": "영향 시작 월 YYYY-MM, 기사에 없으면 null"},
    "product_scope": {"type": "string"},
    "magnitude_pct": {"type": ["number", "null"]},
    "evidence_quote": {"type": "string", "description": "근거가 되는 기사 원문 구절 그대로"}},
    "required": ["end_customer", "direction", "effect_from", "product_scope", "magnitude_pct", "evidence_quote"]}


def llm_event(n):
    user = (f"기업 목록: {EC_TEXT}\n기사 공개일: {n['published']}\n"
            f"상대 시점(내년, 2분기 등)은 공개일 기준으로 YYYY-MM으로 바꾼다. 분기는 첫 달.\n기사: {n['text']}")
    vals = [H.quote_in(n["text"], "evidence_quote"), H.grounded_numbers(n["text"], ["magnitude_pct"]),
            H.month_fmt("effect_from")]

    def fb():  # 규칙 대체: 기업명 문자열 일치만, 시점은 비움
        ec = next((k for k, v in EC_NAME.items() if v in n["text"]), "UNKNOWN")
        return {"end_customer": ec, "direction": "하향" if any(w in n["text"] for w in ["감산", "단종", "둔화"]) else "상향",
                "effect_from": None, "product_scope": "", "magnitude_pct": None, "evidence_quote": n["text"][:20]}
    out, meta = hx.llm_task("3 근거구조화", "근거구조화", EVENT_SYS, user, EVENT_SCHEMA, vals, fb, n["id"])
    return {"id": n["id"], "published": n["published"], "text": n["text"], **out, "_meta": meta}


EXPLAIN_SYS = ("너는 수요예측 모델의 결과를 수요계획 담당자에게 설명한다. 주어진 숫자만 쓰고 새 숫자를 만들지 않는다. "
               "수량을 얼마로 하라고 권하지 않는다. 2문장 이내, 한국어. " + NO_DOT)
EXPLAIN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "explanation": {"type": "string"}, "main_driver": {"type": "string", "enum": list(DRIVER_GROUPS)}},
    "required": ["explanation", "main_driver"]}


def llm_explain(row):
    contrib = {g: row["기여_" + g] for g in DRIVER_GROUPS}
    user = (f"품목 {row['품목']}, 납기 {row['납기월']}\n고객 FCST {row['고객FCST']}톤, 모델 예측 {row['독립예측']}톤, "
            f"80% 구간 {row['구간하단']}~{row['구간상단']}톤, 접수 주문 {row['접수주문']}톤\n"
            f"모델 기준값 {row['기여_기준값']}톤에서 요인별 기여(톤): " + ", ".join(f"{k} {v:+.1f}" for k, v in contrib.items())
            + f"\n검토 사유: {row['검토사유']}\n모델이 FCST와 다른 이유를 가장 큰 요인 중심으로 설명하라.")
    vals = [H.grounded_numbers(user, ["explanation"]), H.forbidden("explanation", ["하세요", "해야 합니다", "권장", "·"])]

    def fb():
        k = max(contrib, key=lambda x: abs(contrib[x]))
        return {"explanation": f"가장 큰 요인은 {k}({contrib[k]:+.1f}톤)입니다.", "main_driver": k}
    out, meta = hx.llm_task("13 예측설명", "예측설명", EXPLAIN_SYS, user, EXPLAIN_SCHEMA, vals, fb, f"{row['품목']} {row['납기월']}")
    return out, meta


QUESTION_SYS = ("너는 수요계획팀 에이전트다. 영업 담당자에게 보낼 확인 질문을 쓴다. 주어진 사실과 숫자만 쓰고, "
                "수량을 바꾸라고 요구하거나 권하지 않는다. 판단 기한 날짜를 반드시 넣는다. 정중하고 짧게(300자 안팎, 최대 600자), 납기별 숫자는 고객 FCST와 모델만 쓴다, 질문은 물음표로 끝낸다. "
                "기사는 예측 모델에 들어가지 않으므로 모델이 기사를 반영했다고 쓰지 않는다. 검토 사유를 바꾸어 말하지 않는다. " + NO_DOT)
QUESTION_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "subject": {"type": "string"}, "question": {"type": "string"},
    "ask_fields": {"type": "array", "items": {"type": "string", "enum": [
        "최종고객", "적용제품", "물량변화율", "시작월", "선구매여부", "취소이연", "거래처재고", "영향시점", "FCST근거"]}}},
    "required": ["subject", "question", "ask_fields"]}


def next_deadline(grp, as_of=REVIEW_DATE):
    """아직 지나지 않은 가장 빠른 판단 기한. 모두 지났으면 None"""
    fut = [d for d in grp["판단기한"] if pd.Timestamp(d) >= as_of]
    return min(fut) if fut else None


def llm_question(item, grp):
    lines = [f"받는 사람: {grp['담당'].iloc[0]}", f"품목: {item} (거래처 {grp['거래처'].iloc[0]}, 그레이드 {grp['그레이드'].iloc[0]})",
             f"가장 빠른 판단 기한: {next_deadline(grp)}"]
    for _, r in grp.iterrows():
        lines.append(f"- 납기 {r['납기월']}: 고객 FCST {r['고객FCST']}톤, 모델 {r['독립예측']}톤(구간 {r['구간하단']}~{r['구간상단']}), "
                     f"접수 주문 {r['접수주문']}톤, 사유: {r['검토사유']}")
    topics = sorted(set(" / ".join(grp["확인주제"]).split(" / ")) - {"-"})
    ev_ids = sorted({t.split("의")[0] for t in topics if t.startswith("N")})
    for e in EVENTS_OK:
        if e["id"] in ev_ids:
            lines.append(f"관련 기사 {e['id']}: {e['text']}")
    lines.append("확인할 주제: " + "; ".join(topics))
    user = "\n".join(lines)
    vals = [H.grounded_numbers(user, ["question", "subject"]),
            H.forbidden("question", ["조정하세요", "변경하세요", "수정하세요", "로 확정", "줄이세요", "늘리세요", "·", "모델은 기사", "모델에 반영", "반영되어 있습니다"]),
            lambda o: [] if "?" in o.get("question", "") else ["물음표 없음"],
            lambda o: [] if len(o.get("question", "")) <= 600 else ["600자 초과"], H.forbidden("subject", ["·"])]

    def fb():
        return {"subject": f"[수요검토] {item} 확인 요청", "question": f"{item} 관련 확인 부탁드립니다: {'; '.join(topics)}? "
                f"판단 기한 {next_deadline(grp)}", "ask_fields": ["FCST근거"]}
    out, meta = hx.llm_task("6 질문작성", "질문작성", QUESTION_SYS, user, QUESTION_SCHEMA, vals, fb, item)
    return {"품목": item, "받는사람": grp["담당"].iloc[0], **out, "주제": "; ".join(topics), "_meta": meta, "_prompt": user}


ANSWER_SYS = ("너는 수요계획팀 에이전트다. 영업 담당자 답변을 읽고 확인된 사실만 JSON으로 정리하고 다음 행동을 고른다. "
              "답변에 없는 숫자나 기업은 쓰지 않는다. 다음 행동: 연결표_갱신(최종 고객이 확인됨), 수량_환산(감산, 증산, 선구매 등 "
              "물량 사실이 숫자로 확인됨), 연결표_갱신_및_수량_환산, 재질문(보류이거나 질문한 핵심에 답이 없음), 종결. "
              "변화가 없다는 답은 답변완료, 변화없음. 여러 고객사에 나뉘어 최종 고객을 특정할 수 없다는 답은 답변완료, 종결(미귀속 유지). "
              "현재 연결표와 같은 최종 고객이면 연결표_갱신이 필요 없다. 날짜는 입력의 기준일만 따르고 실제 오늘 날짜는 쓰지 않는다. "
              "변화율은 해당 품목의 월 주문량 대비로 본다. 담당자가 변화율과 시작 월을 숫자로 답했으면 '정도', '것으로 봅니다' 같은 표현이 있어도 "
              "답변완료로 보고 수량_환산을 고른다. 선구매로 줄어드는 물량은 reduce_by_month에 양수 톤으로 적는다. " + NO_DOT)
ANSWER_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "status": {"type": "string", "enum": ["답변완료", "보류", "무관"]},
    "end_customer": {"type": ["string", "null"], "enum": EC_LIST + [None]},
    "fact_type": {"type": "string", "enum": ["감산", "증산", "선구매", "이연", "변화없음", "없음"]},
    "rate_pct": {"type": ["number", "null"], "description": "감산은 음수, 증산은 양수"},
    "start_month": {"type": ["string", "null"]},
    "reduce_by_month": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "month": {"type": "string"}, "tons": {"type": "number"}}, "required": ["month", "tons"]}},
    "next_action": {"type": "string", "enum": ["연결표_갱신", "수량_환산", "연결표_갱신_및_수량_환산", "재질문", "종결"]},
    "follow_up": {"type": ["string", "null"]},
    "quote": {"type": "string", "description": "핵심 근거가 되는 답변 원문 구절 그대로"}},
    "required": ["status", "end_customer", "fact_type", "rate_pct", "start_month", "reduce_by_month", "next_action",
                 "follow_up", "quote"]}


def answer_consistency(o):
    errs = []
    na, ft = o.get("next_action"), o.get("fact_type")
    if "수량_환산" in (na or "") and not (ft in ("감산", "증산") and o.get("rate_pct") is not None and o.get("start_month")) \
            and not (ft == "선구매" and o.get("reduce_by_month")):
        errs.append("수량_환산인데 물량 사실(변화율과 시작 월, 또는 월별 감소량)이 없음")
    if "연결표_갱신" in (na or "") and not o.get("end_customer"):
        errs.append("연결표_갱신인데 최종 고객이 없음")
    if o.get("status") == "보류" and na != "재질문":
        errs.append("보류인데 재질문이 아님")
    if ft == "감산" and (o.get("rate_pct") or 0) > 0:
        errs.append("감산인데 변화율이 양수")
    return errs


def llm_answer(q, reply):
    cur = F.LINK0.get(q["품목"])
    user = (f"기업 목록: {EC_TEXT}\n기준일: 2026-10-09 (11월, 12월은 2026년, 1월은 2027년)\n"
            f"현재 연결표: {q['품목']}의 최종 고객 = {cur or '미확인'}\n"
            f"보낸 질문: {q['question']}\n질문한 항목: {', '.join(q['ask_fields'])}\n답변: {reply}")
    vals = [H.quote_in(reply, "quote"), H.grounded_numbers(reply + " 2026 2027 11 12 1 01", ["rate_pct", "reduce_by_month"]),
            H.grounded_numbers(user, ["follow_up"]),
            lambda o: ["변화율이 비현실적(절댓값 50% 초과): 수량 환산하지 말고 재질문"] if o.get("rate_pct") is not None and abs(o["rate_pct"]) > 50 else [],
            lambda o: ["월별 감소량은 양수여야 함"] if any(x["tons"] <= 0 for x in o.get("reduce_by_month") or []) else [],
            H.month_fmt("start_month"), answer_consistency,
            lambda o: [] if not o.get("end_customer") or EC_NAME[o["end_customer"]].split()[-1] in reply
            else ["답변에 없는 최종 고객"]]

    def fb():
        return {"status": "보류", "end_customer": None, "fact_type": "없음", "rate_pct": None, "start_month": None,
                "reduce_by_month": [], "next_action": "재질문", "follow_up": None, "quote": reply[:15]}
    out, meta = hx.llm_task("9 답변해석", "답변해석", ANSWER_SYS, user, ANSWER_SCHEMA, vals, fb, q["품목"])
    return out, meta


BRIEF_SYS = ("너는 수요계획팀 에이전트다. 팀장에게 보낼 월간 수요검토 요약을 쓴다. 주어진 숫자만 쓰고 새 숫자를 만들지 않는다. "
             "헤드라인 1문장과 핵심 4개 이내, 결정이 필요한 것을 먼저. 날짜와 남은 일수는 입력의 기준일 기준이며 실제 오늘 날짜는 언급하지 않는다. " + NO_DOT)
BRIEF_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "headline": {"type": "string"}, "bullets": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
    "required": ["headline", "bullets"]}

# ===========================================================================
# 3) 월간 실행
# ===========================================================================
print("[1] 데이터 점검과 1회차 예측 ...")
DQ = hx.call("1 데이터점검", "데이터점검")
LINK0 = dict(F.LINK0)
FC1 = hx.call("2 예측", "예측실행", link_table=LINK0)

print("[2] 근거 구조화 (LLM) ...")
NEWS_OK = [n for n in V.NEWS if pd.Timestamp(n["published"]) <= REVIEW_DATE]
NEWS_LATE = [n["id"] for n in V.NEWS if pd.Timestamp(n["published"]) > REVIEW_DATE]
EVENTS_OK = hx.run_parallel(llm_event, [(n,) for n in NEWS_OK], workers=6)
EV_TABLE = pd.DataFrame([{k: v for k, v in e.items() if k != "_meta"} | {"출처": e["_meta"]["source"]} for e in EVENTS_OK])
for c in ["end_customer", "direction", "effect_from"]:
    EV_TABLE["정답_" + c] = [next(n["gold"][c] for n in V.NEWS if n["id"] == i) for i in EV_TABLE.id]
    EV_TABLE["일치_" + c] = [eq(a, b) for a, b in zip(EV_TABLE[c], EV_TABLE["정답_" + c])]

EVL1 = hx.call("4 근거연결", "근거연결", events=EVENTS_OK, link_table=LINK0)
REV1 = hx.call("5 검토표", "검토표작성", fc=FC1, ev_link=EVL1)

print("[3] 확인 질문 작성 (LLM, 담당자별 묶음) ...")
ASK = REV1[(REV1.확인주제 != "-") & (REV1.상태 != "확정 구간")]


def item_priority(grp):
    """질문 우선순위: 차이 물량(FCST 신뢰도 가중) + 선구매, 채택 근거, 연결 확인 가점"""
    d = grp.거래처.iloc[0]
    gap = float((grp.고객FCST - grp.독립예측).abs()[grp.상태 == "검토"].sum())
    t = " / ".join(grp.확인주제)
    return gap * (1 + abs(REL_MAP.get(d, 0))) + (30 if "선구매" in t else 0) + (20 if "변화가 이 그레이드" in t else 0) \
        + (5 if "최종 고객과 적용 제품" in t else 0)


PRIO = []
for it in dict.fromkeys(ASK.품목):
    g = ASK[ASK.품목 == it]
    PRIO.append({"품목": it, "담당": g.담당.iloc[0], "우선순위점수": round(item_priority(g), 1),
                 "가장빠른기한": next_deadline(g)})
PRIO = pd.DataFrame(PRIO).sort_values(["담당", "우선순위점수"], ascending=[True, False])
PRIO["순위"] = PRIO.groupby("담당").cumcount() + 1
PRIO["처리"] = np.where(PRIO.순위 <= QUESTION_CAP, "이번 회차 질문", "다음 회차로 미룸 (AI 보정 FCST로 시작)")
ASK_ITEMS = list(PRIO[PRIO.순위 <= QUESTION_CAP].품목)
DEFERRED = list(PRIO[PRIO.순위 > QUESTION_CAP].품목)
for it in DEFERRED:
    hx._log("6 질문작성", "질문예산", "계산", "미룸", f"{it}: 담당자별 상한 {QUESTION_CAP}건 초과")

BATCH_SYS = ("너는 수요계획팀 에이전트다. 영업 담당자 한 명에게 보낼 확인 요청 메시지 한 통을 쓴다. 품목마다 질문을 하나씩 쓰고, "
             "주어진 사실과 숫자만 쓴다. 수량을 바꾸라고 요구하거나 권하지 않는다. 품목별 질문에는 그 품목의 판단 기한 날짜를 넣고 물음표로 끝낸다. "
             "납기별 숫자는 고객 FCST와 모델만 쓴다. 기사는 예측 모델에 들어가지 않으므로 모델이 기사를 반영했다고 쓰지 않는다. "
             "본문은 인사와 회신 기한 안내만 짧게 쓴다. " + NO_DOT)


def batch_schema(items):
    return {"type": "object", "additionalProperties": False, "properties": {
        "subject": {"type": "string"}, "body": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "item": {"type": "string", "enum": items}, "question": {"type": "string"},
            "ask_fields": {"type": "array", "items": {"type": "string", "enum": [
                "최종고객", "적용제품", "물량변화율", "시작월", "선구매여부", "취소이연", "거래처재고", "영향시점", "FCST근거"]}}},
            "required": ["item", "question", "ask_fields"]}}},
        "required": ["subject", "body", "items"]}


def item_facts(item, grp):
    lines = [f"[품목 {item}] 거래처 {grp['거래처'].iloc[0]}, 그레이드 {grp['그레이드'].iloc[0]}, 판단 기한 {next_deadline(grp)}"]
    for _, r in grp.iterrows():
        lines.append(f"- 납기 {r['납기월']}: 고객 FCST {r['고객FCST']}톤, 모델 {r['독립예측']}톤(구간 {r['구간하단']}~{r['구간상단']}), "
                     f"접수 주문 {r['접수주문']}톤, 사유: {r['검토사유']}")
    topics = sorted(set(" / ".join(grp["확인주제"]).split(" / ")) - {"-"})
    ev_ids = sorted({t.split("의")[0] for t in topics if t.startswith("N")})
    for e in EVENTS_OK:
        if e["id"] in ev_ids:
            lines.append(f"  관련 기사 {e['id']}: {e['text']}")
    lines.append("  확인할 주제: " + "; ".join(topics))
    return "\n".join(lines), topics


def llm_batch(rep, items):
    blocks, topics_map = [], {}
    for it in items:
        txt, tp = item_facts(it, ASK[ASK.품목 == it])
        blocks.append(txt)
        topics_map[it] = tp
    user = f"받는 사람: {rep}\n회신 요청일: {ANSWER_DUE.date()}\n\n" + "\n\n".join(blocks)
    bad = ["조정하세요", "변경하세요", "수정하세요", "로 확정", "줄이세요", "늘리세요", "·", "모델은 기사", "모델에 반영", "반영되어 있습니다"]

    def v_items(o):
        errs = []
        got = [x["item"] for x in o.get("items", [])]
        if sorted(got) != sorted(items):
            errs.append(f"품목 누락 또는 중복: {got}")
        for x in o.get("items", []):
            if "?" not in x["question"]:
                errs.append(f"{x['item']} 물음표 없음")
            if len(x["question"]) > 450:
                errs.append(f"{x['item']} 450자 초과")
            errs += [f"{x['item']} 금지 표현 '{b}'" for b in bad if b in x["question"]]
        return errs
    vals = [H.grounded_numbers(user, ["subject", "body"]), v_items, H.forbidden("body", bad),
            lambda o: H.grounded_numbers(user, ["q"])({"q": [x["question"] for x in o.get("items", [])]})]

    def fb():
        return {"subject": f"[수요검토] 확인 요청 {len(items)}건", "body": f"회신 요청일 {ANSWER_DUE.date()}",
                "items": [{"item": it, "question": f"{it} 관련 확인 부탁드립니다: {'; '.join(topics_map[it])}? "
                           f"판단 기한 {next_deadline(ASK[ASK.품목 == it])}", "ask_fields": ["FCST근거"]} for it in items]}
    out, meta = hx.llm_task("6 질문작성", "질문묶음작성", BATCH_SYS, user, batch_schema(items), vals, fb, rep)
    return rep, out, meta, topics_map


BATCHES = hx.run_parallel(llm_batch, [(rep, list(g.품목)) for rep, g in PRIO[PRIO.순위 <= QUESTION_CAP].groupby("담당")], workers=6)
QUESTIONS, OUTBOX = [], []
for rep, out, meta, tmap in BATCHES:
    hx._log("7 발송", "사내발송", "사내발송", "실행", f"{rep} <- {out['subject']} ({len(out['items'])}품목)")
    OUTBOX.append({"발송일": REVIEW_DATE.date(), "받는사람": rep, "제목": out["subject"], "품목수": len(out["items"]),
                   "본문": out["body"] + "\n\n" + "\n".join(f"{i + 1}. [{x['item']}] {x['question']}" for i, x in enumerate(out["items"])),
                   "작성": meta["source"]})
    for x in out["items"]:
        QUESTIONS.append({"품목": x["item"], "받는사람": rep, "subject": out["subject"], "question": x["question"],
                          "ask_fields": x["ask_fields"], "주제": "; ".join(tmap[x["item"]]), "_meta": meta})

print("[4] 답변 수집과 해석 (LLM) ...")
DEFAULT_REPLY = "특별히 들은 변화는 없습니다. 고객 계획대로 진행될 것 같습니다."
COMPOUNDER_REPLY = "컴파운드 원료로 여러 고객사에 나눠 들어가서 특정 최종 고객으로 나누기는 어렵습니다."
replies = []
for q in QUESTIONS:
    it = q["품목"]
    if it in V.REPLIES:
        rp = V.REPLIES[it]
        text, gold = (rp["text"], rp["gold"]) if rp else (None, None)
    elif V.ITEMS.set_index("item").loc[it, "유형"] == "컴파운더":
        text, gold = COMPOUNDER_REPLY, {"status": "답변완료", "end_customer": None, "fact_type": "없음"}
    elif F.LINK0.get(it):
        text, gold = DEFAULT_REPLY, {"status": "답변완료", "fact_type": "변화없음"}
    else:
        text, gold = None, None  # 최종 고객 미확인 품목은 기한까지 답이 없는 경우로 둔다
    replies.append((q, text, gold))
NOT_ASKED_DESIGNED = sorted(set(k for k, v in V.REPLIES.items() if v) - set(ASK_ITEMS))

parsed = hx.run_parallel(lambda q, t, g: (llm_answer(q, t) if t else (None, None)), [(q, t, g) for q, t, g in replies], workers=6)

LINK1 = dict(LINK0)
CHANGELOG, FACTS, ANS_ROWS, OPEN_ITEMS = [], {}, [], []
for (q, text, gold), (out, meta) in zip(replies, parsed):
    it = q["품목"]
    if text is None:
        ANS_ROWS.append({"품목": it, "답변": "(기한까지 답변 없음)", "상태": "무응답", "다음행동": "재질문"})
        OPEN_ITEMS.append(it)
        continue
    row = {"품목": it, "답변": text, "상태": out["status"], "최종고객": out["end_customer"], "사실": out["fact_type"],
           "변화율%": out["rate_pct"], "시작월": out["start_month"],
           "월별감소": "; ".join(f"{x['month']} {x['tons']:g}톤" for x in out["reduce_by_month"]) or "-",
           "다음행동": out["next_action"], "후속질문": out["follow_up"], "인용": out["quote"], "해석": meta["source"]}
    for k in ["status", "end_customer", "fact_type", "rate_pct", "start_month"]:
        if k in gold:
            row["일치_" + k] = eq(out[k], gold[k]) or (k == "rate_pct" and out[k] is not None and gold[k] is not None
                                                     and abs(out[k] - gold[k]) < 1e-6)
    if "reduce" in gold:
        row["일치_reduce"] = {x["month"]: x["tons"] for x in out["reduce_by_month"]} == gold["reduce"]
    ANS_ROWS.append(row)
    na = out["next_action"]
    if "연결표_갱신" in na and out["end_customer"] and LINK1.get(it) != out["end_customer"]:
        CHANGELOG.append({"품목": it, "변경전": LINK1.get(it), "변경후": out["end_customer"], "확인자": q["받는사람"],
                          "확인일": ANSWER_DUE.date(), "근거": out["quote"]})
        hx._log("10 실행", "연결표갱신", "제한쓰기", "실행", f"{it}: {LINK1.get(it)} -> {out['end_customer']} (변경 이력 기록)")
        LINK1[it] = out["end_customer"]
    if "수량_환산" in na:
        FACTS[it] = out
    if na == "재질문":
        OPEN_ITEMS.append(it)
ANSWERS = pd.DataFrame(ANS_ROWS)

# 재질문과 에스컬레이션 (규칙: 판단 기한까지 남은 일수)
REMIND = []
for it in OPEN_ITEMS:
    grp = REV1[(REV1.품목 == it)]
    dl = next_deadline(grp, REMIND_DATE)
    if dl is None:
        hx._log("11 재질문", "사내발송", "사내발송", "생략", f"{it}: 판단 기한 모두 경과, 사후 평가로 넘김")
        continue
    left = (pd.Timestamp(dl) - REMIND_DATE).days
    owner = grp.담당.iloc[0]
    esc = left <= ESCALATE_DAYS
    REMIND.append({"품목": it, "대상": owner + (f", 참조 {V.TEAM_LEAD[owner.split()[0]]}" if esc else ""),
                   "판단기한": dl, "남은일수": left, "조치": "재질문 + 팀장 에스컬레이션" if esc else "재질문",
                   "문구": f"[재요청] {it} 확인 요청에 아직 답이 없습니다. 판단 기한 {dl}까지 {left}일 남았습니다. "
                         "답이 없으면 이번 회차는 AI 보정 FCST를 시작값으로 검토합니다."})
    hx._log("11 재질문", "사내발송", "사내발송", "실행", f"{it}: {'에스컬레이션' if esc else '재질문'}")
REMIND = pd.DataFrame(REMIND)

print("[5] 답변 반영 후 재실행과 제안값 ...")
FC2 = hx.call("12 재실행", "예측실행", link_table=LINK1)
EVL2 = hx.call("12 재실행", "근거연결", events=EVENTS_OK, link_table=LINK1)
REV2 = hx.call("12 재실행", "검토표작성", fc=FC2, ev_link=EVL2)
props, notes = [], []
for _, r in REV2.iterrows():
    _h = (pd.Period(r.납기월, "M") - REVIEW_ORIGIN).n
    _c = START_CHOICE[_h]
    _base = r.AI보정FCST if _c == "AI 보정" else r.독립예측 if _c == "독립 예측" else round((r.AI보정FCST + r.독립예측) / 2, 1)
    v, nt = hx.call("12 재실행", "수량환산", base=_base, accepted=r.접수주문, target=r.납기월, fact=FACTS.get(r.품목))
    nt = nt.replace("AI 보정 FCST", {"AI 보정": "AI 보정 FCST", "독립 예측": "독립 예측", "평균": "독립 예측과 AI 보정 평균"}[_c])
    props.append(v)
    notes.append(nt)
REV2["에이전트제안값"] = props
RESOLVED = set(ANSWERS[(ANSWERS.상태 == "답변완료") & (ANSWERS.다음행동 != "재질문")].품목) if len(ANSWERS) else set()
REV2.loc[(REV2.상태 == "검토") & REV2.품목.isin(RESOLVED), "상태"] = "확인 완료(승인만)"
REV2["제안근거"] = notes
ans_map = ANSWERS.set_index("품목")
REV2["질문답변"] = [("답변: " + str(ans_map.loc[i, "답변"])) if i in ans_map.index else "-" for i in REV2.품목]

print("[6] 예측 설명 (LLM) ...")
EXP_ROWS = REV2[REV2.상태.isin(["검토", "확인 완료(승인만)"])]
exps = hx.run_parallel(lambda r: llm_explain(r), [(r,) for _, r in EXP_ROWS.iterrows()], workers=6)
REV2["모델설명"] = "-"
REV2.loc[EXP_ROWS.index, "모델설명"] = [o["explanation"] for o, m in exps]
REV2["판단"], REV2["승인값"], REV2["승인자"] = "", "", ""

# 권한 자체 점검: 에이전트가 승인값을 쓰려 하면 차단되어야 함
try:
    hx.call("16 자체점검", "승인값기록")
    PERM_OK = False
except H.PermissionDenied:
    PERM_OK = True

# ===========================================================================
# 4) 사후 평가 (가상 실적), 다음 달 lot 계획, 효율
# ===========================================================================
REV2["실적"] = [V.ACT[(i, pd.Period(t, "M"))] for i, t in zip(REV2.품목, REV2.납기월)]
fact_items = set(FACTS)
POST = []
for name, sub in [("전체 90건", REV2), ("답변으로 사실이 확인된 품목", REV2[REV2.품목.isin(fact_items)])]:
    POST.append({"범위": name, "건수": len(sub), "WAPE_고객FCST": F.wape(sub.실적, sub.고객FCST),
                 "WAPE_독립예측": F.wape(sub.실적, sub.독립예측), "WAPE_AI보정FCST": F.wape(sub.실적, sub.AI보정FCST),
                 "WAPE_에이전트제안값": F.wape(sub.실적, sub.에이전트제안값)})
POST = pd.DataFrame(POST)

nxt = REV2[REV2.납기월 == str(TARGETS[0])].groupby("그레이드").에이전트제안값.sum().rename("제안물량_톤").reset_index()
LOTPLAN = nxt.merge(COLOR_MIX, on="그레이드")
LOTPLAN["색상별_톤"] = (LOTPLAN.제안물량_톤 * LOTPLAN.색상비중).round(1)
LOTPLAN = LOTPLAN.merge(MES_G[["그레이드", "lot크기_중앙값"]], on="그레이드")
LOTPLAN["예상_lot수"] = np.ceil(LOTPLAN.색상별_톤 / LOTPLAN.lot크기_중앙값).astype(int)
LOTPLAN = LOTPLAN[["그레이드", "색상", "색상비중", "색상별_톤", "lot크기_중앙값", "예상_lot수"]]

# 최종 고객별 수요 화면: 거래처 x 그레이드 예측을 연결표(답변 반영 후)로 묶는다. 연결 안 된 물량은 미귀속으로 따로 표시
_ec = V.END_CUSTOMERS.set_index("code")
_v = REV2.copy()
_v["최종고객"] = _v.품목.map(lambda i: LINK1.get(i) or "미귀속")
_v["산업"] = _v.최종고객.map(lambda c: _ec.loc[c, "산업"] if c in _ec.index else "미귀속")
_v["최종고객명"] = _v.최종고객.map(lambda c: _ec.loc[c, "name"] if c in _ec.index else "미귀속(컴파운더, 대리점 경유 등)")
_v["반폭"] = (_v.구간상단 - _v.구간하단) / 2
EC_VIEW = (_v.groupby(["산업", "최종고객명", "납기월"]).agg(품목수=("품목", "nunique"), 고객FCST=("고객FCST", "sum"),
           독립예측=("독립예측", "sum"), 에이전트제안값=("에이전트제안값", "sum"), 실적_가상=("실적", "sum"),
           범위반폭=("반폭", lambda x: float(np.sqrt((x ** 2).sum()))))
           .reset_index())
EC_VIEW["80%범위(근사)"] = [f"{a - w:.0f}~{a + w:.0f}" for a, w in zip(EC_VIEW.독립예측, EC_VIEW.범위반폭)]
EC_VIEW = EC_VIEW.drop(columns="범위반폭").round(1)
_v["실적_가상"] = _v.실적
IND_VIEW = _v.groupby(["산업", "납기월"])[["고객FCST", "독립예측", "에이전트제안값", "실적_가상"]].sum().round(1).reset_index()
_v["제품군"] = _v.품목.map(lambda i: F.IT.loc[i, "제품군"])
FAM_VIEW = (_v[_v.납기월 == str(TARGETS[0])].pivot_table(index="제품군", columns="산업", values="에이전트제안값", aggfunc="sum", fill_value=0)
            .round(0))
FAM_VIEW["합계"] = FAM_VIEW.sum(axis=1)
_n = EC_VIEW[EC_VIEW.최종고객명 != "미귀속(컴파운더, 대리점 경유 등)"]
EC_WAPE = {"고객FCST": F.wape(_n.실적_가상, _n.고객FCST), "에이전트제안값": F.wape(_n.실적_가상, _n.에이전트제안값)}
UNATTR_SHARE = float(_v[_v.최종고객 == "미귀속"].독립예측.sum() / _v.독립예측.sum())

LOG = pd.DataFrame(hx.log)
llm_log = LOG[LOG.권한 == "LLM"]
n_units = len(REV2)
n_review1 = int((REV1.상태 == "검토").sum())
n_review2 = int((REV2.상태 == "검토").sum())
n_resolved = int((REV2.상태 == "확인 완료(승인만)").sum())
n_fallback = int((LOG.결과 == "규칙 대체").sum())
first_pass = int(((llm_log.결과 == "통과") & (llm_log.시도 == 1)).sum())
n_tasks = first_pass + int(((llm_log.결과 == "통과") & (llm_log.시도 == 2)).sum()) + n_fallback
ev_acc = float(EV_TABLE[[c for c in EV_TABLE if c.startswith("일치_")]].values.mean())
ans_cols = [c for c in ANSWERS if c.startswith("일치_")]
ans_acc = float(np.nanmean(ANSWERS[ans_cols].astype(float).values))
EFF = pd.DataFrame([
    ("검토 단위(거래처 x 그레이드 x 납기 월)", n_units, "건"),
    ("사람이 수량을 검토할 단위 (1회차)", n_review1, f"건, 전체의 {n_review1 / n_units:.0%}"),
    ("답변으로 확인돼 제안값 승인만 남은 단위", n_resolved, "건"),
    ("사람이 직접 판단할 단위 (답변 반영 후)", n_review2, f"건, 전체의 {n_review2 / n_units:.0%}"),
    ("확정 구간으로 알림만 남긴 단위", int((REV1.상태 == "확정 구간").sum()), "건"),
    ("담당자별 묶음 메시지", len(OUTBOX), f"통, 질문 품목 {len(QUESTIONS)}개 (사람 작성 0건)"),
    ("질문 예산으로 다음 회차로 미룬 품목", len(DEFERRED), f"품목, 담당자당 상한 {QUESTION_CAP}"),
    ("예측 범위 보정 수준 (모델 선택 구간에서 선택)", F.CALIB_LEVEL, ""),
    ("도착한 답변을 해석해 정리", int(ANSWERS.상태.ne("무응답").sum()), "건 (사람 정리 0건)"),
    ("답변으로 갱신한 연결표", len(CHANGELOG), "건 (변경 이력 기록)"),
    ("답변 사실로 계산한 수량 제안 품목", len(FACTS), "품목"),
    ("자동 재질문, 팀장 에스컬레이션", f"{len(REMIND)}, {int(REMIND.조치.str.contains('에스컬레이션').sum()) if len(REMIND) else 0}", "건"),
    ("LLM 작업 수 (검증 1회 통과, 재시도 통과, 규칙 대체)", f"{n_tasks} ({first_pass}, {n_tasks - first_pass - n_fallback}, {n_fallback})", "건"),
    ("LLM 비용 (새로 호출한 경우 기록 기준)", round(hx.cost, 3), "달러"),
    ("LLM 처리 시간 합계 (병렬 4개 이전)", round(hx.llm_seconds, 1), "초"),
    ("LLM 근거 구조화 필드 정확도 (가상 정답 대비)", round(ev_acc, 3), ""),
    ("LLM 답변 해석 필드 정확도 (가상 정답 대비)", round(ans_acc, 3), ""),
    ("연결률 (직전 월 주문량 기준) 답변 전 -> 후", None, ""),
    ("전체 실행 시간", None, "초"),
], columns=["지표", "값", "단위"])


def link_rate(lt):
    vol = {it: V.ACT[(it, REVIEW_ORIGIN - 1)] for it in V.ITEMS.item}
    vol = {k: (0 if np.isnan(v) else v) for k, v in vol.items()}
    return sum(v for it, v in vol.items() if lt.get(it)) / sum(vol.values())


RATE1, RATE2 = link_rate(LINK0), link_rate(LINK1)
EFF.loc[EFF.지표.str.startswith("연결률"), "값"] = f"{RATE1:.1%} -> {RATE2:.1%}"

# 날짜, 규칙 자체 점검
assert not (set(EV_TABLE.id) & set(NEWS_LATE)), "검토일 이후 공개 근거가 사용됨"
assert PERM_OK, "승인값 기록이 차단되지 않음"

print("[6b] 답변 해석 스트레스 시험 (LLM) ...")
STRESS_Q = {"question": "이 품목의 최종 고객, 물량 변화율, 시작 월을 확인 부탁드립니다. 판단 기한은 2026-10-14입니다. 알려주실 수 있을까요?",
            "ask_fields": ["최종고객", "물량변화율", "시작월"]}
st_out = hx.run_parallel(lambda i, c: llm_answer({**STRESS_Q, "품목": f"TEST-{i:02d}"}, c[1]), [(i, c) for i, c in enumerate(V.STRESS, 1)], workers=6)
srows = []
for (cat, text, gold, safe), (o, meta) in zip(V.STRESS, st_out):
    row = {"유형": cat, "답변": text, "상태": o["status"], "사실": o["fact_type"], "변화율%": o["rate_pct"], "시작월": o["start_month"],
           "월별감소": "; ".join(f"{x['month']} {x['tons']:g}톤" for x in o["reduce_by_month"]) or "-", "최종고객": o["end_customer"],
           "다음행동": o["next_action"], "수량환산금지": safe, "해석": meta["source"]}
    checks = [o["status"] in gold["status"]]
    for k in ["fact_type", "rate_pct", "start_month", "end_customer"]:
        if k in gold:
            checks.append(eq(o[k], gold[k]))
    if "reduce" in gold:
        checks.append({x["month"]: abs(x["tons"]) for x in o["reduce_by_month"]} == gold["reduce"])
    row["필드정답"] = all(checks)
    row["필드정답률"] = sum(checks) / len(checks)
    quant = "수량_환산" in o["next_action"]
    row["위험(불확실한데 수량 환산)"] = bool(safe and quant)
    row["확실한 답 수량 환산"] = bool((not safe) and quant)
    srows.append(row)
STRESS_T = pd.DataFrame(srows)
ST_FIELD = float(STRESS_T.필드정답률.mean())
ST_UNSAFE = int(STRESS_T["위험(불확실한데 수량 환산)"].sum())
ST_SAFE_N = int(STRESS_T.수량환산금지.sum())
ST_CONV = int(STRESS_T["확실한 답 수량 환산"].sum())
ST_CONV_N = int((~STRESS_T.수량환산금지).sum())
EFF = pd.concat([EFF, pd.DataFrame([
    ("답변 해석 스트레스 시험 20건 필드 정답률", round(ST_FIELD, 3), ""),
    ("불확실한 답을 수량으로 바꾼 건수", f"{ST_UNSAFE} / {ST_SAFE_N}", "건 (0이 목표)"),
    ("확실한 답을 수량 제안으로 바꾼 건수", f"{ST_CONV} / {ST_CONV_N}", "건"),
], columns=["지표", "값", "단위"])], ignore_index=True)

print("[7] 월간 브리핑 (LLM) ...")
p0 = POST.iloc[0]
facts_txt = (f"기준일 2026-10-10. 검토 단위 {n_units}건 중 답변으로 확인돼 승인만 남은 단위 {n_resolved}건, 사람이 직접 판단할 단위 {n_review2}건. "
             f"담당자 {len(OUTBOX)}명에게 묶음 메시지 {len(OUTBOX)}통으로 {len(QUESTIONS)}품목 질문, 질문 상한으로 {len(DEFERRED)}품목은 다음 회차로 미룸. "
             f"답변 {int(ANSWERS.상태.ne('무응답').sum())}건 해석, 연결표 갱신 {len(CHANGELOG)}건, 수량 제안 {len(FACTS)}품목, "
             f"재질문 {len(REMIND)}건. 연결률 {RATE1:.1%}에서 {RATE2:.1%}. "
             f"판단 기한이 가장 빠른 검토 품목: " + ", ".join(f"{r.품목} {r.납기월}(기한 {r.판단기한})"
                                                    for _, r in REV2[REV2.상태 == '검토'].head(3).iterrows())
             + ". 무응답 또는 보류: " + (", ".join(f"{r.품목}(기한 {r.판단기한}, {r.남은일수}일 남음)" for _, r in REMIND.iterrows()) or "없음"))
brief, _ = hx.llm_task("14 브리핑", "브리핑", BRIEF_SYS, facts_txt, BRIEF_SCHEMA,
                       [H.grounded_numbers(facts_txt, ["headline", "bullets"]), H.forbidden("headline", ["·"]),
                        lambda o: ["가운뎃점 사용"] if any("·" in b for b in o.get("bullets", [])) else [],
                        lambda o: ["실제 오늘 날짜 언급"] if "오늘" in json.dumps(o, ensure_ascii=False) else []],
                       lambda: {"headline": "월간 수요검토 결과", "bullets": [facts_txt[:200]]})
LOG = pd.DataFrame(hx.log)
ELAPSED = time.time() - T_START
EFF.loc[EFF.지표 == "전체 실행 시간", "값"] = round(ELAPSED, 1)
EFF.loc[EFF.지표.str.startswith("LLM 비용"), "값"] = round(hx.cost, 3)

# ===========================================================================
# 5) 출력
# ===========================================================================
SHOW = ["품목", "납기월", "상태", "판단기한", "패턴", "고객FCST", "독립예측", "구간하단", "구간상단", "AI보정FCST",
        "접수주문", "에이전트제안값", "제안근거", "검토사유", "모델설명", "질문답변", "연결확인", "구간신뢰",
        "FCST신뢰도이력", "판단", "승인값", "승인자"]
xlsx = os.path.join(OUT, "월간검토_결과_가상데이터.xlsx")
with pd.ExcelWriter(xlsx) as w:
    pd.DataFrame({"안내": [
        "모든 데이터는 가상이며 실제 고객, 거래, 코드, 수치가 아님. 결과는 기능 작동 확인이며 효과의 증명이 아님",
        "숫자는 ML(LightGBM 분위수 모델 등)이 만들고, LLM은 숫자를 만들지 않음. LLM이 정리한 사실은 수량환산 도구를 거쳐서만 제안값이 됨",
        "뉴스는 질문과 근거 분류에만 쓰고 수량에 넣지 않음. 수량 제안은 담당자가 답변으로 확인한 사실만 반영",
        "판단 기한 = 납기 월 1일 - MES lot 이력 리드타임 80분위(그레이드별)",
        f"LLM: Claude Code CLI({H.LLM_MODEL}), 모드 {H.MODE}. llm_records 폴더의 기록으로 같은 결과를 재생할 수 있음",
        "사후 평가의 실적은 가상 데이터의 2026년 11월~2027년 1월 값",
    ]}).to_excel(w, sheet_name="안내", index=False)
    REV2[SHOW].to_excel(w, sheet_name="1_최종검토표_제안값", index=False)
    EFF.to_excel(w, sheet_name="2_효율지표", index=False)
    POST.to_excel(w, sheet_name="3_사후평가_10월회차", index=False)
    BT_SUMMARY.to_excel(w, sheet_name="4_백테스트", index=False)
    pd.concat([CHAMP_TABLE.assign(표="모델선택구간"), PAT_SUMMARY.assign(표="최종평가구간")]).to_excel(
        w, sheet_name="5_패턴별_모델선택", index=False)
    HIER.to_excel(w, sheet_name="6_계층별_오차", index=False)
    EV_TABLE.to_excel(w, sheet_name="7_근거구조화_LLM", index=False)
    pd.DataFrame(OUTBOX).to_excel(w, sheet_name="8_질문발송함_담당자별", index=False)
    PRIO.to_excel(w, sheet_name="8b_질문우선순위_예산", index=False)
    STRESS_T.to_excel(w, sheet_name="18_답변해석_스트레스시험", index=False)
    pd.DataFrame(CALIB_TRIALS, columns=["보정수준", "모델선택구간_포함률"]).to_excel(w, sheet_name="19_보정수준_선택", index=False)
    FLAG_EVAL.to_excel(w, sheet_name="20_예외선별_평가", index=False)
    START_TABLE.to_excel(w, sheet_name="21_시작값_선택", index=False)
    INV_SIM.to_excel(w, sheet_name="22_재고시뮬레이션", index=False)
    INV_SIM2.to_excel(w, sheet_name="22b_생산기준값_비용균형", index=False)
    EC_VIEW.to_excel(w, sheet_name="23_최종고객별_수요", index=False)
    IND_VIEW.to_excel(w, sheet_name="24_산업별_수요", index=False)
    FAM_VIEW.to_excel(w, sheet_name="25_제품군x산업_다음달")
    ANSWERS.to_excel(w, sheet_name="9_답변해석_LLM", index=False)
    pd.DataFrame(CHANGELOG).to_excel(w, sheet_name="10_연결표_변경이력", index=False)
    REMIND.to_excel(w, sheet_name="11_재질문_에스컬레이션", index=False)
    REV1[[c for c in SHOW if c in REV1.columns]].to_excel(w, sheet_name="12_검토표_1회차", index=False)
    LOTPLAN.to_excel(w, sheet_name="13_다음달_lot계획", index=False)
    MES_G.to_excel(w, sheet_name="14_MES_리드타임_재고", index=False)
    REL.to_excel(w, sheet_name="15_FCST신뢰도이력", index=False)
    DQ.to_excel(w, sheet_name="16_데이터점검", index=False)
    LOG.to_excel(w, sheet_name="17_에이전트_실행기록", index=False)


def pct(x):
    return f"{x:.1%}"


fin = BT_SUMMARY[BT_SUMMARY.구간 == "최종평가"]
L = ["월간 수요검토 에이전트 프로토타입 v7 (가상 데이터)",
     f"예측 모델: {'LightGBM' if F.HAS_LGB else 'scikit-learn 대체'} | LLM: Claude Code CLI {H.LLM_MODEL} (모드 {H.MODE})", "",
     "[1] 수치 예측 (ML). 최종 평가: 모델 선택에 쓰지 않은 마지막 6개 시점"]
for _, r in fin.iterrows():
    L.append(f"  {int(r['예측거리(개월)'])}개월 앞 WAPE: 기준 {pct(r.WAPE_기준모델)}, 고객FCST {pct(r.WAPE_고객FCST)}, "
             f"독립예측 {pct(r.WAPE_패턴별챔피언)}, AI보정FCST {pct(r.WAPE_AI보정FCST)} | 구간 포함률 {pct(r.구간포함률_보정전)} -> {pct(r['구간포함률_보정후(목표80%)'])}")
L.append("  패턴별 챔피언: " + ", ".join(f"{r.패턴} {r.챔피언}" for _, r in CHAMP_TABLE.iterrows()))
L.append("  계층별 1개월 앞 WAPE(독립예측): " + ", ".join(f"{r.계층} {pct(r.WAPE_챔피언)}" for _, r in HIER.iterrows()))
_fe = FLAG_EVAL[FLAG_EVAL.구간 == "최종평가"]
_old = _fe[_fe.규칙.str.startswith("v6")].iloc[0]; _new = _fe[(_fe.규칙.str.startswith("v7")) & (_fe.검토예산 == F.REVIEW_SHARE)].iloc[0]
_rnd = _fe[_fe.규칙.str.contains("무작위")].iloc[0]; _vol = _fe[_fe.규칙.str.contains("물량")].iloc[0]
L.append(f"  예외 선별(최종평가, 큰 오차 = 고객 FCST 오차 10톤 이상이면서 20% 이상, 비율 {_new.큰오차_비율:.0%}): "
         f"v7 규칙 검토 {_new.검토비율:.0%}에서 정밀도 {_new.정밀도:.0%}, 재현율 {_new.재현율:.0%}, 오차 {_new.잡아낸오차비중:.0%} 포착 | "
         f"v6 규칙 {_old.정밀도:.0%}, {_old.잡아낸오차비중:.0%} | 무작위 {_rnd.정밀도:.0%}, {_rnd.잡아낸오차비중:.0%} | 물량 상위 {_vol.정밀도:.0%}, {_vol.잡아낸오차비중:.0%}")
L.append(f"  접수 주문을 뺀 LightGBM 1개월 앞 WAPE {pct(fin.iloc[0].WAPE_LGB_접수주문제외)} (기준 모델 {pct(fin.iloc[0].WAPE_기준모델)})")
L.append(f"  데이터 모드: {V.DATA_MODE}")
L.append(f"  생산 기준값 계수 {PLAN_K:.2f} (모델 선택 구간에서 초과 + 1.5 x 부족 최소): " + " | ".join(f"{r['계획 기준']} 비용지수 {r['비용지수(초과+1.5x부족)']:.2f}일" for _, r in INV_SIM2.iterrows()))
L.append("  제안 시작값(모델 선택 구간에서 선택): " + ", ".join(f"{h}개월 {c}" for h, c in START_CHOICE.items()))
L.append("  재고 시뮬레이션(최종평가 1개월 앞, 그 값대로 생산했다면): " + " | ".join(
    f"{r['계획 기준']} 초과 {r.초과생산_톤:.0f}톤({r.초과_재고일수:.1f}일), 부족 {r.부족_톤:.0f}톤({r.부족_일수:.1f}일)" for _, r in INV_SIM.iterrows()))
L.append(f"  예측 범위 보정 수준 {F.CALIB_LEVEL:.2f} (모델 선택 구간 포함률: " + ", ".join(f"{a:.2f} -> {b:.1%}" for a, b in CALIB_TRIALS) + ")")
L.append("  판단 기한용 MES 리드타임 80분위(일): " + ", ".join(f"{k} {v:g}" for k, v in LEAD80.items()))
L.append("")
L.append("[2] LLM 작업 (하네스 검증 포함)")
L.append(f"  근거 구조화 {len(EVENTS_OK)}건, 필드 정확도 {pct(ev_acc)} | 질문 작성 {len(QUESTIONS)}건 | 답변 해석 "
         f"{int(ANSWERS.상태.ne('무응답').sum())}건, 필드 정확도 {pct(ans_acc)} | 예측 설명 {len(exps)}건 | 브리핑 1건")
L.append(f"  검증 1회 통과 {first_pass}건, 재시도 후 통과 {n_tasks - first_pass - n_fallback}건, 규칙 대체 {n_fallback}건")
L.append(f"  비용 {hx.cost:.3f}달러, LLM 처리 시간 합계 {hx.llm_seconds:.0f}초, 새 호출 {hx.llm_calls}회 (나머지는 기록 재생)")
L.append(f"  답변 해석 스트레스 시험 {len(STRESS_T)}건: 필드 정답률 {pct(ST_FIELD)}, 불확실한 답을 수량으로 바꾼 건 {ST_UNSAFE}/{ST_SAFE_N}, 확실한 답을 수량 제안으로 바꾼 건 {ST_CONV}/{ST_CONV_N}")
L.append(f"  검토일 이후 공개 근거 {', '.join(NEWS_LATE)} 제외 확인, 승인값 기록 도구 차단 확인")
if NOT_ASKED_DESIGNED:
    L.append(f"  참고: 답변이 준비됐지만 질문 대상이 아니었던 품목 {', '.join(NOT_ASKED_DESIGNED)}")
L.append("")
L.append("[3] 업무 루프")
L.append(f"  검토 단위 {n_units}건 -> 1회차 검토 {n_review1}건, 확정 구간 알림 {(REV1.상태 == '확정 구간').sum()}건")
L.append(f"  선구매 확인 주제가 붙은 품목: {', '.join(sorted(set(REV1[REV1.확인주제.str.contains('선구매')].품목))) or '없음'}")
L.append(f"  질문 예산: 담당자당 {QUESTION_CAP}품목, 묶음 메시지 {len(OUTBOX)}통으로 {len(QUESTIONS)}품목 질문, {len(DEFERRED)}품목은 다음 회차로 미룸")
L.append(f"  질문 {len(QUESTIONS)}품목 발송 -> 답변 {int(ANSWERS.상태.ne('무응답').sum())}건 해석, 연결표 갱신 {len(CHANGELOG)}건, "
         f"수량 제안 {len(FACTS)}품목, 재질문 {len(REMIND)}건")
L.append(f"  연결률 {pct(RATE1)} -> {pct(RATE2)} | 답변 반영 후: 승인만 남은 단위 {n_resolved}건, 직접 판단 {n_review2}건")
for _, r in ANSWERS.iterrows():
    L.append(f"   - {r.품목}: {r.상태}, {r.get('사실', '-')}, 다음 행동 {r.다음행동}")
L.append("")
L.append("[3b] 최종 고객별 수요 (거래처 x 그레이드 예측을 연결표로 묶음, 답변 반영 후)")
for _, r in EC_VIEW[EC_VIEW.납기월 == str(TARGETS[0])].iterrows():
    L.append(f"  {r.산업} {r.최종고객명} {r.납기월}: 품목 {r.품목수}, 고객FCST {r.고객FCST:.0f}톤, 독립예측 {r.독립예측:.0f}톤({r['80%범위(근사)']}), 제안값 {r.에이전트제안값:.0f}톤, 가상 실적 {r.실적_가상:.0f}톤")
L.append("  다음 달 제안값, 제품군 x 산업(톤): " + "; ".join(f"{fam} " + ", ".join(f"{c} {v:.0f}" for c, v in row.items()) for fam, row in FAM_VIEW.iterrows()))
L.append(f"  미귀속 물량 비중(독립예측 기준) {UNATTR_SHARE:.0%} | 최종 고객 단위 3개월 WAPE: 고객FCST {EC_WAPE['고객FCST']:.1%}, 제안값 {EC_WAPE['에이전트제안값']:.1%}")
L.append("")
L.append("[4] 10월 회차 사후 평가 (가상 실적 2026-11 ~ 2027-01)")
for _, r in POST.iterrows():
    L.append(f"  {r.범위}: 고객FCST {pct(r.WAPE_고객FCST)}, 독립예측 {pct(r.WAPE_독립예측)}, AI보정FCST {pct(r.WAPE_AI보정FCST)}, "
             f"에이전트 제안값 {pct(r.WAPE_에이전트제안값)}")
L.append("")
L.append("[5] 월간 브리핑 (LLM 작성, 숫자 근거 검증 통과)")
L.append("  " + brief["headline"])
for b in brief["bullets"]:
    L.append("  - " + b)
L.append("")
L.append(f"전체 실행 {ELAPSED:.0f}초. 상세: output/월간검토_결과_가상데이터.xlsx")
PRUNED = hx.prune_records() if H.MODE in ("auto", "claude") and n_fallback == n_fallback else 0
import review_ui
review_ui.write_review_html(REV2, REVIEW_ORIGIN, os.path.join(OUT, "검토화면.html"))
TEXT = "\n".join(L)
with open(os.path.join(OUT, "실행_요약.txt"), "w", encoding="utf-8") as f:
    f.write(TEXT)
with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"bt_final": fin.to_dict("records"), "post": POST.to_dict("records"), "eff": EFF.astype(str).to_dict("records"),
               "rate": [RATE1, RATE2], "ev_acc": ev_acc, "ans_acc": ans_acc, "cost": hx.cost, "llm_calls": hx.llm_calls,
               "n_units": n_units, "n_review1": n_review1, "n_review2": n_review2, "brief": brief}, f, ensure_ascii=False,
              indent=1, default=str)
print(TEXT)
