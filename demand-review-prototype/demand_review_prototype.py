# -*- coding: utf-8 -*-
"""
LOTTE AX Challenge 2026 - 월간 수요검토 에이전트 미니 프로토타입 (가상 데이터)

목적: 기획안의 핵심 흐름이 실제로 돌아가는지 가상 데이터로 확인한다.
  1) 가상 데이터 생성 (거래처 6, 그레이드 3, 최종 고객 3, 공표 시차가 있는 최종 고객 신호, 편향된 고객 FCST)
  2) 백테스트: 기준 모델 vs LightGBM(주문 이력만) vs LightGBM(최종 고객 신호 추가), 마지막 6개 시점은 최종 평가용
  3) 예측구간(10, 90% 분위수)과 포함률, 구간 폭
  4) 검토 대상 규칙과 품목별 검토표, 근거 분류, 확인 질문
  5) 연결 지도 성장(답변 저장 후 같은 근거가 채택되는 장면, 연결률 변화)
  6) 거래처별 FCST 신뢰도 이력, 모의 승인 기반 사후 평가

주의:
  - 모든 데이터는 가상이며 롯데케미칼의 실제 고객, 거래, 수치가 아니다.
  - LLM이 맡을 부분(뉴스에서 이벤트 추출, 질문 문장 작성)은 이 프로토타입에서 규칙 기반으로 대신한다.
  - 사후 평가의 승인값은 사람이 실제로 승인한 값이 아니라 규칙으로 만든 '모의 승인'이다.

실행: python demand_review_prototype.py
필요: pandas, numpy, scikit-learn, openpyxl (lightgbm이 있으면 사용, 없으면 scikit-learn으로 자동 대체)
"""
import os
import sys
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingRegressor

SEED = 7
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1) 가상 데이터
# ---------------------------------------------------------------------------
MONTHS = pd.period_range("2023-01", "2027-01", freq="M")
END_CUSTOMERS = {"E_A": "완성차 A", "E_B": "완성차 B", "E_C": "가전 C"}

# 품목 = 거래처 × 그레이드. true_ec = 실제 최종 고객(가상), linked = 주문에 최종 고객이 기록돼 있는지
ITEMS = pd.DataFrame([
    ("D1", "G1", "E_A", True, 120), ("D1", "G2", "E_A", True, 80),
    ("D2", "G1", "E_B", True, 150), ("D2", "G3", "E_B", True, 60),
    ("D3", "G2", "E_C", True, 90),  ("D3", "G3", "E_C", True, 70),
    ("D4", "G1", "E_A", True, 110), ("D4", "G2", "E_C", False, 50),
    ("D5", "G2", "E_B", False, 100), ("D5", "G3", "E_B", False, 40),
    ("D6", "G1", "E_C", False, 65), ("D6", "G3", "E_A", True, 55),
], columns=["거래처", "그레이드", "true_ec", "linked", "base"])
ITEMS["item"] = ITEMS["거래처"] + "-" + ITEMS["그레이드"]

# 최종 고객 생산지수(가상): 추세 + 계절 + 충격. E_A는 2026년 하반기 감산 충격
t = np.arange(len(MONTHS))
signal = {}
for ec, amp, trend in [("E_A", 0.08, 0.002), ("E_B", 0.06, 0.001), ("E_C", 0.10, -0.001)]:
    s = 100 * (1 + trend * t) * (1 + amp * np.sin(2 * np.pi * (t % 12) / 12)) + rng.normal(0, 2, len(t))
    signal[ec] = s
shock_start = MONTHS.get_loc(pd.Period("2026-06", "M"))
signal["E_A"][shock_start:] *= 0.82          # E_A 감산
signal["E_B"][MONTHS.get_loc(pd.Period("2025-09", "M")):MONTHS.get_loc(pd.Period("2026-01", "M"))] *= 1.12
SIG = pd.DataFrame(signal, index=MONTHS)
SIG_PUBLISH_LAG = 1  # t월 지수는 t+1월 초에 공개

# 실제 주문량(최종 순주문량): 최종 고객 생산지수의 2개월 뒤 수요 + 거래처 재고 조정 잡음
rows = []
for _, it in ITEMS.iterrows():
    for i, m in enumerate(MONTHS):
        drv = SIG[it.true_ec].iloc[max(i - 2, 0)] / 100.0
        season = 1 + 0.05 * np.cos(2 * np.pi * (m.month - 3) / 12)
        q = it.base * drv * season * rng.lognormal(0, 0.08)
        rows.append((it["item"], m, q))
ACT = {(i, m): q for i, m, q in rows}  # (품목, 월) -> 실제 주문량

# 거래처별 FCST 편향(가상): D2는 과대, D4는 과소 경향
FCST_BIAS = {"D1": 0.02, "D2": 0.22, "D3": 0.0, "D4": -0.12, "D5": 0.08, "D6": 0.0}
ACCEPT_SHARE = {1: 0.55, 2: 0.25, 3: 0.08}   # 예측 시점에 이미 접수된 비율(평균)


_FCST_CACHE, _ACC_CACHE = {}, {}


def customer_fcst(item, target, origin):
    """같은 (품목, 납기월, 예측 시점)이면 항상 같은 값(회차가 바뀌어도 고객 FCST는 그대로)"""
    key = (item, target, origin)
    if key not in _FCST_CACHE:
        d = item.split("-")[0]
        h = (target - origin).n
        _FCST_CACHE[key] = ACT[(item, target)] * (1 + FCST_BIAS[d]) * rng.lognormal(0, 0.06 + 0.03 * h)
    return _FCST_CACHE[key]


def accepted(item, target, origin):
    key = (item, target, origin)
    if key not in _ACC_CACHE:
        h = (target - origin).n
        share = np.clip(ACCEPT_SHARE[h] + rng.normal(0, 0.05), 0, 0.95)
        _ACC_CACHE[key] = ACT[(item, target)] * share
    return _ACC_CACHE[key]


# ---------------------------------------------------------------------------
# 2) 특징 생성 (예측 시점 origin = 해당 월 1일, origin 이전 실적만 사용)
# ---------------------------------------------------------------------------
def features(item_row, origin, target, use_signal, link_table):
    it = item_row["item"]
    h = (target - origin).n
    hist = [ACT[(it, origin - k)] for k in range(1, 7)]
    f = {
        "item_code": int(ITEMS.index[ITEMS.item == it][0]),
        "h": h,
        "target_month": target.month,
        "lag1": hist[0], "lag2": hist[1], "lag3": hist[2],
        "ma3": np.mean(hist[:3]), "ma6": np.mean(hist),
        "yoy": ACT[(it, target - 12)],
        "accepted": accepted(it, target, origin),
    }
    if use_signal:
        ec = link_table.get(it)
        last_pub = origin - SIG_PUBLISH_LAG  # origin 시점에 공개된 가장 최근 월
        if ec is not None:
            f["sig_last"] = SIG[ec][last_pub]
            f["sig_chg3"] = SIG[ec][last_pub] / SIG[ec][last_pub - 3] - 1
        else:
            f["sig_last"] = np.nan
            f["sig_chg3"] = np.nan
    return f


def build(origins, use_signal, link_table, with_target=True):
    recs = []
    for o in origins:
        for _, r in ITEMS.iterrows():
            for h in (1, 2, 3):
                tg = o + h
                f = features(r, o, tg, use_signal, link_table)
                f.update({"item": r["item"], "origin": o, "target": tg})
                if with_target:
                    f["y"] = ACT[(r["item"], tg)]
                recs.append(f)
    return pd.DataFrame(recs)


def fit_predict(train, test, cols, alpha=None):
    X, y = train[cols], train["y"]
    if HAS_LGB:
        params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                      subsample=0.9, colsample_bytree=0.9, random_state=SEED, verbose=-1)
        if alpha is None:
            m = lgb.LGBMRegressor(objective="regression", **params)
        else:
            m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        m.fit(X, y)
        return m.predict(test[cols])
    Xf, Tf = X.fillna(-1), test[cols].fillna(-1)
    if alpha is None:
        m = GradientBoostingRegressor(random_state=SEED, n_estimators=200, max_depth=3)
    else:
        m = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=SEED, n_estimators=200, max_depth=3)
    m.fit(Xf, y)
    return m.predict(Tf)


def wape(a, f):
    return float(np.sum(np.abs(a - f)) / np.sum(np.abs(a)))


def bias(a, f):
    return float(np.sum(f - a) / np.sum(a))


# ---------------------------------------------------------------------------
# 3) 백테스트
# ---------------------------------------------------------------------------
LINK0 = {r["item"]: (r.true_ec if r.linked else None) for _, r in ITEMS.iterrows()}
ALL_ORIGINS = pd.period_range("2024-07", "2026-10", freq="M")
BT_ORIGINS = pd.period_range("2025-07", "2026-09", freq="M")   # 백테스트 시점
SELECT_ORIGINS = BT_ORIGINS[:-6]                                  # 모델 선택용
FINAL_ORIGINS = BT_ORIGINS[-6:]                                   # 최종 평가용(선택에 사용 안 함)

base_cols = ["item_code", "h", "target_month", "lag1", "lag2", "lag3", "ma3", "ma6", "yoy", "accepted"]
sig_cols = base_cols + ["sig_last", "sig_chg3"]

data_nosig = build(ALL_ORIGINS, False, LINK0)
data_sig = build(ALL_ORIGINS, True, LINK0)

results = []
for o in BT_ORIGINS:
    # 학습: 타깃 월이 o 이전(실적 확인)인 행만
    tr_n = data_nosig[(data_nosig.target < o)]
    tr_s = data_sig[(data_sig.target < o)]
    te_n = data_nosig[data_nosig.origin == o].copy()
    te_s = data_sig[data_sig.origin == o].copy()
    te_n["pred_base"] = te_n["ma3"]  # 기준 모델: 최근 3개월 평균
    te_n["pred_lgb"] = fit_predict(tr_n, te_n, base_cols)
    te_s["pred_lgb_sig"] = fit_predict(tr_s, te_s, sig_cols)
    te_s["q10"] = fit_predict(tr_s, te_s, sig_cols, alpha=0.1)
    te_s["q90"] = fit_predict(tr_s, te_s, sig_cols, alpha=0.9)
    m = te_n[["item", "origin", "target", "h", "y", "pred_base", "pred_lgb"]].merge(
        te_s[["item", "origin", "target", "pred_lgb_sig", "q10", "q90"]], on=["item", "origin", "target"])
    m["fcst"] = [customer_fcst(i, tg, oo) for i, tg, oo in zip(m.item, m.target, m.origin)]
    results.append(m)
BT = pd.concat(results, ignore_index=True)
BT["q10"] = np.minimum(BT.q10, BT.pred_lgb_sig)
BT["q90"] = np.maximum(BT.q90, BT.pred_lgb_sig)
BT["period"] = np.where(BT.origin.isin(FINAL_ORIGINS), "최종평가", "모델선택")

# 구간 보정: 모델선택 구간에서 '구간을 벗어난 정도'의 80% 분위수를 예측거리별로 구해 구간을 넓힌다
def calib_width(df):
    sc = np.maximum(df.q10 - df.y, df.y - df.q90)
    n = len(sc)
    return float(np.quantile(sc, min(1.0, np.ceil((n + 1) * 0.8) / n)))

CALIB = {h: calib_width(BT[(BT.period == "모델선택") & (BT.h == h)]) for h in (1, 2, 3)}
BT["q10_raw"], BT["q90_raw"] = BT.q10, BT.q90
BT["q10"] = BT.q10 - BT.h.map(CALIB)
BT["q90"] = BT.q90 + BT.h.map(CALIB)
BT["linked"] = BT.item.map(lambda i: LINK0[i] is not None)

summ = []
for per in ["모델선택", "최종평가"]:
    for h in (1, 2, 3):
        s = BT[(BT.period == per) & (BT.h == h)]
        summ.append({"구간": per, "예측거리(개월)": h,
                     "WAPE_기준모델": wape(s.y, s.pred_base), "WAPE_LGB_주문이력만": wape(s.y, s.pred_lgb),
                     "WAPE_LGB_신호추가": wape(s.y, s.pred_lgb_sig), "WAPE_고객FCST": wape(s.y, s.fcst),
                     "편향_LGB_신호추가": bias(s.y, s.pred_lgb_sig),
                     "구간포함률_보정전": float(((s.y >= s.q10_raw) & (s.y <= s.q90_raw)).mean()),
                     "구간포함률_보정후(목표80%)": float(((s.y >= s.q10) & (s.y <= s.q90)).mean()),
                     "평균구간폭(톤)": float((s.q90 - s.q10).mean())})
SUMMARY = pd.DataFrame(summ)

# 신호 효과는 연결된 품목에서 따로 확인
link_eff = []
for per in ["최종평가"]:
    for lk in (True, False):
        s = BT[(BT.period == per) & (BT.linked == lk)]
        link_eff.append({"구간": per, "최종고객연결": "연결됨" if lk else "미연결",
                         "WAPE_LGB_주문이력만": wape(s.y, s.pred_lgb), "WAPE_LGB_신호추가": wape(s.y, s.pred_lgb_sig)})
LINK_EFF = pd.DataFrame(link_eff)

# ---------------------------------------------------------------------------
# 4) 거래처별 FCST 신뢰도 이력 (최근 6개 시점, 1개월 앞 기준)
# ---------------------------------------------------------------------------
recent = BT[(BT.origin.isin(FINAL_ORIGINS)) & (BT.h == 1)].copy()
recent["거래처"] = recent.item.str.split("-").str[0]
rel = []
for d, g in recent.groupby("거래처"):
    err = (g.fcst - g.y) / g.y
    rel.append({"거래처": d, "회차": len(g.origin.unique()), "건수": len(g), "평균오차(부호)": float(err.mean()),
                "과대건수(+15%초과)": int((err > 0.15).sum()), "과소건수(-15%미만)": int((err < -0.15).sum()),
                "WAPE": wape(g.y, g.fcst)})
REL = pd.DataFrame(rel).sort_values("평균오차(부호)", ascending=False)
REL_MAP = REL.set_index("거래처")["평균오차(부호)"].to_dict()


def rel_text(d):
    r = REL.set_index("거래처").loc[d]
    direction = "과대" if r["평균오차(부호)"] > 0.05 else ("과소" if r["평균오차(부호)"] < -0.05 else "중립")
    return f"최근 {int(r['회차'])}회 평균 {r['평균오차(부호)']:+.0%} ({direction})"


# ---------------------------------------------------------------------------
# 5) 근거 분류 (LLM 추출을 규칙으로 대신): 채택, 참고 보관, 시차 불명확, 연결 부족
# ---------------------------------------------------------------------------
REVIEW_ORIGIN = pd.Period("2026-10", "M")
TARGETS = [REVIEW_ORIGIN + h for h in (1, 2, 3)]
EVENTS = [  # 가상의 공개 실적, 뉴스 이벤트 (실제 기사 아님)
    {"id": "N1", "end_customer": "E_A", "direction": "하향", "effect_from": pd.Period("2026-11", "M"),
     "published": "2026-10-01", "text": "[가상] 완성차 A, 9월 판매 전년 대비 감소, 4분기 일부 라인 감산 계획"},
    {"id": "N2", "end_customer": "E_B", "direction": "상향", "effect_from": None,
     "published": "2026-09-20", "text": "[가상] 완성차 B, 신차 출시 검토 (시기 미정)"},
    {"id": "N3", "end_customer": "E_C", "direction": "하향", "effect_from": pd.Period("2027-04", "M"),
     "published": "2026-10-02", "text": "[가상] 가전 C, 내년 2분기 일부 모델 단종 예정"},
    {"id": "N5", "end_customer": "E_B", "direction": "상향", "effect_from": pd.Period("2026-12", "M"),
     "published": "2026-10-02", "text": "[가상] 완성차 B, 12월부터 주말 특근으로 생산 확대"},
    {"id": "N4", "end_customer": "E_X", "direction": "상향", "effect_from": pd.Period("2026-12", "M"),
     "published": "2026-10-03", "text": "[가상] 가전 X, 연말 판촉 확대 (연결표에 없는 기업)"},
]


def classify(ev, item_row, link_table):
    """품목 하나에 대해 이벤트의 적용 범위를 판정"""
    linked_ec = link_table.get(item_row["item"])
    known_ecs = {v for v in link_table.values() if v}
    if ev["end_customer"] not in known_ecs and linked_ec != ev["end_customer"]:
        # 연결표 어디에도 없는 기업: 고객명만 있는 정보
        return "참고 보관"
    if linked_ec is None:
        return "연결 부족"
    if linked_ec != ev["end_customer"]:
        return None  # 이 품목과 무관
    if ev["effect_from"] is None:
        return "시차 불명확"
    if ev["effect_from"] > TARGETS[-1]:
        return "참고 보관"  # 예측 기간 밖
    return "채택"


def classify_all(link_table):
    out = []
    for ev in EVENTS:
        for _, r in ITEMS.iterrows():
            c = classify(ev, r, link_table)
            if c is None:
                continue
            if c == "연결 부족" and r.true_ec != ev["end_customer"] and ev["end_customer"] in END_CUSTOMERS:
                # 미연결 품목에는 '이 이벤트가 이 품목과 관련 있는지' 질문만 남긴다
                pass
            out.append({"이벤트": ev["id"], "최종고객": ev["end_customer"], "품목": r["item"], "분류": c,
                        "방향": ev["direction"], "원문": ev["text"], "공개일": ev["published"]})
    df = pd.DataFrame(out)
    # 참고 보관은 이벤트당 한 줄만
    keep = df[df.분류 != "참고 보관"]
    ref = df[df.분류 == "참고 보관"].drop_duplicates("이벤트").assign(품목="-")
    return pd.concat([keep, ref], ignore_index=True)


# ---------------------------------------------------------------------------
# 6) 기준월 예측과 검토표
# ---------------------------------------------------------------------------
def review(link_table, label):
    tr = build(pd.period_range("2024-07", REVIEW_ORIGIN - 1, freq="M"), True, link_table)
    tr = tr[tr.target < REVIEW_ORIGIN]
    te = build([REVIEW_ORIGIN], True, link_table, with_target=False)
    te["모델예측"] = fit_predict(tr, te, sig_cols)
    te["구간하단"] = np.minimum(fit_predict(tr, te, sig_cols, alpha=0.1), te["모델예측"]) - te.h.map(CALIB)
    te["구간상단"] = np.maximum(fit_predict(tr, te, sig_cols, alpha=0.9), te["모델예측"]) + te.h.map(CALIB)
    te["고객FCST"] = [customer_fcst(i, tg, REVIEW_ORIGIN) for i, tg in zip(te.item, te.target)]
    te["접수주문"] = te["accepted"]
    ev = classify_all(link_table)
    adopted = ev[ev.분류 == "채택"]

    sheets = []
    for _, r in te.iterrows():
        d = r["item"].split("-")[0]
        reasons, questions, evidence = [], [], []
        if r.고객FCST > r.구간상단:
            reasons.append("고객 FCST가 구간 상단 초과")
            questions.append("고객 FCST 상향 근거(신규 프로젝트, 재고 확충)가 있는가?")
        elif r.고객FCST < r.구간하단:
            reasons.append("고객 FCST가 구간 하단 미만")
            questions.append("고객 FCST 하향 사유(감산, 공급사 변경)가 있는가?")
        if r.모델예측 < r.접수주문:
            reasons.append("모델 예측이 접수 주문보다 낮음")
            questions.append("접수 주문 중 취소나 납기 이연 가능 물량이 있는가?")
        hit = adopted[(adopted.품목 == r["item"]) & (adopted.이벤트.isin([e["id"] for e in EVENTS
                                                                     if e["effect_from"] is not None and e["effect_from"] <= r.target]))]
        for _, h in hit.iterrows():
            reasons.append(f"채택 근거({h.방향})")
            evidence.append(f"{h.이벤트}: {h.원문} (공개 {h.공개일}, 연결 수준: 최종고객-거래처-그레이드 확인)")
            questions.append(f"{h.이벤트}의 {h.방향} 변화가 이 그레이드 적용 제품에 해당하는가, 거래처 재고 수준은?")
        miss_link = ev[(ev.품목 == r["item"]) & (ev.분류 == "연결 부족")]
        if len(miss_link):
            questions.append(f"[연결 확인] {r['item']}의 최종 고객은 누구인가? (관련 가능 이벤트: {', '.join(miss_link.이벤트)})")
        for _, mrow in ev[(ev.품목 == r["item"]) & (ev.분류 == "시차 불명확")].iterrows():
            questions.append(f"[시차 확인] {mrow.이벤트}의 영향 시점은 언제인가?")
        if not reasons:
            status = "정상"
        else:
            status = "검토"
        gap = abs(r.고객FCST - r.모델예측)
        weight = 1 + abs(REL_MAP.get(d, 0))
        start_value = r.고객FCST
        sheets.append({
            "검토회차": label, "기준월": str(REVIEW_ORIGIN), "품목": r["item"], "납기월": str(r.target),
            "상태": status, "검토사유": "; ".join(reasons) if reasons else "-",
            "고객FCST": round(r.고객FCST, 1), "모델예측": round(r.모델예측, 1),
            "구간(10~90%)": f"{r.구간하단:.0f}~{r.구간상단:.0f}", "접수주문": round(r.접수주문, 1),
            "FCST신뢰도이력": rel_text(d), "근거": " | ".join(evidence) if evidence else "-",
            "확인질문": " / ".join(dict.fromkeys(questions)) if questions else "-",
            "검토시작값": round(start_value, 1), "판단": "", "승인값": "", "사유": "", "승인자": "", "승인시각": "",
            "_정렬점수": gap * weight if status == "검토" else -1,
        })
    sheet = pd.DataFrame(sheets).sort_values("_정렬점수", ascending=False).drop(columns="_정렬점수")
    return sheet, ev


def link_rate(link_table):
    vol = {it: ACT[(it, REVIEW_ORIGIN - 1)] for it in ITEMS.item}
    tot = sum(vol.values())
    return sum(v for it, v in vol.items() if link_table.get(it)) / tot


sheet1, ev1 = review(LINK0, "1회차(답변 전)")
rate1 = link_rate(LINK0)

# 연결 확인 질문에 담당자가 답했다고 가정(가상): D5-G2 -> E_B, D4-G2 -> E_C
ANSWERS = [("D5-G2", "E_B", "수요계획 담당자(가상)", "2026-10-07"),
           ("D4-G2", "E_C", "수요계획 담당자(가상)", "2026-10-07")]
LINK1 = dict(LINK0)
changelog = []
for it, ec, who, when in ANSWERS:
    changelog.append({"버전": 2, "품목": it, "변경전": LINK1[it], "변경후": ec, "확인자": who, "확인일": when,
                      "출처": "확인 질문 답변"})
    LINK1[it] = ec
CHANGELOG = pd.DataFrame(changelog)
sheet2, ev2 = review(LINK1, "2회차(답변 저장 후)")
rate2 = link_rate(LINK1)

ev_compare = ev1.merge(ev2[["이벤트", "품목", "분류"]], on=["이벤트", "품목"], how="outer",
                       suffixes=("_답변전", "_답변후"))

# ---------------------------------------------------------------------------
# 7) 사후 평가 (모의 승인): 최종평가 구간에서 규칙으로 만든 승인값과 비교
# ---------------------------------------------------------------------------
fin = BT[BT.period == "최종평가"].copy()
fin["거래처"] = fin.item.str.split("-").str[0]
outside = (fin.fcst > fin.q90) | (fin.fcst < fin.q10)
fin["승인값_모의"] = np.where(outside, fin.pred_lgb_sig, fin.fcst)
post = []
for h in (1, 2, 3):
    s = fin[fin.h == h]
    post.append({"예측거리(개월)": h, "WAPE_고객FCST": wape(s.y, s.fcst), "WAPE_모델": wape(s.y, s.pred_lgb_sig),
                 "WAPE_모의승인값": wape(s.y, s["승인값_모의"]),
                 "조정한건수": int(((s.fcst > s.q90) | (s.fcst < s.q10)).sum()), "전체건수": len(s)})
POST = pd.DataFrame(post)

# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
xlsx = os.path.join(OUT, "검토_결과_가상데이터.xlsx")
with pd.ExcelWriter(xlsx) as w:
    pd.DataFrame({"안내": [
        "모든 데이터는 가상이며 실제 고객, 거래, 수치가 아님",
        "LLM이 맡을 이벤트 추출과 질문 작성은 이 프로토타입에서 규칙으로 대신함",
        "사후평가의 승인값은 규칙으로 만든 모의 승인값임",
        f"예측 모델: {'LightGBM' if HAS_LGB else 'scikit-learn GradientBoosting(LightGBM 미설치로 대체)'}",
    ]}).to_excel(w, sheet_name="안내", index=False)
    SUMMARY.to_excel(w, sheet_name="1_백테스트", index=False)
    LINK_EFF.to_excel(w, sheet_name="2_신호효과_연결여부별", index=False)
    sheet1.to_excel(w, sheet_name="3_검토표_1회차", index=False)
    ev_compare.to_excel(w, sheet_name="4_근거분류_답변전후", index=False)
    CHANGELOG.to_excel(w, sheet_name="5_연결표_변경이력", index=False)
    sheet2.to_excel(w, sheet_name="6_검토표_2회차", index=False)
    REL.to_excel(w, sheet_name="7_FCST신뢰도이력", index=False)
    POST.to_excel(w, sheet_name="8_사후평가_모의승인", index=False)


def pct(x):
    return f"{x:.1%}"


lines = []
lines.append("월간 수요검토 에이전트 미니 프로토타입 (가상 데이터)")
lines.append(f"예측 모델: {'LightGBM' if HAS_LGB else 'scikit-learn GradientBoosting (LightGBM 대체)'}")
lines.append("")
lines.append("[1] 백테스트 (최종평가 구간, 모델 선택에 쓰지 않은 마지막 6개 시점)")
for _, r in SUMMARY[SUMMARY.구간 == "최종평가"].iterrows():
    lines.append(f"  {int(r['예측거리(개월)'])}개월 앞: WAPE 기준모델 {pct(r.WAPE_기준모델)}, 주모델(주문이력만) {pct(r.WAPE_LGB_주문이력만)}, "
                 f"주모델(신호추가) {pct(r.WAPE_LGB_신호추가)}, 고객FCST {pct(r.WAPE_고객FCST)} | 구간 포함률 보정 전 {pct(r.구간포함률_보정전)} -> 보정 후 {pct(r['구간포함률_보정후(목표80%)'])}")
lines.append("[2] 신호 효과 (최종평가 구간)")
for _, r in LINK_EFF.iterrows():
    lines.append(f"  {r.최종고객연결}: 주문이력만 {pct(r.WAPE_LGB_주문이력만)} -> 신호추가 {pct(r.WAPE_LGB_신호추가)}")
n_review = (sheet1.상태 == "검토").sum()
lines.append(f"[3] 1회차 검토표: 전체 {len(sheet1)}건 중 검토 {n_review}건, 정상 {len(sheet1) - n_review}건 (구간 보정 폭: " + ", ".join(f"{h}개월 {CALIB[h]:.1f}톤" for h in (1, 2, 3)) + ")")
for _, r in sheet1[sheet1.근거 != "-"].head(2).iterrows():
    lines.append(f"  * 근거 채택 예: {r.품목} {r.납기월}: {r.근거[:60]}...")
top = sheet1[sheet1.상태 == "검토"].head(3)
for _, r in top.iterrows():
    lines.append(f"  - {r.품목} {r.납기월}: {r.검토사유} | FCST {r.고객FCST} vs 모델 {r.모델예측} ({r['구간(10~90%)']}) | {r.FCST신뢰도이력}")
lines.append("[4] 연결 지도 성장")
lines.append(f"  연결률(직전월 주문량 기준): 답변 전 {pct(rate1)} -> 답변 저장 후 {pct(rate2)}")
lines.append(f"  2회차 검토표: 검토 {(sheet2.상태 == '검토').sum()}건")
moved = ev_compare[(ev_compare.분류_답변전 == "연결 부족") & (ev_compare.분류_답변후 == "채택")]
for _, r in moved.iterrows():
    lines.append(f"  {r.이벤트} x {r.품목}: 연결 부족 -> 채택")
lines.append("[5] 거래처별 FCST 신뢰도 이력 (최종평가 6개 회차, 1개월 앞)")
for _, r in REL.iterrows():
    lines.append(f"  {r.거래처}: 최근 {int(r.회차)}회차 {int(r.건수)}건, 평균 오차 {r['평균오차(부호)']:+.0%}, 과대 {r['과대건수(+15%초과)']}건, 과소 {r['과소건수(-15%미만)']}건")
lines.append("[6] 사후 평가 (모의 승인: FCST가 구간 밖이면 모델값으로 조정)")
for _, r in POST.iterrows():
    lines.append(f"  {int(r['예측거리(개월)'])}개월 앞: FCST {pct(r.WAPE_고객FCST)}, 모델 {pct(r.WAPE_모델)}, 모의승인 {pct(r.WAPE_모의승인값)} "
                 f"(조정 {int(r.조정한건수)}/{int(r.전체건수)}건)")
lines.append("")
lines.append(f"상세 결과: {xlsx}")
text = "\n".join(lines)
with open(os.path.join(OUT, "실행_요약.txt"), "w", encoding="utf-8") as f:
    f.write(text)
print(text)
