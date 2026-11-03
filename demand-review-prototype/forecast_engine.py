# -*- coding: utf-8 -*-
"""
수치 예측 엔진 (ML이 맡는 부분). LLM은 이 모듈의 숫자를 만들거나 바꾸지 않는다.

  1) 수요 패턴 자동 분류: 예측 시점까지의 이력으로 안정, 변동, 간헐, 덩어리, 신규 (ADI, CV^2 기준)
  2) 후보 모델: 최근 3개월 평균(기준), Croston-SBA(간헐 수요용), LightGBM 전역 분위수 모델
  3) 패턴별 챔피언 자동 선택: 모델 선택 구간 WAPE가 가장 낮은 모델 (최종 평가 구간은 선택에 쓰지 않음)
  4) 80% 예측구간: 분위수 + 시점별 보정(그 시점에 실적이 확인된 과거 오차만 사용, 품목 규모로 정규화)
  5) AI 보정 FCST: 고객 FCST를 입력으로 받아 거래처별 과거 편향을 학습해 고친 값 (독립 예측과 별도)
"""
import numpy as np
import pandas as pd

import virtual_data as V

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:  # pragma: no cover
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingRegressor

SEED = V.SEED
PAT_KO = {"smooth": "안정", "erratic": "변동", "intermittent": "간헐", "lumpy": "덩어리", "new": "신규"}
PAT_CODE = {"안정": 0, "변동": 1, "간헐": 2, "덩어리": 3, "신규": 4}
ITEM_IDX = {it: i for i, it in enumerate(V.ITEMS.item)}
IT = V.ITEMS.set_index("item")
FAM_CODE = {"ABS": 0, "PC": 1, "컴파운드": 2}
DTYPE_CODE = {t: i for i, t in enumerate(sorted(V.DISTRIBUTORS.유형.unique()))}
DIST_CODE = {d: i for i, d in enumerate(V.DISTRIBUTORS.거래처)}


def act(item, m):
    return V.ACT.get((item, m), np.nan)


def history(item, origin, n):
    return np.array([act(item, origin - k) for k in range(1, n + 1)], dtype=float)  # 최근 월부터


def classify_pattern(item, origin):
    h = history(item, origin, 24)
    h = h[~np.isnan(h)]
    if len(h) < 12:
        return "신규"
    nz = h[h > 0]
    if len(nz) == 0:
        return "간헐"
    adi = len(h) / len(nz)
    cv2 = (np.std(nz) / np.mean(nz)) ** 2 if len(nz) > 1 else 0.0
    if adi < 1.32:
        return "안정" if cv2 < 0.49 else "변동"
    return "간헐" if cv2 < 0.49 else "덩어리"


def sba(item, origin, alpha=0.1):
    """Croston-SBA: 주문이 있는 달의 크기와 간격을 따로 평활"""
    h = history(item, origin, 36)[::-1]
    h = h[~np.isnan(h)]
    z = p = None
    q = 1
    for v in h:
        if v > 0:
            if z is None:
                z, p = v, q
            else:
                z = z + alpha * (v - z)
                p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
    if z is None:
        return 0.0
    return (1 - alpha / 2) * z / p


def features(item, origin, target, link_table, fcst=None):
    r = IT.loc[item]
    h = (target - origin).n
    hist = history(item, origin, 12)
    ma6 = np.nanmean(hist[:6]) if np.any(~np.isnan(hist[:6])) else np.nan
    ma12 = np.nanmean(hist) if np.any(~np.isnan(hist)) else np.nan
    acc = V.accepted(item, target, origin)
    f = {
        "item_code": ITEM_IDX[item], "dist_code": DIST_CODE[r.거래처], "dtype_code": DTYPE_CODE[r.유형],
        "fam_code": FAM_CODE[r.제품군], "h": h, "target_month": target.month,
        "lag1": hist[0], "lag2": hist[1], "lag3": hist[2],
        "ma3": np.nanmean(hist[:3]) if np.any(~np.isnan(hist[:3])) else np.nan, "ma6": ma6, "ma12": ma12,
        "zero_share12": float(np.mean(hist[~np.isnan(hist)] == 0)) if np.any(~np.isnan(hist)) else np.nan,
        "yoy": act(item, target - 12), "accepted": acc,
        "acc_ratio": acc / (ma6 + 1) if not np.isnan(acc) and not np.isnan(ma6) else np.nan,
        "months_active": float(np.sum(~np.isnan(history(item, origin, 36)))),
        "pattern_code": PAT_CODE[classify_pattern(item, origin)],
    }
    ec = link_table.get(item)
    last_pub = origin - V.SIG_PUBLISH_LAG
    if ec is not None:
        f["sig_last"] = V.SIG[ec][last_pub]
        f["sig_chg3"] = V.SIG[ec][last_pub] / V.SIG[ec][last_pub - 3] - 1
    else:
        f["sig_last"] = np.nan
        f["sig_chg3"] = np.nan
    f["market_chg3"] = V.SIG["MARKET"][last_pub] / V.SIG["MARKET"][last_pub - 3] - 1
    # 선구매 점검용: 계절 영향을 뺀 전년 동월 대비 변화 (모델 입력에는 쓰지 않음)
    has_yoy = (last_pub - 12) >= V.MONTHS[0]
    f["sig_yoy"] = V.SIG[ec][last_pub] / V.SIG[ec][last_pub - 12] - 1 if (ec is not None and has_yoy) else np.nan
    f["market_yoy"] = V.SIG["MARKET"][last_pub] / V.SIG["MARKET"][last_pub - 12] - 1 if has_yoy else np.nan
    f["fcst"] = V.customer_fcst(item, target, origin) if fcst is None else fcst
    f["fcst_ratio"] = f["fcst"] / (ma6 + 1) if not np.isnan(ma6) else np.nan
    return f


BASE_COLS = ["item_code", "dtype_code", "fam_code", "h", "target_month", "lag1", "lag2", "lag3", "ma3", "ma6",
             "ma12", "zero_share12", "yoy", "accepted", "acc_ratio", "months_active", "pattern_code"]
SIG_COLS = BASE_COLS + ["sig_last", "sig_chg3"]
CORR_COLS = SIG_COLS + ["dist_code", "fcst", "fcst_ratio"]


def build(origins, link_table, with_target=True):
    recs = []
    for o in origins:
        for item in V.ITEMS.item:
            for h in (1, 2, 3):
                tg = o + h
                if np.isnan(act(item, tg)) and with_target:
                    continue
                if np.all(np.isnan(history(item, o, 3))):
                    continue  # 예측 시점에 주문 이력이 전혀 없으면 예측하지 않음
                f = features(item, o, tg, link_table)
                f.update({"item": item, "origin": o, "target": tg, "pattern": classify_pattern(item, o)})
                if with_target:
                    f["y"] = act(item, tg)
                recs.append(f)
    return pd.DataFrame(recs)


def fit_predict(train, test, cols, alpha=0.5):
    X, y = train[cols], train["y"]
    if HAS_LGB:
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300, learning_rate=0.05,
                              num_leaves=15, min_child_samples=10, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.9, random_state=SEED, verbose=-1)
        m.fit(X, y)
        return np.maximum(m.predict(test[cols]), 0)
    m = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=SEED, n_estimators=200, max_depth=3)
    m.fit(X.fillna(-1), y)
    return np.maximum(m.predict(test[cols].fillna(-1)), 0)


def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sum(np.abs(a - f)) / np.sum(np.abs(a)))


def bias(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sum(f - a) / np.sum(a))


# ---------------------------------------------------------------------------
# 백테스트
# ---------------------------------------------------------------------------
LINK0 = {r.item: (r.true_ec if (r.linked and isinstance(r.true_ec, str)) else None) for r in V.ITEMS.itertuples()}
TRAIN_ORIGINS = pd.period_range("2024-01", "2026-10", freq="M")
BT_ORIGINS = pd.period_range("2025-04", "2026-09", freq="M")
SELECT_ORIGINS = BT_ORIGINS[:-6]
FINAL_ORIGINS = BT_ORIGINS[-6:]
MODELS = ["기준(3개월평균)", "Croston-SBA", "LightGBM"]
MIN_CALIB_N = 20
CALIB_LEVEL = 0.8          # choose_level()이 모델 선택 구간에서 다시 정한다
CALIB_CANDIDATES = (0.80, 0.85, 0.90)


def run_backtest(link_table=LINK0):
    data = build(TRAIN_ORIGINS, link_table)
    out = []
    for o in BT_ORIGINS:
        tr = data[data.target < o]
        te = data[data.origin == o].copy()
        te["p_base"] = te["ma3"].fillna(0)
        te["p_sba"] = [sba(i, o) for i in te.item]
        te["p_lgb_hist"] = fit_predict(tr, te, BASE_COLS)
        te["p_lgb"] = fit_predict(tr, te, SIG_COLS)
        te["lgb_q10"] = np.minimum(fit_predict(tr, te, SIG_COLS, 0.1), te.p_lgb)
        te["lgb_q90"] = np.maximum(fit_predict(tr, te, SIG_COLS, 0.9), te.p_lgb)
        te["p_corr"] = fit_predict(tr, te, CORR_COLS)
        out.append(te)
    bt = pd.concat(out, ignore_index=True)
    bt["scale"] = np.maximum(bt.ma12.fillna(bt.ma3).fillna(1), 1)
    bt["period"] = np.where(bt.origin.isin(FINAL_ORIGINS), "최종평가", "모델선택")
    return bt


def select_champions(bt):
    """패턴별로 모델 선택 구간 WAPE가 가장 낮은 모델을 고른다"""
    sel = bt[bt.period == "모델선택"]
    rows, champ = [], {}
    cols = {"기준(3개월평균)": "p_base", "Croston-SBA": "p_sba", "LightGBM": "p_lgb"}
    for pat in ["안정", "변동", "간헐", "덩어리", "신규"]:
        s = sel[sel.pattern == pat]
        if len(s) < 10:
            champ[pat] = "LightGBM"  # 표본이 적으면 전역 모델 사용
            rows.append({"패턴": pat, "선택구간_건수": len(s), **{m: np.nan for m in MODELS}, "챔피언": "LightGBM(표본 부족 기본값)"})
            continue
        sc = {m: wape(s.y, s[c]) for m, c in cols.items()}
        champ[pat] = min(sc, key=sc.get)
        rows.append({"패턴": pat, "선택구간_건수": len(s), **sc, "챔피언": champ[pat]})
    return champ, pd.DataFrame(rows)


def apply_champion(df, champ):
    col = {"기준(3개월평균)": "p_base", "Croston-SBA": "p_sba", "LightGBM": "p_lgb"}
    df["champion"] = df.pattern.map(champ)
    df["p_champ"] = [r[col[c]] for c, (_, r) in zip(df.champion, df.iterrows())]
    return df


def calib_params(bt, origin, h, pattern, champion, level=None):
    level = CALIB_LEVEL if level is None else level
    """origin 시점에 실적이 확인된(target < origin) 과거 오차만으로 구간 보정값 계산. 표본이 적으면 같은 h 전체로 넓힘"""
    past = bt[(bt.h == h) & (bt.target < origin)]
    p2 = past[past.pattern == pattern]
    if len(p2) >= MIN_CALIB_N:
        past = p2
    if len(past) < MIN_CALIB_N:
        return None, len(past)
    if champion == "LightGBM":
        s = np.maximum(past.lgb_q10 - past.y, past.y - past.lgb_q90) / past.scale
        n = len(s)
        return ("lgb", float(np.quantile(s, min(1.0, np.ceil((n + 1) * level) / n)))), n
    col = {"기준(3개월평균)": "p_base", "Croston-SBA": "p_sba"}[champion]
    r = (past.y - past[col]) / past.scale
    return ("resid", float(np.quantile(r, (1 - level) / 2)), float(np.quantile(r, (1 + level) / 2))), len(past)


def add_intervals(df, bt, level=None):
    lo, hi, ns = [], [], []
    cache = {}
    for _, r in df.iterrows():
        key = (r.origin, r.h, r.pattern, r.champion)
        if key not in cache:
            cache[key] = calib_params(bt, r.origin, r.h, r.pattern, r.champion, level)
        prm, n = cache[key]
        ns.append(n)
        if r.champion == "LightGBM":
            c = prm[1] if prm else 0.0
            lo.append(max(0.0, r.lgb_q10 - c * r.scale))
            hi.append(r.lgb_q90 + c * r.scale)
        else:
            if prm is None:
                lo.append(max(0.0, r.p_champ * 0.5))
                hi.append(r.p_champ * 1.5)
            else:
                lo.append(max(0.0, r.p_champ + prm[1] * r.scale))
                hi.append(max(r.p_champ, r.p_champ + prm[2] * r.scale))
    df["q10"], df["q90"], df["calib_n"] = lo, hi, ns
    return df


def backtest_summary(bt):
    rows = []
    for per in ["모델선택", "최종평가"]:
        for h in (1, 2, 3):
            s = bt[(bt.period == per) & (bt.h == h)]
            rows.append({"구간": per, "예측거리(개월)": h, "건수": len(s),
                         "WAPE_기준모델": wape(s.y, s.p_base), "WAPE_LGB_주문이력만": wape(s.y, s.p_lgb_hist),
                         "WAPE_LGB_생산지수추가": wape(s.y, s.p_lgb), "WAPE_패턴별챔피언": wape(s.y, s.p_champ),
                         "WAPE_고객FCST": wape(s.y, s.fcst), "WAPE_AI보정FCST": wape(s.y, s.p_corr),
                         "편향_고객FCST": bias(s.y, s.fcst), "편향_AI보정FCST": bias(s.y, s.p_corr),
                         "구간포함률_보정전": float(((s.y >= s.lgb_q10) & (s.y <= s.lgb_q90)).mean()),
                         "구간포함률_보정후(목표80%)": float(((s.y >= s.q10) & (s.y <= s.q90)).mean())})
    return pd.DataFrame(rows)


def pattern_summary(bt):
    rows = []
    s0 = bt[bt.period == "최종평가"]
    for pat, s in s0.groupby("pattern"):
        rows.append({"패턴": pat, "건수": len(s), "품목수": s.item.nunique(), "챔피언": s.champion.iloc[0],
                     "WAPE_기준모델": wape(s.y, s.p_base), "WAPE_SBA": wape(s.y, s.p_sba), "WAPE_LightGBM": wape(s.y, s.p_lgb),
                     "WAPE_챔피언": wape(s.y, s.p_champ), "구간포함률": float(((s.y >= s.q10) & (s.y <= s.q90)).mean())})
    return pd.DataFrame(rows)


def hierarchy_summary(bt):
    """같은 예측을 계층별로 합산해 오차 비교 (합계는 아래 단계의 합과 항상 같다)"""
    s = bt[(bt.period == "최종평가") & (bt.h == 1)].copy()
    s["제품군"] = s.item.map(IT.제품군)
    s["계열"] = s.item.map(lambda i: IT.loc[i, "제품군"] + "-" + IT.loc[i, "계열"])
    s["그레이드"] = s.item.map(IT.그레이드)
    rows = []
    for lvl in ["item", "그레이드", "계열", "제품군"]:
        g = s.groupby([lvl, "origin"])[["y", "p_champ", "fcst"]].sum()
        rows.append({"계층": {"item": "거래처x그레이드"}.get(lvl, lvl), "단위수": g.index.get_level_values(0).nunique(),
                     "WAPE_챔피언": wape(g.y, g.p_champ), "WAPE_고객FCST": wape(g.y, g.fcst)})
    return pd.DataFrame(rows)


def mes_lead_and_inventory():
    lots = V.MES_LOTS.copy()
    lots["리드타임"] = (pd.to_datetime(lots.출하일) - pd.to_datetime(lots.생산확정일)).dt.days
    g = lots.groupby("그레이드").agg(lot수=("lot", "count"), 리드타임_중앙값=("리드타임", "median"),
                                    리드타임_80분위=("리드타임", lambda x: float(np.quantile(x, 0.8))),
                                    재고일수_평균=("창고대기일", "mean"), lot크기_중앙값=("lot크기_톤", "median")).reset_index()
    return g, lots


def color_mix(lots):
    w = lots.groupby(["그레이드", "색상"]).lot크기_톤.sum()
    return (w / w.groupby(level=0).transform("sum")).rename("색상비중").reset_index()


def choose_level(bt):
    """보정 수준을 모델 선택 구간에서만 고른다: 포함률이 80% 이상이 되는 가장 낮은 수준 (최종 평가 구간은 보지 않음)"""
    sel = bt[bt.period == "모델선택"].copy()
    rows = []
    for lv in CALIB_CANDIDATES:
        t = add_intervals(sel.copy(), bt, lv)
        rows.append((lv, float(((t.y >= t.q10) & (t.y <= t.q90)).mean())))
    for lv, cov in rows:
        if cov >= 0.80:
            return lv, rows
    return CALIB_CANDIDATES[-1], rows
