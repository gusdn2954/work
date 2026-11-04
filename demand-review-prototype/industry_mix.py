# -*- coding: utf-8 -*-
"""
최종 고객을 특정하지 못한 물량의 산업 비중 추정 (가상 데이터)

  - 입력: 품목의 월별 주문량, 산업별 공개 생산지수(공표 시차 1개월). 영업 질문은 쓰지 않는다.
  - 방법: 주문량을 산업 지수의 비음수 결합으로 맞추고(NNLS), 계수를 비중으로 바꾼다.
          시차는 정답을 아는 품목에서 적합도가 가장 좋은 값을 고른다.
          간헐 품목은 3개월 이동합으로 평탄화한다. 월을 재표집해 비중의 80% 범위를 낸다.
  - 검증: 정답을 아는 품목은 연결 정보를 가리고 산업을 맞히는지 본다.
          컴파운더 품목은 가상 데이터에서 6개 고객사 평균을 따르므로 정답 비중이 정해져 있다.
  - 쓰임: 회사까지 특정한 물량(연결표)과 산업까지만 특정한 물량(이 추정)을 나눠 산업별 수요에 반영한다.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls

import virtual_data as V

EC = V.END_CUSTOMERS.set_index("code")
INDUSTRIES = list(dict.fromkeys(V.END_CUSTOMERS["산업"]))
# 산업별 공개 생산지수: 그 산업 고객사 지수의 평균
IND_IDX = pd.DataFrame({k: V.SIG[list(EC[EC.산업 == k].index)].mean(axis=1) for k in INDUSTRIES})
SPARSE = {"intermittent", "lumpy"}


def true_mix(item):
    ec = V.ITEMS.set_index("item").loc[item, "true_ec"]
    if isinstance(ec, str):
        return {k: float(EC.loc[ec, "산업"] == k) for k in INDUSTRIES}
    n = V.END_CUSTOMERS.groupby("산업").size()  # 컴파운더 품목: 6개 고객사 평균을 따른다
    return {k: float(n[k] / n.sum()) for k in INDUSTRIES}


def _series(item, origin, lag, months=36):
    it = V.ITEMS.set_index("item").loc[item]
    m_end = origin - 1
    ms = [m for m in pd.period_range(m_end - months + 1, m_end, freq="M") if m >= pd.Period(it.start, "M")]
    y = pd.Series([V.ACT[(item, m)] for m in ms], index=ms, dtype=float)
    X = pd.DataFrame({k: [IND_IDX.loc[m - lag, k] for m in ms] for k in INDUSTRIES}, index=ms)
    if it.pattern in SPARSE or (y == 0).mean() > 0.3:
        y = y.rolling(3).sum()
        X = X.rolling(3).sum()
    ok = y.notna() & X.notna().all(axis=1)
    y, X = y[ok], X[ok]
    X = X / X.mean()  # 산업 지수의 수준 차이를 없애고 움직임만 비교
    return y.values, X.values


def fit_mix(item, origin, lag, n_boot=200, seed=0):
    y, X = _series(item, origin, lag)
    if len(y) < 12 or y.sum() <= 0:
        return None
    c, _ = nnls(X, y)
    w = c / c.sum() if c.sum() > 0 else np.full(len(INDUSTRIES), 1 / len(INDUSTRIES))
    r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        cb, _ = nnls(X[idx], y[idx])
        boots.append(cb / cb.sum() if cb.sum() > 0 else np.full(len(INDUSTRIES), 1 / len(INDUSTRIES)))
    boots = np.array(boots) if boots else np.array([w])
    lo, hi = np.quantile(boots, 0.1, axis=0), np.quantile(boots, 0.9, axis=0)
    return {"w": dict(zip(INDUSTRIES, w)), "lo": dict(zip(INDUSTRIES, lo)), "hi": dict(zip(INDUSTRIES, hi)), "r2": float(r2)}


def choose_lag(origin, link, lags=(0, 1, 2, 3)):
    """정답을 아는(연결된) 품목에서 평균 적합도가 가장 좋은 시차. 미연결 품목 정보는 쓰지 않는다"""
    known = [i for i, e in link.items() if e]
    score = {}
    for L in lags:
        r = [fit_mix(i, origin, L, n_boot=0) for i in known]
        score[L] = float(np.mean([x["r2"] for x in r if x]))
    return max(score, key=score.get), score


def evaluate(origin, link):
    """정답 가림 검증. link: 운영 중 연결표(None = 회사 미특정)"""
    lag, lag_score = choose_lag(origin, link)
    rows = []
    for item in V.ITEMS.item:
        f = fit_mix(item, origin, lag)
        if f is None:
            continue
        tm = true_mix(item)
        top = max(f["w"], key=f["w"].get)
        true_top = max(tm, key=tm.get)
        single = max(tm.values()) == 1.0
        vol = np.nanmean([V.ACT[(item, origin - k)] for k in range(1, 7)])
        rows.append({
            "품목": item, "연결표": link.get(item) or "회사 미특정", "정답": "단일 산업 " + true_top if single else "여러 산업 혼합",
            "최근6개월_평균물량": vol, "적합도_R2": f["r2"], "추정_1순위산업": top,
            **{f"추정_{k}": f["w"][k] for k in INDUSTRIES}, **{f"범위_{k}": f"{f['lo'][k]:.0%}~{f['hi'][k]:.0%}" for k in INDUSTRIES},
            **{f"정답_{k}": tm[k] for k in INDUSTRIES},
            "1순위_적중": (top == true_top) if single else np.nan,
            "정답산업_추정비중": f["w"][true_top] if single else np.nan,
            "비중_평균절대오차": float(np.mean([abs(f["w"][k] - tm[k]) for k in INDUSTRIES])),
        })
    return pd.DataFrame(rows), lag, lag_score


def allocation_test(df, link, months):
    """과거 실적을 산업별로 나눌 때의 오차. 정답 = 품목 실적 x 정답 비중.
       비교: 회사 미특정 물량을 빼고 보기, 연결된 물량의 산업 비율로 나누기, 이번 추정 비중으로 나누기"""
    est = df.set_index("품목")
    res = []
    for m in months:
        truth = {k: 0.0 for k in INDUSTRIES}
        known = {k: 0.0 for k in INDUSTRIES}
        unk_items = []
        for item in V.ITEMS.item:
            a = V.ACT[(item, m)]
            if np.isnan(a):
                continue
            tm = true_mix(item)
            for k in INDUSTRIES:
                truth[k] += a * tm[k]
            ec = link.get(item)
            if ec:
                known[EC.loc[ec, "산업"]] += a
            else:
                unk_items.append((item, a))
        unk = sum(a for _, a in unk_items)
        share = {k: known[k] / sum(known.values()) for k in INDUSTRIES}
        exclude = dict(known)
        prop = {k: known[k] + unk * share[k] for k in INDUSTRIES}
        mix = dict(known)
        for item, a in unk_items:
            if item in est.index:
                for k in INDUSTRIES:
                    mix[k] += a * est.loc[item, f"추정_{k}"]
            else:
                for k in INDUSTRIES:
                    mix[k] += a * share[k]
        for k in INDUSTRIES:
            res.append({"월": str(m), "산업": k, "정답": truth[k], "미특정_제외": exclude[k], "연결물량_비율배분": prop[k], "주문패턴_추정배분": mix[k]})
    r = pd.DataFrame(res)
    w = lambda col: float((r[col] - r.정답).abs().sum() / r.정답.sum())
    summ = {"미특정_제외": w("미특정_제외"), "연결물량_비율배분": w("연결물량_비율배분"), "주문패턴_추정배분": w("주문패턴_추정배분"),
            "회사미특정_물량비중": float(1 - r.미특정_제외.sum() / r.정답.sum())}
    return r, summ


if __name__ == "__main__":
    import forecast_engine as F
    origin = pd.Period("2026-10", "M")
    pd.set_option("display.width", 250)
    df, lag, ls = evaluate(origin, F.LINK0)
    print("시차 선택:", lag, {k: round(v, 3) for k, v in ls.items()})
    single = df[df.정답.str.startswith("단일")]
    unl = single[single.연결표 == "회사 미특정"]
    print(f"단일 산업 품목 {len(single)}개: 1순위 적중 {single['1순위_적중'].mean():.0%}, 물량가중 {np.average(single['1순위_적중'], weights=single.최근6개월_평균물량):.0%}, 정답 산업 평균 추정비중 {single.정답산업_추정비중.mean():.0%}")
    print(f"  그중 연결표에 없는 품목 {len(unl)}개: 1순위 적중 {unl['1순위_적중'].mean():.0%}")
    mixed = df[~df.정답.str.startswith("단일")]
    print(f"컴파운더 혼합 품목 {len(mixed)}개: 비중 평균절대오차 {mixed.비중_평균절대오차.mean():.1%}")
    print(df[["품목", "연결표", "정답", "적합도_R2", "추정_1순위산업"] + [f"추정_{k}" for k in INDUSTRIES]].round(2).to_string())
    r, s = allocation_test(df, F.LINK0, pd.period_range("2026-05", "2026-10", freq="M"))
    print({k: round(v, 3) for k, v in s.items()})


# ---------------------------------------------------------------------------
# 단서 결합: 거래처 단서(같은 거래처의 연결된 다른 품목), 그레이드 단서(같은 그레이드의 연결 물량), 주문 패턴
# 검증 때는 대상 품목 자신의 연결 정보를 빼고 계산한다
# ---------------------------------------------------------------------------
def _vol(item, origin):
    v = np.nanmean([V.ACT[(item, origin - k)] for k in range(1, 13)])
    return 0.0 if np.isnan(v) else float(v)


def clue_mix(item, origin, link, by):
    it = V.ITEMS.set_index("item")
    acc = {k: 0.0 for k in INDUSTRIES}
    for j, ec in link.items():
        if j == item or not ec:
            continue
        if it.loc[j, by] == it.loc[item, by]:
            acc[EC.loc[ec, "산업"]] += _vol(j, origin)
    s = sum(acc.values())
    return ({k: v / s for k, v in acc.items()}, s) if s > 0 else (None, 0.0)


def overall_mix(origin, link, item=None):
    acc = {k: 0.0 for k in INDUSTRIES}
    for j, ec in link.items():
        if ec and j != item:
            acc[EC.loc[ec, "산업"]] += _vol(j, origin)
    s = sum(acc.values())
    return {k: v / s for k, v in acc.items()}


def combined_mix(item, origin, link, lag=None, pattern=None):
    """채택 방식: 거래처 단서(같은 거래처의 연결된 다른 품목의 산업 비율)가 있으면 그것, 없으면 전체 비율.
       주문 패턴과 그레이드 단서는 정답 가림 검증에서 전체 비율보다 나빠 채택하지 않았다"""
    base = overall_mix(origin, link, item)
    d, _ = clue_mix(item, origin, link, "거래처")
    if d:
        return d, "거래처 단서"
    return base, "전체 비율"


def evaluate_all(origin, link):
    """정답 가림 검증: 방법별 1순위 적중과 비중 오차. 연결된 품목은 자신의 연결을 가리고 맞힌다"""
    lag, _ = choose_lag(origin, link)
    rows = []
    for item in V.ITEMS.item:
        pat = fit_mix(item, origin, lag)
        if pat is None:
            continue
        tm = true_mix(item)
        single = max(tm.values()) == 1.0
        true_top = max(tm, key=tm.get)
        methods = {"전체 비율": overall_mix(origin, link, item), "주문 패턴만": pat["w"]}
        d, _ = clue_mix(item, origin, link, "거래처")
        g, _ = clue_mix(item, origin, link, "그레이드")
        methods["거래처 단서만"] = d or methods["전체 비율"]
        methods["그레이드 단서만"] = g or methods["전체 비율"]
        cm, src = combined_mix(item, origin, link, lag, pat)
        methods["채택: 거래처 단서, 없으면 전체 비율"] = cm
        for name, w in methods.items():
            top = max(w, key=w.get)
            rows.append({"품목": item, "연결표": link.get(item) or "회사 미특정", "단일산업": single, "방법": name,
                         "결합_근거": src, "물량": _vol(item, origin),
                         "1순위_적중": float(top == true_top) if single else np.nan,
                         "정답산업_비중": w[true_top] if single else np.nan,
                         "비중_평균절대오차": float(np.mean([abs(w[k] - tm[k]) for k in INDUSTRIES])),
                         **{f"추정_{k}": w[k] for k in INDUSTRIES}})
    return pd.DataFrame(rows), lag


def allocation_by(est_df, link, months):
    """est_df: 품목별 추정_산업 열. 회사 미특정 품목의 실적을 이 비중으로 나눈 산업별 합계와 정답의 오차"""
    est = est_df.set_index("품목")
    tot_err, tot = 0.0, 0.0
    for m in months:
        truth = {k: 0.0 for k in INDUSTRIES}
        alloc = {k: 0.0 for k in INDUSTRIES}
        for item in V.ITEMS.item:
            a = V.ACT[(item, m)]
            if np.isnan(a):
                continue
            tm = true_mix(item)
            ec = link.get(item)
            for k in INDUSTRIES:
                truth[k] += a * tm[k]
                alloc[k] += a * ((EC.loc[ec, "산업"] == k) if ec else est.loc[item, f"추정_{k}"])
        tot_err += sum(abs(alloc[k] - truth[k]) for k in INDUSTRIES)
        tot += sum(truth.values())
    return tot_err / tot


def run_eval(origin=pd.Period("2026-10", "M"), link=None, months=None):
    import forecast_engine as F
    link = link if link is not None else F.LINK0
    months = months if months is not None else pd.period_range(origin - 5, origin, freq="M")
    ev, lag = evaluate_all(origin, link)
    summ = []
    for name, g in ev.groupby("방법", sort=False):
        s = g[g.단일산업]
        su = s[s.연결표 == "회사 미특정"]
        mx = g[~g.단일산업]
        unk = g[g.연결표 == "회사 미특정"]
        summ.append({"방법": name, "단일산업_1순위적중": s["1순위_적중"].mean(),
                     "단일산업_물량가중적중": np.average(s["1순위_적중"], weights=s.물량),
                     "회사미특정_단일산업_적중": su["1순위_적중"].mean(),
                     "컴파운더혼합_비중오차": mx.비중_평균절대오차.mean(),
                     "산업배분_오차_최근6개월": allocation_by(unk, link, months)})
    summ = pd.DataFrame(summ)
    # 비교 기준: 회사 미특정 물량을 산업에서 빼고 보는 경우
    excl = unk.assign(**{f"추정_{k}": 0.0 for k in INDUSTRIES})
    base_excl = allocation_by(excl, link, months)
    return ev, summ, base_excl, lag


if __name__ == "__main__":
    ev, summ, base_excl, lag = run_eval()
    print("시차", lag, "| 미특정 물량을 빼고 볼 때 산업 배분 오차", round(base_excl, 3))
    print(summ.round(3).to_string())
    print(ev[ev.방법.str.startswith("채택")][["품목", "연결표", "결합_근거", "1순위_적중", "비중_평균절대오차"] + [f"추정_{k}" for k in INDUSTRIES]].round(2).to_string())


# ---------------------------------------------------------------------------
# 스트레스 시험: 거래처가 여러 산업에 섞여 있으면 거래처 단서가 얼마나 떨어지는가
# 단일 고객 품목의 실제 최종 고객을 확률 p로 다른 고객사로 바꾼 가상 연결 구조를 여러 번 만들고(연결된 품목의 연결표도 같이 바뀜),
# 연결된 품목의 정답을 가린 채 산업 적중률과 최근 6개월 산업 배분 오차를 잰다. 주문 실적은 그대로 쓴다
# ---------------------------------------------------------------------------
def stress_test(origin=pd.Period("2026-10", "M"), mix_rates=(0.0, 0.2, 0.4, 0.6), n_draw=200, seed=0):
    import forecast_engine as F
    rng = np.random.default_rng(seed)
    items = list(V.ITEMS.item)
    it = V.ITEMS.set_index("item")
    linked0 = {i: bool(F.LINK0.get(i)) for i in items}
    ecs = list(EC.index)
    vol = {i: _vol(i, origin) for i in items}
    months = pd.period_range(origin - 5, origin, freq="M")
    act = {i: sum(0.0 if np.isnan(V.ACT[(i, m)]) else V.ACT[(i, m)] for m in months) for i in items}
    n_ind = V.END_CUSTOMERS.groupby("산업").size()
    mixed_truth = {k: float(n_ind[k] / n_ind.sum()) for k in INDUSTRIES}
    rows = []
    for p in mix_rates:
        for d in range(n_draw):
            truth = {}
            for i in items:
                e = it.loc[i, "true_ec"]
                if isinstance(e, str) and rng.random() < p:
                    e = rng.choice([x for x in ecs if x != e])
                truth[i] = e if isinstance(e, str) else None
            link = {i: (truth[i] if linked0[i] else None) for i in items}
            def tm(i):
                return {k: float(EC.loc[truth[i], "산업"] == k) for k in INDUSTRIES} if truth[i] else mixed_truth
            def est(i, lk):
                acc = {k: 0.0 for k in INDUSTRIES}
                for j, e in lk.items():
                    if j != i and e and it.loc[j, "거래처"] == it.loc[i, "거래처"]:
                        acc[EC.loc[e, "산업"]] += vol[j]
                s = sum(acc.values())
                if s > 0:
                    return {k: v / s for k, v in acc.items()}, True
                acc = {k: 0.0 for k in INDUSTRIES}
                for j, e in lk.items():
                    if j != i and e:
                        acc[EC.loc[e, "산업"]] += vol[j]
                s = sum(acc.values())
                return {k: v / s for k, v in acc.items()}, False
            # 정답 가림: 연결된 품목 하나씩 가리고 맞히기
            hits, clue_used = [], []
            for i in items:
                if not link[i]:
                    continue
                w, used = est(i, link)
                t = tm(i)
                hits.append(float(max(w, key=w.get) == max(t, key=t.get)))
                clue_used.append(used)
            # 운영: 연결 안 된 품목 실적을 추정 비중으로 나눈 산업 합계 오차
            truth_ind = {k: 0.0 for k in INDUSTRIES}
            alloc = {k: 0.0 for k in INDUSTRIES}
            alloc_prop = {k: 0.0 for k in INDUSTRIES}
            base = {k: 0.0 for k in INDUSTRIES}
            for j, e in link.items():
                if e:
                    base[EC.loc[e, "산업"]] += vol[j]
            sb = sum(base.values())
            base = {k: v / sb for k, v in base.items()}
            for i in items:
                t = tm(i)
                for k in INDUSTRIES:
                    truth_ind[k] += act[i] * t[k]
                if link[i]:
                    for k in INDUSTRIES:
                        a = act[i] * float(EC.loc[link[i], "산업"] == k)
                        alloc[k] += a
                        alloc_prop[k] += a
                else:
                    w, _ = est(i, link)
                    for k in INDUSTRIES:
                        alloc[k] += act[i] * w[k]
                        alloc_prop[k] += act[i] * base[k]
            tot = sum(truth_ind.values())
            rows.append({"섞임비율": p, "회차": d, "가림_적중": np.mean(hits),
                         "배분오차_거래처단서": sum(abs(alloc[k] - truth_ind[k]) for k in INDUSTRIES) / tot,
                         "배분오차_전체비율": sum(abs(alloc_prop[k] - truth_ind[k]) for k in INDUSTRIES) / tot})
    r = pd.DataFrame(rows)
    summ = r.groupby("섞임비율").agg(가림_적중=("가림_적중", "mean"), 배분오차_거래처단서=("배분오차_거래처단서", "mean"),
                                    배분오차_전체비율=("배분오차_전체비율", "mean"))
    summ["거래처단서가_나은_비율"] = r.assign(w=r.배분오차_거래처단서 < r.배분오차_전체비율).groupby("섞임비율").w.mean()
    return summ.reset_index()
