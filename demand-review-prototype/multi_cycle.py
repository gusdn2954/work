# -*- coding: utf-8 -*-
"""
6개월 연속 회차 시뮬레이션: "쓸수록 나아지는가"를 달별로 확인한다 (가상 데이터, LLM 미사용)

  - 회차: 2026-05 ~ 2026-10 예측 기준월, 매 회차 그 시점까지의 실적으로 다시 학습
  - 검토 예산: 고객 FCST와 AI 보정 FCST 차이 상위 25% (중요도 10톤 이상)
  - 질문 예산: 담당자당 4품목, 이월 품목은 다음 회차 우선순위 가점
  - 가상 영업 응답기: 답변을 구조화된 형태로 바로 돌려준다. 자유 문장 해석은 10월 회차 LLM 실행과 스트레스 시험에서 따로 검증했다.
      연결 질문: 실제 최종 고객이 있으면 80% 확률로 답, 컴파운더처럼 귀속이 안 되면 90% 확률로 "특정 불가" 확정
      수량 검토 질문: 75% 답변(확인 완료), 15% 보류, 10% 무응답
  - 답변된 연결은 연결표에 저장되어 다음 회차부터 생산지수 입력과 근거 연결에 쓰이고, "특정 불가"는 다시 묻지 않는다
"""
import os

import numpy as np
import pandas as pd

import virtual_data as V
import forecast_engine as F

HERE = os.path.dirname(os.path.abspath(__file__))
CYCLES = pd.period_range("2026-05", "2026-10", freq="M")
QUESTION_CAP = 4
OWNER = V.ITEMS.set_index("item")["담당"].to_dict()
TRUE_EC = {r.item: (r.true_ec if isinstance(r.true_ec, str) else None) for r in V.ITEMS.itertuples()}


def forecast(origin, link, bt):
    tr = F.build(pd.period_range("2024-01", origin - 1, freq="M"), link)
    tr = tr[tr.target < origin]
    te = F.build([origin], link, with_target=False)
    te["p_lgb"] = F.fit_predict(tr, te, F.SIG_COLS)
    te["lgb_q10"] = np.minimum(F.fit_predict(tr, te, F.SIG_COLS, 0.1), te.p_lgb)
    te["lgb_q90"] = np.maximum(F.fit_predict(tr, te, F.SIG_COLS, 0.9), te.p_lgb)
    te["p_corr"] = F.fit_predict(tr, te, F.CORR_COLS)
    te["p_base"] = te.ma3.fillna(0)
    te["p_sba"] = [F.sba(i, origin) for i in te.item]
    te["scale"] = np.maximum(te.ma12.fillna(te.ma3).fillna(1), 1)
    te = F.apply_champion(te, CHAMP)
    te = F.add_intervals(te, bt)
    te["y"] = [V.ACT[(i, t)] for i, t in zip(te.item, te.target)]
    return te


def link_rate(link, origin):
    vol = {it: V.ACT[(it, origin - 1)] for it in V.ITEMS.item}
    vol = {k: (0 if np.isnan(v) else v) for k, v in vol.items()}
    return sum(v for it, v in vol.items() if link.get(it)) / sum(vol.values())


def run(seed=11, learn=True):
    rng = np.random.default_rng(seed)
    link = dict(F.LINK0)
    unattributable = set()
    carry = {}
    rows, units = [], []
    for o in CYCLES:
        link_start = dict(link)
        fc = forecast(o, link, BT)
        flag, thr = F.review_flag(fc)
        fc["flag"] = flag
        fc["gap"] = (fc.fcst - fc.p_corr).abs()
        need_link = [it for it in V.ITEMS.item if link.get(it) is None and it not in unattributable and it in set(fc.item)]
        # 질문 후보: 검토 대상이 있는 품목 + 연결 확인이 필요한 품목
        cand = {}
        for it, g in fc.groupby("item"):
            score = float(g.gap[g.flag].sum()) + (5.0 if it in need_link else 0.0) + carry.get(it, 0.0)
            if g.flag.any() or it in need_link:
                cand[it] = score
        pr = pd.DataFrame([{"item": k, "owner": OWNER[k], "score": v} for k, v in cand.items()])
        asked, deferred = [], []
        if len(pr):
            pr = pr.sort_values(["owner", "score"], ascending=[True, False])
            pr["rank"] = pr.groupby("owner").cumcount() + 1
            asked = list(pr[pr["rank"] <= QUESTION_CAP].item)
            deferred = list(pr[pr["rank"] > QUESTION_CAP].item)
        carry = {it: carry.get(it, 0.0) + 20.0 for it in deferred}
        resolved, answered = set(), 0
        for it in asked:
            # 연결 질문
            if it in need_link:
                if TRUE_EC[it] and rng.random() < 0.8:
                    if learn:
                        link[it] = TRUE_EC[it]
                elif not TRUE_EC[it] and rng.random() < 0.9:
                    if learn:
                        unattributable.add(it)
            # 수량 검토 질문
            u = rng.random()
            if u < 0.75:
                resolved.add(it)
                answered += 1
            elif u < 0.90:
                answered += 1  # 보류: 답은 왔지만 판단은 사람에게
        fc["status"] = np.where(~fc.flag, "정상", np.where(fc.item.isin(resolved), "승인만", "직접 판단"))
        fc["proposal"] = fc.p_corr  # 사실 확인이 없는 회차라 제안값 = AI 보정 FCST
        h1 = fc[fc.h == 1]
        big = ((fc.fcst - fc.y).abs() >= 10) & ((fc.fcst - fc.y).abs() >= 0.2 * fc.y.clip(lower=1))
        rows.append({
            "회차": str(o), "검토단위": len(fc), "검토대상": int(fc.flag.sum()), "검토정밀도": float((fc.flag & big).sum() / max(1, fc.flag.sum())),
            "질문품목": len(asked), "묶음메시지": len({OWNER[i] for i in asked}), "이월품목": len(deferred),
            "답변률": answered / max(1, len(asked)), "연결확인필요": len(need_link),
            "직접판단": int((fc.status == "직접 판단").sum()), "승인만": int((fc.status == "승인만").sum()),
            "연결률_회차시작": link_rate(link_start, o), "연결률_회차끝": link_rate(link, o),
            "WAPE1_고객FCST": F.wape(h1.y, h1.fcst), "WAPE1_독립예측": F.wape(h1.y, h1.p_champ), "WAPE1_AI보정": F.wape(h1.y, h1.p_corr),
            "WAPE전체_독립예측": F.wape(fc.y, fc.p_champ), "WAPE전체_AI보정": F.wape(fc.y, fc.p_corr),
        })
        units.append(fc.assign(회차=str(o)))
        print(o, "done", flush=True)
    return pd.DataFrame(rows), pd.concat(units, ignore_index=True)


BT = F.run_backtest()
CHAMP, _ = F.select_champions(BT)
BT = F.apply_champion(BT, CHAMP)
F.CALIB_LEVEL, _ = F.choose_level(BT)
BT = F.add_intervals(BT, BT)

if __name__ == "__main__":
    res, units = run(learn=True)
    base, _ = run(learn=False)
    cmp = res[["회차", "연결률_회차시작", "질문품목", "묶음메시지", "연결확인필요", "WAPE전체_독립예측", "WAPE전체_AI보정"]].merge(
        base[["회차", "연결률_회차시작", "질문품목", "연결확인필요", "WAPE전체_독립예측", "WAPE전체_AI보정"]], on="회차", suffixes=("_학습", "_학습안함"))
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    res.to_csv(os.path.join(HERE, "output", "연속회차_6개월.csv"), index=False, encoding="utf-8-sig")
    cmp.to_csv(os.path.join(HERE, "output", "연속회차_학습효과비교.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())
    print(cmp.round(3).to_string())
    tail = cmp[cmp.회차 >= "2026-07"]
    print("7~10월 평균 WAPE(전체 예측 거리) 독립예측: 학습", round(tail.WAPE전체_독립예측_학습.mean(), 4), "학습안함", round(tail.WAPE전체_독립예측_학습안함.mean(), 4),
          "| AI보정: 학습", round(tail.WAPE전체_AI보정_학습.mean(), 4), "학습안함", round(tail.WAPE전체_AI보정_학습안함.mean(), 4))
