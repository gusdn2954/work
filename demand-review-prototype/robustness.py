# -*- coding: utf-8 -*-
"""안정성 검사: 가상 데이터를 시드별로 다시 만들어 ML 백테스트와 예외 선별 평가를 반복하고 평균과 편차를 낸다 (LLM 미사용).
환경변수 DEMAND_DATA=designed로 바꾸면 설계형 데이터로 반복한다."""
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3, 4, 5, 7]
CODE = r'''
import json, forecast_engine as F
bt = F.run_backtest(); champ, _ = F.select_champions(bt); bt = F.apply_champion(bt, champ)
F.CALIB_LEVEL, _ = F.choose_level(bt); bt = F.add_intervals(bt, bt); bt = F.add_selected(bt)
s = F.backtest_summary(bt); s = s[s.구간 == "최종평가"]
fe = F.eval_flags(bt, shares=(0.25,)); fe = fe[fe.구간 == "최종평가"]
print("JSON" + json.dumps({"level": F.CALIB_LEVEL, "rows": s.to_dict("records"), "flags": fe.to_dict("records")}, ensure_ascii=False))
'''


def run(seed):
    env = dict(os.environ, DEMAND_SEED=str(seed))
    p = subprocess.run([sys.executable, "-c", CODE], cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8")
    line = [x for x in p.stdout.splitlines() if x.startswith("JSON")][0]
    return json.loads(line[4:])


FLAGS = []
if __name__ == "__main__":
    rows = []
    for sd in SEEDS:
        r = run(sd)
        for x in r["rows"]:
            rows.append({"시드": sd, "보정수준": r["level"], "예측거리": x["예측거리(개월)"], "WAPE_기준모델": x["WAPE_기준모델"],
                         "WAPE_고객FCST": x["WAPE_고객FCST"], "WAPE_독립예측": x["WAPE_패턴별챔피언"],
                         "WAPE_AI보정FCST": x["WAPE_AI보정FCST"], "WAPE_접수주문제외": x["WAPE_LGB_접수주문제외"],
                         "WAPE_두예측평균": x["WAPE_두예측평균"], "WAPE_품목별선택": x["WAPE_품목별선택"], "WAPE_거래처별선택": x["WAPE_거래처별선택"],
                         "구간포함률": x["구간포함률_보정후(목표80%)"]})
        for f in r["flags"]:
            FLAGS.append({"시드": sd, "규칙": f["규칙"], "정밀도": f["정밀도"], "재현율": f["재현율"], "잡아낸오차비중": f["잡아낸오차비중"], "검토비율": f["검토비율"]})
        print("seed", sd, "done", flush=True)
    df = pd.DataFrame(rows)
    df["AI보정_개선률"] = 1 - df.WAPE_AI보정FCST / df.WAPE_고객FCST
    g = df.groupby("예측거리")[["WAPE_기준모델", "WAPE_접수주문제외", "WAPE_고객FCST", "WAPE_독립예측", "WAPE_AI보정FCST", "WAPE_두예측평균", "WAPE_품목별선택", "WAPE_거래처별선택", "AI보정_개선률", "구간포함률"]]
    summ = g.mean().round(4).add_suffix("_평균").join(g.std().round(4).add_suffix("_편차"))
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "output", "안정성검사_시드별.csv"), index=False, encoding="utf-8-sig")
    summ.to_csv(os.path.join(HERE, "output", "안정성검사_요약.csv"), encoding="utf-8-sig")
    print(summ.to_string())
    print("AI 보정 FCST가 고객 FCST보다 나빴던 경우:", int((df.WAPE_AI보정FCST > df.WAPE_고객FCST).sum()), "/", len(df))
    print("독립 예측이 기준 모델보다 나빴던 경우:", int((df.WAPE_독립예측 > df.WAPE_기준모델).sum()), "/", len(df))
    print("AI 보정이 독립 예측보다 나빴던 경우:", int((df.WAPE_AI보정FCST > df.WAPE_독립예측).sum()), "/", len(df))
    df["둘중_나은값"] = df[["WAPE_독립예측", "WAPE_AI보정FCST"]].min(axis=1)
    for c in ["WAPE_AI보정FCST", "WAPE_두예측평균", "WAPE_품목별선택", "WAPE_거래처별선택"]:
        print(f"{c}: 독립 예측보다 나쁨 {int((df[c] > df.WAPE_독립예측).sum())}/{len(df)}, AI 보정보다 나쁨 {int((df[c] > df.WAPE_AI보정FCST).sum())}/{len(df)}, 둘 중 나은 값 대비 평균 차 {(df[c] - df.둘중_나은값).mean():+.4f}")
    fl = pd.DataFrame(FLAGS)
    fs = fl.groupby("규칙")[["검토비율", "정밀도", "재현율", "잡아낸오차비중"]].agg(["mean", "std"]).round(3)
    fl.to_csv(os.path.join(HERE, "output", "안정성검사_예외선별.csv"), index=False, encoding="utf-8-sig")
    fs.to_csv(os.path.join(HERE, "output", "안정성검사_예외선별_요약.csv"), encoding="utf-8-sig")
    print(fs.to_string())
