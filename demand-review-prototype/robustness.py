# -*- coding: utf-8 -*-
"""안정성 검사: 가상 데이터를 시드별로 다시 만들어 ML 백테스트를 반복하고 평균과 편차를 낸다 (LLM 미사용)"""
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
F.CALIB_LEVEL, _ = F.choose_level(bt); bt = F.add_intervals(bt, bt)
s = F.backtest_summary(bt); s = s[s.구간 == "최종평가"]
print("JSON" + json.dumps({"level": F.CALIB_LEVEL, "rows": s.to_dict("records")}, ensure_ascii=False))
'''


def run(seed):
    env = dict(os.environ, DEMAND_SEED=str(seed))
    p = subprocess.run([sys.executable, "-c", CODE], cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8")
    line = [x for x in p.stdout.splitlines() if x.startswith("JSON")][0]
    return json.loads(line[4:])


if __name__ == "__main__":
    rows = []
    for sd in SEEDS:
        r = run(sd)
        for x in r["rows"]:
            rows.append({"시드": sd, "보정수준": r["level"], "예측거리": x["예측거리(개월)"], "WAPE_기준모델": x["WAPE_기준모델"],
                         "WAPE_고객FCST": x["WAPE_고객FCST"], "WAPE_독립예측": x["WAPE_패턴별챔피언"],
                         "WAPE_AI보정FCST": x["WAPE_AI보정FCST"], "구간포함률": x["구간포함률_보정후(목표80%)"]})
        print("seed", sd, "done", flush=True)
    df = pd.DataFrame(rows)
    df["AI보정_개선률"] = 1 - df.WAPE_AI보정FCST / df.WAPE_고객FCST
    g = df.groupby("예측거리")[["WAPE_기준모델", "WAPE_고객FCST", "WAPE_독립예측", "WAPE_AI보정FCST", "AI보정_개선률", "구간포함률"]]
    summ = g.mean().round(4).add_suffix("_평균").join(g.std().round(4).add_suffix("_편차"))
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "output", "안정성검사_시드별.csv"), index=False, encoding="utf-8-sig")
    summ.to_csv(os.path.join(HERE, "output", "안정성검사_요약.csv"), encoding="utf-8-sig")
    print(summ.to_string())
    print("AI 보정 FCST가 고객 FCST보다 나빴던 경우:", int((df.WAPE_AI보정FCST > df.WAPE_고객FCST).sum()), "/", len(df))
    print("독립 예측이 기준 모델보다 나빴던 경우:", int((df.WAPE_독립예측 > df.WAPE_기준모델).sum()), "/", len(df))
