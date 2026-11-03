# -*- coding: utf-8 -*-
"""
가상 데이터 (모두 가상이며 롯데케미칼의 실제 고객, 거래, 코드, 수치가 아님)

구성
  - 코드 계층: 제품군 > 계열 > 그레이드 > 색상
  - 최종 고객 6곳(자동차, 가전, IT기기, 의료기기)과 직접 거래처 12곳(부품사, 사출업체, 완제품사 직거래, 컴파운더)
  - 수요 패턴이 다른 품목(안정, 변동, 간헐, 덩어리, 신규)
  - MES lot 이력(생산 확정일, 출하일, 창고 대기일)으로 그레이드별 리드타임과 재고 일수 계산
  - 공개 뉴스(자유 문장), 영업 담당자 답변(자유 문장)과 정답 라벨 (LLM 정확도 평가용)
  - 2026년 11월 이후 숨은 변화: 완성차 A 감산, IT기기 E 신모델, 컴파운더 D09의 가격 기대 선구매
"""
import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)
MONTHS = pd.period_range("2023-01", "2027-01", freq="M")

# ---------------------------------------------------------------------------
# 코드 계층 (가상 코드)
# ---------------------------------------------------------------------------
GRADES = pd.DataFrame([
    ("AB-G10", "ABS", "범용"), ("AB-H20", "ABS", "내열"),
    ("PC-N30", "PC", "일반"), ("PC-F40", "PC", "난연"),
    ("CP-M50", "컴파운드", "PC/ABS"), ("CP-R60", "컴파운드", "유리섬유강화"),
], columns=["그레이드", "제품군", "계열"])
COLORS = ["NAT", "BLK", "WHT", "GRY"]

END_CUSTOMERS = pd.DataFrame([
    ("E_A", "완성차 A사", "자동차"), ("E_B", "완성차 B사", "자동차"),
    ("E_C", "가전 C사", "가전"), ("E_D", "가전 D사", "가전"),
    ("E_E", "IT기기 E사", "IT기기"), ("E_F", "의료기기 F사", "의료기기"),
], columns=["code", "name", "산업"])

DISTRIBUTORS = pd.DataFrame([
    ("D01", "부품사", "영업1팀 담당 가"), ("D02", "부품사", "영업1팀 담당 나"),
    ("D03", "사출업체", "영업2팀 담당 다"), ("D04", "사출업체", "영업2팀 담당 라"),
    ("D05", "사출업체", "영업3팀 담당 마"), ("D06", "사출업체", "영업3팀 담당 바"),
    ("D07", "부품사", "영업1팀 담당 가"), ("D08", "완제품사 직거래", "영업2팀 담당 다"),
    ("D09", "컴파운더", "영업4팀 담당 사"), ("D10", "컴파운더", "영업4팀 담당 사"),
    ("D11", "사출업체", "영업3팀 담당 바"), ("D12", "부품사", "영업1팀 담당 나"),
], columns=["거래처", "유형", "담당"])
TEAM_LEAD = {"영업1팀": "영업1팀장", "영업2팀": "영업2팀장", "영업3팀": "영업3팀장", "영업4팀": "영업4팀장"}

# 품목 = 거래처 x 그레이드 (고객 FCST 제출 단위)
# true_ec: 실제 최종 고객(가상, None은 여러 고객에 섞여 귀속 불가), linked: 주문 기록이나 승인 이력으로 연결이 확인됐는지
# pattern: 데이터를 만들 때 쓴 성격(모델은 이 값을 모르고 이력으로 다시 분류한다), start: 첫 주문 월
ITEMS = pd.DataFrame([
    ("D01", "AB-G10", "E_A", True, 120, "smooth", "2023-01"),
    ("D01", "AB-H20", "E_A", True, 80, "smooth", "2023-01"),
    ("D01", "CP-M50", "E_A", True, 60, "smooth", "2023-01"),
    ("D02", "AB-G10", "E_B", True, 150, "smooth", "2023-01"),
    ("D02", "PC-N30", "E_B", True, 70, "smooth", "2023-01"),
    ("D02", "CP-R60", "E_B", False, 40, "erratic", "2023-01"),
    ("D03", "AB-G10", "E_C", True, 90, "smooth", "2023-01"),
    ("D03", "AB-H20", "E_C", True, 50, "smooth", "2023-01"),
    ("D03", "PC-F40", "E_D", False, 35, "smooth", "2023-01"),
    ("D04", "AB-G10", "E_D", True, 110, "smooth", "2023-01"),
    ("D04", "CP-M50", "E_C", False, 45, "smooth", "2023-01"),
    ("D05", "PC-N30", "E_F", False, 30, "smooth", "2023-01"),
    ("D05", "PC-F40", "E_F", True, 12, "intermittent", "2023-01"),
    ("D06", "PC-F40", "E_E", True, 55, "smooth", "2023-01"),
    ("D06", "CP-M50", "E_E", True, 40, "erratic", "2023-01"),
    ("D06", "AB-G10", "E_E", False, 25, "intermittent", "2023-01"),
    ("D07", "CP-R60", "E_A", True, 65, "smooth", "2023-01"),
    ("D07", "AB-H20", "E_B", False, 45, "smooth", "2023-01"),
    ("D08", "AB-G10", "E_D", True, 75, "smooth", "2023-01"),
    ("D08", "PC-N30", "E_D", True, 20, "lumpy", "2023-01"),
    ("D09", "AB-G10", None, False, 140, "smooth", "2023-01"),
    ("D09", "PC-N30", None, False, 90, "smooth", "2023-01"),
    ("D10", "PC-F40", None, False, 25, "lumpy", "2023-01"),
    ("D10", "CP-R60", None, False, 18, "intermittent", "2023-01"),
    ("D11", "PC-F40", "E_E", True, 20, "smooth", "2026-03"),
    ("D12", "CP-M50", "E_A", True, 25, "smooth", "2026-01"),
    ("D12", "AB-H20", "E_A", True, 15, "intermittent", "2023-01"),
    ("D04", "PC-N30", "E_C", True, 28, "smooth", "2023-01"),
    ("D08", "CP-M50", "E_D", True, 22, "erratic", "2023-01"),
    ("D02", "AB-H20", "E_B", True, 60, "smooth", "2023-01"),
], columns=["거래처", "그레이드", "true_ec", "linked", "base", "pattern", "start"])
ITEMS["item"] = ITEMS["거래처"] + "-" + ITEMS["그레이드"]
ITEMS = ITEMS.merge(GRADES, on="그레이드").merge(DISTRIBUTORS, on="거래처")

# ---------------------------------------------------------------------------
# 최종 고객 생산지수(가상, 공표 시차 1개월)
# ---------------------------------------------------------------------------
t = np.arange(len(MONTHS))
_sig = {}
for ec, amp, trend in [("E_A", 0.08, 0.002), ("E_B", 0.06, 0.001), ("E_C", 0.10, -0.001),
                       ("E_D", 0.09, 0.0005), ("E_E", 0.05, 0.003), ("E_F", 0.02, 0.002)]:
    _sig[ec] = 100 * (1 + trend * t) * (1 + amp * np.sin(2 * np.pi * (t % 12) / 12)) + rng.normal(0, 2, len(t))
_sig["E_A"][MONTHS.get_loc(pd.Period("2026-06", "M")):] *= 0.82            # 공개된 감산
_sig["E_B"][MONTHS.get_loc(pd.Period("2025-09", "M")):MONTHS.get_loc(pd.Period("2026-01", "M"))] *= 1.12
_sig["E_A"][MONTHS.get_loc(pd.Period("2026-11", "M")):] *= 0.85            # 숨은 추가 감산(11월부터)
_sig["E_E"][MONTHS.get_loc(pd.Period("2026-11", "M")):] *= 1.10            # 숨은 신모델 생산(11월부터)
SIG = pd.DataFrame(_sig, index=MONTHS)
SIG["MARKET"] = SIG[[c for c in SIG.columns]].mean(axis=1)                 # 연결이 없는 품목의 비교용 시장 지표
SIG_PUBLISH_LAG = 1

# ---------------------------------------------------------------------------
# 실제 주문량 (최종 순주문량). ACT_BASE: 11월 이후 숨은 변화가 없었을 경우, ACT: 실제
# ---------------------------------------------------------------------------
HIDDEN = {  # (품목) -> {월: 배수 또는 가감 톤}
    "E_A_CUT": 0.85, "D07-CP-R60": 0.92, "E_E_UP": 1.10,
    "D09-AB-G10": {"2026-11": +60, "2026-12": -40, "2027-01": -20},
}
ACT, ACT_BASE = {}, {}
for _, it in ITEMS.iterrows():
    start = pd.Period(it.start, "M")
    for i, m in enumerate(MONTHS):
        if m < start:
            ACT[(it["item"], m)] = np.nan
            ACT_BASE[(it["item"], m)] = np.nan
            continue
        ec = it.true_ec if isinstance(it.true_ec, str) else "MARKET"
        sig_base = SIG[ec].iloc[max(i - 2, 0)] / 100.0
        season = 1 + 0.05 * np.cos(2 * np.pi * (m.month - 3) / 12)
        ramp = min(1.0, 0.4 + 0.1 * (m - start).n) if it.start != "2023-01" else 1.0
        level = it.base * sig_base * season * ramp
        if it.pattern == "smooth":
            q = level * rng.lognormal(0, 0.08)
        elif it.pattern == "erratic":
            q = level * rng.lognormal(0, 0.35)
        elif it.pattern == "intermittent":
            q = (level / 0.45) * rng.lognormal(0, 0.15) if rng.random() < 0.45 else 0.0
        else:  # lumpy
            q = (level / 0.4) * rng.lognormal(0, 0.6) if rng.random() < 0.4 else 0.0
        ACT_BASE[(it["item"], m)] = q
        # 숨은 변화(11월 이후): 고객 FCST와 공개 지표는 아직 모름
        q2 = q
        if m >= pd.Period("2026-11", "M"):
            if it.true_ec == "E_A":
                q2 = q * (HIDDEN["D07-CP-R60"] if it["item"] == "D07-CP-R60" else HIDDEN["E_A_CUT"])
            if it.true_ec == "E_E":
                q2 = q * HIDDEN["E_E_UP"]
            if it["item"] == "D09-AB-G10":
                q2 = max(0.0, q + HIDDEN["D09-AB-G10"].get(str(m), 0))
        ACT[(it["item"], m)] = q2

# 거래처별 고객 FCST 편향(가상): D02는 과대, D04는 과소 경향
FCST_BIAS = {"D01": 0.02, "D02": 0.22, "D03": 0.0, "D04": -0.12, "D05": 0.05, "D06": 0.0,
             "D07": 0.03, "D08": 0.0, "D09": 0.08, "D10": 0.15, "D11": 0.10, "D12": 0.0}
ACCEPT_SHARE = {1: 0.55, 2: 0.25, 3: 0.08}
_FC, _AC = {}, {}


def customer_fcst(item, target, origin):
    """고객 FCST: 고객이 아는 범위(ACT_BASE)에 거래처 편향과 잡음. D09의 11월 선구매 주문은 고객이 알고 FCST에 넣음"""
    key = (item, target, origin)
    if key not in _FC:
        d = item.split("-")[0]
        h = (target - origin).n
        base = ACT_BASE[(item, target)]
        if item == "D09-AB-G10" and str(target) == "2026-11":
            base = ACT[(item, target)]
        if np.isnan(base):
            _FC[key] = np.nan
        else:
            noise = rng.lognormal(0, 0.06 + 0.03 * h)
            # 간헐, 덩어리 품목은 고객도 월별 시점을 맞히기 어려워 평균 수준을 제출
            pat = ITEMS.set_index("item").loc[item, "pattern"]
            if pat in ("intermittent", "lumpy"):
                hist = [ACT_BASE[(item, origin - k)] for k in range(1, 13)]
                base = float(np.nanmean(hist))
            _FC[key] = base * (1 + FCST_BIAS[d]) * noise
    return _FC[key]


def accepted(item, target, origin):
    """예측 시점에 이미 접수된 주문. D09의 11월 선구매분은 미리 들어옴"""
    key = (item, target, origin)
    if key not in _AC:
        h = (target - origin).n
        a = ACT[(item, target)]
        if np.isnan(a):
            _AC[key] = np.nan
        else:
            share = np.clip(ACCEPT_SHARE[h] + rng.normal(0, 0.05), 0, 0.95)
            if item == "D09-AB-G10" and str(target) == "2026-11" and str(origin) == "2026-10":
                share = 0.9
            _AC[key] = a * share
    return _AC[key]


# ---------------------------------------------------------------------------
# MES lot 이력 (가상, 일반화한 형식). 리드타임 = 생산 확정일 ~ 출하일, 재고 일수 = 생산 완료 ~ 출하 대기일
# ---------------------------------------------------------------------------
_LT = {"AB-G10": 18, "AB-H20": 24, "PC-N30": 28, "PC-F40": 35, "CP-M50": 40, "CP-R60": 45}
_INV = {"AB-G10": 12, "AB-H20": 16, "PC-N30": 14, "PC-F40": 20, "CP-M50": 22, "CP-R60": 25}
lots = []
n = 0
for g in GRADES.그레이드:
    for m in pd.period_range("2025-10", "2026-09", freq="M"):
        for c in COLORS:
            for _ in range(rng.integers(1, 4)):
                n += 1
                confirm = m.start_time + pd.Timedelta(days=int(rng.integers(0, 25)))
                lt = max(7, int(rng.normal(_LT[g], 4)))
                inv = max(2, int(rng.normal(_INV[g], 5)))
                lots.append({"lot": f"LOT-{n:05d}", "그레이드": g, "색상": c, "생산확정일": confirm.date(),
                             "출하일": (confirm + pd.Timedelta(days=lt)).date(), "창고대기일": inv,
                             "lot크기_톤": float(rng.choice([20, 25, 30, 40])) if c != "NAT" else float(rng.choice([30, 40, 50]))})
MES_LOTS = pd.DataFrame(lots)

# ---------------------------------------------------------------------------
# 공개 뉴스(가상, 자유 문장)와 정답 라벨
# ---------------------------------------------------------------------------
NEWS = [
    {"id": "N1", "published": "2026-10-01",
     "text": "[가상 기사] 완성차 A사가 9월 글로벌 판매 부진에 따라 11월부터 일부 차종 생산 라인을 15% 감산하기로 했다고 밝혔다.",
     "gold": {"end_customer": "E_A", "direction": "하향", "effect_from": "2026-11"}},
    {"id": "N2", "published": "2026-09-20",
     "text": "[가상 기사] 완성차 B사가 전기차 신모델 출시를 검토 중이다. 출시 시기는 아직 정해지지 않았다.",
     "gold": {"end_customer": "E_B", "direction": "상향", "effect_from": None}},
    {"id": "N3", "published": "2026-10-02",
     "text": "[가상 기사] 가전 C사는 내년 2분기부터 일부 냉장고 모델을 단종할 예정이다.",
     "gold": {"end_customer": "E_C", "direction": "하향", "effect_from": "2027-04"}},
    {"id": "N4", "published": "2026-10-03",
     "text": "[가상 기사] 가전 X사가 연말 판촉 행사를 확대한다.",
     "gold": {"end_customer": "UNKNOWN", "direction": "상향", "effect_from": None}},
    {"id": "N5", "published": "2026-10-02",
     "text": "[가상 기사] 완성차 B사는 12월부터 주말 특근을 실시해 일부 차종 생산을 늘린다.",
     "gold": {"end_customer": "E_B", "direction": "상향", "effect_from": "2026-12"}},
    {"id": "N6", "published": "2026-10-12",
     "text": "[가상 기사] 완성차 A사가 11월 추가 감산을 발표했다.",
     "gold": {"end_customer": "E_A", "direction": "하향", "effect_from": "2026-11"}},
    {"id": "N7", "published": "2026-10-05",
     "text": "[가상 기사] IT기기 E사는 신형 노트북을 내년 1월 출시하며, 초기 물량 생산을 11월부터 시작한다.",
     "gold": {"end_customer": "E_E", "direction": "상향", "effect_from": "2026-11"}},
    {"id": "N8", "published": "2026-10-04",
     "text": "[가상 기사] 의료기기 F사는 국내 신규 공장을 2027년 하반기에 가동할 계획이다.",
     "gold": {"end_customer": "E_F", "direction": "상향", "effect_from": "2027-07"}},
    {"id": "N9", "published": "2026-10-06",
     "text": "[가상 기사] 가전 D사가 연말 할인 행사를 준비하고 있으며 구체적인 일정은 공개하지 않았다.",
     "gold": {"end_customer": "E_D", "direction": "상향", "effect_from": None}},
    {"id": "N10", "published": "2026-10-06",
     "text": "[가상 기사] 자동차 업계 전반에 하반기 판매 둔화 우려가 커지고 있다.",
     "gold": {"end_customer": "UNKNOWN", "direction": "하향", "effect_from": None}},
]

# ---------------------------------------------------------------------------
# 영업 담당자 답변(가상, 자유 문장). 에이전트가 해당 품목에 질문을 보냈을 때만 도착한다.
# None은 기한까지 답이 없는 경우.
# ---------------------------------------------------------------------------
REPLIES = {
    "D01-AB-G10": {"text": "A사 감산 맞습니다. 11월부터 해당 차종 라인 15% 감산이고 저희 AB-G10이 들어가는 범퍼 부품도 포함됩니다.",
                   "gold": {"status": "답변완료", "end_customer": "E_A", "fact_type": "감산", "rate_pct": -15, "start_month": "2026-11"}},
    "D07-CP-R60": {"text": "감산 차종에는 일부만 들어갑니다. 고객 확인 결과 11월부터 저희 물량이 8% 줄어듭니다.",
                   "gold": {"status": "답변완료", "end_customer": "E_A", "fact_type": "감산", "rate_pct": -8, "start_month": "2026-11"}},
    "D02-AB-G10": {"text": "B사 12월 특근 차종에는 저희 그레이드가 안 들어가서 물량 변화는 없습니다. 고객 FCST에는 연말 재고 확충분이 섞여 있어 실제 주문은 평소 수준일 겁니다.",
                   "gold": {"status": "답변완료", "end_customer": "E_B", "fact_type": "변화없음", "rate_pct": None, "start_month": None}},
    "D06-PC-F40": {"text": "E사 신형 노트북 하우징에 PC-F40이 들어갑니다. 11월부터 월 10% 정도 늘 것으로 봅니다.",
                   "gold": {"status": "답변완료", "end_customer": "E_E", "fact_type": "증산", "rate_pct": 10, "start_month": "2026-11"}},
    "D09-AB-G10": {"text": "원료가 인상 예고가 있어서 12월과 1월 물량 일부를 11월로 앞당겨 주문했습니다. 대략 60톤이고, 12월에 40톤, 1월에 20톤 줄어들 예정입니다.",
                   "gold": {"status": "답변완료", "end_customer": None, "fact_type": "선구매", "rate_pct": None, "start_month": "2026-12",
                            "reduce": {"2026-12": 40, "2027-01": 20}}},
    "D05-PC-N30": {"text": "D05의 PC-N30은 의료기기 F사 수액세트 하우징용입니다. 물량 변화는 따로 들은 게 없습니다.",
                   "gold": {"status": "답변완료", "end_customer": "E_F", "fact_type": "변화없음", "rate_pct": None, "start_month": None}},
    "D04-CP-M50": {"text": "이 물량은 가전 C사 세탁기 부품으로 나갑니다.",
                   "gold": {"status": "답변완료", "end_customer": "E_C", "fact_type": "없음", "rate_pct": None, "start_month": None}},
    "D03-PC-F40": {"text": "다음 주 고객 미팅 후에 다시 회신드리겠습니다.",
                   "gold": {"status": "보류", "end_customer": None, "fact_type": "없음", "rate_pct": None, "start_month": None}},
    "D08-AB-G10": {"text": "D사 판촉은 12월에 진행하는데, 판매 증가분은 완제품 재고로 대응해서 저희 주문 변화는 없다고 합니다.",
                   "gold": {"status": "답변완료", "end_customer": "E_D", "fact_type": "변화없음", "rate_pct": None, "start_month": None}},
    "D07-AB-H20": None,
}
