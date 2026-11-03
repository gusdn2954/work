# -*- coding: utf-8 -*-
"""최종 검토표를 담당자가 보고 승인, 조정, 보류를 입력하는 단일 HTML 화면으로 만든다 (서버 불필요, CSV 내보내기)"""
import json
import html as _html

import pandas as pd

TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>월간 수요검토 화면</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1b1b1a;--muted:#6d6c66;--line:#e4e3df;--acc:#2a78d6;--good:#1b8a5a;--warn:#c27a00;--bad:#c2410c;--chip:#eef3fb}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--muted:#a3a29b;--line:#34332f;--acc:#6aa5ee;--good:#4cc38a;--warn:#e0a83a;--bad:#f07a4a;--chip:#22324a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 'Noto Sans KR','Apple SD Gothic Neo',sans-serif}
header{padding:16px;max-width:1280px;margin:auto}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:12px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}.kpi b{font-size:20px;display:block}.kpi span{color:var(--muted);font-size:12px}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:5px 10px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc);font-weight:700}
main{max-width:1280px;margin:auto;padding:0 16px 32px}
.row{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:8px 0;display:grid;grid-template-columns:220px 1fr 230px;gap:12px}
@media (max-width:860px){.row{grid-template-columns:1fr}}
.badge{display:inline-block;font-size:12px;border-radius:999px;padding:1px 8px;background:var(--chip)}
.s-검토{color:var(--bad)}.s-확인{color:var(--good)}.s-확정{color:var(--warn)}.s-정상{color:var(--muted)}
.nums{display:grid;grid-template-columns:repeat(4,auto);gap:2px 12px;font-variant-numeric:tabular-nums}.nums i{color:var(--muted);font-style:normal;font-size:12px}
.note{color:var(--muted);font-size:12.5px;margin-top:4px}.prop{font-weight:700;color:var(--acc)}
.act input{width:90px;font:inherit;padding:4px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.act .done{font-size:12px;color:var(--good);margin-top:4px}
</style></head><body>
<header><h1>월간 수요검토 화면 (가상 데이터)</h1>
<div class="sub">검토 기준월 __ORIGIN__. 제안값은 AI 보정 FCST에 담당자가 답변으로 확인한 사실만 반영한 값입니다. 승인값은 사람만 입력합니다.</div>
<div class="kpis" id="kpis"></div>
<div class="bar" id="filters"></div>
<div class="bar"><button id="csv">승인 결과 CSV로 내보내기</button><span class="sub" id="saved"></span></div></header>
<main id="list"></main>
<script>
const ROWS = __DATA__;
const KEY = 'demand-review-decisions';
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { dec = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch (e) {} ; draw(); };
const ORDER = ['검토', '확인 완료(승인만)', '확정 구간', '정상'];
let filt = '검토';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function kpis(){ const c = s => ROWS.filter(r => r.상태 === s).length;
  const done = Object.keys(dec).length;
  document.getElementById('kpis').innerHTML = [
    ['검토 단위', ROWS.length], ['직접 판단', c('검토')], ['승인만', c('확인 완료(승인만)')], ['확정 구간 알림', c('확정 구간')], ['입력 완료', done]
  ].map(([a,b]) => `<div class="kpi"><b>${b}</b><span>${a}</span></div>`).join(''); }
function filters(){ document.getElementById('filters').innerHTML = ['전체', ...ORDER].map(f =>
  `<button class="${f===filt?'on':''}" data-f="${f}">${f} (${f==='전체'?ROWS.length:ROWS.filter(r=>r.상태===f).length})</button>`).join('');
  document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { filt = b.dataset.f; draw(); }); }
function draw(){ kpis(); filters();
  const rows = ROWS.filter(r => filt==='전체' || r.상태===filt);
  document.getElementById('list').innerHTML = rows.map(r => { const k = r.품목 + '|' + r.납기월; const d = dec[k];
    const cls = r.상태.startsWith('확인') ? '확인' : r.상태.startsWith('확정') ? '확정' : r.상태;
    return `<div class="row"><div><b>${esc(r.품목)}</b> <span class="badge s-${cls}">${esc(r.상태)}</span><div class="note">납기 ${esc(r.납기월)}, 판단 기한 ${esc(r.판단기한)}<br>${esc(r.패턴)} 패턴, ${esc(r.FCST신뢰도이력)}</div></div>
    <div><div class="nums"><i>고객 FCST</i><i>독립 예측 (80% 범위)</i><i>AI 보정 FCST</i><i>제안값</i>
      <span>${r.고객FCST}</span><span>${r.독립예측} (${r.구간하단}~${r.구간상단})</span><span>${r.AI보정FCST}</span><span class="prop">${r.에이전트제안값}</span></div>
      <div class="note">계산: ${esc(r.제안근거)}</div><div class="note">사유: ${esc(r.검토사유)}</div>
      ${r.모델설명 && r.모델설명 !== '-' ? `<div class="note">모델 설명: ${esc(r.모델설명)}</div>` : ''}
      ${r.질문답변 && r.질문답변 !== '-' ? `<div class="note">${esc(r.질문답변)}</div>` : ''}</div>
    <div class="act"><button data-a="승인" data-k="${esc(k)}">승인</button> <button data-a="보류" data-k="${esc(k)}">보류</button><br>
      <input type="number" step="0.1" placeholder="조정값" data-in="${esc(k)}"> <button data-a="조정" data-k="${esc(k)}">조정</button>
      ${d ? `<div class="done">${esc(d.판단)}${d.승인값!==''?' '+d.승인값+'톤':''}</div>` : ''}</div></div>`; }).join('') || '<p class="sub">해당 품목이 없습니다.</p>';
  document.querySelectorAll('[data-a]').forEach(b => b.onclick = () => { const k = b.dataset.k, r = ROWS.find(x => x.품목+'|'+x.납기월===k);
    if (b.dataset.a === '조정') { const v = document.querySelector(`[data-in="${CSS.escape(k)}"]`).value; if (v === '') return; dec[k] = {판단:'조정', 승인값:Number(v)}; }
    else if (b.dataset.a === '승인') dec[k] = {판단:'승인', 승인값:r.에이전트제안값}; else dec[k] = {판단:'보류', 승인값:''};
    save(); }); }
document.getElementById('csv').onclick = () => {
  const head = ['품목','납기월','상태','고객FCST','독립예측','AI보정FCST','에이전트제안값','판단','승인값'];
  const lines = [head.join(',')].concat(ROWS.map(r => { const d = dec[r.품목+'|'+r.납기월] || {판단:'',승인값:''};
    return [r.품목,r.납기월,r.상태,r.고객FCST,r.독립예측,r.AI보정FCST,r.에이전트제안값,d.판단,d.승인값].join(','); }));
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(['﻿'+lines.join('\n')], {type:'text/csv'}));
  a.download = '승인결과.csv'; a.click(); };
draw();
</script></body></html>"""


def write_review_html(rev, origin, path):
    cols = ["품목", "납기월", "상태", "판단기한", "패턴", "고객FCST", "독립예측", "구간하단", "구간상단", "AI보정FCST",
            "에이전트제안값", "제안근거", "검토사유", "모델설명", "질문답변", "FCST신뢰도이력"]
    d = rev[cols].copy()
    d["판단기한"] = d["판단기한"].astype(str)
    data = json.dumps(d.to_dict("records"), ensure_ascii=False, default=str).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__DATA__", data).replace("__ORIGIN__", _html.escape(str(origin))))
