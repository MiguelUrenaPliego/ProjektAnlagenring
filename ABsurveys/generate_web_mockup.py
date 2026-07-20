# coding: utf-8
"""
generate_web_mockup.py

Builds ABsurveys/web.html: a static, frontend-only mockup of the AB survey.
It mirrors the real React/FastAPI app's flow (backend/server.py, frontend/src/*)
but runs entirely client-side with data baked into the page as JSON, with no
network calls, no backend, and no answers ever persisted anywhere.
Intended for hosting the mockup alongside map.html on a static file host.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
REAL_SURVEY_URL = "https://cs-futurecities.media.mit.edu/"


def parse_list(value: Any) -> list[str]:
    """Same semantics as backend/utils.py:parse_list."""
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            return [x.strip().strip('"').strip("'") for x in ast.literal_eval(s) if x.strip()]
        except (ValueError, SyntaxError):
            return [x.strip().strip('"').strip("'") for x in s[1:-1].split(",") if x.strip()]
    return [s]


def read_csv(path: str) -> list[dict[str, str]]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_config() -> dict:
    with open(os.path.join(ROOT_PATH, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def build_data() -> dict:
    config = load_config()
    scenario = config["scenario"]
    scenario_list = scenario if isinstance(scenario, list) else [scenario]

    scenarios_rows = read_csv(os.path.join(ROOT_PATH, config["scenarios_path"]))
    questions_rows = read_csv(os.path.join(ROOT_PATH, config["questions_path"]))
    languages_rows = read_csv(os.path.join(ROOT_PATH, config["languages_path"]))
    images_rows = read_csv(os.path.join(ROOT_PATH, config["images_path"]))
    images_dir = os.path.dirname(config["images_path"])

    # question_id -> {question_type, autocontinue}
    question_meta = {
        row["question_id"]: {
            "type": row.get("question_type", ""),
            "autocontinue": str(row.get("autocontinue", "")).strip().lower() == "true",
        }
        for row in questions_rows
    }

    # language -> {question_id -> {question, options, user_options}}
    translations: dict[str, dict[str, dict]] = {}
    languages: list[dict] = []
    for lang_row in languages_rows:
        lang = lang_row["language"]
        languages.append({"language": lang, "symbol": lang_row["symbol"]})
        lang_csv_path = os.path.join(ROOT_PATH, "questions", lang_row["file"])
        lang_rows = read_csv(lang_csv_path)
        translations[lang] = {
            row["question_id"]: {
                "question": row.get("question", ""),
                "options": parse_list(row.get("options", "")),
                "user_options": parse_list(row.get("user_options", "")),
            }
            for row in lang_rows
        }

    # img_type -> [{img_id, path, incompatible_ids}]
    images_by_type: dict[str, list[dict]] = {}
    for row in images_rows:
        scen = row.get("scenario", "")
        if scen not in scenario_list:
            continue
        img_type = row.get("img_type", "")
        rel_path = "/".join([images_dir, row["path"]]).replace("\\", "/")
        images_by_type.setdefault(img_type, []).append({
            "img_id": row["img_id"],
            "path": rel_path,
            "incompatible_ids": parse_list(row.get("incompatible_ids", "")),
        })

    # Survey groups, mirroring backend/server.py:get_survey (one row -> one group)
    groups: list[dict] = []
    for row in scenarios_rows:
        if row.get("scenario") not in scenario_list:
            continue
        qids = parse_list(row.get("question_id", ""))
        img_types = parse_list(row.get("img_type", ""))
        questions = []
        for qid in qids:
            meta = question_meta.get(qid, {"type": "", "autocontinue": False})
            questions.append({
                "question_id": qid,
                "type": meta["type"],
                "autocontinue": meta["autocontinue"],
            })
        groups.append({
            "img_types": img_types,  # resolved randomly client-side, like the real app
            "questions": questions,
        })

    return {
        "languages": languages,
        "defaultLanguage": languages[0]["language"] if languages else "english",
        "translations": translations,
        "imagesByType": images_by_type,
        "groups": groups,
        "realSurveyUrl": REAL_SURVEY_URL,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Survey Mockup</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f9fafb;
  --bg-card: #f3f4f6;
  --border-subtle: #e5e7eb;
  --border-accent: #c7d2fe;
  --text-primary: #111827;
  --text-secondary: #4b5563;
  --text-muted: #9ca3af;
  --accent-primary: #4f46e5;
  --accent-success: #10b981;
  --shadow-soft: 0 4px 24px rgba(0, 0, 0, 0.08);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
  --radius-xl: 28px;
  --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
  --transition-smooth: 300ms cubic-bezier(0.4, 0, 0.2, 1);
  font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--text-primary);
  background: var(--bg-primary);
  background-image:
    radial-gradient(ellipse at 20% 0%, rgba(79, 70, 229, 0.05) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 100%, rgba(99, 102, 241, 0.04) 0%, transparent 50%);
}
h1, h2, h3 { margin: 0; }
button { font-family: inherit; cursor: pointer; }

.demo-banner {
  background: #fffbeb;
  border-bottom: 1px solid #fde68a;
  color: #92400e;
  padding: 10px 20px;
  font-size: 0.9rem;
  text-align: center;
  line-height: 1.5;
}
.demo-banner a { color: #92400e; font-weight: 700; }
.demo-banner strong { font-weight: 700; }

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.25rem 2.5rem;
  border-bottom: 1px solid var(--border-subtle);
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(20px);
  position: sticky;
  top: 0;
  z-index: 50;
}
.brand-name { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.02em; }
.brand-name span:nth-child(2) { color: var(--accent-primary); font-weight: 700; }
.header-right { display: flex; align-items: center; gap: 1rem; }
.home-link {
  text-decoration: none;
  color: var(--text-secondary);
  font-weight: 600;
  font-size: 0.9rem;
  padding: 0.5rem 0.9rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
}
.home-link:hover { border-color: var(--accent-primary); color: var(--accent-primary); }

.lang-switcher { position: relative; }
.lang-btn {
  display: flex; align-items: center; gap: 0.4rem;
  padding: 0.5rem 1rem; background: var(--bg-card);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-md);
  font-size: 0.875rem; font-weight: 600; color: var(--text-primary);
}
.lang-dropdown {
  position: absolute; top: calc(100% + 8px); right: 0;
  background: white; border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md); box-shadow: 0 10px 25px rgba(0,0,0,0.1);
  min-width: 100px; overflow: hidden; z-index: 200; display: none;
}
.lang-dropdown.open { display: block; }
.lang-item {
  display: block; width: 100%; padding: 0.75rem 1.25rem; background: none;
  border: none; text-align: center; font-size: 0.9rem; font-weight: 500;
  color: var(--text-primary);
}
.lang-item:hover { background: var(--bg-secondary); }
.lang-item.active { color: var(--accent-primary); font-weight: 700; }

.app-main {
  min-height: calc(100vh - 110px);
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 2rem;
}

.survey-card {
  width: 100%; max-width: 1100px; background: white;
  border: 1px solid var(--border-subtle); border-radius: var(--radius-xl);
  box-shadow: var(--shadow-soft); padding: 2.5rem;
  animation: fadeUp 0.4s ease-out;
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

.question-block { display: flex; flex-direction: column; gap: 1.5rem; }
.question-title { font-size: 1.75rem; font-weight: 700; text-align: center; line-height: 1.3; }
.question-input {
  display: block; width: 100%; max-width: 600px; margin: 0 auto;
  padding: 1rem 1.25rem; border: 2px solid var(--border-subtle);
  border-radius: var(--radius-md); font-size: 1.1rem; font-family: inherit;
  outline: none;
}
.question-input:focus { border-color: var(--accent-primary); box-shadow: 0 0 0 4px rgba(79,70,229,0.1); }
.question-options { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; max-width: 900px; margin: 0 auto; }
.option-btn {
  padding: 0.75rem 1.5rem; border: 2px solid var(--border-subtle);
  border-radius: var(--radius-md); background: white; color: var(--text-primary);
  font-size: 1rem; font-weight: 600; font-family: inherit; user-select: none;
  transition: all var(--transition-fast);
}
.option-btn:hover { border-color: var(--accent-primary); background: rgba(79,70,229,0.04); transform: translateY(-2px); }
.option-btn.selected { border-color: var(--accent-success); background: rgba(16,185,129,0.07); color: #065f46; transform: scale(1.03); }
.continue-btn {
  padding: 0.9rem 3rem; background: var(--text-primary); color: white;
  border: none; border-radius: var(--radius-md); font-size: 1rem;
  font-weight: 700; letter-spacing: 0.02em; transition: all var(--transition-smooth);
}
.continue-btn:hover:not(:disabled) { background: #1f2937; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.2); }
.continue-btn:disabled { background: var(--bg-card); color: var(--text-muted); cursor: not-allowed; }
.simple-questions { display: flex; flex-direction: column; gap: 2.5rem; }
.simple-footer { display: flex; justify-content: center; padding-top: 0.5rem; }

.ab-group { display: flex; flex-direction: column; gap: 2rem; }
.ab-title { font-size: 2rem; font-weight: 700; text-align: center; line-height: 1.25; }
.ab-grid { display: grid; grid-template-columns: 1fr 80px 1fr; gap: 12px; align-items: center; }
.ab-image-card {
  position: relative; width: 100%; background: none; border: 3px solid transparent;
  border-radius: var(--radius-lg); overflow: hidden; padding: 0; display: block;
  transition: all var(--transition-smooth);
}
.ab-image-card img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; border-radius: calc(var(--radius-lg) - 3px); }
.ab-image-card:hover:not(:disabled) { border-color: var(--border-accent); transform: scale(1.02); box-shadow: 0 8px 24px rgba(0,0,0,0.1); }
.ab-image-card.selected { border-color: var(--accent-success); box-shadow: 0 0 0 4px rgba(16,185,129,0.2); transform: scale(1.02); }
.ab-image-card.dimmed { opacity: 0.45; transform: scale(0.98); }
.ab-selected-badge {
  position: absolute; top: 12px; right: 12px; width: 36px; height: 36px;
  background: var(--accent-success); border-radius: 50%; display: flex;
  align-items: center; justify-content: center; color: white; font-weight: 900;
}
.ab-center { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; }
.ab-arrow-btn { background: none; border: none; color: var(--accent-primary); font-size: 2.75rem; font-weight: 900; line-height: 1; padding: 4px; }
.ab-arrow-btn:hover:not(:disabled) { transform: scale(1.3); color: #4338ca; }
.ab-arrow-btn.chosen { color: var(--accent-success); transform: scale(1.15); }
.ab-equal-btn {
  background: none; border: 2px solid var(--border-subtle); border-radius: 50%;
  width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; font-weight: 900; color: var(--text-secondary);
}
.ab-equal-btn:hover:not(:disabled) { border-color: #f59e0b; color: #f59e0b; transform: scale(1.1); }
.ab-equal-btn.chosen { border-color: var(--accent-success); color: var(--accent-success); }

.done-card { text-align: center; padding: 3rem 4rem; }
.done-icon {
  display: inline-flex; align-items: center; justify-content: center; width: 72px; height: 72px;
  background: linear-gradient(135deg, #10b981, #059669); border-radius: 50%; margin-bottom: 1.5rem;
  color: white; font-size: 2rem; font-weight: 900;
}
.done-card h2 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.75rem; }
.done-card p { color: var(--text-secondary); font-size: 1rem; line-height: 1.6; margin: 0.35rem 0; }
.done-card .real-link {
  display: inline-block; margin-top: 1.5rem; padding: 0.9rem 2rem;
  background: var(--accent-primary); color: white; text-decoration: none;
  border-radius: var(--radius-md); font-weight: 700;
}
.done-card .real-link:hover { background: #4338ca; }
.progress { text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem; }

@media (max-width: 800px) {
  .ab-grid { grid-template-columns: 1fr; gap: 1.5rem; }
  .ab-center { flex-direction: row; justify-content: center; gap: 2rem; order: 2; }
  .ab-title { font-size: 1.5rem; }
}
@media (max-width: 700px) {
  .survey-card { padding: 1.5rem 1rem; }
}
</style>
</head>
<body>

<div class="demo-banner">
  <strong data-i18n-shell="demo_label">SURVEY MOCKUP.</strong>
  <span data-i18n-shell="demo_text">This is a frontend mockup. No answers are saved.</span>
  <span data-i18n-shell="demo_cta">The real survey is hosted at</span>
  <a href="__REAL_SURVEY_URL__" target="_blank" rel="noopener">__REAL_SURVEY_URL__</a>
</div>

<header class="header">
  <div class="brand-name"><span>Survey</span><span>AB</span> <span style="color:#9ca3af;font-weight:600;font-size:0.85rem;">(mockup)</span></div>
  <div class="header-right">
    <a class="home-link" href="index.html" data-i18n-shell="back_home">&larr; Home</a>
    <div class="lang-switcher">
      <button class="lang-btn" id="langBtn">EN</button>
      <div class="lang-dropdown" id="langDropdown"></div>
    </div>
  </div>
</header>

<main class="app-main" id="appMain"></main>

<script id="survey-data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('survey-data').textContent);

const SHELL_TEXT = {
  english: {
    demo_label: "SURVEY MOCKUP.",
    demo_text: "This is a frontend mockup. No answers are saved.",
    demo_cta: "The real survey is hosted at",
    back_home: "\\u2190 Home",
    loading: "Loading images\\u2026",
    thanks_title: "Thank you for trying the mockup!",
    thanks_text: "This was a survey mockup. Your responses were not saved anywhere.",
    thanks_cta: "Take the real survey",
    continue: "Continue",
  },
  german: {
    demo_label: "UMFRAGE-MOCKUP.",
    demo_text: "Dies ist ein reines Frontend-Mockup. Es werden keine Antworten gespeichert.",
    demo_cta: "Die echte Umfrage finden Sie unter",
    back_home: "\\u2190 Startseite",
    loading: "Bilder werden geladen\\u2026",
    thanks_title: "Danke, dass Sie das Mockup ausprobiert haben!",
    thanks_text: "Dies war ein Umfrage-Mockup. Ihre Antworten wurden nirgendwo gespeichert.",
    thanks_cta: "Zur echten Umfrage",
    continue: "Weiter",
  },
};

let currentLanguage = DATA.defaultLanguage;
let currentIndex = 0;
let activePair = null;
const pairHistoryByKey = {}; // key -> { images:Set, pairs:[[a,b]] }

function t(key) {
  const dict = SHELL_TEXT[currentLanguage] || SHELL_TEXT.english;
  return dict[key] !== undefined ? dict[key] : (SHELL_TEXT.english[key] || key);
}

function applyShellText() {
  document.querySelectorAll('[data-i18n-shell]').forEach(el => {
    el.textContent = t(el.getAttribute('data-i18n-shell'));
  });
}

function setupLanguageSwitcher() {
  const btn = document.getElementById('langBtn');
  const dropdown = document.getElementById('langDropdown');
  const current = DATA.languages.find(l => l.language === currentLanguage);
  btn.textContent = current ? current.symbol : currentLanguage.toUpperCase().slice(0, 2);

  dropdown.innerHTML = '';
  DATA.languages.forEach(lang => {
    const item = document.createElement('button');
    item.className = 'lang-item' + (lang.language === currentLanguage ? ' active' : '');
    item.textContent = lang.symbol;
    item.onclick = () => {
      currentLanguage = lang.language;
      dropdown.classList.remove('open');
      applyShellText();
      setupLanguageSwitcher();
      renderCurrentGroup();
    };
    dropdown.appendChild(item);
  });

  btn.onclick = () => dropdown.classList.toggle('open');
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.lang-switcher')) dropdown.classList.remove('open');
  });
}

function questionText(qid) {
  const trans = (DATA.translations[currentLanguage] || {})[qid];
  return trans || { question: qid, options: [], user_options: [] };
}

// Resolve one img_type for a group (mirrors backend/server.py random.choice)
function resolveImgType(group) {
  if (!group.img_types || group.img_types.length === 0) return '';
  return group.img_types[Math.floor(Math.random() * group.img_types.length)];
}

function incompatibleSet(pool, anchorId) {
  const anchor = pool.find(p => p.img_id === anchorId);
  return new Set(anchor ? anchor.incompatible_ids : []);
}

// Lightweight client-side mirror of backend/pairing.py: prefers unseen images,
// avoids incompatible pairs and pairs already shown this session, else relaxes.
function pickPair(imgType, questionId) {
  const pool = DATA.imagesByType[imgType] || [];
  if (pool.length < 2) return null;

  const key = imgType + '|' + questionId;
  const history = pairHistoryByKey[key] || { images: new Set(), pairs: [] };
  pairHistoryByKey[key] = history;

  let candidatesA = pool.filter(p => !history.images.has(p.img_id));
  if (candidatesA.length === 0) candidatesA = pool;
  const a = candidatesA[Math.floor(Math.random() * candidatesA.length)];

  const incompat = incompatibleSet(pool, a.img_id);
  const isRepeatPair = (bId) => history.pairs.some(([x, y]) =>
    (x === a.img_id && y === bId) || (x === bId && y === a.img_id));

  let candidatesB = pool.filter(p =>
    p.img_id !== a.img_id && !incompat.has(p.img_id) && !isRepeatPair(p.img_id));
  let unseenB = candidatesB.filter(p => !history.images.has(p.img_id));
  if (unseenB.length > 0) candidatesB = unseenB;
  if (candidatesB.length === 0) {
    candidatesB = pool.filter(p => p.img_id !== a.img_id);
  }
  const b = candidatesB[Math.floor(Math.random() * candidatesB.length)];

  history.images.add(a.img_id);
  history.images.add(b.img_id);
  history.pairs.push([a.img_id, b.img_id]);

  const swap = Math.random() < 0.5;
  const left = swap ? b : a;
  const right = swap ? a : b;
  return { A: left.path, B: right.path, img_id_A: left.img_id, img_id_B: right.img_id };
}

function isDone() { return currentIndex >= DATA.groups.length; }

function goNext() {
  currentIndex += 1;
  activePair = null;
  renderCurrentGroup();
}

function renderCurrentGroup() {
  const main = document.getElementById('appMain');
  main.innerHTML = '';

  if (isDone()) {
    main.appendChild(renderDone());
    return;
  }

  const progress = document.createElement('div');
  progress.className = 'progress';
  progress.textContent = `${currentIndex + 1} / ${DATA.groups.length}`;
  main.appendChild(progress);

  const card = document.createElement('div');
  card.className = 'survey-card';
  main.appendChild(card);

  const group = DATA.groups[currentIndex];
  const firstQ = group.questions[0];
  const isAB = firstQ && String(firstQ.type).toUpperCase() === 'AB';

  if (isAB) {
    renderABGroup(card, group);
  } else {
    renderSimpleGroup(card, group);
  }
}

function renderDone() {
  const wrap = document.createElement('div');
  wrap.className = 'done-card';
  wrap.innerHTML = `
    <div class="done-icon">&#10003;</div>
    <h2>${t('thanks_title')}</h2>
    <p>${t('thanks_text')}</p>
    <a class="real-link" href="${DATA.realSurveyUrl}" target="_blank" rel="noopener">${t('thanks_cta')} &rarr;</a>
  `;
  return wrap;
}

function renderABGroup(card, group) {
  const imgType = resolveImgType(group);
  const firstQ = group.questions[0];
  const pair = pickPair(imgType, firstQ.question_id);

  if (!pair) {
    card.innerHTML = `<div class="ab-loading">${t('loading')}</div>`;
    setTimeout(goNext, 400);
    return;
  }

  const qText = questionText(firstQ.question_id);
  let selected = null;

  const wrap = document.createElement('div');
  wrap.className = 'ab-group';
  wrap.innerHTML = `
    <h2 class="ab-title">${qText.question}</h2>
    <div class="ab-grid">
      <div class="ab-image-wrapper">
        <button class="ab-image-card" data-choice="A"><img src="${pair.A}" alt="Option A" /></button>
      </div>
      <div class="ab-center">
        <button class="ab-arrow-btn" data-choice="A" title="Choose left">&larr;</button>
        <button class="ab-equal-btn" data-choice="=" title="Equal">=</button>
        <button class="ab-arrow-btn" data-choice="B" title="Choose right">&rarr;</button>
      </div>
      <div class="ab-image-wrapper">
        <button class="ab-image-card" data-choice="B"><img src="${pair.B}" alt="Option B" /></button>
      </div>
    </div>
  `;
  card.appendChild(wrap);

  const buttons = wrap.querySelectorAll('[data-choice]');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      if (selected !== null) return;
      selected = btn.getAttribute('data-choice');

      const cardA = wrap.querySelector('.ab-image-card[data-choice="A"]');
      const cardB = wrap.querySelector('.ab-image-card[data-choice="B"]');
      const arrowA = wrap.querySelector('.ab-arrow-btn[data-choice="A"]');
      const arrowB = wrap.querySelector('.ab-arrow-btn[data-choice="B"]');
      const eqBtn = wrap.querySelector('.ab-equal-btn');

      if (selected === 'A') { cardA.classList.add('selected'); cardB.classList.add('dimmed'); arrowA.classList.add('chosen'); }
      else if (selected === 'B') { cardB.classList.add('selected'); cardA.classList.add('dimmed'); arrowB.classList.add('chosen'); }
      else { eqBtn.classList.add('chosen'); }

      buttons.forEach(b => b.disabled = true);
      setTimeout(goNext, 350);
    });
  });
}

function renderSimpleGroup(card, group) {
  const answers = {};
  const hasManualContinue = group.questions.some(q => q.autocontinue === false);

  const container = document.createElement('div');
  container.className = 'simple-questions';
  card.appendChild(container);

  function allAnswered() {
    return group.questions.every(q => answers[q.question_id] !== undefined && answers[q.question_id] !== '');
  }

  function maybeAutoAdvance() {
    if (!hasManualContinue && allAnswered()) goNext();
    else updateContinueState();
  }

  function updateContinueState() {
    const btn = container.querySelector('.continue-btn');
    if (btn) btn.disabled = !allAnswered();
  }

  group.questions.forEach(q => {
    const qText = questionText(q.question_id);
    const block = document.createElement('div');
    block.className = 'question-block';

    const title = document.createElement('h2');
    title.className = 'question-title';
    title.textContent = qText.question;
    block.appendChild(title);

    if (q.type === 'int') {
      const input = document.createElement('input');
      input.className = 'question-input';
      input.type = 'text';
      input.inputMode = 'numeric';
      input.placeholder = 'Enter a number';
      input.addEventListener('input', () => {
        input.value = input.value.replace(/\\D/g, '');
        answers[q.question_id] = input.value;
        maybeAutoAdvance();
      });
      block.appendChild(input);
    } else {
      const displayOptions = qText.user_options.length > 0 ? qText.user_options : qText.options;
      const optsWrap = document.createElement('div');
      optsWrap.className = 'question-options';
      displayOptions.forEach((label, i) => {
        const value = qText.options[i] !== undefined ? qText.options[i] : label;
        const optBtn = document.createElement('button');
        optBtn.className = 'option-btn';
        optBtn.textContent = label;
        optBtn.addEventListener('click', () => {
          optsWrap.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
          optBtn.classList.add('selected');
          answers[q.question_id] = value;
          maybeAutoAdvance();
        });
        optsWrap.appendChild(optBtn);
      });
      block.appendChild(optsWrap);
    }

    container.appendChild(block);
  });

  if (hasManualContinue) {
    const footer = document.createElement('div');
    footer.className = 'simple-footer';
    const btn = document.createElement('button');
    btn.className = 'continue-btn';
    btn.textContent = t('continue');
    btn.disabled = true;
    btn.addEventListener('click', () => { if (allAnswered()) goNext(); });
    footer.appendChild(btn);
    container.appendChild(footer);
  }
}

applyShellText();
setupLanguageSwitcher();
renderCurrentGroup();
</script>

</body>
</html>
"""


def render_html(data: dict) -> str:
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__REAL_SURVEY_URL__", data["realSurveyUrl"])
    return html


def main() -> None:
    data = build_data()
    html = render_html(data)
    out_path = os.path.join(ROOT_PATH, "web.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote mockup survey to: {out_path}")
    print(f"Groups: {len(data['groups'])}, languages: {len(data['languages'])}, "
          f"image types: {list(data['imagesByType'].keys())}")


if __name__ == "__main__":
    main()
