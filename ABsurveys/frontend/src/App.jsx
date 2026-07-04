import { useState, useEffect, useRef } from 'react';
import SurveyCard from './components/SurveyCard.jsx';
import Header from './components/Header.jsx';
import './App.css';

const API = '/api';

// Track pairing history per "scenario|img_type|question_id" key, so that
// Rule 6 (no repeat pairs) and Rule 7 (no repeat images, where possible)
// can be enforced across all occurrences of the same AB question for this user.
const pairHistoryByKey = {};

export default function App() {
  const [languages, setLanguages] = useState([]);
  const [currentLanguage, setCurrentLanguage] = useState('english');
  const [survey, setSurvey] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userId, setUserId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activePair, setActivePair] = useState(null); // pair for current AB group
  const [pairLoading, setPairLoading] = useState(false);
  const initDone = useRef(false);

  useEffect(() => {
    if (initDone.current) return;
    initDone.current = true;
    init();
  }, []);

  async function init() {
    setLoading(true);
    const [langs, uid] = await Promise.all([
      fetchLanguages(),
      createUser(),
    ]);
    setLanguages(langs);
    setUserId(uid);
    const s = await loadSurvey(currentLanguage);
    setSurvey(s);
    setLoading(false);
  }

  async function fetchLanguages() {
    try {
      const res = await fetch(`${API}/languages`);
      return await res.json();
    } catch { return []; }
  }

  async function createUser() {
    try {
      const res = await fetch(`${API}/new-user`, { method: 'POST' });
      const data = await res.json();
      return data.user_id;
    } catch { return Math.floor(Math.random() * 1_000_000); }
  }

  async function loadSurvey(lang) {
    try {
      const res = await fetch(`${API}/survey?language=${lang}`);
      return await res.json();
    } catch { return []; }
  }

  async function handleLanguageChange(lang) {
    setCurrentLanguage(lang);
    setCurrentIndex(0);
    setActivePair(null);
    setLoading(true);
    const s = await loadSurvey(lang);
    setSurvey(s);
    setLoading(false);
  }

  async function saveAnswer(question, pairId, value, pair) {
    try {
      await fetch(`${API}/save-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id:      userId,
          language:     currentLanguage,
          type:         question.type,
          user_options: question.user_options,
          scenario:     question.scenario,
          question_id:  question.question_id,
          pair_id:      pairId,
          img_id_A:     pair ? pair.img_id_A : null,
          img_id_B:     pair ? pair.img_id_B : null,
          img_type:     question.img_type || '',
          answer:       value,
          user_input:   value,
          question:     question.question,
          options:      question.options,
        }),
      });
    } catch {}
  }

  // Fetch a fresh AB pair for the current group
  async function fetchPairForGroup(group) {
    const firstQ = group.questions[0];
    if (!firstQ || String(firstQ.type).toUpperCase() !== 'AB') return null;

    const pairKey = `${firstQ.scenario}|${firstQ.img_type}|${firstQ.question_id}`;
    const history = pairHistoryByKey[pairKey];
    const prevPairIds = history ? history.lastIds : [];
    const seenImages = history ? Array.from(history.images) : [];
    const seenPairs = history ? history.pairs : [];
    const isNewSurvey = !history;

    setPairLoading(true);
    try {
      const res = await fetch(`${API}/next-pair`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario:      firstQ.scenario,
          img_type:      firstQ.img_type,
          question_id:   firstQ.question_id,
          prev_pair_ids: prevPairIds,
          seen_images:   seenImages,
          seen_pairs:    seenPairs,
          is_new_survey: isNewSurvey,
        }),
      });
      const { pair } = await res.json();
      if (pair) {
        const entry = pairHistoryByKey[pairKey] || { images: new Set(), pairs: [], lastIds: [] };
        entry.images.add(pair.img_id_A);
        entry.images.add(pair.img_id_B);
        entry.pairs.push([pair.img_id_A, pair.img_id_B]);
        entry.lastIds = [pair.img_id_A, pair.img_id_B];
        pairHistoryByKey[pairKey] = entry;
      }
      return pair || null;
    } catch { return null; }
    finally { setPairLoading(false); }
  }

  // Called by SurveyCard when ready to advance
  async function goNext() {
    const nextIndex = currentIndex + 1;
    setActivePair(null);
    setCurrentIndex(nextIndex);

    if (nextIndex < survey.length) {
      const nextGroup = survey[nextIndex];
      const firstQ = nextGroup?.questions?.[0];
      if (firstQ && String(firstQ.type).toUpperCase() === 'AB') {
        const pair = await fetchPairForGroup(nextGroup);
        setActivePair(pair);
      }
    }
  }

  // On first load of survey, fetch pair for first AB group
  useEffect(() => {
    if (!survey.length) return;
    const group = survey[0];
    const firstQ = group?.questions?.[0];
    if (firstQ && String(firstQ.type).toUpperCase() === 'AB') {
      fetchPairForGroup(group).then(p => {
        setActivePair(p);
      });
    }
  }, [survey]);

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        <p className="loading-text">Loading</p>
      </div>
    );
  }

  const isDone = currentIndex >= survey.length;

  return (
    <div className="app">
      <Header
        languages={languages}
        currentLanguage={currentLanguage}
        onLanguageChange={handleLanguageChange}
      />
      <main className="app-main">
        {isDone ? (
          <div className="done-card">
            <div className="done-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 6L9 17l-5-5" />
              </svg>
            </div>
            <h2>Thank you for participating!</h2>
            <p>Your responses have been saved.</p>
          </div>
        ) : (
          <SurveyCard
            key={currentIndex}
            group={survey[currentIndex]}
            pair={activePair}
            pairLoading={pairLoading}
            onSaveAnswer={saveAnswer}
            onNext={goNext}
          />
        )}
      </main>
    </div>
  );
}
