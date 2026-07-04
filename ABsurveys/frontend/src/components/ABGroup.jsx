import { useState } from 'react';
import QuestionBlock from './QuestionBlock.jsx';
import './ABGroup.css';

export default function ABGroup({ group, pair, pairLoading, onSaveAnswer, onNext }) {
  const [selected, setSelected] = useState(null); // 'A' | 'B' | '='
  const [followupAnswered, setFollowupAnswered] = useState(false);
  const questions = group.questions;
  const firstQ = questions[0];
  const followups = questions.slice(1);
  const hasFollowup = followups.length > 0;

  if (pairLoading || !pair) {
    return (
      <div className="ab-loading">
        <div className="loading-spinner-sm" />
        <span>Loading images…</span>
      </div>
    );
  }

  async function handleChoice(value) {
    if (selected !== null) return;
    setSelected(value);
    await onSaveAnswer(firstQ, `${pair.img_id_A}_${pair.img_id_B}`, value, pair);

    if (!hasFollowup) {
      onNext();
    }
  }

  async function handleFollowupAnswer(q, value) {
    await onSaveAnswer(q, pair.pair_id || '', value, null);
    setFollowupAnswered(true);
    onNext();
  }

  // Replace {AB_selection} placeholder in follow-up question text
  const selectionLabel = selected === 'A' ? 'Left' : selected === 'B' ? 'Right' : 'Equal';

  return (
    <div className="ab-group">
      {/* Question title */}
      <h2 className="ab-title">{firstQ.question}</h2>

      {/* Image comparison */}
      <div className="ab-grid">
        {/* Image A */}
        <div className="ab-image-wrapper">
          <button
            className={`ab-image-card ${selected === 'A' ? 'selected' : ''} ${selected && selected !== 'A' ? 'dimmed' : ''}`}
            onClick={() => handleChoice('A')}
            disabled={selected !== null}
          >
            <img src={pair.A} alt="Option A" />
            {selected === 'A' && (
              <div className="ab-selected-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
              </div>
            )}
          </button>
        </div>

        {/* Center controls */}
        <div className="ab-center">
          <button
            className={`ab-arrow-btn ${selected === 'A' ? 'chosen' : ''}`}
            onClick={() => handleChoice('A')}
            disabled={selected !== null}
            title="Choose left"
          >
            ←
          </button>
          <button
            className={`ab-equal-btn ${selected === '=' ? 'chosen' : ''}`}
            onClick={() => handleChoice('=')}
            disabled={selected !== null}
            title="Equal"
          >
            =
          </button>
          <button
            className={`ab-arrow-btn ${selected === 'B' ? 'chosen' : ''}`}
            onClick={() => handleChoice('B')}
            disabled={selected !== null}
            title="Choose right"
          >
            →
          </button>
        </div>

        {/* Image B */}
        <div className="ab-image-wrapper">
          <button
            className={`ab-image-card ${selected === 'B' ? 'selected' : ''} ${selected && selected !== 'B' ? 'dimmed' : ''}`}
            onClick={() => handleChoice('B')}
            disabled={selected !== null}
          >
            <img src={pair.B} alt="Option B" />
            {selected === 'B' && (
              <div className="ab-selected-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
              </div>
            )}
          </button>
        </div>
      </div>

      {/* Follow-up questions (shown after selection) */}
      {selected !== null && hasFollowup && !followupAnswered && (
        <div className="ab-followup">
          {followups.map(q => {
            const resolved = {
              ...q,
              question: q.question.replace('{AB_selection}', selectionLabel),
            };
            return (
              <QuestionBlock
                key={q.question_id}
                question={resolved}
                onAnswer={(val) => handleFollowupAnswer(resolved, val)}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
