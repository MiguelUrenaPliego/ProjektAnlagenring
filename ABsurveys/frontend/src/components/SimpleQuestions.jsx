import { useState } from 'react';
import QuestionBlock from './QuestionBlock.jsx';
import './SimpleQuestions.css';

export default function SimpleQuestions({ group, onSaveAnswer, onNext }) {
  const [answers, setAnswers] = useState({});
  const [navigating, setNavigating] = useState(false);
  const questions = group.questions;

  const hasManualContinue = questions.some(
    q => String(q.autocontinue).toLowerCase() === 'false'
  );

  function allAnswered() {
    return questions.every(
      q => answers[q.question_id] !== undefined && answers[q.question_id] !== ''
    );
  }

  async function handleAnswer(question, value) {
    const updated = { ...answers, [question.question_id]: value };
    setAnswers(updated);
    await onSaveAnswer(question, '', value, null);

    if (!hasManualContinue) {
      const allDone = questions.every(
        q => updated[q.question_id] !== undefined && updated[q.question_id] !== ''
      );
      if (allDone && !navigating) {
        setNavigating(true);
        onNext();
      }
    }
  }

  function handleContinue() {
    if (!allAnswered() || navigating) return;
    setNavigating(true);
    onNext();
  }

  return (
    <div className="simple-questions">
      {questions.map(q => (
        <QuestionBlock
          key={q.question_id}
          question={q}
          currentAnswer={answers[q.question_id]}
          onAnswer={(val) => handleAnswer(q, val)}
        />
      ))}

      {hasManualContinue && (
        <div className="simple-footer">
          <button
            className="continue-btn"
            disabled={!allAnswered() || navigating}
            onClick={handleContinue}
          >
            Continue
          </button>
        </div>
      )}
    </div>
  );
}
