import ABGroup from './ABGroup.jsx';
import SimpleQuestions from './SimpleQuestions.jsx';
import './SurveyCard.css';

export default function SurveyCard({ group, pair, pairLoading, onSaveAnswer, onNext }) {
  if (!group) return null;

  const firstQ = group.questions[0];
  const isAB = firstQ && String(firstQ.type).toUpperCase() === 'AB';

  return (
    <div className="survey-card">
      {isAB ? (
        <ABGroup
          group={group}
          pair={pair}
          pairLoading={pairLoading}
          onSaveAnswer={onSaveAnswer}
          onNext={onNext}
        />
      ) : (
        <SimpleQuestions
          group={group}
          onSaveAnswer={onSaveAnswer}
          onNext={onNext}
        />
      )}
    </div>
  );
}
