import { useState } from 'react';
import './QuestionBlock.css';

export default function QuestionBlock({ question, currentAnswer, onAnswer }) {
  const { type, question: text, options = [], user_options = [] } = question;
  const [multiSelected, setMultiSelected] = useState([]);

  const displayOptions = user_options.length > 0 ? user_options : options;

  return (
    <div className="question-block">
      <h2 className="question-title">{text}</h2>

      {type === 'int' && (
        <input
          className="question-input"
          type="text"
          inputMode="numeric"
          placeholder="Enter a number"
          defaultValue={currentAnswer ?? ''}
          onChange={(e) => {
            const cleaned = e.target.value.replace(/\D/g, '');
            e.target.value = cleaned;
            onAnswer(cleaned);
          }}
        />
      )}

      {type === 'choice' && (
        <div className="question-options">
          {displayOptions.map((label, i) => {
            const value = options[i] ?? label;
            return (
              <button
                key={label}
                className={`option-btn ${currentAnswer === value ? 'selected' : ''}`}
                onClick={() => onAnswer(value)}
              >
                {label}
              </button>
            );
          })}
        </div>
      )}

      {type && type.startsWith('multichoice') && (
        <div className="question-options-multi">
          <div className="question-options">
            {displayOptions.map((label, i) => {
              const value = options[i] ?? label;
              const isSelected = multiSelected.includes(value);
              return (
                <button
                  key={label}
                  className={`option-btn ${isSelected ? 'selected' : ''}`}
                  onClick={() => {
                    setMultiSelected(prev =>
                      isSelected ? prev.filter(v => v !== value) : [...prev, value]
                    );
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>
          <div className="multi-footer">
            <button
              className="continue-btn"
              disabled={multiSelected.length === 0}
              onClick={() => {
                if (multiSelected.length > 0) onAnswer(multiSelected);
              }}
            >
              Continue
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
