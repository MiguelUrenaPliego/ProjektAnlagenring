import { useState } from 'react';
import './Header.css';

export default function Header({ languages, currentLanguage, onLanguageChange }) {
  const [open, setOpen] = useState(false);
  const current = languages.find(l => l.language === currentLanguage);

  return (
    <header className="header">
      <div className="brand">
        <div className="brand-name">
          <span>Survey</span><span>AB</span>
        </div>
      </div>

      <div className="header-right">
        {languages.length > 0 && (
          <div className="lang-switcher">
            <button
              className="lang-btn"
              onClick={() => setOpen(o => !o)}
              aria-label="Select language"
            >
              {current?.symbol ?? currentLanguage.toUpperCase().slice(0, 2)}
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {open && (
              <div className="lang-dropdown">
                {languages.map(lang => (
                  <button
                    key={lang.language}
                    className={`lang-item ${lang.language === currentLanguage ? 'active' : ''}`}
                    onClick={() => {
                      onLanguageChange(lang.language);
                      setOpen(false);
                    }}
                  >
                    {lang.symbol}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
