import React, { useState } from 'react';
import './App.css';

const App: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setAudioUrl(null);

    // TODO: Replace with actual API call
    setTimeout(() => {
      const fakeUrl = 'https://example.com/fake-music-file.mp3'; // Replace later
      setAudioUrl(fakeUrl);
      setLoading(false);
    }, 2000);
  };

  // const response = await fetch('https://your-backend.com/generate', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ prompt }),
  // });
  // const { audioUrl } = await response.json();
  // setAudioUrl(audioUrl);
  

  return (
    <div className="app-container">
      <h1 className="app-title">What Music Do You Want To Make?</h1>
      {/* <p className="app-subtitle">
        Create any music style effortlessly — your sound, your rules, your creativity.
      </p> */}
      <textarea
        className="app-textarea"
        placeholder="Enter your music prompt..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <button
        className="app-button"
        onClick={handleGenerate}
        disabled={loading || !prompt}
      >
        {loading ? 'Generating...' : 'Generate'}
      </button>

      {audioUrl && (
        <div className="app-audio-container">
          <audio controls src={audioUrl} className="app-audio" />
          <div className="app-download-link">
            <a href={audioUrl} download="generated-music.mp3">Download</a>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
