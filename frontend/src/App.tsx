import React, { useRef, useState } from 'react';
import './App.css';

const App: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [midiUrl, setMidiUrl] = useState<string | null>(null);


  const handleGenerate = async () => {
    setLoading(true);
    setAudioUrl(null);
  
    const formData = new FormData();
    formData.append('prompt', prompt);
  
  //   const response = await fetch('http://localhost:8000/generate', {
  //     method: 'POST',
  //     body: formData,
  //   });
  
  //   if (response.ok) {
  //     const blob = await response.blob();
  //     const blobUrl = URL.createObjectURL(blob);
  //     setAudioUrl(blobUrl);
  //   } else {
  //     alert("Error generating music.");
  //   }
  
  //   setLoading(false);
  // };
  const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  body: formData,
});

if (response.ok) {
  const blob = await response.blob();
  const contentType = response.headers.get("Content-Type");

  if (contentType?.includes("audio/wav")) {
    const blobUrl = URL.createObjectURL(blob);
    setAudioUrl(blobUrl);
  } else if (contentType?.includes("audio/midi")) {
    const midiBlob = new Blob([blob], { type: "audio/midi" });
    const midiUrl = URL.createObjectURL(midiBlob);
    setMidiUrl(midiUrl);
  }

  setLoading(false);
} else {
  alert("Error generating music.");
  setLoading(false);
}}

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };
  
  const handleTimeUpdate = () => {
    const audio = audioRef.current;
    if (audio) setCurrentTime(audio.currentTime);
  };
  
  const handleLoadedMetadata = () => {
    const audio = audioRef.current;
    if (audio) setDuration(audio.duration);
  };
  
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    const time = parseFloat(e.target.value);
    if (audio) audio.currentTime = time;
    setCurrentTime(time);
  };
  
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };
  

  return (
    <div className="app-container">
      <h1 className="app-title">What Music Do You Want To Make?</h1>
      <textarea
        className="app-textarea"
        placeholder="Enter your music prompt..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <button
        type='button'
        className="app-button"
        onClick={handleGenerate}
        disabled={loading || !prompt}
      >
        {loading ? 'Generating...' : 'Generate'}
          </button>
    {audioUrl && (
      <div className="app-audio-container">
        <div className="audio-player">
          <button className="play-button" onClick={togglePlay}>
            {isPlaying ? (
              <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
                <rect x="6" y="5" width="4" height="14" rx="1" />
                <rect x="14" y="5" width="4" height="14" rx="1" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          <div className="timeline">
            <span className="time">{formatTime(currentTime)}</span>
            <input
              type="range"
              min={0}
              max={duration}
              step={0.01}
              value={currentTime}
              onChange={handleSeek}
              className="slider"
            />
            <span className="time">{formatTime(duration)}</span>
          </div>

          <button className="download-button">
            <a href={midiUrl || ''} download="generated.mid">Download MIDI</a>
          </button>
        </div>

        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
          className="hidden"
        />
      </div>
    )}


    </div>
  );
};

export default App;
