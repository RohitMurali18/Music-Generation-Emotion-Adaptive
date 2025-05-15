---

# Music Generation Emotion Adaptive

This project consists of a frontend and a backend for generating music based on emotional input. The backend is powered by FastAPI, and the frontend is built with React.

## Prerequisites

- **Node.js** and **npm**: Required to run the frontend.
- **Python 3.8+**: Required to run the backend.
- **Conda**: Recommended for managing Python environments.

## Setup

### Backend

1. **Install Python Dependencies**:
   Run:
   ```bash
   pip install requirements.txt
   ```

2. Install FluidSynth

- This project uses `fluidsynth` to convert MIDI files to audio. Install it using:

#### macOS
```bash
brew install fluid-synth
```

#### Windows
```bash
Download from: https://github.com/FluidSynth/fluidsynth/releases
Extract and add the folder with fluidsynth.exe to your system PATH
Restart your terminal or IDE
```

3. **Run the Backend**:
   Ensure you are in the project root directory and run:
   ```bash
   uvicorn api_cache:app --reload
   ```

### Frontend

1. **Navigate to the Frontend Directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js Dependencies**:
   Run the following command to install the necessary packages:
   ```bash
   npm install
   ```

3. **Run the Frontend**:
   Start the React development server with:
   ```bash
   npm start
   ```
   This will open the frontend in your default web browser, typically running on `http://localhost:3000`.

## Usage

- **Access the Frontend**: Open your web browser and go to `http://localhost:3000`.
- **Interact with the Backend**: The frontend will communicate with the backend to generate music based on the input provided.

## Notes

- Ensure that both the frontend and backend are running simultaneously for the application to function correctly.
---
