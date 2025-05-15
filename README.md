---

# Music Generation Emotion Adaptive

This project consists of a frontend and a backend for generating music based on emotional input. The backend is powered by FastAPI, and the frontend is built with React.

## Prerequisites

- **Node.js** and **npm**: Required to run the frontend.
- **Python 3.8+**: Required to run the backend.
- **Conda**: Recommended for managing Python environments.

## Setup

### Backend

1. **Create a Conda Environment**:
   ```bash
   conda create --name music python=3.8
   conda activate music
   ```

2. **Install Python Dependencies**:
   Run:
   ```bash
   pip install fastapi==0.110.1 uvicorn==0.29.0 torch==2.2.2 pretty_midi==0.2.10 midi2audio==0.1.1 python-dotenv aiofiles==23.2.1 pydantic==2.7.1 numpy scipy types-python-dateutil typing-extensions
   ```

3. **Run the Backend**:
   Ensure you are in the project root directory and run:
   ```bash
   uvicorn api_cache:app --reload
   ```
   This will start the FastAPI server on `http://127.0.0.1:8000`.

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
- If you encounter any issues, check the console output for error messages and ensure all dependencies are installed correctly.

---
