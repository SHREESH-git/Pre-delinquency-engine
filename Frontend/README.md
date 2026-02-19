# EarlyShield | Risk Protection Portal

EarlyShield is an integrated risk management system featuring a hybrid ML/DL risk engine, a real-time behavioral stress monitoring system, and an interactive dashboard.

## Project Structure

- `/backend`: FastAPI service, ML models, and risk engine.
- `/dashboard`: Vite + TypeScript frontend.
- `/content_backup (2)`: Reference models and scripts.

---

## 🚀 Getting Started

### 1. Start the Backend (API & ML Engine)
The backend runs on **Python 3.10+**. Use the existing virtual environment.

1. Open a new terminal.
2. Navigate to the `backend` directory:
   ```powershell
   cd d:\Games\Frontend\backend
3. Create a virtual environment to avoid conflicts with local python environment:
   python -m venv .venv

4. Activate the virtual environment (located in the root folder):
   ```powershell
   # From d:\Games\Frontend\backend, go up one level to find .venv
   ..\.venv\Scripts\Activate.ps1
   ```
   *Note: If you get a "not recognized" error, ensure you are in the `backend` folder and using `..` to reference the parent.*

5. Start the server:
   ```powershell
   python main.py
   ```
   *The backend will be available at `http://localhost:8000`.*
if ERROR occurs, try:
   ```powershell
   netstat -ano | findstr :8000
find the running process id and kill running process:
   taskkill /F /PID <PID>
   ```
### 2. Start the Frontend (Dashboard)
The frontend uses **NPM/Vite**.

1. Open a second terminal.
2. Navigate to the dashboard directory:
   ```powershell
   cd d:\Games\Frontend\dashboard
   ```
3. Install dependencies:
   ```powershell
   npm install
   ```
4. Start the development server:
   ```powershell
   npm run dev
   ```
   *The dashboard will be available at `http://localhost:5173`.*

---

## 🧪 Testing the Integration

### Manual Dashboard Test
1. Open the Dashboard in your browser (`http://localhost:5173`).
2. Navigate to the **Customer Safety** tab in the sidebar.
3. Search for a test customer ID: `101`.
4. Observe the **"Click & Respond"** feedback:
   - A success toast will appear when the profile loads.
   - Click **Run ML Inference** to see the real-time risk scores.
   - Click **View Synthetic Data** to verify the raw history being fetched from the backend.

### Backend API Test
If you want to verify the API independently:
1. Ensure the backend is running.
2. Run the diagnostic script:
   ```powershell
   python test_api.py
   ```
   *This script verifies the Root status, the Hybrid ML Prediction, and the Kafka Stream status.*

---

## 🛠 Features
- **Hybrid ML/DL Ensemble**: Uses XGBoost, LightGBM, CatBoost, and LSTM.
- **EL Engine**: Live calculation of PD, LGD, and EAD.
- **Interactive UI**: Toast notifications for "click and response" feedback.
- **Data Transparency**: Modal windows to verify synthetic data ingestion.
