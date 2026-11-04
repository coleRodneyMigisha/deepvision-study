# DeepVision Oral Defense: Full Deployment Guide
### Google AI Studio (Gemini) + Render (Backend) + Cloudflare Pages (Frontend)

This package contains everything needed to deploy the **DeepVision Oral Exam Simulator & Interactive Labs** to the public internet so you can access it anywhere from your laptop, tablet, or phone.

---

## 🏗️ Architecture Overview

```
[Student Browser / Phone]
        │
        ▼
[Cloudflare Pages] (Global Edge CDN / SSL / Custom Domain)
        │  (Calls /api/evaluate-oral)
        ▼
[Render.com Web Service] (FastAPI Python Server)
        │  (Authenticated via GEMINI_API_KEY)
        ▼
[Google AI Studio] (Gemini 1.5 Flash - Real-Time Oral Exam Socratic Scoring)
```

---

## 📦 Directory Structure

```
deepvision-deploy/
├── backend/
│   ├── main.py              # FastAPI server with CORS & Gemini API integration
│   ├── requirements.txt     # Python dependencies (fastapi, uvicorn, pydantic)
│   ├── render.yaml          # Render 1-click Blueprint configuration
│   ├── Dockerfile           # Optional container deployment
│   └── .env.example         # Template for environment variables
├── frontend/
│   ├── index.html           # Complete single-page interactive study app
│   ├── _headers             # Cloudflare security & CORS headers
│   └── _redirects           # Cloudflare SPA fallback routing
└── README.md                # This deployment guide
```

---

## 🚀 Step 1: Get Your Free Google AI Studio API Key

1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **"Get API key"** in the top left navigation.
4. Click **"Create API key"** (select a project or create a default one).
5. Copy your API key (starts with `AIzaSy...`). Keep it handy for Step 2!

---

## ⚡ Step 2: Deploy the Backend to Render (Free)

1. Create a free account at [Render.com](https://render.com/).
2. Push this project to a GitHub repository (or push just the `backend/` directory).
3. On your Render Dashboard:
   - Click **"New +"** ➔ **"Web Service"**.
   - Connect your GitHub repository.
   - Configure the following settings:
     - **Name**: `deepvision-backend` (or your choice)
     - **Region**: Oregon (US West) or closest to you
     - **Branch**: `main`
     - **Root Directory**: `backend` (if your repo contains both folders)
     - **Runtime**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type**: `Free`
4. Under **"Environment Variables"**:
   - Add: `GEMINI_API_KEY` = `<paste your key from Google AI Studio>`
   - Add: `GEMINI_MODEL` = `gemini-1.5-flash`
5. Click **"Deploy Web Service"**.
6. Once deployed, copy your public Render URL (e.g., `https://deepvision-backend.onrender.com`).
   - *Note on Free Tier*: Free Render services spin down after 15 minutes of inactivity. When you open the app, the first request may take ~30 seconds while the server wakes up.

---

## 🌐 Step 3: Deploy the Frontend to Cloudflare Pages (Free & Instant)

You have two easy options on Cloudflare:

### Option A: 60-Second Direct Drag-and-Drop (No Git Required)
1. Log in to your [Cloudflare Dashboard](https://dash.cloudflare.com/).
2. Navigate to **Compute (Workers) ➔ Pages**.
3. Click **"Upload assets"** or **"Create an application" ➔ "Pages" ➔ "Upload assets"**.
4. Project name: `deepvision` (or whatever you like).
5. Drag and drop the `frontend` folder (containing `index.html`, `_headers`, and `_redirects`).
6. Click **"Deploy site"**.
7. Your app is instantly live globally at `https://deepvision.pages.dev`!

### Option B: Connect via GitHub
1. In Cloudflare Pages, click **"Connect to Git"**.
2. Select your repository.
3. Set **Build output directory** to: `frontend`.
4. Click **"Save and Deploy"**. Every Git push will automatically redeploy your site.

---

## 🔗 Step 4: Connect Cloudflare Frontend to Render Backend

1. Open your live Cloudflare Pages URL (`https://your-site.pages.dev`).
2. In the top navigation bar, click the button: **"Backend: Setup"**.
3. Paste your Render backend URL (e.g., `https://deepvision-backend.onrender.com`).
4. Click **"Test Connection"** to verify the health check.
5. Click **"Save & Connect"**.

The setting is automatically saved in your browser's local storage. From now on:
- Every oral checkpoint question you answer is sent to Gemini on Google AI Studio.
- Gemini evaluates your oral defense against the **3-Beat Formula** (Definition, Mechanism, Trade-off), awards a score, identifies missing technical terminology, and reveals the professor's gold-standard answer!

---

## 🛠️ Testing Locally (Optional)

To run both services locally before deploying:

```bash
# 1. Run Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key_here"
uvicorn main:app --reload --port 8000

# 2. Open Frontend
# Simply double-click frontend/index.html or run:
cd ../frontend
python3 -m http.server 3000
# Then set backend URL in the UI to: http://localhost:8000
```
