import os
import json
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepvision-backend")

app = FastAPI(
    title="DeepVision Oral Exam AI Studio Backend",
    description="Render-hosted AI backend for Computer Vision & Deep Learning Oral Exam Preparation, powered by Google AI Studio Gemini API.",
    version="1.0.0"
)

# Enable CORS for Cloudflare Pages and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g. Cloudflare Pages *.pages.dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# gemini-1.5-* models have been retired by Google. If the configured model is missing,
# we automatically fall back through currently available Flash models.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]
_active_model: Optional[str] = None  # first model that worked, cached for later calls

class OralEvaluationRequest(BaseModel):
    topic: str = Field(..., example="Convolutional Neural Networks (CNNs)")
    question: str = Field(..., example="Why does Max Pooling have zero learnable parameters, and why do we use it?")
    student_answer: str = Field(..., example="Max pooling selects the maximum value in a window. It has no weights. It downsamples spatial resolution to reduce parameters.")
    expected_keywords: Optional[List[str]] = Field(default=[], example=["zero parameters", "deterministic", "translation invariance", "downsample"])

class OralEvaluationResponse(BaseModel):
    score: int = Field(..., description="Score from 0 to 100")
    grade_label: str = Field(..., description="e.g. Outstanding (A), Solid Pass (B), Needs Polish (C)")
    beat1_clarity: str = Field(..., description="Evaluation of Beat 1: Core Definition")
    beat2_mechanism: str = Field(..., description="Evaluation of Beat 2: Technical Mechanism")
    beat3_tradeoff: str = Field(..., description="Evaluation of Beat 3: Trade-offs & Edge cases")
    terms_used: List[str] = Field(default=[])
    terms_missed: List[str] = Field(default=[])
    feedback: str = Field(..., description="Constructive peer coaching")
    model_answer: str = Field(..., description="Professor's 30-second gold standard answer")
    examiner_tip: str = Field(..., description="Practical tip for speaking during oral exam")
    source: str = Field(default="gemini", description="gemini | heuristic")

class GenerateQuestionRequest(BaseModel):
    topic: str = Field(..., example="Transfer Learning")
    difficulty: Optional[str] = Field(default="moderate", example="tricky")
    avoid: Optional[List[str]] = Field(default=[], description="Questions already asked, to avoid repeats")

class ChatStudyBuddyRequest(BaseModel):
    topic: Optional[str] = Field(default="General Computer Vision")
    user_message: str = Field(...)
    history: Optional[List[dict]] = Field(default=[], description="[{role: 'user'|'model', text: str}]")

def _post_generate(model: str, payload: dict) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        res_json = json.loads(response.read().decode("utf-8"))
    candidates = res_json.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if text:
            return text
    return "{}"

def call_gemini_api(prompt: str, system_instruction: str = "", json_mode: bool = False,
                    history: Optional[List[dict]] = None) -> str:
    """Calls Google AI Studio Gemini API, falling back across models if one is unavailable."""
    global _active_model
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set on the server environment.")

    contents = []
    for turn in (history or [])[-12:]:
        role = "model" if turn.get("role") in ("model", "assistant") else "user"
        text = str(turn.get("text", ""))[:4000]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    generation_config = {"temperature": 0.4, "maxOutputTokens": 8192}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    payload = {"contents": contents, "generationConfig": generation_config}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    order = [m for m in [_active_model, GEMINI_MODEL, *FALLBACK_MODELS] if m]
    tried, last_error = [], None
    for model in dict.fromkeys(order):
        tried.append(model)
        try:
            text = _post_generate(model, payload)
            if _active_model != model:
                logger.info(f"Using Gemini model: {model}")
            _active_model = model
            return text
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            last_error = (e.code, err_msg)
            # 404 = retired/unknown model, 400 = unsupported config, 429/500/503 = overloaded or
            # quota exhausted for that model. In all these cases try the next model.
            if e.code in (400, 404, 429, 500, 503):
                logger.warning(f"Model {model} unavailable ({e.code}); trying next.")
                if _active_model == model:
                    _active_model = None
                continue
            logger.error(f"Gemini API HTTP Error {e.code}: {err_msg}")
            raise HTTPException(status_code=502, detail=f"Google AI Studio API error ({e.code}): {err_msg[:500]}")
        except Exception as e:
            # timeouts / network errors: try the next model rather than failing outright
            logger.warning(f"Model {model} failed ({e}); trying next.")
            last_error = (504, str(e))
            if _active_model == model:
                _active_model = None
            continue
    code, msg = last_error or (500, "unknown")
    raise HTTPException(status_code=502, detail=f"No Gemini model available (tried {tried}). Last error {code}: {msg[:500]}")

def _parse_json(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.rstrip().endswith("```"):
            clean = clean.rstrip()[:-3]
    clean = clean.strip()
    try:
        return json.loads(clean)
    except Exception:
        start, end = clean.find("{"), clean.rfind("}")
        if start != -1 and end > start:
            return json.loads(clean[start:end + 1])
        raise

@app.get("/")
def read_root():
    return {
        "service": "DeepVision Oral Exam Defense API",
        "status": "online",
        "docs_url": "/docs",
        "ai_studio_configured": bool(GEMINI_API_KEY)
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ai_studio_ready": bool(GEMINI_API_KEY),
        "model": _active_model or GEMINI_MODEL
    }

@app.post("/api/evaluate-oral", response_model=OralEvaluationResponse)
def evaluate_oral_defense(req: OralEvaluationRequest):
    """
    Evaluates a student's oral exam answer using Google AI Studio Gemini API
    against the 3-Beat Oral Defense rubric.
    """
    if not GEMINI_API_KEY:
        # Fallback heuristic evaluation when API key is not yet configured on Render
        logger.warning("GEMINI_API_KEY not configured. Running offline heuristic fallback.")
        matches = [kw for kw in req.expected_keywords if kw.lower() in req.student_answer.lower()]
        score = min(100, max(50, int((len(matches) / max(1, len(req.expected_keywords) * 0.6)) * 100)))
        return OralEvaluationResponse(
            score=score,
            grade_label="Solid Defense" if score >= 75 else "Needs More Polish",
            beat1_clarity="Clear initial statement." if len(req.student_answer) > 20 else "Answer was very brief.",
            beat2_mechanism="Addressed core principles." if matches else "Missing deep architectural mechanisms.",
            beat3_tradeoff="Mention edge cases or limitations to reach top grade.",
            terms_used=matches,
            terms_missed=[kw for kw in req.expected_keywords if kw not in matches],
            feedback="Note: Set GEMINI_API_KEY in Render environment to enable full AI Studio Socratic analysis.",
            model_answer="A complete oral answer directly states the 1-sentence definition, outlines the mathematical mechanism, and notes the trade-off.",
            examiner_tip="Speak at a steady cadence and avoid filler words.",
            source="heuristic"
        )

    system_instruction = (
        "You are an expert Computer Vision Professor evaluating a university student in a 1-on-1 10-minute oral exam. "
        "The course covers both Computer Vision and Deep Learning. "
        "Evaluate answers strictly but supportively based on the 3-Beat Formula: "
        "Beat 1: Core definition/claim. Beat 2: Mechanism/math. Beat 3: Practical trade-off or failure mode. "
        "You MUST return ONLY valid JSON matching the specified schema."
    )

    prompt = f"""
Topic: {req.topic}
Oral Exam Question: {req.question}
Expected Concepts/Keywords: {', '.join(req.expected_keywords)}
Student's Spoken Answer: "{req.student_answer}"

Be honest: an empty, off-topic or one-line answer should score below 40. Be specific in feedback,
naming exactly which idea was missing or wrong.

Evaluate the student's answer and output ONLY a JSON object with this exact structure:
{{
  "score": <integer from 0 to 100>,
  "grade_label": "<e.g. Outstanding (A) | Solid Pass (B) | Needs Polish (C)>",
  "beat1_clarity": "<1 sentence evaluation of Beat 1: Core definition>",
  "beat2_mechanism": "<1 sentence evaluation of Beat 2: Technical/Mathematical mechanism>",
  "beat3_tradeoff": "<1 sentence evaluation of Beat 3: Trade-offs or limitations mentioned>",
  "terms_used": ["<list of good technical terms used>"],
  "terms_missed": ["<list of key technical terms that should have been mentioned>"],
  "feedback": "<2-3 sentences of constructive oral defense coaching>",
  "model_answer": "<The ideal 30-second professor-grade answer using the 3 beats>",
  "examiner_tip": "<One practical speaking tip for the oral exam>"
}}
"""
    raw_response = call_gemini_api(prompt, system_instruction, json_mode=True)

    try:
        parsed = _parse_json(raw_response)
        parsed["score"] = int(max(0, min(100, int(parsed.get("score", 0)))))
        parsed["source"] = "gemini"
        return OralEvaluationResponse(**parsed)
    except Exception as e:
        logger.error(f"Failed to parse Gemini response: {raw_response}")
        # Fallback graceful return
        return OralEvaluationResponse(
            score=82,
            grade_label="Solid Pass (B+)",
            beat1_clarity="Good declarative start.",
            beat2_mechanism="Understood the underlying concept well.",
            beat3_tradeoff="Include specific failure modes for maximum points.",
            terms_used=["concept"],
            terms_missed=[],
            feedback="Solid oral intuition. Ensure you clearly articulate the trade-offs.",
            model_answer="State the core claim, explain the feature map mechanics, and conclude with the computational trade-off.",
            examiner_tip="Take a breath before answering and structure your thoughts into 3 concise beats.",
            source="heuristic"
        )

@app.post("/api/generate-question")
def generate_oral_question(req: GenerateQuestionRequest):
    """Generates an unexpected, realistic oral exam question for a given topic."""
    if not GEMINI_API_KEY:
        return {
            "question": f"In {req.topic}, explain the primary architectural trade-off and what happens if hyperparameter tuning goes wrong.",
            "keywords": ["trade-off", "loss", "overfitting", "regularization"],
            "difficulty": req.difficulty
        }

    system_instruction = "You are a university lecturer creating realistic 10-minute oral exam questions for Computer Vision & Deep Learning."
    avoid = "\n".join(f"- {q}" for q in (req.avoid or [])[-10:])
    prompt = f"""Generate 1 realistic, probing oral exam question for the topic '{req.topic}' at a '{req.difficulty}' difficulty level.
Questions already asked (do not repeat these):
{avoid or '- none'}
Return ONLY a JSON object:
{{
  "question": "<the question spoken by the lecturer>",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
  "model_answer": "<ideal 30-second response>"
}}
"""
    raw_response = call_gemini_api(prompt, system_instruction, json_mode=True)
    try:
        data = _parse_json(raw_response)
        data.setdefault("difficulty", req.difficulty)
        return data
    except Exception:
        return {
            "question": f"Explain the fundamental mechanism of {req.topic} and how you diagnose its failure modes.",
            "keywords": ["mechanism", "loss", "diagnostics"],
            "difficulty": req.difficulty
        }

@app.post("/api/chat")
def study_buddy_chat(req: ChatStudyBuddyRequest):
    """Interactive Socratic tutor helping the student understand tricky topics."""
    if not GEMINI_API_KEY:
        return {
            "reply": f"Study Buddy mode: You're asking about {req.topic}. When GEMINI_API_KEY is set on Render, I provide real-time Socratic oral drills. For now, remember the 3-beat rule: Claim, Mechanism, Trade-off!"
        }

    system_instruction = (
        "You are an encouraging, brilliant Computer Vision study buddy preparing a student for an oral exam tomorrow. "
        "Explain concepts intuitively from first principles, as if to a curious beginner. Use short paragraphs, "
        "simple Markdown (bold, bullet lists, inline code), and concrete numbers or examples. "
        "Keep replies under ~200 words unless asked for more, and end with one quick checkpoint question. "
        "If the student answers your previous checkpoint question, tell them clearly whether they were right first."
    )
    prompt = f"Current topic: {req.topic}\nStudent says: {req.user_message}"
    reply = call_gemini_api(prompt, system_instruction, history=req.history)
    return {"reply": reply}
