import os
import json
import logging
import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("study-site-backend")

app = FastAPI(
    title="Multi-Course AI Study Platform Backend",
    description=(
        "Render-hosted AI backend for Multi-Course Interactive Study Platform "
        "(Computer Vision & Deep Learning + Data Science Systems & Technologies), "
        "powered by Google AI Studio Gemini API."
    ),
    version="2.0.0"
)

# Enable CORS for Cloudflare Pages and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
FALLBACK_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
_active_model: Optional[str] = None
_daily_challenge_cache: Dict[str, dict] = {}


COURSE_PROFILES = {
    "cv": {
        "name": "Computer Vision & Deep Learning (CSC3104)",
        "professor_role": (
            "You are an expert Computer Vision & Deep Learning Professor evaluating a university student "
            "in a 1-on-1 oral exam and interactive lab."
        ),
    },
    "ds": {
        "name": "Data Science: Systems & Technologies (DSC3103)",
        "professor_role": (
            "You are an expert Data Engineering & Data Systems Professor evaluating a university student "
            "in DSC3103 Data Science (Systems & Technologies), covering data profiling/cleaning, ETL/ELT pipelines, "
            "DuckDB star schemas, Apache Airflow orchestration, PySpark distributed batch processing, "
            "event-time streaming/watermarks, and FastAPI/Docker model deployment."
        ),
    },
}


def get_course_profile(course_id: Optional[str], course_name: Optional[str] = None) -> dict:
    cid = (course_id or "").lower().strip()
    if cid in COURSE_PROFILES:
        return COURSE_PROFILES[cid]
    # Fallback check against course_name
    cname = (course_name or "").lower().strip()
    if "data science" in cname or "dsc" in cname:
        return COURSE_PROFILES["ds"]
    if "vision" in cname or "cv" in cname or "deep" in cname:
        return COURSE_PROFILES["cv"]
    label = course_name or course_id or "University Computer Science"
    return {
        "name": label,
        "professor_role": f"You are an expert university professor teaching {label}.",
    }


class OralEvaluationRequest(BaseModel):
    course_id: Optional[str] = Field(default=None, example="ds")
    course: Optional[str] = Field(default=None, description="Alias for course_id")
    course_name: Optional[str] = Field(default=None, example="Data Science: Systems & Technologies")
    topic: str = Field(..., example="Convolutional Neural Networks (CNNs)")
    question: str = Field(..., example="Why does Max Pooling have zero learnable parameters, and why do we use it?")
    student_answer: str = Field(..., example="Max pooling selects the maximum value in a window. It has no weights. It downsamples spatial resolution to reduce parameters.")
    expected_keywords: Optional[List[str]] = Field(default=[], example=["zero parameters", "deterministic", "translation invariance", "downsample"])

    def resolved_course_id(self) -> str:
        return self.course_id or self.course or "ds"


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
    course_id: Optional[str] = Field(default=None, example="ds")
    course: Optional[str] = Field(default=None, description="Alias for course_id")
    course_name: Optional[str] = Field(default=None)
    topic: str = Field(..., example="Transfer Learning")
    difficulty: Optional[str] = Field(default="moderate", example="tricky")
    avoid: Optional[List[str]] = Field(default=[], description="Questions already asked, to avoid repeats")

    def resolved_course_id(self) -> str:
        return self.course_id or self.course or "ds"


class ChatStudyBuddyRequest(BaseModel):
    course_id: Optional[str] = Field(default=None, example="ds")
    course: Optional[str] = Field(default=None, description="Alias for course_id")
    course_name: Optional[str] = Field(default=None)
    topic: Optional[str] = Field(default="General Study Topic")
    user_message: str = Field(...)
    history: Optional[List[dict]] = Field(default=[], description="[{role: 'user'|'model', text: str}]")

    def resolved_course_id(self) -> str:
        return self.course_id or self.course or "ds"


class DailyChallengeRequest(BaseModel):
    course_id: Optional[str] = Field(default=None, example="ds")
    course: Optional[str] = Field(default=None, description="Alias for course_id")
    course_name: Optional[str] = Field(default=None)
    topic: str = Field(..., example="Streaming & Event-Time Watermarks")
    lab_title: Optional[str] = Field(default="")
    date_seed: Optional[str] = Field(default=None, description="YYYY-MM-DD date seed; defaults to today UTC")
    variation: Optional[int] = Field(default=1, description="Variation number so user can try multiple versions")
    avoid_questions: Optional[List[str]] = Field(default=[])

    def resolved_course_id(self) -> str:
        return self.course_id or self.course or "ds"


def _post_generate(model: str, payload: dict) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        res_json = json.loads(response.read().decode("utf-8"))
    candidates = res_json.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if text:
            return text
    return "{}"


def call_gemini_api(
    prompt: str,
    system_instruction: str = "",
    json_mode: bool = False,
    history: Optional[List[dict]] = None,
) -> str:
    """Calls Google AI Studio Gemini API, falling back across Flash models if one is unavailable or rate-limited."""
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

    generation_config = {"temperature": 0.45, "maxOutputTokens": 8192}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    payload: Dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
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
            if e.code in (400, 404, 429, 500, 503):
                logger.warning(f"Model {model} unavailable ({e.code}); trying next.")
                if _active_model == model:
                    _active_model = None
                continue
            logger.error(f"Gemini API HTTP Error {e.code}: {err_msg}")
            raise HTTPException(status_code=502, detail=f"Google AI Studio API error ({e.code}): {err_msg[:500]}")
        except Exception as e:
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
            return json.loads(clean[start : end + 1])
        raise


@app.get("/")
def read_root():
    return {
        "service": "StudyHub Multi-Course AI Platform API",
        "status": "online",
        "courses": list(COURSE_PROFILES.keys()),
        "docs_url": "/docs",
        "ai_studio_configured": bool(GEMINI_API_KEY),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ai_studio_ready": bool(GEMINI_API_KEY),
        "model": _active_model or GEMINI_MODEL,
        "courses": list(COURSE_PROFILES.keys()),
    }


@app.post("/api/evaluate-oral", response_model=OralEvaluationResponse)
def evaluate_oral_defense(req: OralEvaluationRequest):
    """
    Evaluates a student's oral exam or checkpoint answer using Google AI Studio Gemini API
    against the 3-Beat Oral Defense rubric (Claim, Mechanism, Trade-off).
    """
    profile = get_course_profile(req.resolved_course_id(), req.course_name)
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured. Running offline heuristic fallback.")
        matches = [kw for kw in (req.expected_keywords or []) if kw.lower() in req.student_answer.lower()]
        score = min(100, max(50, int((len(matches) / max(1, len(req.expected_keywords or [1]) * 0.6)) * 100)))
        return OralEvaluationResponse(
            score=score,
            grade_label="Solid Defense" if score >= 75 else "Needs More Polish",
            beat1_clarity="Clear initial statement." if len(req.student_answer) > 20 else "Answer was very brief.",
            beat2_mechanism="Addressed core principles." if matches else "Missing deep architectural mechanisms.",
            beat3_tradeoff="Mention edge cases or limitations to reach top grade.",
            terms_used=matches,
            terms_missed=[kw for kw in (req.expected_keywords or []) if kw not in matches],
            feedback="Note: Set GEMINI_API_KEY in Render environment to enable full AI Studio Socratic analysis.",
            model_answer="A complete oral answer directly states the 1-sentence definition, outlines the technical mechanism, and notes the engineering trade-off.",
            examiner_tip="Speak at a steady cadence and structure your answer into Claim, Mechanism, and Trade-off.",
            source="heuristic",
        )

    system_instruction = (
        f"{profile['professor_role']} "
        "Evaluate answers strictly but supportively based on the 3-Beat Formula: "
        "Beat 1: Core definition/claim. Beat 2: Mechanism/math/system architecture. Beat 3: Practical trade-off or failure mode. "
        "You MUST return ONLY valid JSON matching the specified schema."
    )

    prompt = f"""
Course: {profile['name']}
Topic: {req.topic}
Oral Exam Question: {req.question}
Expected Concepts/Keywords: {', '.join(req.expected_keywords or [])}
Student's Spoken Answer: "{req.student_answer}"

Be honest: an empty, off-topic or one-line answer should score below 40. Be specific in feedback,
naming exactly which idea was missing or wrong.

Evaluate the student's answer and output ONLY a JSON object with this exact structure:
{{
  "score": <integer from 0 to 100>,
  "grade_label": "<e.g. Outstanding (A) | Solid Pass (B) | Needs Polish (C)>",
  "beat1_clarity": "<1 sentence evaluation of Beat 1: Core definition>",
  "beat2_mechanism": "<1 sentence evaluation of Beat 2: Technical/Mathematical/System mechanism>",
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
    except Exception:
        logger.error(f"Failed to parse Gemini response: {raw_response}")
        return OralEvaluationResponse(
            score=82,
            grade_label="Solid Pass (B+)",
            beat1_clarity="Good declarative start.",
            beat2_mechanism="Understood the underlying concept well.",
            beat3_tradeoff="Include specific failure modes for maximum points.",
            terms_used=["concept"],
            terms_missed=[],
            feedback="Solid intuition. Ensure you clearly articulate the trade-offs.",
            model_answer="State the core claim, explain the mechanism, and conclude with the operational trade-off.",
            examiner_tip="Take a breath before answering and structure your thoughts into 3 concise beats.",
            source="heuristic",
        )


@app.post("/api/generate-question")
def generate_oral_question(req: GenerateQuestionRequest):
    """Generates an unexpected, realistic oral exam question for a given course and topic."""
    profile = get_course_profile(req.resolved_course_id(), req.course_name)
    if not GEMINI_API_KEY:
        return {
            "question": f"In {req.topic}, explain the primary architectural trade-off and how you diagnose failure modes in production.",
            "keywords": ["trade-off", "mechanism", "failure mode", "diagnostics"],
            "difficulty": req.difficulty,
        }

    system_instruction = f"{profile['professor_role']} Create realistic 10-minute oral exam questions."
    avoid = "\n".join(f"- {q}" for q in (req.avoid or [])[-10:])
    prompt = f"""Generate 1 realistic, probing oral exam question for the course '{profile['name']}' on the topic '{req.topic}' at a '{req.difficulty}' difficulty level.
Questions already asked (do not repeat these):
{avoid or '- none'}
Return ONLY a JSON object:
{{
  "question": "<the question spoken by the lecturer>",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "<keyword4>"],
  "model_answer": "<ideal 30-second response following Claim, Mechanism, Trade-off>"
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
            "keywords": ["mechanism", "validation", "diagnostics"],
            "difficulty": req.difficulty,
        }


@app.post("/api/daily-challenge")
def generate_daily_challenge(req: DailyChallengeRequest):
    """
    Generates a daily changing, multi-step numerical/engineering lab challenge
    so students can attempt fresh variations of a lab concept every day.
    """
    cid = req.resolved_course_id()
    today = req.date_seed or datetime.date.today().isoformat()
    cache_key = f"{today}:{cid}:{req.topic}:{req.variation}"
    if cache_key in _daily_challenge_cache and not req.avoid_questions:
        return _daily_challenge_cache[cache_key]

    profile = get_course_profile(cid, req.course_name)
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured for AI daily challenge")

    system_instruction = (
        f"{profile['professor_role']} "
        "You design rigorous, educational, numerical interactive lab challenges where the student must compute "
        "a concrete numeric answer (integer or decimal) by applying formulas and systems concepts from the lab. "
        "Make the problem realistic, multi-step, and pedagogical so the student genuinely learns while solving it."
    )
    avoid_str = "\n".join(f"- {q}" for q in (req.avoid_questions or [])[-6:])
    prompt = f"""Create a new daily numerical lab challenge for date {today} (Variation #{req.variation}).
Course: {profile['name']}
Module Topic: {req.topic}
Lab Title: {req.lab_title or req.topic}
Avoid repeating these previous challenge prompts:
{avoid_str or '- none'}

Requirements:
1. The question (`q`) must give concrete numbers/parameters and ask for a single numeric value (integer or decimal rounded to at most 2 decimal places). Use HTML `<b>` and `<code>` tags sparingly to highlight key numbers.
2. The numeric answer (`answer`) MUST be a pure number (float or int) that matches your step-by-step calculation. Double-check the arithmetic!
3. Set `tol` (numeric tolerance, e.g. 0.02 for decimals or 0 for exact integers).
4. Provide a `hint` that guides the student on which formula or system rule to apply without giving away the final number.
5. Provide a thorough step-by-step `explain` string showing the exact calculation and the conceptual takeaway.

Return ONLY valid JSON with this exact schema:
{{
  "title": "<short 3-6 word title for this challenge variation>",
  "q": "<the problem statement with concrete numbers>",
  "answer": <numeric float or int>,
  "tol": <numeric float tolerance, e.g. 0.02 or 0>,
  "hint": "<1-2 sentence hint>",
  "explain": "<step-by-step worked calculation and engineering insight>",
  "date_seed": "{today}",
  "variation": {req.variation},
  "source": "gemini"
}}
"""
    raw_response = call_gemini_api(prompt, system_instruction, json_mode=True)
    data = _parse_json(raw_response)
    data["answer"] = float(data["answer"])
    data["tol"] = float(data.get("tol", 0.02))
    data["date_seed"] = today
    data["variation"] = req.variation
    data["source"] = "gemini"
    _daily_challenge_cache[cache_key] = data
    return data


@app.post("/api/chat")
def study_buddy_chat(req: ChatStudyBuddyRequest):
    """Interactive Socratic tutor helping the student master any course module."""
    cid = req.resolved_course_id()
    profile = get_course_profile(cid, req.course_name)
    if not GEMINI_API_KEY:
        return {
            "reply": (
                f"Study Buddy mode ({profile['name']}): You're asking about {req.topic}. "
                "When GEMINI_API_KEY is set on Render, I provide real-time Socratic oral drills. "
                "For now, remember the 3-beat rule: Claim, Mechanism, Trade-off!"
            )
        }

    system_instruction = (
        f"You are an encouraging, brilliant university study buddy for '{profile['name']}' ({cid.upper()}) preparing a student for an exam and lab defense. "
        f"STRICT INSTRUCTION: Only reference concepts, terminology, algorithms, and examples relevant to '{profile['name']}'. "
        f"Do NOT reference unrelated courses (for example, if tutoring {profile['name']}, do NOT mention Computer Vision, pixels, image recognition, or CNNs unless specifically asked by the student). "
        "Explain concepts intuitively from first principles, as if to a curious student. Use short paragraphs, "
        "simple Markdown (bold, bullet lists, inline code), and concrete numbers or examples from the labs. "
        "Keep replies under ~200 words unless asked for more, and end with one quick checkpoint question. "
        "If the student answers your previous checkpoint question, tell them clearly whether they were right first."
    )
    prompt = f"Course: {profile['name']}\nCurrent topic: {req.topic}\nStudent says: {req.user_message}"
    reply = call_gemini_api(prompt, system_instruction, history=req.history)
    return {"reply": reply}
