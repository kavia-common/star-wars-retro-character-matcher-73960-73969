import os
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ----- Domain Models -----

class QuizAnswer(BaseModel):
    question_id: int = Field(..., description="ID of the question (1-5).")
    answer_id: str = Field(..., description="Identifier for chosen answer option.")

class QuizSubmission(BaseModel):
    answers: List[QuizAnswer] = Field(..., description="List of answers for the 5-question quiz.")
    selfie_temp_id: Optional[str] = Field(None, description="Optional selfie temp ID if selfie already uploaded.")

class CharacterTraits(BaseModel):
    name: str = Field(..., description="Character name")
    traits: List[str] = Field(..., description="List of personality traits")

class MatchResult(BaseModel):
    character: CharacterTraits = Field(..., description="Matched character and traits")
    witty_writeup: str = Field(..., description="Witty description of why this match fits, in 80s-retro Star Wars glam style.")
    score_breakdown: Dict[str, int] = Field(..., description="Trait score aggregation for transparency")
    selfie_temp_id: Optional[str] = Field(None, description="Optional selfie temp ID carried forward")

class MashupRequest(BaseModel):
    selfie_temp_id: str = Field(..., description="Temporary id or URL returned from selfie upload step.")
    character_name: str = Field(..., description="Matched character name to use for style/theme of mashup.")

class MashupResult(BaseModel):
    mashup_url: HttpUrl = Field(..., description="Public URL of generated mashup image.")
    public_id: str = Field(..., description="Cloudinary public id of generated asset.")

class ShareRequest(BaseModel):
    channel: str = Field(..., description="Share channel: facebook, twitter, sms")
    message: Optional[str] = Field(None, description="Optional custom message/caption")
    target_phone_e164: Optional[str] = Field(None, description="E.164 formatted phone for SMS when channel=sms")
    image_url: HttpUrl = Field(..., description="URL of the mashup image to share.")
    page_url: Optional[HttpUrl] = Field(None, description="Optional page URL to include for social shares")

class ShareResponse(BaseModel):
    status: str = Field(..., description="ok or error")
    link: Optional[HttpUrl] = Field(None, description="Generated social share link when applicable")
    sid: Optional[str] = Field(None, description="Twilio SID when SMS is sent")

# ----- Config helpers -----

def require_env(name: str) -> str:
    """Fetch env var and raise helpful error if missing."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

# ----- Simple in-memory catalogs and logic for demo -----

# Simple character trait catalog. In production, this could be in DB.
CHARACTER_CATALOG: List[CharacterTraits] = [
    CharacterTraits(name="Luke Skywalker", traits=["hopeful", "brave", "loyal", "idealistic"]),
    CharacterTraits(name="Darth Vader", traits=["ambitious", "disciplined", "intense", "strategic"]),
    CharacterTraits(name="Princess Leia", traits=["witty", "leader", "resilient", "compassionate"]),
    CharacterTraits(name="Han Solo", traits=["charming", "rebellious", "resourceful", "risk-taker"]),
    CharacterTraits(name="Yoda", traits=["wise", "patient", "mysterious", "humble"]),
]

# Map quiz answers to trait deltas
ANSWER_TO_TRAITS: Dict[str, List[str]] = {
    # q1 options
    "q1_a": ["hopeful", "idealistic", "leader"],
    "q1_b": ["rebellious", "risk-taker"],
    "q1_c": ["disciplined", "strategic"],
    "q1_d": ["wise", "patient"],
    # q2 options
    "q2_a": ["witty", "charming"],
    "q2_b": ["resilient", "loyal"],
    "q2_c": ["intense", "ambitious"],
    "q2_d": ["mysterious", "humble"],
    # q3 options
    "q3_a": ["leader", "strategic"],
    "q3_b": ["resourceful", "risk-taker"],
    "q3_c": ["patient", "wise"],
    "q3_d": ["compassionate", "hopeful"],
    # q4 options
    "q4_a": ["brave", "loyal"],
    "q4_b": ["charming", "witty"],
    "q4_c": ["intense", "disciplined"],
    "q4_d": ["mysterious", "wise"],
    # q5 options
    "q5_a": ["idealistic", "hopeful"],
    "q5_b": ["rebellious", "resourceful"],
    "q5_c": ["ambitious", "strategic"],
    "q5_d": ["humble", "compassionate"],
}

# ----- Services -----

class OpenAIService:
    """Service wrapper for OpenAI text generation."""

    def __init__(self):
        # PUBLIC_INTERFACE
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    # PUBLIC_INTERFACE
    def witty_writeup(self, character: CharacterTraits, trait_scores: Dict[str, int]) -> str:
        """
        Generate a playful, 80's-retro Star Wars themed witty write-up explaining
        why the user matched the character.
        Falls back to a templated message if OPENAI_API_KEY not configured.
        """
        base_prompt = (
            "Write a 120-180 word witty, cheeky 1980s glam Star Wars style blurb for a quiz result.\n"
            f"Matched character: {character.name}\n"
            f"Traits and scores: {trait_scores}\n"
            "Tone: neon, retro, tongue-in-cheek, VHS nostalgia, disco sparkle. Avoid copyrighted song lyrics.\n"
            "Second-person voice, positive and fun. End with a playful one-liner."
        )
        if not self.api_key:
            # Fallback deterministic copy
            top_traits = sorted(trait_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            tlist = ", ".join([t for t, _ in top_traits]) if top_traits else ", ".join(character.traits[:3])
            return (
                f"Dial up the synths—your inner {character.name} just stepped out of a retro hyperspace limo! "
                f"Your vibe beams {tlist}, making you a perfect fit for this star-splashed icon. "
                "Picture a fog machine, glittering stars, and your best power stance—because subtlety is for droids. "
                "Whether you're zinging one-liners or saving the day with swagger, you've got that VHS-certified charm. "
                "Stay luminous, star child—your destiny is neon."
            )
        # Using httpx directly to avoid adding heavy dependencies; models API v1.
        import httpx
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a witty retro copywriter."},
                    {"role": "user", "content": base_prompt},
                ],
                "temperature": 0.9,
                "max_tokens": 300,
            }
            resp = httpx.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            # Graceful fallback
            top_traits = sorted(trait_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            tlist = ", ".join([t for t, _ in top_traits]) if top_traits else ", ".join(character.traits[:3])
            return (
                f"Cosmic static! Our holo-writer is snoozing. Still, your aura screams {character.name}. "
                f"Those {tlist} vibes? Chef’s kiss. Glitter on, superstar."
            )

class CloudinaryService:
    """Service for uploading files and creating simple transformations via Cloudinary."""

    def __init__(self):
        self.cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "")
        self.api_key = os.getenv("CLOUDINARY_API_KEY", "")
        self.api_secret = os.getenv("CLOUDINARY_API_SECRET", "")
        # Note: If not configured, we provide mock URLs for local dev.

    # PUBLIC_INTERFACE
    async def upload_selfie(self, file: UploadFile) -> Dict[str, str]:
        """
        Upload a selfie to Cloudinary. Returns dict with public_id and url.
        If Cloudinary env vars are missing, returns a mock URL for development.
        """
        if not (self.cloud_name and self.api_key and self.api_secret):
            # Dev fallback: produce a pseudo URL using filename
            fake_id = f"dev/{file.filename}"
            return {"public_id": fake_id, "url": f"https://example.com/{fake_id}"}

        import hashlib
        import time
        import httpx

        # Cloudinary unsigned upload requires an upload_preset; with signed we must sign the request.
        upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET")  # Optional if using unsigned
        timestamp = str(int(time.time()))
        params_to_sign = {"timestamp": timestamp}
        if upload_preset:
            params_to_sign["upload_preset"] = upload_preset

        # Generate signature
        to_sign = "&".join(f"{k}={params_to_sign[k]}" for k in sorted(params_to_sign))
        to_sign += self.api_secret
        signature = hashlib.sha1(to_sign.encode("utf-8")).hexdigest()

        # Read file bytes
        content = await file.read()
        files = {"file": (file.filename, content, file.content_type)}
        data = {
            "api_key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
        }
        if upload_preset:
            data["upload_preset"] = upload_preset

        url = f"https://api.cloudinary.com/v1_1/{self.cloud_name}/image/upload"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, data=data, files=files)
            r.raise_for_status()
            payload = r.json()
            return {"public_id": payload.get("public_id", ""), "url": payload.get("secure_url", payload.get("url", ""))}

    # PUBLIC_INTERFACE
    async def create_mashup(self, selfie_public_id: str, character_name: str) -> Dict[str, str]:
        """
        Create a themed mashup using Cloudinary transformation.
        This demo applies a simple overlay text and retro style; production could use generative background or templates.
        Returns dict with public_id and url.
        """
        if not (self.cloud_name and self.api_key and self.api_secret):
            # Dev fallback: simple placeholder
            fake_id = f"mashups/{character_name}-{selfie_public_id}".replace("/", "_")
            return {"public_id": fake_id, "url": f"https://example.com/{fake_id}.jpg"}

        import time
        import hashlib
        import httpx

        # We'll use explicit API to generate a derived image via incoming transformation
        timestamp = str(int(time.time()))
        # Transformation: add neon frame effect + overlay text (character name) in retro font if available
        transformation = "e_vibrance:60,e_colorize:20,co_rgb:ff77ff/l_text:Arial_60_bold:" + httpx.utils.quote(character_name.replace(" ", "%20")) + ",co_rgb:ffff00,g_south,y_30/x_0"

        params_to_sign = {
            "public_id": selfie_public_id,
            "timestamp": timestamp,
            "transformation": transformation,
        }
        to_sign = "&".join(f"{k}={params_to_sign[k]}" for k in sorted(params_to_sign))
        to_sign += self.api_secret
        signature = hashlib.sha1(to_sign.encode("utf-8")).hexdigest()

        url = f"https://api.cloudinary.com/v1_1/{self.cloud_name}/image/explicit"
        data = {
            "api_key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "public_id": selfie_public_id,
            "type": "upload",
            "eager": transformation,  # request derived
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, data=data)
            r.raise_for_status()
            payload = r.json()
            # Fetch derived or construct transformation URL
            # Build a URL manually for the transformation if not present
            base = f"https://res.cloudinary.com/{self.cloud_name}/image/upload"
            transformed_url = f"{base}/{transformation}/{selfie_public_id}.jpg"
            out_id = payload.get("public_id", selfie_public_id)
            return {"public_id": out_id, "url": transformed_url}

class TwilioService:
    """Service wrapper to send SMS via Twilio."""

    def __init__(self):
        self.sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "")

    # PUBLIC_INTERFACE
    def send_sms(self, to_e164: str, body: str) -> str:
        """
        Send an SMS message using Twilio REST API.
        Returns message SID on success. If env not configured, returns 'dev-sid'.
        """
        if not (self.sid and self.token and self.from_number):
            return "dev-sid"

        import httpx

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json"
        auth = (self.sid, self.token)
        data = {
            "From": self.from_number,
            "To": to_e164,
            "Body": body,
        }
        r = httpx.post(url, data=data, auth=auth, timeout=20)
        r.raise_for_status()
        payload = r.json()
        return payload.get("sid", "unknown-sid")

# ----- Core Quiz logic -----

# PUBLIC_INTERFACE
def aggregate_traits(answers: List[QuizAnswer]) -> Dict[str, int]:
    """Aggregate trait scores from user answers."""
    scores: Dict[str, int] = {}
    for ans in answers:
        traits = ANSWER_TO_TRAITS.get(ans.answer_id, [])
        for t in traits:
            scores[t] = scores.get(t, 0) + 1
    return scores

# PUBLIC_INTERFACE
def choose_character(scores: Dict[str, int]) -> CharacterTraits:
    """Choose best matching character based on overlapping traits."""
    best: Optional[CharacterTraits] = None
    best_score = -1
    for c in CHARACTER_CATALOG:
        score = sum(scores.get(t, 0) for t in c.traits)
        if score > best_score:
            best_score = score
            best = c
    return best or CHARACTER_CATALOG[0]

# ----- FastAPI app -----

openai_service = OpenAIService()
cloudinary_service = CloudinaryService()
twilio_service = TwilioService()

tags_metadata = [
    {"name": "health", "description": "Service health and info endpoints"},
    {"name": "quiz", "description": "Quiz submission and character matching"},
    {"name": "characters", "description": "Character traits and catalog"},
    {"name": "media", "description": "Selfie upload and mashup image"},
    {"name": "share", "description": "Social and SMS sharing"},
]

app = FastAPI(
    title="Star Wars Retro Character Generator API",
    description="APIs for quiz, character matching, witty write-ups, selfie/mashup handling, and sharing.",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"], summary="Health Check", description="Simple health check endpoint.")
def health_check():
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.get("/characters/traits", response_model=List[CharacterTraits], tags=["characters"], summary="List character traits", description="Get the catalog of Star Wars characters and their associated traits.")
def get_characters_traits():
    """Return static character catalog for front-end visualization."""
    return CHARACTER_CATALOG

# PUBLIC_INTERFACE
@app.post("/quiz/submit", response_model=MatchResult, tags=["quiz"], summary="Submit quiz", description="Submit 5-question quiz answers, receive character match with witty write-up and trait score breakdown.")
def submit_quiz(payload: QuizSubmission):
    """
    Process quiz answers, compute trait scores, select best character match,
    and generate witty write-up via OpenAI (with a graceful fallback).
    """
    if not payload.answers or len(payload.answers) < 3:
        raise HTTPException(status_code=400, detail="At least 3 answers required for a meaningful match.")

    scores = aggregate_traits(payload.answers)
    character = choose_character(scores)
    writeup = openai_service.witty_writeup(character, scores)

    return MatchResult(character=character, witty_writeup=writeup, score_breakdown=scores, selfie_temp_id=payload.selfie_temp_id)

# PUBLIC_INTERFACE
@app.post("/selfie/upload", tags=["media"], summary="Upload selfie", description="Upload a selfie file; returns a temporary id and URL. Uses Cloudinary when configured.")
async def upload_selfie(file: UploadFile = File(...)):
    """
    Accept a multipart/form-data file upload for selfie.
    When Cloudinary env vars are configured, the file is uploaded there.
    Otherwise, returns a mock URL for development.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type; image required.")

    try:
        result = await cloudinary_service.upload_selfie(file)
        # We treat public_id as temp id for continuity
        return {"selfie_temp_id": result["public_id"], "url": result["url"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# PUBLIC_INTERFACE
@app.post("/image/mashup", response_model=MashupResult, tags=["media"], summary="Create mashup", description="Create an 80s-retro mashup image based on the matched character and uploaded selfie.")
async def image_mashup(req: MashupRequest):
    """
    Create a retro-styled mashup image using Cloudinary transformations.
    """
    if not req.selfie_temp_id:
        raise HTTPException(status_code=400, detail="selfie_temp_id is required")
    if not req.character_name:
        raise HTTPException(status_code=400, detail="character_name is required")

    try:
        result = await cloudinary_service.create_mashup(req.selfie_temp_id, req.character_name)
        return MashupResult(mashup_url=result["url"], public_id=result["public_id"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mashup failed: {str(e)}")

# PUBLIC_INTERFACE
@app.post("/share", response_model=ShareResponse, tags=["share"], summary="Share result", description="Create share links for Facebook/Twitter, or send SMS via Twilio when channel=sms.")
def share(req: ShareRequest):
    """
    Share via:
    - facebook: returns a share URL with query params
    - twitter: returns an intent URL with prefilled text
    - sms: sends SMS via Twilio and returns SID
    """
    channel = req.channel.lower().strip()
    message = req.message or "Behold my 80s Star Wars glam mashup!"
    image_url = str(req.image_url)
    page_url = str(req.page_url) if req.page_url else image_url

    if channel == "facebook":
        # Facebook share dialog URL
        share_link = f"https://www.facebook.com/sharer/sharer.php?u={page_url}"
        return ShareResponse(status="ok", link=share_link)
    elif channel == "twitter":
        import urllib.parse
        text = urllib.parse.quote_plus(message)
        url = urllib.parse.quote_plus(page_url)
        share_link = f"https://twitter.com/intent/tweet?text={text}&url={url}"
        return ShareResponse(status="ok", link=share_link)
    elif channel == "sms":
        if not req.target_phone_e164:
            raise HTTPException(status_code=400, detail="target_phone_e164 is required for SMS")
        body = f"{message}\n{page_url}"
        try:
            sid = twilio_service.send_sms(req.target_phone_e164, body)
            return ShareResponse(status="ok", sid=sid)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SMS failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported channel. Use facebook, twitter, or sms.")
