# Star Wars Retro Character Generator - Backend (FastAPI)

Provides REST APIs for:
- POST /quiz/submit — process 5-question quiz, compute trait scores, match to a Star Wars character, and generate a witty retro write-up (OpenAI optional)
- GET /characters/traits — list character catalog and traits
- POST /selfie/upload — upload selfie (Cloudinary optional; dev fallback returns mock URLs)
- POST /image/mashup — create a neon 80s mashup image (Cloudinary transform; dev fallback returns mock URL)
- POST /share — generate social share links (FB/Twitter) or send SMS via Twilio

## Run locally

1. Create and fill `.env` (see `.env.example`).
2. Install Python dependencies:
   pip install -r requirements.txt
3. Run the API:
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
4. Docs: http://localhost:8000/docs

## Environment variables

See `.env.example` for all supported variables:
- OPENAI_API_KEY: OpenAI API key for witty write-ups (optional, fallback to templated text if absent)
- OPENAI_MODEL: Defaults to `gpt-4o-mini`
- CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET: Cloudinary credentials (optional for dev)
- CLOUDINARY_UPLOAD_PRESET: Optional, if using unsigned uploads
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER: For SMS (optional; dev fallback SID if absent)
- CORS_ALLOW_ORIGINS: Comma-separated list of allowed origins (default "*")

## Notes

- This service is demo-friendly: if external integrations are not configured, it falls back gracefully so the app remains functional.
- In production, secure environment variables using your secrets manager and restrict CORS.
