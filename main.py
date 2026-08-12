"""
GeoMeta - aTem AI Backend
FastAPI tabanli, Groq API'sini proxy'leyen backend.
Groq API Key ASLA Android uygulamasinda bulunmaz; sadece bu backend'de
ortam degiskeni olarak tutulur (Render -> Environment -> GROQ_API_KEY).

Calistirma (yerel):
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Render Deploy:
    Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
    Environment:   GROQ_API_KEY=xxxx
"""

import os
import time
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ATEM_SYSTEM_PROMPT = (
    "Sen GeoMeta uygulamasinin yapay zeka asistani aTem'sin. "
    "GPS, harita, alan olcumu, mesafe olcumu, arazi olcumu, metrekare, donum, "
    "dekar, hektar, acre, kilometre ve AR olcum konularinda kullaniciya "
    "anlasilir bilgiler ver. Cevaplarini Turkce ver. Matematiksel donusumlerde "
    "dogru sonuc vermeye dikkat et. GPS olcumlerinin cihaz, sinyal, cevre ve "
    "kosullara bagli olarak hata icerebilecegini acikla. Gercek bir olcum "
    "sonucu elde etmediysen kesin fiziksel olcum sonucu varmis gibi davranma. "
    "Kullanicinin cihazindaki GPS veya kamera verilerine gercekten erisimin "
    "yoksa bunu acikca belirt. Kisa sorulara kisa cevap ver. Detay isteyen "
    "kullaniciya adim adim aciklama yap. Bilmedigin bilgiyi uydurma. Senin "
    "adin aTem."
)

app = FastAPI(title="GeoMeta aTem Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basit bellek-ici rate limit (IP basina dakikada N istek)
RATE_LIMIT_PER_MINUTE = 20
_request_log: Dict[str, list] = {}


def check_rate_limit(client_ip: str):
    now = time.time()
    window_start = now - 60
    history = _request_log.get(client_ip, [])
    history = [t for t in history if t > window_start]
    if len(history) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Çok fazla istek gönderildi. Lütfen bir dakika sonra tekrar deneyin.")
    history.append(now)
    _request_log[client_ip] = history


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {"status": "ok", "service": "GeoMeta aTem Backend"}


@app.get("/health")
def health():
    return {"status": "healthy", "groq_configured": bool(GROQ_API_KEY)}


@app.post("/api/ai/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")

    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Sunucu yapılandırması eksik (GROQ_API_KEY).")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": ATEM_SYSTEM_PROMPT},
            {"role": "user", "content": payload.message.strip()},
        ],
        "temperature": 0.4,
        "max_tokens": 800,
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(GROQ_URL, headers=headers, json=body)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Yapay zeka servisinden yanıt alınamadı (zaman aşımı).")
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Yapay zeka servisine ulaşılamadı.")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groq API hatası: {response.status_code}")

    data = response.json()
    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Yapay zeka servisinden geçersiz yanıt alındı.")

    return ChatResponse(answer=answer)
