"""
GeoMeta - aTem AI Backend
FastAPI tabanli, Groq API'sini proxy'leyen backend.
Groq API Key ASLA Android uygulamasinda bulunmaz; sadece bu backend'de
ortam degiskeni olarak tutulur (Render -> Environment -> GROQ_API_KEY).

aTem Merkezi Uygulama Katalogu entegrasyonu:
- CATALOG_URL: Meta uygulama katalogunun JSON adresi (Render Environment'ta
  manuel olarak ayarlanir, kodda asla sabit yazilmaz).
- APP_ID: Bu uygulamanin (GeoMeta) katalogdaki kimligi (Render Environment'ta
  manuel olarak ayarlanir, kodda asla sabit yazilmaz).
Katalog bellekte ~10 dakika TTL ile cache'lenir; ETag/Last-Modified destekleniyorsa
kosullu istekle (304) gereksiz veri transferi engellenir.

Calistirma (yerel):
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Render Deploy:
    Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
    Environment:   GROQ_API_KEY=xxxx, CATALOG_URL=xxxx, APP_ID=xxxx
"""

import json
import os
import re
import time
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- aTem Merkezi Uygulama Katalogu ayarlari ---
# Bu iki deger SADECE ortam degiskeninden okunur, koda asla sabit yazilmaz.
CATALOG_URL = os.environ.get("CATALOG_URL", "")
APP_ID = os.environ.get("APP_ID", "")

CATALOG_TTL_SECONDS = 600  # ~10 dakika

# Projeden tespit edilen kimlik bilgileri (uygulamanin kendi kaynak kodundan
# okunur; bir sir/secret degildir, applicationId zaten APK icinde acik haldedir).
# APP_ID env degiskeni bos ya da katalogla birebir eslesmezse, ikincil eslesme
# kriteri olarak kullanilir.
SELF_APP_NAME = "GeoMeta"
SELF_PACKAGE_NAME = "com.geometa.app"

ATEM_BASE_INSTRUCTIONS = (
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

PLAY_STORE_URL_REGEX = re.compile(r"https?://play\.google\.com/[^\s\)\]\"']+")

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

# Katalog cache (bellek-ici)
_catalog_cache: Dict[str, object] = {
    "apps": [],
    "fetched_at": 0.0,
    "etag": None,
    "last_modified": None,
}


def check_rate_limit(client_ip: str):
    now = time.time()
    window_start = now - 60
    history = _request_log.get(client_ip, [])
    history = [t for t in history if t > window_start]
    if len(history) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Çok fazla istek gönderildi. Lütfen bir dakika sonra tekrar deneyin.")
    history.append(now)
    _request_log[client_ip] = history


async def get_catalog() -> List[dict]:
    """Meta uygulama katalogunu (~10 dk TTL) bellekte cache'ler.
    ETag/Last-Modified destekleniyorsa kosullu istek (304) ile gereksiz
    veri transferini engeller. CATALOG_URL bos ise veya istek basarisiz
    olursa mevcut cache (varsa) ile devam edilir."""
    now = time.time()
    if _catalog_cache["apps"] and (now - float(_catalog_cache["fetched_at"])) < CATALOG_TTL_SECONDS:
        return _catalog_cache["apps"]  # type: ignore[return-value]

    if not CATALOG_URL:
        return _catalog_cache["apps"]  # type: ignore[return-value]

    headers = {}
    if _catalog_cache["etag"]:
        headers["If-None-Match"] = str(_catalog_cache["etag"])
    if _catalog_cache["last_modified"]:
        headers["If-Modified-Since"] = str(_catalog_cache["last_modified"])

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(CATALOG_URL, headers=headers)
    except httpx.RequestError:
        _catalog_cache["fetched_at"] = now
        return _catalog_cache["apps"]  # type: ignore[return-value]

    if response.status_code == 304:
        _catalog_cache["fetched_at"] = now
        return _catalog_cache["apps"]  # type: ignore[return-value]

    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError:
            _catalog_cache["fetched_at"] = now
            return _catalog_cache["apps"]  # type: ignore[return-value]

        apps = payload.get("apps", payload) if isinstance(payload, dict) else payload
        if isinstance(apps, list):
            _catalog_cache["apps"] = apps
        _catalog_cache["fetched_at"] = now
        if response.headers.get("ETag"):
            _catalog_cache["etag"] = response.headers.get("ETag")
        if response.headers.get("Last-Modified"):
            _catalog_cache["last_modified"] = response.headers.get("Last-Modified")
    else:
        _catalog_cache["fetched_at"] = now

    return _catalog_cache["apps"]  # type: ignore[return-value]


def find_self_app(apps: List[dict]) -> Optional[dict]:
    """Kendi uygulama kaydini once APP_ID, sonra package_name uzerinden bulur."""
    if not apps:
        return None

    normalized_app_id = APP_ID.strip().lower() if APP_ID else ""
    if normalized_app_id:
        for entry in apps:
            entry_id = str(entry.get("app_id", "")).strip().lower()
            if entry_id == normalized_app_id:
                return entry

    normalized_package = SELF_PACKAGE_NAME.strip().lower()
    for entry in apps:
        entry_package = str(entry.get("package_name", "")).strip().lower()
        if entry_package and entry_package == normalized_package:
            return entry

    return None


def build_catalog_block(apps: List[dict], self_app: Optional[dict]) -> str:
    lines = []
    self_id = str(self_app.get("app_id", "")).strip().lower() if self_app else ""
    for entry in apps:
        entry_id = str(entry.get("app_id", "")).strip().lower()
        if self_id and entry_id == self_id:
            continue  # kendi uygulamasini "alternatif oneri" olarak listelemeye gerek yok
        name = entry.get("name", "")
        desc = entry.get("description") or entry.get("short_description") or ""
        lines.append(f"- app_id={entry.get('app_id', '')} | isim={name} | aciklama={desc}")
    return "\n".join(lines) if lines else "(katalogda baska uygulama bulunamadi ya da katalog su an erisilemiyor)"


def build_system_prompt(apps: List[dict], self_app: Optional[dict]) -> str:
    if self_app:
        self_name = self_app.get("name", SELF_APP_NAME)
        self_desc = self_app.get("description") or self_app.get("short_description") or ""
    else:
        self_name = SELF_APP_NAME
        self_desc = ""

    catalog_block = build_catalog_block(apps, self_app)

    identity = f"Sen {self_name} uygulamasinin yapay zeka asistani aTem'sin."
    if self_desc:
        identity += f" {self_name} hakkinda: {self_desc}"

    catalog_policy = (
        "Kullanici bu uygulamada olmayan bir ozellik ya da baska bir Meta "
        "uygulamasi ile ilgili bir sey isterse, SADECE asagidaki 'Meta Uygulama "
        "Katalogu' listesinde yer alan uygulamalar arasindan en alakali "
        "olan(lar)ini oner. Listede olmayan hicbir uygulamayi onerme, uydurma, "
        "adini degistirme ya da hatirliyormus gibi davranma. Katalogda alakali "
        "bir uygulama yoksa oneri yapma ve bunu kullaniciya acikca belirt.\n\n"
        f"Meta Uygulama Katalogu:\n{catalog_block}"
    )

    output_schema = (
        "Cevabini SADECE gecerli bir JSON nesnesi olarak, asagidaki semaya "
        "birebir uyacak sekilde ver. JSON disinda hicbir metin, aciklama ya da "
        "kod bloku isareti (```) ekleme:\n"
        '{"reply": "kullaniciya gosterilecek serbest metin cevap (icinde asla '
        'ciplak URL veya link olmamali)", "recommendations": '
        '[{"app_id": "katalogdaki app_id", "name": "uygulama adi", '
        '"description": "kisa aciklama", "play_store_url": "katalogdaki play_store_url"}]}\n'
        "recommendations bos bir dizi [] olabilir (oneri yoksa boyle birak). "
        "play_store_url alanini SADECE yukaridaki katalogdan oldugu gibi kopyala; "
        "kendi URL'ini asla uretme, tahmin etme ya da degistirme. app_id degeri "
        "katalogda gecen degerle birebir ayni olmali."
    )

    return "\n\n".join([identity, ATEM_BASE_INSTRUCTIONS, catalog_policy, output_schema])


def _catalog_lookup(apps: List[dict]) -> Dict[str, dict]:
    return {str(a.get("app_id", "")).strip().lower(): a for a in apps if a.get("app_id")}


def _to_recommendation(entry: dict) -> dict:
    return {
        "app_id": entry.get("app_id", ""),
        "name": entry.get("name", ""),
        "description": entry.get("description") or entry.get("short_description") or "",
        "play_store_url": entry.get("play_store_url", ""),
    }


def fallback_parse(content: str, apps: List[dict]) -> Dict[str, object]:
    """Model, istenen JSON semasina uymazsa: metindeki play.google.com
    linklerini regex ile yakalayip katalogla eslestirerek recommendations
    dizisine tasir, reply metninden ciplak linkleri temizler."""
    urls = PLAY_STORE_URL_REGEX.findall(content)
    reply = PLAY_STORE_URL_REGEX.sub("", content).strip()

    recommendations: List[dict] = []
    seen_ids = set()
    for url in urls:
        match = next(
            (a for a in apps if a.get("play_store_url") and (a["play_store_url"] in url or url in a["play_store_url"])),
            None,
        )
        if match:
            app_id = str(match.get("app_id", ""))
            if app_id and app_id not in seen_ids:
                recommendations.append(_to_recommendation(match))
                seen_ids.add(app_id)

    return {"reply": reply, "recommendations": recommendations}


def parse_model_output(content: str, apps: List[dict]) -> Dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return fallback_parse(content, apps)

    if not isinstance(parsed, dict) or "reply" not in parsed:
        return fallback_parse(content, apps)

    reply = str(parsed.get("reply", "")).strip()
    reply = PLAY_STORE_URL_REGEX.sub("", reply).strip()

    lookup = _catalog_lookup(apps)
    recommendations: List[dict] = []
    raw_recommendations = parsed.get("recommendations")
    if isinstance(raw_recommendations, list):
        seen_ids = set()
        for raw in raw_recommendations:
            if not isinstance(raw, dict):
                continue
            raw_id = str(raw.get("app_id", "")).strip().lower()
            catalog_entry = lookup.get(raw_id)
            if catalog_entry and raw_id not in seen_ids:
                # play_store_url ve diger alanlar HER ZAMAN katalogdan alinir;
                # modelin uydurabilecegi degerlere guvenilmez.
                recommendations.append(_to_recommendation(catalog_entry))
                seen_ids.add(raw_id)

    return {"reply": reply, "recommendations": recommendations}


class ChatRequest(BaseModel):
    message: str


class Recommendation(BaseModel):
    app_id: str
    name: str
    description: str = ""
    play_store_url: str = ""


class ChatResponse(BaseModel):
    answer: str  # geriye donuk uyumluluk (eski istemciler icin `reply` ile ayni deger)
    reply: str
    recommendations: List[Recommendation] = []


@app.get("/")
def root():
    return {"status": "ok", "service": "GeoMeta aTem Backend"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "catalog_configured": bool(CATALOG_URL),
        "app_id_configured": bool(APP_ID),
        "catalog_cached_apps": len(_catalog_cache["apps"]),  # type: ignore[arg-type]
    }


@app.post("/api/ai/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")

    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Sunucu yapılandırması eksik (GROQ_API_KEY).")

    apps = await get_catalog()
    self_app = find_self_app(apps)
    system_prompt = build_system_prompt(apps, self_app)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
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
        raw_content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Yapay zeka servisinden geçersiz yanıt alındı.")

    parsed = parse_model_output(raw_content, apps)
    reply_text = str(parsed.get("reply") or "").strip() or raw_content
    recommendations = parsed.get("recommendations") or []

    return ChatResponse(answer=reply_text, reply=reply_text, recommendations=recommendations)  # type: ignore[arg-type]
