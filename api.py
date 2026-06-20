
import os
import uuid
import shutil
import cv2
import numpy as np
import easyocr
import pandas as pd
from rapidfuzz import fuzz
import re
import unicodedata
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = "dataset/medicaments.csv"
CSV_SEP      = ";"
CSV_ENCODING = "utf-8"
SEUIL_REJET  = 50
TMP_DIR      = "tmp_uploads"
os.makedirs(TMP_DIR, exist_ok=True)

COL_CODE        = "CODE"
COL_NOM_FR      = "NOM"
COL_NOM_AR      = "الاسم"
COL_MOLECULE_FR = "DCI1"
COL_MOLECULE_AR = "معلومات الدواء"
COL_DOSAGE      = "DOSAGE1"
COL_UNITE_FR    = "UNITE_DOSAGE1"
COL_UNITE_AR    = "وحدة الجرعة"
COL_FORME_FR    = "FORME"
COL_FORME_AR    = "صيغة"
COL_COND_FR     = "PRESENTATION"
COL_COND_AR     = "شكل الدواء"

W_NAME   = 0.70
W_DOSAGE = 0.20
W_FORME  = 0.10


FORMS_KW = [
    "COMPRIME", "COMPRIMES", "COMPRIME PELLICULE", "COMPRIME ENROBE",
    "COMPRIME SECABLE", "COMPRIME EFFERVESCENT",
    "GELULE", "GELULES",
    "SIROP", "FLACON", "POMMADE", "SOLUTION", "SOLUTION BUVABLE",
    "CAPSULE", "SACHET", "INJECTABLE", "SUSPENSION",
    "PATCH", "CREME", "GEL", "SPRAY", "POUDRE",
    "POMMADE OPHTALMIQUE", "COLLYRE", "SUPPOSITOIRE",
    "LYOPHILISAT", "EMULSION", "LOTION", "MOUSSE",
    "GRANULES", "GRANULE", "AMPOULE",
]
FORMS_KW.sort(key=len, reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT CSV (au démarrage, une seule fois)
# ─────────────────────────────────────────────────────────────────────────────
df = None
for enc in [CSV_ENCODING, "windows-1256", "latin-1"]:
    try:
        df = pd.read_csv(CSV_PATH, sep=CSV_SEP, encoding=enc, dtype=str)
        df = df.fillna("")
        df.columns = df.columns.str.strip()
        for col in df.columns:
            df[col] = df[col].str.strip()
        print(f"✔ CSV chargé ({len(df)} médicaments) — encodage : {enc}")
        break
    except Exception:
        continue

if df is None:
    raise RuntimeError("❌ Impossible de charger le CSV")

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT OCR (au démarrage, une seule fois — évite de recharger à chaque appel)
# ─────────────────────────────────────────────────────────────────────────────
print("✔ Chargement des modèles OCR…")
reader_fr = easyocr.Reader(["fr", "en"], gpu=False)
reader_ar = easyocr.Reader(["ar"],       gpu=False)
print("✔ Modèles OCR prêts")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_latin(text: str) -> str:
    text = str(text).translate(ARABIC_DIGITS)
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_arabic(text: str) -> str:
    text = str(text).translate(ARABIC_DIGITS)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F 0-9]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def safe(row, col: str) -> str:
    val = row.get(col, "")
    return "" if str(val).strip() in ("nan", "NaN", "") else str(val).strip()

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE OCR + MATCHING
# ─────────────────────────────────────────────────────────────────────────────
def run_ocr(reader, paths, threshold=0.35):
    out = []
    for p in paths:
        for (_, text, conf) in reader.readtext(p):
            if conf >= threshold:
                out.append(text)
    return " ".join(out)

def score_name_latin(ocr: str, name: str) -> float:
    if not name: return 0.0
    n = normalize_latin(name)
    if not n: return 0.0
    return float(max(fuzz.token_set_ratio(ocr, n),
                     fuzz.partial_ratio(ocr, n),
                     fuzz.token_sort_ratio(ocr, n)))

def score_name_arabic(ocr_ar: str, name_ar: str) -> float:
    if not name_ar or not ocr_ar: return 0.0
    n = normalize_arabic(name_ar)
    if not n: return 0.0
    return float(max(fuzz.token_set_ratio(ocr_ar, n),
                     fuzz.partial_ratio(ocr_ar, n)))

def score_dosage(ocr_nums, dosage_str: str) -> float:
    if not dosage_str or not ocr_nums: return 0.0
    d = normalize_latin(dosage_str)
    best = 0.0
    for n in ocr_nums:
        if len(n) == 1 and n.isdigit():
            s = float(fuzz.ratio(n, d))
        else:
            s = float(fuzz.partial_ratio(n, d))
        if s > best:
            best = s
    return best

def score_forme(ocr_full: str, ocr_forms_kw: list, forme_str: str) -> float:
    """
    FIX v2 — double stratégie :
      1. Comparaison directe entre le texte OCR complet et la forme du CSV
         → fonctionne même si aucun mot-clé n'a été extrait
      2. Comparaison via les mots-clés détectés dans l'OCR
         → donne un bonus si un mot-clé exact est présent
    """
    if not forme_str:
        return 0.0
    f = normalize_latin(forme_str)
    if not f:
        return 0.0

    score_direct = float(fuzz.partial_ratio(ocr_full, f))

    score_kw = 0.0
    if ocr_forms_kw:
        score_kw = float(max(fuzz.partial_ratio(kw, f) for kw in ocr_forms_kw))

    return max(score_direct, score_kw)

def preprocess(img_path: str):
    img  = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image illisible")
    h, _ = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, 1200 / h)
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    if np.mean(gray) < 100:
        gray = cv2.bitwise_not(gray)
    _, otsu  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 15, 4)
    gray_bin = cv2.bitwise_or(otsu, adaptive)
    base     = img_path.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    path_g   = base + "_gray.jpg"
    path_b   = base + "_bin.jpg"
    cv2.imwrite(path_g, gray)
    cv2.imwrite(path_b, gray_bin)
    return [path_g, path_b]

def run_pipeline(img_path: str) -> dict:
    """Lance le pipeline complet sur une image et retourne le résultat."""
    
    processed_imgs = preprocess(img_path)

    
    raw_latin  = run_ocr(reader_fr, processed_imgs, threshold=0.35)
    raw_arabic = run_ocr(reader_ar, processed_imgs, threshold=0.40)
    ocr_latin  = normalize_latin(raw_latin)
    ocr_arabic = normalize_arabic(raw_arabic)

    ocr_words   = ocr_latin.split()
    ocr_numbers = [w for w in ocr_words if any(c.isdigit() for c in w)]

    
    ocr_forms_detected = [kw for kw in FORMS_KW if kw in ocr_latin]

    # Matching
    results = []
    for _, row in df.iterrows():
        nom_fr   = safe(row, COL_NOM_FR)
        nom_ar   = safe(row, COL_NOM_AR)
        mol_fr   = safe(row, COL_MOLECULE_FR)
        mol_ar   = safe(row, COL_MOLECULE_AR)
        dosage   = safe(row, COL_DOSAGE)
        unite_fr = safe(row, COL_UNITE_FR)
        unite_ar = safe(row, COL_UNITE_AR)
        forme_fr = safe(row, COL_FORME_FR)
        forme_ar = safe(row, COL_FORME_AR)
        cond_fr  = safe(row, COL_COND_FR)
        cond_ar  = safe(row, COL_COND_AR)
        code     = safe(row, COL_CODE)

        sn = max(score_name_latin(ocr_latin,  nom_fr),
                 score_name_latin(ocr_latin,  mol_fr),
                 score_name_arabic(ocr_arabic, nom_ar),
                 score_name_arabic(ocr_arabic, mol_ar))
        sd    = score_dosage(ocr_numbers, dosage)

        sf    = score_forme(ocr_latin, ocr_forms_detected, forme_fr)
        final = W_NAME * sn + W_DOSAGE * sd + W_FORME * sf

        results.append({
            "code":     code,
            "nom_fr":   nom_fr,
            "nom_ar":   nom_ar,
            "mol_fr":   mol_fr,
            "mol_ar":   mol_ar,
            "dosage":   f"{dosage} {unite_fr}".strip() if dosage else "",
            "dosage_ar":f"{dosage} {unite_ar}".strip() if dosage and unite_ar else "",
            "forme_fr": forme_fr,
            "forme_ar": forme_ar,
            "cond_fr":  cond_fr,
            "cond_ar":  cond_ar,
            "score_nom":    round(sn, 2),
            "score_dosage": round(sd, 2),
            "score_forme":  round(sf, 2),
            "score_final":  round(final, 2),
        })

    results.sort(key=lambda x: x["score_final"], reverse=True)

    # Nettoyage fichiers temporaires
    for p in processed_imgs:
        try: os.remove(p)
        except: pass

    return results

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="API Reconnaissance Médicaments",
    description="Identifie un médicament à partir d'une image de sa boîte (FR + AR)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT PRINCIPAL : POST /predict
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit une image et retourne le médicament identifié.

    Paramètre :
      - file : image JPG ou PNG,... de la boîte de médicament

    Retourne :
      - reconnu     : true si le score >= seuil
      - confiance   : "haute" / "bonne" / "moyenne" / "rejeté"
      - medicament  : fiche complète (nom, DCI, dosage, forme, présentation)
      - scores      : détail des scores (nom, dosage, forme, final)
    """
    filename = file.filename or ""
    ext = filename.lower().split(".")[-1]
    content_type = (file.content_type or "").lower()
    ext_ok  = ext in ("jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif")
    type_ok = any(t in content_type for t in ("image", "octet-stream", "jpeg", "png", "webp"))
    if not ext_ok and not type_ok:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : '{ext}'. Envoyer une image JPG, PNG ou WEBP."
        )

    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.jpg")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        results = run_pipeline(tmp_path)
        best    = results[0]

        score = best["score_final"]
        if   score >= 75: confiance = "haute"
        elif score >= 60: confiance = "bonne"
        elif score >= SEUIL_REJET: confiance = "moyenne"
        else:             confiance = "rejeté"

        reconnu = score >= SEUIL_REJET

        response = {
            "reconnu":   reconnu,
            "confiance": confiance,
            "scores": {
                "nom":    best["score_nom"],
                "dosage": best["score_dosage"],
                "forme":  best["score_forme"],
                "final":  best["score_final"],
            },
        }

        if reconnu:
            response["medicament"] = {
                "nom_fr":           best["nom_fr"],
                "nom_ar":           best["nom_ar"],
                "dci_fr":           best["mol_fr"],
                "dci_ar":           best["mol_ar"],
                "dosage_fr":        best["dosage"],
                "dosage_ar":        best["dosage_ar"],
                "forme_fr":         best["forme_fr"],
                "forme_ar":         best["forme_ar"],
                "presentation_fr":  best["cond_fr"],
                "presentation_ar":  best["cond_ar"],
            }
        else:
            response["message"] = {
               "attention": "Médicament non reconnu",
               "details": [
                         "Veuillez reprendre la photo en vous assurant que :",
                         "la boîte est bien cadrée",
                         "la lumière est suffisante",
                         "le texte est net et lisible",
                         "Si le problème persiste, ce médicament est peut-être absent de la base de données"
    ]
}

            if score >= 35:
                response["suggestion"] = {
                    "nom":   best["nom_fr"],
                    "score": best["score_final"],
                    "note":  "Non confirmé — vérifier manuellement",
                }

        return response

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT SANTÉ : GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Vérifie que l'API est opérationnelle."""
    return {
        "status":      "ok",
        "medicaments": len(df),
        "ocr_fr":      "chargé",
        "ocr_ar":      "chargé",
    }

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT INFOS : GET /
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "api":     "Reconnaissance Médicaments",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Identifier un médicament depuis une image",
            "GET  /health":  "Vérifier l'état de l'API",
            "GET  /docs":    "Documentation interactive (Swagger)",
        }
    }
