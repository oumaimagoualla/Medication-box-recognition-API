"""
Script d'évaluation du pipeline de reconnaissance de médicaments — DEV SET
==========================================================================
- Calcule l'accuracy Top-1
- Affiche les cas d'erreur détaillés
- Génère un rapport CSV des résultats
- Génère 4 graphiques sauvegardés en PNG

Prérequis :
  1. L'API doit tourner : python -m uvicorn api:app --port 8000
  2. Créer dataset/dev/annotations.csv avec le format :
       image;nom_correct
       oradexon.jpg;ORADEXON
       dolamine.jpg;DOLAMINE

pip install requests pandas matplotlib
"""

import os
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_URL        = "http://localhost:8000/predict"
ANNOTATIONS    = "dataset/annotations.csv"
IMAGES_DIR     = "dataset/images"
RAPPORT_CSV    = "dataset/rapport_evaluation.csv"
GRAPHIQUES_DIR = "dataset/graphiques"
SEP            = ";"

os.makedirs(GRAPHIQUES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def ligne(car="─", n=60):
    return car * n

def appeler_api(img_path: str) -> dict:
    with open(img_path, "rb") as f:
        r = requests.post(API_URL, files={"file": f}, timeout=300)
    if r.status_code != 200:
        return {"erreur": f"HTTP {r.status_code}"}
    return r.json()

def normaliser(texte: str) -> str:
    return str(texte).strip().upper()

# ─────────────────────────────────────────────────────────────────────────────
# VÉRIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
print(ligne("="))
print("  ÉVALUATION DU PIPELINE OCR – MÉDICAMENTS  [DEV SET]")
print(ligne("="))
print(f"\n  Jeu évalué : {ANNOTATIONS}")
print(f"  Images     : {IMAGES_DIR}")

try:
    health = requests.get("http://localhost:8000/health", timeout=5)
    info   = health.json()
    print(f"\n✔ API opérationnelle  ({info.get('medicaments', '?')} médicaments chargés)")
except Exception:
    print("\n❌ L'API ne répond pas. Lance d'abord :")
    print("   python -m uvicorn api:app --port 8000")
    exit(1)

if not os.path.exists(ANNOTATIONS):
    print(f"\n❌ Fichier annotations introuvable : {ANNOTATIONS}")
    exit(1)

annotations = []
with open(ANNOTATIONS, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=SEP)
    for row in reader:
        img_path = os.path.join(IMAGES_DIR, row["image"].strip())
        if os.path.exists(img_path):
            annotations.append({
                "image":       row["image"].strip(),
                "img_path":    img_path,
                "nom_correct": normaliser(row["nom_correct"]),
            })
        else:
            print(f"⚠️  Image introuvable, ignorée : {img_path}")

if not annotations:
    print("\n❌ Aucune image valide trouvée dans annotations.csv")
    exit(1)

print(f"✔ {len(annotations)} images à tester\n")

# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE D'ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print(ligne())
print(f"  {'IMAGE':<30} {'ATTENDU':<20} {'PRÉDIT':<20} {'SCORE':>6}  {'OK?'}")
print(ligne())

resultats  = []
top1_ok    = 0
rejetes    = 0
erreurs    = []

for i, ann in enumerate(annotations, 1):
    print(f"  [{i}/{len(annotations)}] {ann['image']:<28} ", end="", flush=True)

    try:
        reponse = appeler_api(ann["img_path"])
    except Exception as e:
        print(f"ERREUR API : {e}")
        resultats.append({**ann, "predit": "ERREUR", "score": 0,
                          "top1": False, "statut": "erreur_api",
                          "est_arabe": "_ar" in ann["image"].lower()})
        continue

    reconnu   = reponse.get("reconnu", False)
    scores    = reponse.get("scores", {})
    score     = scores.get("final", 0)
    predit    = normaliser(
        reponse.get("medicament", {}).get("nom_fr", "NON RECONNU")
    ) if reconnu else "NON RECONNU"

    correct_top1 = (predit == ann["nom_correct"])

    if not reconnu:
        rejetes += 1
        statut  = "rejeté"
        symbole = "⚠️ "
    elif correct_top1:
        top1_ok += 1
        statut  = "correct"
        symbole = "✅"
    else:
        statut  = "erreur"
        symbole = "❌"
        erreurs.append({
            "image":   ann["image"],
            "attendu": ann["nom_correct"],
            "predit":  predit,
            "score":   score,
        })

    print(f"{ann['nom_correct']:<20} {predit:<20} {score:5.1f}  {symbole}")

    resultats.append({
        "image":       ann["image"],
        "nom_correct": ann["nom_correct"],
        "predit":      predit,
        "score":       round(score, 2),
        "top1":        correct_top1,
        "reconnu":     reconnu,
        "statut":      statut,
        "est_arabe":   "_ar" in ann["image"].lower(),
    })

# ─────────────────────────────────────────────────────────────────────────────
# RÉSULTATS GLOBAUX
# ─────────────────────────────────────────────────────────────────────────────
total       = len(annotations)
acc_top1    = top1_ok / total * 100
pct_rejetes = rejetes / total * 100
pct_erreurs = (total - top1_ok - rejetes) / total * 100

print(f"\n{ligne('═')}")
print("  RÉSULTATS GLOBAUX")
print(ligne("═"))
print(f"\n  Images testées        : {total}")
print(f"  Correctes (Top-1)     : {top1_ok}  →  Accuracy Top-1 = {acc_top1:.1f}%")
print(f"  Rejetées (score<50)   : {rejetes}  →  {pct_rejetes:.1f}% des images")
print(f"  Erreurs (mauvais méd) : {total - top1_ok - rejetes}  →  Taux d'erreur = {pct_erreurs:.1f}%")

print(f"\n  {ligne(n=50)}")
if acc_top1 >= 80:
    print("  ✅ Performances BONNES  (>= 80%)")
elif acc_top1 >= 60:
    print("  ⚠️  Performances MOYENNES (60-80%)")
else:
    print("  ❌ Performances FAIBLES  (< 60%)")

if erreurs:
    print(f"\n{ligne('─')}")
    print("  DÉTAIL DES ERREURS")
    print(ligne("─"))
    for e in erreurs:
        print(f"\n  Image    : {e['image']}")
        print(f"  Attendu  : {e['attendu']}")
        print(f"  Prédit   : {e['predit']}  (score {e['score']:.1f})")

df_res = pd.DataFrame(resultats)
df_res.to_csv(RAPPORT_CSV, sep=SEP, index=False, encoding="utf-8")
print(f"\n{ligne('─')}")
print(f"  ✔ Rapport sauvegardé : {RAPPORT_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{ligne('─')}")
print("  GÉNÉRATION DES GRAPHIQUES…")

VERT   = "#1D9E75"
ROUGE  = "#D85A30"
ORANGE = "#FAC775"
BLEU   = "#378ADD"
GRIS   = "#888780"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ── Graphique 1 : Résultats globaux (camembert) ──────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))
nb_erreurs = total - top1_ok - rejetes - sum(1 for r in resultats if r["statut"] == "erreur_api")
nb_api_err = sum(1 for r in resultats if r["statut"] == "erreur_api")

labels, values, colors = [], [], []
if top1_ok   > 0: labels.append(f"Correctes\n({top1_ok})");   values.append(top1_ok);   colors.append(VERT)
if rejetes   > 0: labels.append(f"Rejetées\n({rejetes})");    values.append(rejetes);    colors.append(ORANGE)
if nb_erreurs> 0: labels.append(f"Erreurs\n({nb_erreurs})");  values.append(nb_erreurs); colors.append(ROUGE)
if nb_api_err> 0: labels.append(f"Timeout API\n({nb_api_err})"); values.append(nb_api_err); colors.append(GRIS)

wedges, texts, autotexts = ax1.pie(
    values, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")

ax1.set_title(f"Résultats globaux — Dev set — Accuracy : {acc_top1:.1f}%",
              fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
path1 = os.path.join(GRAPHIQUES_DIR, "1_resultats_globaux.png")
fig1.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  ✔ {path1}")

MIN_IMAGES_PAR_LANGUE = 3

res_fr_imgs = [r for r in resultats if not r["est_arabe"]]
res_ar_imgs = [r for r in resultats if r["est_arabe"]]

acc_fr = sum(1 for r in res_fr_imgs if r["statut"] == "correct") / max(len(res_fr_imgs), 1) * 100
acc_ar = sum(1 for r in res_ar_imgs if r["statut"] == "correct") / max(len(res_ar_imgs), 1) * 100

if len(res_fr_imgs) >= MIN_IMAGES_PAR_LANGUE and len(res_ar_imgs) >= MIN_IMAGES_PAR_LANGUE:
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bars = ax2.bar(["Images\nFrançaises", "Images\nArabes"],
                   [acc_fr, acc_ar], color=[BLEU, VERT],
                   width=0.4, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [acc_fr, acc_ar]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy : Images françaises vs arabes — Dev set", fontsize=13, fontweight="bold")
    ax2.axhline(y=acc_top1, color=GRIS, linestyle="--", linewidth=1,
                label=f"Moyenne globale ({acc_top1:.1f}%)")
    ax2.legend(fontsize=10)
    ax2.set_yticks(range(0, 111, 10))
    path2 = os.path.join(GRAPHIQUES_DIR, "2_fr_vs_ar.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✔ {path2}")
else:
    print(f"  ⚠️  Graphique FR vs AR ignoré (échantillon trop petit : "
          f"{len(res_fr_imgs)} FR / {len(res_ar_imgs)} AR, "
          f"minimum {MIN_IMAGES_PAR_LANGUE} par langue requis)")

fig3, ax3 = plt.subplots(figsize=(8, 5))
scores_corrects = [r["score"] for r in resultats if r["statut"] == "correct"]
scores_erreurs  = [r["score"] for r in resultats if r["statut"] == "erreur"]
scores_rejetes  = [r["score"] for r in resultats if r["statut"] == "rejeté" and r["score"] > 0]

if scores_corrects:
    ax3.hist(scores_corrects, bins=10, range=(0, 100), color=VERT,
             alpha=0.7, label=f"Correctes ({len(scores_corrects)})", edgecolor="white")
if scores_erreurs:
    ax3.hist(scores_erreurs,  bins=10, range=(0, 100), color=ROUGE,
             alpha=0.7, label=f"Erreurs ({len(scores_erreurs)})", edgecolor="white")
if scores_rejetes:
    ax3.hist(scores_rejetes,  bins=5,  range=(0, 50),  color=ORANGE,
             alpha=0.7, label=f"Rejetées ({len(scores_rejetes)})", edgecolor="white")

ax3.axvline(x=50, color="black", linestyle="--", linewidth=1.5, label="Seuil de rejet (50)")
ax3.set_xlabel("Score final (/100)", fontsize=11)
ax3.set_ylabel("Nombre d'images", fontsize=11)
ax3.set_title("Distribution des scores par catégorie — Dev set", fontsize=13, fontweight="bold")
ax3.legend(fontsize=10)
ax3.set_xlim(0, 105)
path3 = os.path.join(GRAPHIQUES_DIR, "3_distribution_scores.png")
fig3.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  ✔ {path3}")

fig4, ax4 = plt.subplots(figsize=(6, 5))
bars4 = ax4.bar(["Accuracy", "Taux\nd'erreur", "Taux de\nrejet"],
                [acc_top1, pct_erreurs, pct_rejetes],
                color=[VERT, ROUGE, ORANGE], width=0.45,
                edgecolor="white", linewidth=1.5)
for bar, val in zip(bars4, [acc_top1, pct_erreurs, pct_rejetes]):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax4.set_ylim(0, 115)
ax4.set_ylabel("Pourcentage (%)", fontsize=11)
ax4.set_title("Répartition des résultats — Dev set", fontsize=13, fontweight="bold")
ax4.set_yticks(range(0, 111, 10))
path4 = os.path.join(GRAPHIQUES_DIR, "4_metriques.png")
fig4.savefig(path4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  ✔ {path4}")

print(f"\n  ✔ Tous les graphiques sauvegardés dans : {GRAPHIQUES_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{ligne('═')}")
print("  RÉSUMÉ POUR MON RAPPORT DE PROJET")
print(ligne("═"))
print(f"""
  Système   : Pipeline OCR + Matching bilingue (FR/AR)
  Jeu       : Dev set  (dataset/annotations.csv)
  Dataset   : {total} images de boîtes de médicaments

  Accuracy (Top-1) : {acc_top1:.1f}%
  Taux d'erreur    : {pct_erreurs:.1f}%
  Taux de rejet    : {pct_rejetes:.1f}%

  FR uniquement    : {acc_fr:.1f}%
  AR uniquement    : {acc_ar:.1f}%

  Date évaluation  : {datetime.now().strftime('%d/%m/%Y %H:%M')}
""")
print(ligne("═"))