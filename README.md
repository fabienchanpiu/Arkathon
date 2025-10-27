
# 🌌 ASP CIE — Abstract Country Orbits (2010–2024)

**Objectif** : rester **abstrait** tout en encodant l’information.  
Chaque **pays** devient un **anneau** concentrique. Chaque **compagnie** du pays dépose des **traces** sur l’anneau :
- **Arc** (longueur ∝ `CIE_PAX`, couleur = f(`CIE_PAX`))
- **Étoiles/spikes** vers l’extérieur (densité ∝ `CIE_VOL`, couleur = f(`CIE_VOL`))
- **Bulle interne** (rayon ∝ `CIE_FRP` ou `CIE_PKT`, couleur = f(valeur))

Aucune étiquette : composition 100% **abstraite**. Les angles sont déterminés par un **hachage** reproductible (seed).

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Utilisation
```bash
# Test rapide (2024)
python src/art_asp_cie.py --input data/ASP_CIE_2024.csv --algo abstract_country --out out/abstract_country_2024.png

# Dézipper toutes les années et faire un rendu
unzip data/asp-cie-2010-2024.zip -d data
python src/art_asp_cie.py --input data/asp-cie-2010-2024/ASP_CIE_2024.csv --algo abstract_country --out out/abstract_country_2024.png

# Paramètres utiles
# - N d’anneaux (pays), largeur/écart, palette, seed
python src/art_asp_cie.py --input data/ASP_CIE_2024.csv --algo abstract_country   --max_countries 10 --ring_width 0.04 --ring_gap 0.035 --palette plasma --seed 7   --out out/abstract_country_tuned.png

# - Focus sur certains pays
python src/art_asp_cie.py --input data/ASP_CIE_2024.csv --algo abstract_country   --country_filter France,United\ Kingdom,Spain --out out/abstract_country_focus.png
```
