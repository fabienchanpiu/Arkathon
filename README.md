Ce projet transforme des données compagnies aériennes **ASP_CIE** en **œuvres abstraites** :  
chaque **pays** devient un **anneau** concentrique et chaque **compagnie** y laisse des **traces** (arc, spikes, bulle) dont la **taille** et la **couleur** encodent des métriques.

## Installation rapide
```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install pandas numpy matplotlib
```

Les dépendances sont aussi listées dans `requirements.txt` :

```bash
# après activation du virtualenv
python -m pip install -r requirements.txt
```

## Utilisation
```bash
# Fichier direct
python art_abstract_country_oop.py --input data/ASP_CIE_2024.csv --out out/abstract.png

# Dossier + choix interactif du CSV
python art_abstract_country_oop.py --input data/ --ask --out out/abstract.png

# Filtrer quelques pays
python art_abstract_country_oop.py --input data/ASP_CIE_2024.csv   --country_filter "France,United Kingdom,Spain" --out out/focus.png

# Forcer le mode “small-data” (formes plus visibles)
python art_abstract_country_oop.py --input data/mini.csv --small_data_threshold 9999
```

## Signification visuelle
- **Anneau** = **pays** (top N par somme de `CIE_PAX`).  
- **Arc** (portion de l’anneau) = **compagnie** du pays : **longueur ∝ `CIE_PAX`** et **couleur = f(`CIE_PAX`)**.  
- **Spikes** (petites aiguilles vers l’extérieur) = **vols** : **densité & longueur ∝ `CIE_VOL`** et **couleur = f(`CIE_VOL`)**.  
- **Bulle** (petit disque près de l’arc) : **rayon ∝ `CIE_FRP` (cargo)**, **intérieur coloré = f(`CIE_PKT`)** (distance), **bord = f(`CIE_FRP`)**.

> Les valeurs sont **normalisées par pays** (comparaison **intra‑anneau**).

---

## Explication du code

### 1) `ASPDataLoader`
- **But** : lire un **CSV unique** ou **plusieurs** CSV `ASP_CIE_*` d’un **dossier** (séparateur `;`, décimale `,`).  
- **Fonctions clés** :
  - `read_input(path)` : concatène les fichiers, **ne garde que les colonnes communes**, cast en numérique (`CIE_PAX`, `CIE_VOL`, …), dérive `YEAR` à partir de `ANMOIS`.  
  - `list_csvs_in(path)` + `choose_csv_interactive(path)` : **affiche** les CSV disponibles et **demande** un choix si `--ask` est passé.

### 2) `Scale` (normalisation robuste)
- `minmax(a, fallback=0.5)` : renvoie des valeurs **[0,1]**.  
  Si **toutes constantes/NaN**, remplit avec **0.5** → évite des formes invisibles (utile sur très petits datasets).

### 3) `Palette`
- `color_from_value(v, palette, alpha)` : conversion **valeur normalisée → couleur** via `matplotlib.colormaps` (API ≥3.7).

### 4) `AbstractCountryOrbits` (renderer)
- **Paramètres** : `ring_width`, `ring_gap`, `inner_radius`, `max_countries`, `palette`, `bg`, `dpi`, `small_data_threshold`.
- **Pipeline** :
  1. **Sélection des pays** (`_pick_countries`) : top N par somme de `CIE_PAX`.  
  2. **Filtrage optionnel** par `--country_filter`.  
  3. **Ordre des compagnies** (`_deterministic_company_order`) : tri **déterministe** (PAX ↓ puis CIE ↑).  
  4. **Angles** (`_regular_angles`) : **espacements réguliers** sur [0, 2π) → aucun hasard.  
  5. **Normalisation** (`_normalize_metrics`) par pays : `CIE_PAX`, `CIE_VOL`, `CIE_FRP`, `CIE_PKT`.  
  6. **Dessin** par compagnie (`_draw_company_trace`) :  
     - **Arc** : ouverture **[6°,126°]** (ou **≥ `min_span`** en small‑data), couleur = f(PAX).  
     - **Spikes** : **max(`spike_min`, f(VOL))** spikes réguliers, longueur ≥ `spike_len_min`.  
     - **Bulle** : rayon ≥ `bubble_min`, **face = f(PKT)**, **bord = f(FRP)**.  
  6. **Anneaux** concentriques : on avance de `ring_gap` après chaque pays.
- **Small‑data mode** (auto si `len(df) ≤ small_data_threshold`) :  
  Amplifie la lisibilité quand il y a **peu de lignes** :  
  - anneaux **plus larges** (`ring_width ×1.7`), **espaces** (`ring_gap ×1.4`),  
  - **span minimal** des arcs **36°**,  
  - **≥ 6 spikes** par arc & spikes plus longs,  
  - **bulle** plus grande.  
  → évite l’effet “quelques traits perdus sur fond noir”.

### 5) Sortie **unique** (`make_unique_path`)
- Si `out/abstract.png` existe déjà : écrit **`out/abstract_1.png`**, puis `_2`, etc.

### 6) Bloc CLI (où est utilisé `country_filter` ?)
- Dans `main()` :  
  - `--country_filter "France,Spain"` devient la **liste** `["France","Spain"]`.  
  - On instancie `AbstractCountryOrbits(...)`, puis `renderer.render(df, out_path, country_filter=country_filter)` :  
    le rendu **ne garde que** les lignes dont `CIE_PAYS` est dans la liste.

---

## Options CLI

| Option | Description |
|---|---|
| `--input` | CSV unique ou dossier contenant des `ASP_CIE_*.csv` |
| `--ask` | Si `--input` est un dossier, affiche une **liste** et **demande** un CSV |
| `--out` | Chemin PNG de sortie (**suffixe auto** si le fichier existe déjà) |
| `--palette` | Palette Matplotlib (`viridis`, `plasma`, `magma`, …) |
| `--bg` | Couleur de fond (`black`/`white`/hex) |
| `--dpi` | Résolution |
| `--max_countries` | Nombre d’anneaux (pays) |
| `--ring_width`, `--ring_gap`, `--inner_radius` | Géométrie des anneaux |
| `--small_data_threshold` | Seuil d’activation du **small‑data mode** |
| `--country_filter` | Ex: `"France,United Kingdom"` |

---

## Structure
```
.
├── art_abstract_country_oop.py      # script principal (POO)
├── data/
│   └── ASP_CIE_2024.csv             # tes données
└── out/
    └── abstract.png                 # sorties (suffixe auto _1, _2…)
```

## Équipe 5

Mehdi ZERROUKI  
Corentin SANJUAN  
Gabriel FAUCON  
Lilia QUISPE LÒPEZ  
Fabien CHAN PIU  
Minh Dung KIEU
