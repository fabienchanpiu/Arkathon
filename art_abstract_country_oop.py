from __future__ import annotations
import argparse
import re
import unicodedata
from math import cos, sin, pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib import colormaps as cmaps  # Matplotlib ≥ 3.7

# ----------------------------- Constantes ------------------------------
KNOWN_NUMERIC = ["CIE_PAX", "CIE_FRP", "CIE_PEQ", "CIE_PKT", "CIE_TKT", "CIE_PEQKT", "CIE_VOL"]
KNOWN_CATEG  = ["CIE", "CIE_NOM", "CIE_NAT", "CIE_PAYS"]

# Strict: ces colonnes sont obligatoires
REQUIRED_STRICT = {"CIE_PAYS", "CIE_PAX", "CIE_VOL", "CIE_FRP", "CIE_PKT"}

# Aliases (clé normalisée -> nom canonique)
ALIASES: Dict[str, str] = {
    "anmois": "ANMOIS", "annee_mois": "ANMOIS", "an_mois": "ANMOIS", "yearmonth": "ANMOIS", "yyyymm": "ANMOIS",
    "cie": "CIE", "code_cie": "CIE", "compagnie": "CIE", "airline_code": "CIE",
    "cie_nom": "CIE_NOM", "nom_cie": "CIE_NOM", "airline": "CIE_NOM", "compagnie_nom": "CIE_NOM", "name": "CIE_NOM",
    "cie_nat": "CIE_NAT", "nationalite": "CIE_NAT", "nation": "CIE_NAT", "nat": "CIE_NAT",
    "cie_pays": "CIE_PAYS", "pays": "CIE_PAYS", "country": "CIE_PAYS",
    "cie_pax": "CIE_PAX", "pax": "CIE_PAX", "passagers": "CIE_PAX", "passenger": "CIE_PAX", "nb_pax": "CIE_PAX",
    "cie_frp": "CIE_FRP", "frp": "CIE_FRP", "freight": "CIE_FRP", "cargo": "CIE_FRP", "fret": "CIE_FRP",
    "cie_peq": "CIE_PEQ", "peq": "CIE_PEQ",
    "cie_pkt": "CIE_PKT", "pkt": "CIE_PKT", "pkm": "CIE_PKT", "passenger_km": "CIE_PKT",
    "cie_tkt": "CIE_TKT", "tkt": "CIE_TKT", "tickets": "CIE_TKT",
    "cie_peqkt": "CIE_PEQKT", "peqkt": "CIE_PEQKT",
    "cie_vol": "CIE_VOL", "vol": "CIE_VOL", "flights": "CIE_VOL", "nb_vols": "CIE_VOL",
}

# ----------------------------- Normalisation I/O ------------------------
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _normalize_col_key(name: str) -> str:
    # trim + lower + désaccentuation + remplace non-alphanum par '_' + compresse '_'
    s = _strip_accents(str(name).strip().lower())
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def normalize_text(x: str) -> str:
    """Pour valeurs utilisateur/CSV: collapse espaces + trim + lower (ex: '  FrAnCe  ' → 'france')."""
    return re.sub(r"\s+", " ", str(x)).strip().lower()

def make_unique_path(p: Path) -> Path:
    """file.png -> file.png | file_1.png | file_2.png ..."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return p
    stem, suffix = p.stem, "".join(p.suffixes)
    i = 1
    while True:
        cand = p.parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes via normalisation + alias vers les noms canoniques."""
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = _normalize_col_key(col)
        if key in ALIASES:
            rename_map[col] = ALIASES[key]
        else:
            # Si déjà une colonne canonique (mais avec espaces/accents/casse)
            guess = _normalize_col_key(col).upper()
            if guess in KNOWN_NUMERIC + KNOWN_CATEG + ["ANMOIS", "YEAR"]:
                rename_map[col] = guess
    return df.rename(columns=rename_map)

def ensure_required_columns(df: pd.DataFrame, required: set) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        avail_norm = {c: _normalize_col_key(c) for c in df.columns}
        hints = []
        for m in missing:
            ask = [k for k, v in ALIASES.items() if v == m]
            if ask:
                hints.append(f"- '{m}' attendu. Alias acceptés: {', '.join(sorted(set(ask)))}")
            else:
                hints.append(f"- '{m}' attendu.")
        raise SystemExit(
            "Colonnes indispensables manquantes après harmonisation : "
            + ", ".join(missing)
            + "\nColonnes disponibles (normalisées) : "
            + ", ".join([f"{c}→{n}" for c, n in avail_norm.items()])
            + "\nAstuces : corrige le nom dans le CSV OU ajoute un alias dans ALIASES."
        )

# ----------------------------- Data Loader -----------------------------
class ASPDataLoader:
    """Lecture CSV/dossier ASP_CIE_* avec harmonisation + vérif colonnes + normalisation des valeurs."""

    @staticmethod
    def _is_folder(p: Path) -> bool:
        return p.exists() and p.is_dir()

    @staticmethod
    def list_csvs_in(path: Path) -> List[Path]:
        if not ASPDataLoader._is_folder(path):
            return []
        return sorted([f for f in path.rglob("*.csv") if f.name.upper().startswith("ASP_CIE_")])

    @staticmethod
    def choose_csv_interactive(path: Path) -> Path:
        files = ASPDataLoader.list_csvs_in(path)
        if not files:
            raise SystemExit(f"Aucun CSV 'ASP_CIE_*' trouvé dans {path}")
        print("\nCSV disponibles :")
        for i, f in enumerate(files, 1):
            print(f"  [{i}] {f.name}")
        while True:
            s = input(f"Saisis un numéro 1–{len(files)} : ").strip()
            if s.isdigit() and 1 <= int(s) <= len(files):
                chosen = files[int(s) - 1]
                print(f"→ Sélection : {chosen.name}\n")
                return chosen
            print("Choix invalide, recommence.")

    @staticmethod
    def _list_asp_files(p: Path) -> List[Path]:
        if ASPDataLoader._is_folder(p):
            return sorted([f for f in p.rglob("*.csv") if f.name.upper().startswith("ASP_CIE_")])
        return [p]

    @staticmethod
    def _read_asp_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, sep=";", decimal=",")
        except Exception as e:
            raise SystemExit(f"Erreur de lecture CSV '{path}': {e}")

    @classmethod
    def read_input(cls, path: Path) -> pd.DataFrame:
        files = cls._list_asp_files(path)
        if not files:
            raise SystemExit(f"Aucun CSV trouvé dans {path}")
        dfs: List[pd.DataFrame] = []
        for f in files:
            df_one = cls._read_asp_csv(f)
            df_one = harmonize_columns(df_one)
            dfs.append(df_one)

        # Colonnes communes après harmonisation
        common_cols = set(dfs[0].columns)
        for d in dfs[1:]:
            common_cols &= set(d.columns)
        if not common_cols:
            raise SystemExit("Aucune colonne commune entre les CSV (après harmonisation).")

        df = pd.concat([d[list(common_cols)] for d in dfs], ignore_index=True)

        # STRICT: on exige toutes les colonnes clés
        ensure_required_columns(df, REQUIRED_STRICT)

        # Cast numériques
        for c in [c for c in KNOWN_NUMERIC if c in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "ANMOIS" in df.columns:
            df["ANMOIS"] = pd.to_numeric(df["ANMOIS"], errors="coerce").astype("Int64")
            df["YEAR"] = (df["ANMOIS"] // 100).astype("Int64")

        # Normalise catégorielles (trim + collapse spaces + lower)
        for c in [c for c in KNOWN_CATEG if c in df.columns]:
            df[c] = df[c].astype(str).map(normalize_text)

        return df

# ----------------------------- Helpers ---------------------------------
class Scale:
    """Normalisations robustes pour lisibilité, même peu de lignes."""
    @staticmethod
    def minmax(a: np.ndarray, fallback: float = 0.5) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        finite = np.isfinite(a)
        if not finite.any():
            return np.zeros_like(a, dtype=float) + fallback
        amin = np.nanmin(a[finite]); amax = np.nanmax(a[finite])
        if amax - amin < 1e-12:
            return np.zeros_like(a, dtype=float) + fallback
        out = (a - amin) / (amax - amin)
        out[~finite] = fallback
        return out

class Palette:
    """Couleurs via Matplotlib colormaps (API ≥3.7)."""
    @staticmethod
    def color_from_value(vnorm: float, palette: str = "viridis", alpha: float = 0.9) -> Tuple[float, float, float, float]:
        try:
            cmap = cmaps.get_cmap(palette)
        except Exception:
            cmap = cmaps.get_cmap("viridis")
        r, g, b, _ = cmap(float(np.clip(vnorm, 0, 1)))
        return (r, g, b, alpha)

# ----------------------------- Renderer --------------------------------
class AbstractCountryOrbits:
    """
    - Pays = anneaux (top N par PAX)
    - Compagnies = arcs (PAX), spikes (VOL), bulles (FRP/PKT)
    - Déterministe + small-data mode automatique
    """

    def __init__(
        self,
        palette: str = "viridis",
        bg: str = "black",
        dpi: int = 220,
        max_countries: int = 8,
        ring_width: float = 0.05,
        ring_gap: float = 0.03,
        inner_radius: float = 0.18,
        small_data_threshold: int = 30,
    ) -> None:
        self.palette = palette
        self.bg = bg
        self.dpi = dpi
        self.max_countries = max_countries
        self.ring_width = ring_width
        self.ring_gap = ring_gap
        self.inner_radius = inner_radius
        self.small_data_threshold = small_data_threshold
        self.center = (0.5, 0.5)

    # ---------- Préparation ----------
    def _pick_countries(self, df: pd.DataFrame) -> List[str]:
        if "CIE_PAYS" not in df.columns:
            raise SystemExit("Colonne 'CIE_PAYS' manquante.")
        totals = df.groupby("CIE_PAYS")["CIE_PAX"].sum().sort_values(ascending=False)
        return list(totals.index[: self.max_countries])

    @staticmethod
    def _normalize_metrics(sub: pd.DataFrame):
        # Les colonnes sont garanties présentes par le mode strict; on garde néanmoins la robustesse
        n = len(sub)

        def norm_or_fill(col: str) -> np.ndarray:
            if col in sub.columns:
                arr = pd.to_numeric(sub[col], errors="coerce").to_numpy()
                return Scale.minmax(arr, fallback=0.5)
            return np.full(n, 0.5, dtype=float)

        pax_n = norm_or_fill("CIE_PAX")
        vol_n = norm_or_fill("CIE_VOL")
        frp_n = norm_or_fill("CIE_FRP")
        pkt_n = norm_or_fill("CIE_PKT")
        return pax_n, vol_n, frp_n, pkt_n

    @staticmethod
    def _deterministic_company_order(sub: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["CIE_PAX", "CIE"] if c in sub.columns]
        if "CIE_PAX" in cols and "CIE" in cols:
            return sub.sort_values(by=["CIE_PAX", "CIE"], ascending=[False, True], kind="mergesort")
        if "CIE_PAX" in cols:
            return sub.sort_values(by="CIE_PAX", ascending=False, kind="mergesort")
        if "CIE" in cols:
            return sub.sort_values(by="CIE", ascending=True, kind="mergesort")
        return sub

    @staticmethod
    def _regular_angles(n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, 2.0 * pi, n, endpoint=False)

    # ---------- Dessin ----------
    def _draw_company_trace(
        self,
        ax: plt.Axes,
        ang: float,
        span_deg: float,
        r_in: float,
        r_out: float,
        pax_norm: float,
        vol_norm: float,
        frp_norm: float,
        pkt_norm: float,
        spike_min: int,
        spike_len_min: float,
        bubble_min: float,
    ) -> None:
        theta_center_deg = np.degrees(ang)
        theta1 = theta_center_deg - span_deg / 2.0
        theta2 = theta_center_deg + span_deg / 2.0

        # Arc (PAX)
        col_arc = Palette.color_from_value(pax_norm, self.palette, alpha=0.85)
        ax.add_patch(Wedge(self.center, r_out, theta1, theta2,
                           width=(r_out - r_in), facecolor=col_arc, edgecolor=None, linewidth=0))

        # Spikes (VOL) — réguliers
        n_spikes = max(spike_min, int(round(2 + 14 * vol_norm)))
        spike_len = max(spike_len_min, 0.01 + 0.06 * vol_norm)
        for i in range(n_spikes):
            t = (i + 0.5) / n_spikes
            a = np.radians(theta1 + t * (theta2 - theta1))
            x1 = self.center[0] + r_out * cos(a); y1 = self.center[1] + r_out * sin(a)
            x2 = self.center[0] + (r_out + spike_len) * cos(a); y2 = self.center[1] + (r_out + spike_len) * sin(a)
            ax.plot([x1, x2], [y1, y2], linewidth=0.65,
                    color=Palette.color_from_value(vol_norm, self.palette, alpha=0.95))

        # Bulle (FRP/PKT)
        rb = (r_in + r_out) / 2.0
        xb = self.center[0] + rb * cos(ang); yb = self.center[1] + rb * sin(ang)
        bub = max(bubble_min, 0.004 + 0.040 * frp_norm)
        ax.add_patch(Circle((xb, yb), bub,
                            facecolor=Palette.color_from_value(pkt_norm, self.palette, alpha=0.5),
                            edgecolor=Palette.color_from_value(frp_norm, self.palette, alpha=0.95),
                            linewidth=0.5))

    # ---------- Rendu principal ----------
    def render(
        self,
        df: pd.DataFrame,
        out_path: Path,
        country_filter: Optional[List[str]] = None,
    ) -> Path:
        work = df.copy()

        # Filtre pays (déjà normalisé côté CSV; on normalise aussi le filtre CLI)
        if country_filter:
            if "CIE_PAYS" not in work.columns:
                raise SystemExit("Filtrage pays impossible: 'CIE_PAYS' absent.")
            work = work[work["CIE_PAYS"].isin(country_filter)]

        if work.empty:
            raise SystemExit("Aucune ligne après harmonisation/filtrage.")

        # Small-data mode
        total_n  = len(work)
        small_md = total_n <= self.small_data_threshold
        ring_w   = self.ring_width * (1.7 if small_md else 1.0)
        ring_g   = self.ring_gap   * (1.4 if small_md else 1.0)
        min_span = 36.0 if small_md else 12.0
        spike_min     = 6    if small_md else 2
        spike_len_min = 0.035 if small_md else 0.01
        bubble_min    = 0.018 if small_md else 0.004

        countries = self._pick_countries(work)

        fig = plt.figure(figsize=(11, 11), dpi=self.dpi)
        ax  = plt.gca()
        ax.set_facecolor(self.bg); fig.patch.set_facecolor(self.bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

        r_in = self.inner_radius
        for country in countries:
            r_out = r_in + ring_w
            sub = work[work["CIE_PAYS"] == country].copy()
            if sub.empty:
                r_in = r_out + ring_g; continue

            sub    = self._deterministic_company_order(sub)
            angles = self._regular_angles(len(sub))
            pax_n, vol_n, frp_n, pkt_n = self._normalize_metrics(sub)

            for k, _ in enumerate(sub.itertuples(index=False)):
                span_deg = max(min_span, 6.0 + 120.0 * float(pax_n[k]))
                self._draw_company_trace(ax, float(angles[k]), span_deg, r_in, r_out,
                                         float(pax_n[k]), float(vol_n[k]),
                                         float(frp_n[k]), float(pkt_n[k]),
                                         spike_min, spike_len_min, bubble_min)
            r_in = r_out + ring_g

        out_path = make_unique_path(out_path)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return out_path

# --------------------------------- CLI ---------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ASP CIE — Abstract Country Orbits (POO, strict & small-data)")
    p.add_argument("--input", required=True, help="CSV ASP_CIE_YYYY.csv ou dossier de CSV")
    p.add_argument("--out", default="out/abstract_country.png", help="PNG de sortie")
    p.add_argument("--palette", default="viridis")
    p.add_argument("--bg", default="black")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--max_countries", type=int, default=8)
    p.add_argument("--ring_width", type=float, default=0.05)
    p.add_argument("--ring_gap", type=float, default=0.03)
    p.add_argument("--inner_radius", type=float, default=0.18)
    p.add_argument("--small_data_threshold", type=int, default=30,
                   help="N lignes ≤ seuil → small-data mode")
    p.add_argument("--country_filter", help="Liste de pays séparés par des virgules (optionnel)")
    p.add_argument("--ask", action="store_true",
                   help="Si INPUT est un dossier, lister les CSV et demander lequel utiliser")
    return p.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input)

    # Choix interactif si dossier + --ask
    if args.ask and in_path.is_dir():
        chosen = ASPDataLoader.choose_csv_interactive(in_path)
        df = ASPDataLoader.read_input(chosen)     # lit UN seul CSV
    else:
        df = ASPDataLoader.read_input(in_path)    # fichier direct ou concat dossier

    # Filtre pays optionnel (normalisé)
    country_filter = None
    if args.country_filter:
        country_filter = [normalize_text(c) for c in args.country_filter.split(",") if c.strip()]

    renderer = AbstractCountryOrbits(
        palette=args.palette,
        bg=args.bg,
        dpi=args.dpi,
        max_countries=args.max_countries,
        ring_width=args.ring_width,
        ring_gap=args.ring_gap,
        inner_radius=args.inner_radius,
        small_data_threshold=args.small_data_threshold,
    )
    out_final = renderer.render(df, out_path=Path(args.out), country_filter=country_filter)
    print(f"✅ Image générée: {out_final}")

if __name__ == "__main__":
    main()