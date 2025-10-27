
import argparse, math, hashlib
from math import cos, sin, pi
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib import cm


# ----------------------------- IO & utils -----------------------------
KNOWN_NUMERIC = ["CIE_PAX","CIE_FRP","CIE_PEQ","CIE_PKT","CIE_TKT","CIE_PEQKT","CIE_VOL"]
KNOWN_CATEG = ["CIE","CIE_NOM","CIE_NAT","CIE_PAYS"]

def is_folder(p: Path) -> bool:
    return p.exists() and p.is_dir()

def list_asp_files(p: Path) -> List[Path]:
    if is_folder(p):
        return sorted([f for f in p.rglob("*.csv") if f.name.upper().startswith("ASP_CIE_")])
    return [p]

def read_asp_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=';', decimal=',')

def read_input(path: Path) -> pd.DataFrame:
    files = list_asp_files(path)
    if not files:
        raise SystemExit(f"Aucun CSV trouvé dans {path}")
    dfs = [read_asp_csv(f) for f in files]
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    if not common_cols:
        raise SystemExit("Aucune colonne commune entre les CSV.")
    df = pd.concat([d[list(common_cols)] for d in dfs], ignore_index=True)
    # numeric coercion
    for c in [c for c in KNOWN_NUMERIC if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ANMOIS" in df.columns:
        df["ANMOIS"] = pd.to_numeric(df["ANMOIS"], errors="coerce").astype("Int64")
        df["YEAR"] = (df["ANMOIS"] // 100).astype("Int64")
    return df

def minmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=float)
    amin = np.nanmin(a[finite]); amax = np.nanmax(a[finite])
    if amax - amin < 1e-12: return np.zeros_like(a, dtype=float)
    out = (a - amin) / (amax - amin)
    out[~finite] = 0.0
    return out

def hash_to_unit(s: str, salt: int = 0) -> float:
    h = hashlib.sha256((str(s) + str(salt)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def color_from_value(vnorm: float, palette: str = "viridis", alpha: float = 0.9):
    try: cmap = cm.get_cmap(palette)
    except ValueError: cmap = cm.get_cmap("viridis")
    r,g,b,a = cmap(float(np.clip(vnorm, 0, 1)))
    return (r,g,b,alpha)


# ----------------------- Abstract Country Orbits ----------------------
def plot_abstract_country(df: pd.DataFrame, out_path: Path, palette: str = "viridis", bg: str = "black",
                          seed: int = 1, max_countries: int = 8, ring_width: float = 0.05, ring_gap: float = 0.03,
                          country_filter: Optional[List[str]] = None, inner_radius: float = 0.18):
    if "CIE_PAYS" not in df.columns:
        raise SystemExit("CIE_PAYS manquant pour l'algo abstract_country.")
    work = df.copy()
    if country_filter:
        work = work[work["CIE_PAYS"].astype(str).isin(country_filter)]
    if work.empty:
        raise SystemExit("Aucune ligne après filtrage des pays.")

    # Top countries by total PAX
    totals = work.groupby("CIE_PAYS")["CIE_PAX"].sum().sort_values(ascending=False)
    countries = list(totals.index[:max_countries])

    fig = plt.figure(figsize=(11, 11), dpi=220)
    ax = plt.gca()
    ax.set_facecolor(bg); fig.patch.set_facecolor(bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

    rng = np.random.RandomState(seed)
    # Subtle starry background
    n_glow = 1200
    gx, gy = rng.rand(n_glow), rng.rand(n_glow)
    ax.scatter(gx, gy, s=0.5, alpha=0.05, linewidths=0, color="white" if bg!="white" else "black")

    # ring radii
    r_in = inner_radius
    center = (0.5, 0.5)
    for ring_idx, country in enumerate(countries):
        r_out = r_in + ring_width
        sub = work[work["CIE_PAYS"] == country].copy()
        if sub.empty:
            r_in = r_out + ring_gap
            continue

        # Normalize per country (keeps contrast within ring)
        pax_n = minmax(sub["CIE_PAX"].to_numpy())
        vol_n = minmax(sub["CIE_VOL"].to_numpy()) if "CIE_VOL" in sub.columns else np.zeros(len(sub))
        frp_n = minmax(sub["CIE_FRP"].to_numpy()) if "CIE_FRP" in sub.columns else np.zeros(len(sub))
        pkt_n = minmax(sub["CIE_PKT"].to_numpy()) if "CIE_PKT" in sub.columns else np.zeros(len(sub))

        # Deterministic angles by company id + seed
        ids = (sub.get("CIE").astype(str) if "CIE" in sub.columns else sub.index.astype(str)).tolist()
        base_angles = np.array([2*pi*hash_to_unit(i, seed) for i in ids])

        # Draw traces for each company
        for k, (idx, row) in enumerate(sub.iterrows()):
            ang = base_angles[k]

            # Arc span ∝ PAX
            span = (6 + 120 * pax_n[k])   # degrees
            theta1 = np.degrees(ang) - span/2
            theta2 = theta1 + span
            col_arc = color_from_value(pax_n[k], palette, alpha=0.75)
            wedge = Wedge(center, r_out, theta1, theta2, width=(r_out - r_in), facecolor=col_arc, edgecolor=None, linewidth=0)
            ax.add_patch(wedge)

            # Radial spikes ∝ VOL
            n_spikes = int(2 + round(14 * vol_n[k]))
            spike_len = (0.01 + 0.06 * vol_n[k])
            theta0 = ang
            for s in range(n_spikes):
                # small jitter per spike
                jitter = (s / max(1, n_spikes)) * (span*pi/180) - (span*pi/360)
                a = theta0 + jitter
                x1 = center[0] + r_out * cos(a); y1 = center[1] + r_out * sin(a)
                x2 = center[0] + (r_out + spike_len) * cos(a); y2 = center[1] + (r_out + spike_len) * sin(a)
                ax.plot([x1, x2], [y1, y2], linewidth=0.6, color=color_from_value(vol_n[k], palette, alpha=0.95))

            # Inner bubble ∝ FRP (or PKT if FRP absent)
            bub = (0.004 + 0.040 * (frp_n[k] if "CIE_FRP" in sub.columns else pkt_n[k]))
            rb = (r_in + r_out)/2
            xb = center[0] + rb * cos(ang); yb = center[1] + rb * sin(ang)
            ax.add_patch(Circle((xb, yb), bub, facecolor=color_from_value(pkt_n[k], palette, alpha=0.45),
                                edgecolor=color_from_value(frp_n[k], palette, alpha=0.9), linewidth=0.4))

        # advance to next ring
        r_in = r_out + ring_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ------------------------------- CLI ---------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ASP CIE — Abstract Country Orbits")
    p.add_argument("--input", required=True, help="Fichier CSV ASP_CIE_YYYY.csv ou dossier contenant plusieurs CSV")
    p.add_argument("--algo", choices=["abstract_country"], default="abstract_country")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--bg", default="black")
    p.add_argument("--palette", default="viridis")
    p.add_argument("--out", help="PNG de sortie", default="out/abstract_country.png")
    p.add_argument("--max_countries", type=int, default=8)
    p.add_argument("--ring_width", type=float, default=0.05)
    p.add_argument("--ring_gap", type=float, default=0.03)
    p.add_argument("--inner_radius", type=float, default=0.18)
    p.add_argument("--country_filter", help="Liste de pays séparés par des virgules (optionnel)")
    return p.parse_args()


def main():
    args = parse_args()
    df = read_input(Path(args.input))
    countries = None
    if args.country_filter:
        countries = [c.strip() for c in args.country_filter.split(",") if c.strip()]
    out = Path(args.out)
    plot_abstract_country(df, out_path=out, palette=args.palette, bg=args.bg, seed=args.seed,
                          max_countries=args.max_countries, ring_width=args.ring_width, ring_gap=args.ring_gap,
                          country_filter=countries, inner_radius=args.inner_radius)
    print(f"✅ Image générée: {out}")


if __name__ == "__main__":
    main()
