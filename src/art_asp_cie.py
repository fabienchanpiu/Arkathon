
import argparse, math, hashlib, os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from sklearn.preprocessing import OneHotEncoder

KNOWN_NUMERIC = ["CIE_PAX","CIE_FRP","CIE_PEQ","CIE_PKT","CIE_TKT","CIE_PEQKT","CIE_VOL"]
KNOWN_CATEG = ["CIE","CIE_NOM","CIE_NAT","CIE_PAYS"]

def list_asp_files(p: Path):
    if p.exists() and p.is_dir():
        return sorted([f for f in p.rglob("*.csv") if f.name.startswith("ASP_CIE_")])
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
        raise SystemExit("Aucune colonne commune entre les fichiers CSV.")
    df = pd.concat([d[list(common_cols)] for d in dfs], ignore_index=True)
    for c in [c for c in KNOWN_NUMERIC if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ANMOIS" in df.columns:
        df["ANMOIS"] = pd.to_numeric(df["ANMOIS"], errors="coerce").astype("Int64")
        df["YEAR"] = (df["ANMOIS"] // 100).astype("Int64")
    return df

def minmax(a):
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=float)
    amin = np.nanmin(a[finite]); amax = np.nanmax(a[finite])
    if amax - amin < 1e-12: return np.zeros_like(a, dtype=float)
    out = (a - amin) / (amax - amin); out[~finite] = 0.0; return out

def hash_to_unit(s, salt=0):
    h = hashlib.sha256((str(s) + str(salt)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def ensure_out_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def pick_columns_for_asp(df, xcol, ycol, sizecol, categorical, idexpr):
    if xcol is None: xcol = "CIE_FRP" if "CIE_FRP" in df.columns else None
    if ycol is None: ycol = "CIE_PKT" if "CIE_PKT" in df.columns else None
    if sizecol is None: sizecol = "CIE_PAX" if "CIE_PAX" in df.columns else None
    if categorical is None or len(categorical) == 0:
        categorical = [c for c in KNOWN_CATEG if c in df.columns]
    if idexpr is None:
        if "CIE" in df.columns and "ANMOIS" in df.columns:
            idexpr = "{CIE}-{ANMOIS}"
        elif "CIE" in df.columns and "YEAR" in df.columns:
            idexpr = "{CIE}-{YEAR}"
        else:
            idexpr = "ROW-{index}"
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if xcol is None and num_cols: xcol = num_cols[0]
    if ycol is None and len(num_cols) >= 2: ycol = num_cols[1]
    if sizecol is None and len(num_cols) >= 3: sizecol = num_cols[2]
    need = {"xcol": xcol, "ycol": ycol, "sizecol": sizecol}
    missing = [k for k, v in need.items() if v is None]
    if missing: raise SystemExit(f"Colonnes non résolues: {missing}. Spécifiez --xcol/--ycol/--sizecol.")
    return xcol, ycol, sizecol, categorical, idexpr

def materialize_id(df, idexpr):
    def mk(row):
        s = idexpr
        for col in df.columns: s = s.replace("{"+col+"}", str(row.get(col, "")))
        s = s.replace("{index}", str(row.name))
        return s
    return df.apply(mk, axis=1)

def ohe_color_values(df, categorical, seed):
    if not categorical: return np.zeros(len(df))
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    mat = enc.fit_transform(df[categorical].astype(str))
    rng = np.random.RandomState(seed); w = rng.rand(mat.shape[1])
    proj = mat @ w
    if proj.max() - proj.min() < 1e-12: return np.zeros_like(proj)
    return (proj - proj.min()) / (proj.max() - proj.min())

def compute_positions(df, xcol, ycol, sizecol, ids, seed, style):
    x_norm = minmax(df[xcol].to_numpy()); y_norm = minmax(df[ycol].to_numpy()); s_norm = minmax(df[sizecol].to_numpy())
    total = np.maximum(x_norm.sum(), 1e-9); angles = 2 * math.pi * np.cumsum(x_norm) / total
    ranks = s_norm.argsort().argsort(); radius = np.sqrt((ranks + 1) / (len(df) + 1))
    jitter = np.array([(hash_to_unit(str(i), seed) - 0.5) * 0.25 for i in ids])
    if style == "orbit":
        rings = np.clip((radius * 6).astype(int), 0, 6) / 6.0; radius = 0.25 + 0.65 * rings
    elif style == "bars":
        radius = 0.2 + 0.7 * y_norm; angles = np.linspace(0, 2*math.pi, len(df), endpoint=False)
    ang = angles + jitter; rad = 0.15 + 0.8 * radius
    xs = 0.5 + rad * np.cos(ang); ys = 0.5 + rad * np.sin(ang); sizes = 6 + 34 * s_norm
    return xs, ys, sizes, x_norm

def choose_colors(df, x_norm, categorical, seed, palette):
    from matplotlib import cm
    try: cmap = cm.get_cmap(palette)
    except ValueError: cmap = cm.get_cmap("viridis")
    base = np.array([cmap(v) for v in x_norm]); ohe_vals = ohe_color_values(df, categorical, seed)
    tint = np.array([cmap((v + 0.35) % 1.0) for v in ohe_vals]); colors = 0.65 * base + 0.35 * tint; colors[:,3] = 0.7
    return colors

def plot_abstract(df, xcol, ycol, sizecol, ids, categorical, out_path, seed=1, style="spiral", dpi=200, bg="black", palette="viridis", title=None):
    xs, ys, sizes, x_norm = compute_positions(df, xcol, ycol, sizecol, ids, seed, style); colors = choose_colors(df, x_norm, categorical, seed, palette)
    fig = plt.figure(figsize=(10, 10), dpi=dpi); ax = plt.gca(); ax.set_facecolor(bg); fig.patch.set_facecolor(bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    n_glow = max(200, len(df) * 5); rng = np.random.RandomState(seed); gx = rng.rand(n_glow); gy = rng.rand(n_glow)
    ax.scatter(gx, gy, s=1, alpha=0.05, linewidths=0)
    for (x, y, r, col) in zip(xs, ys, sizes, colors):
        c = Circle((x, y), r/100.0, facecolor=col, edgecolor=(col[0], col[1], col[2], 0.9), linewidth=0.4); ax.add_patch(c)
    if title: ax.text(0.5, 0.02, title, ha="center", va="bottom", fontsize=10, color="white" if bg!="white" else "black")
    ensure_out_dir(out_path); plt.savefig(out_path, bbox_inches="tight", pad_inches=0); plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser(description="Art abstrait pour ASP_CIE (2010–2024)")
    p.add_argument("--input", required=True, help="Fichier CSV ASP_CIE_YYYY.csv ou dossier contenant plusieurs CSV")
    p.add_argument("--xcol"); p.add_argument("--ycol"); p.add_argument("--sizecol")
    p.add_argument("--categorical", help="Colonnes catégorielles séparées par des virgules")
    p.add_argument("--idexpr", help="Expression id, ex: '{CIE}-{ANMOIS}'")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--style", choices=["spiral","orbit","bars"], default="spiral")
    p.add_argument("--dpi", type=int, default=200); p.add_argument("--bg", default="black"); p.add_argument("--palette", default="viridis")
    p.add_argument("--out"); p.add_argument("--by", choices=["none","year","company"], default="none")
    p.add_argument("--out_dir")
    return p.parse_args()

def main():
    args = parse_args(); path = Path(args.input); df = read_input(path)
    categorical = [c.strip() for c in args.categorical.split(",")] if args.categorical else None
    xcol, ycol, sizecol, categorical, idexpr = pick_columns_for_asp(df, args.xcol, args.ycol, args.sizecol, categorical, args.idexpr)
    df = df.dropna(subset=[xcol, ycol, sizecol]).copy(); ids = materialize_id(df, idexpr)
    if args.by == "none":
        out = Path(args.out or "out/asp_cie.png"); plot_abstract(df, xcol, ycol, sizecol, ids, categorical, out, seed=args.seed, style=args.style, dpi=args.dpi, bg=args.bg, palette=args.palette); print(f"✅ Image générée: {out}"); return
    out_dir = Path(args.out_dir or "out"); out_dir.mkdir(parents=True, exist_ok=True)
    if args.by == "year":
        if "YEAR" not in df.columns: raise SystemExit("YEAR indisponible pour --by year.")
        for y, part in df.groupby("YEAR", dropna=True):
            if len(part)==0: continue
            out = out_dir / f"asp_cie_{int(y)}.png"; plot_abstract(part, xcol, ycol, sizecol, ids.loc[part.index], categorical, out, seed=args.seed, style=args.style, dpi=args.dpi, bg=args.bg, palette=args.palette)
            print(f"✅ {out}")
    elif args.by == "company":
        if "CIE" not in df.columns: raise SystemExit("CIE indisponible pour --by company.")
        for cie, part in df.groupby("CIE"):
            if len(part)==0: continue
            safe = str(cie).replace("/", "_"); out = out_dir / f"asp_cie_company_{safe}.png"
            plot_abstract(part, xcol, ycol, sizecol, ids.loc[part.index], categorical, out, seed=args.seed, style=args.style, dpi=args.dpi, bg=args.bg, palette=args.palette)
            print(f"✅ {out}")
if __name__ == "__main__": main()
