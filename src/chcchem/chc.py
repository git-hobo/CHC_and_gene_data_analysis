from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from adjustText import adjust_text
from matplotlib.transforms import Bbox


# ---- targeted regexes (case-insensitive) ----
LEN_RE          = re.compile(r"^C(?P<length>\d+)", re.I)
UNKNOWN_RE      = re.compile(r"^C(?P<length>\d+)unknown(?:CHC)?$", re.I)   # supports 'unkown' and 'unknown'
UNSAT_WORD      = re.compile(r"(?P<prefix>di|tri)?ene\b", re.I)
UNSAT_MAP       = {None: 1, "di": 2, "tri": 3}
BRANCH_ANY      = re.compile(r"(?:int)?(?P<count>Mono|Di|Tri)Me(?P<pos>\d+)?", re.I)
BRANCH_COUNT    = {"mono": 1, "di": 2, "tri": 3}
METHYLALKENE_RE = re.compile(r"^C(?P<length>\d+)methylalkene$", re.I)

@dataclass(frozen=True)
class CHC:
    code: str

    # ------------ special-case matches ------------
    def _methylalkene_match(self):
        return METHYLALKENE_RE.match(self.code)

    def _unknown_match(self):
        return UNKNOWN_RE.match(self.code)

    # ------------ mixture handling ------------
    def _split_body(self) -> tuple[int | None, list[str]]:  # (length, parts)
        """
        C24MonoMe2_DiMe5 -> (24, ['MonoMe2','DiMe5'])
        C27MonoMe3_intDiMe -> (27, ['MonoMe3','intDiMe'])
        Non-mixture -> (length, [])
        """
        m = LEN_RE.match(self.code)
        if not m:
            return (None, [])
        L = int(m.group("length"))
        body = self.code[m.end():]
        if not body:
            return (L, [])
        # If body has underscores, it's a mixture of same-chain compounds
        parts = body.split("_") if "_" in body else []
        # Defensive strip (tolerate weird whitespace)
        parts = [p.strip() for p in parts if p.strip()]
        return (L, parts)

    def _is_mixture(self) -> bool:
        _, parts = self._split_body()
        return len(parts) > 0

    def _parse_part(self, part: str) -> dict:
        """
        Parse a single component token (no leading 'C<length>').
        Returns per-part facts and derived labels.
        """
        # ---- special literal: methylalkene ----
        if re.fullmatch(r"methylalkene", part, re.I):
            return {
                "me_count": None,             # unknown
                "me_positions": None,         # unknown
                "db_count": None,             # unknown
                "branched": True,
                "unsaturated": True,
                "subclass": "methylbranched_alkene",
                "backbone": "unsaturated",
            }

        # ---- regular parsing follows ----
        mm = BRANCH_ANY.search(part)
        if mm:
            cnt = BRANCH_COUNT[mm.group("count").lower()]
            pos = mm.group("pos")
            me_positions = ([int(pos)] + [None]*(cnt-1)) if pos else [None]*cnt
        else:
            cnt = 0
            me_positions = []

        um = UNSAT_WORD.search(part)
        db = 0 if not um else UNSAT_MAP[um.group("prefix").lower() if um.group("prefix") else None]
        branched = cnt > 0
        unsat = db > 0

        if branched and unsat:
            subclass = "methylbranched_alkene"
        elif branched:
            subclass = "methylbranched"
        elif unsat:
            subclass = "alkene"
        else:
            subclass = "alkane"
        backbone = "unsaturated" if unsat else "saturated"

        return {
            "me_count": cnt,
            "me_positions": me_positions,
            "db_count": db,
            "branched": branched,
            "unsaturated": unsat,
            "subclass": subclass,
            "backbone": backbone,
        }

    # ------------ internal parse helpers ------------
    def _length(self) -> int | None:
        um = self._unknown_match()
        if um:
            return int(um.group("length"))
        mm = self._methylalkene_match()
        if mm:
            return int(mm.group("length"))
        m = LEN_RE.search(self.code)
        return int(m.group("length")) if m else None

    def _unsaturation_count(self) -> int | None:
        # single-compound only
        if self._unknown_match() or self._methylalkene_match() or self._is_mixture():
            return None
        m = UNSAT_WORD.search(self.code)
        return 0 if not m else UNSAT_MAP[m.group("prefix").lower() if m.group("prefix") else None]

    def _me_count(self) -> int | None:
        # single-compound only
        if self._unknown_match() or self._methylalkene_match() or self._is_mixture():
            return None
        m = BRANCH_ANY.search(self.code)
        if not m:
            return 0
        return BRANCH_COUNT[m.group("count").lower()]

    def _me_positions(self) -> list[int | None] | None:
        """
        Single compound:
          C30MonoMe5   -> [5]
          C30DiMe5     -> [5, None]
          C23intMonoMe -> [None]
          no Me        -> []
        Special cases:
          C30unk?nown, C33methylalkene -> None
        Mixtures:
          flatten across parts, e.g. C24MonoMe2_DiMe5 -> [2, 5, None]
        """
        if self._unknown_match() or self._methylalkene_match():
            return None
        L, parts = self._split_body()
        if parts:  # mixture
            out: list[int | None] = []
            for p in parts:
                out.extend(self._parse_part(p)["me_positions"])
            return out
        # single compound
        m = BRANCH_ANY.search(self.code)
        if not m:
            return []
        count = BRANCH_COUNT[m.group("count").lower()]
        pos = m.group("pos")
        if pos is None:
            return [None] * count
        return [int(pos)] + [None] * (count - 1)

    # ------------ public properties ------------
    def chainlength(self) -> int:
        L = self._length()
        return L

    def backbone(self) -> str:
        if self._unknown_match():
            return "unknown"
        if self._methylalkene_match():
            return "unsaturated"
        L, parts = self._split_body()
        if parts:  # mixture → consensus or "mixed"
            bks = {self._parse_part(p)["backbone"] for p in parts}
            return bks.pop() if len(bks) == 1 else "mixed"
        # single compound
        cnt = self._unsaturation_count()
        return "unsaturated" if (cnt or 0) > 0 else "saturated"

    def subclass(self) -> str:
        if self._unknown_match():
            return "unknown"
        if self._methylalkene_match():
            return "methylbranched_alkene"
        L, parts = self._split_body()
        if parts:  # mixture → consensus or "mixture"
            subs = {self._parse_part(p)["subclass"] for p in parts}
            return subs.pop() if len(subs) == 1 else "mixture"
        # single compound
        me = self._me_positions() or []
        db = self._unsaturation_count() or 0
        if me and db > 0:
            return "methylbranched_alkene"
        if me:
            return "methylbranched"
        if db > 0:
            return "alkene"
        return "alkane"

    def as_dict(self) -> dict:
        L, parts = self._split_body()
        if parts:  # mixture summary (set granular counts/positions to None)
            comp = [self._parse_part(p) for p in parts]
            contains_branch = any(c["branched"] for c in comp)
            contains_unsat  = any(c["unsaturated"] for c in comp)
            return {
                "Compound": self.code,
                "Chain_Length": self.chainlength(),
                "Is_Mixture": True,
                "Backbone": self.backbone(),     # consensus or "mixed"
                "Class": self.subclass(),        # consensus or "mixture"
                "Contains_Unsaturated": contains_unsat,    # (optional) drop if you don't need it
                "Contains_Methylbranched": contains_branch,   # (optional) drop if you don't need it
                "Me_Positions": None,
                "Me_Count": None,
                "Double_Bond_Count": None,
                "Components": [f"C{L}{p}" for p in parts],
            }

        # single compound (unchanged)
        db = self._unsaturation_count()
        me = self._me_count()
        return {
            "Compound": self.code,
            "Chain_Length": self.chainlength(),
            "Is_Mixture": False,
            "Backbone": self.backbone(),
            "Class": self.subclass(),
            "Contains_Unsaturated": (None if db is None else db > 0),    
            "Contains_Methylbranched": (None if me is None else me > 0), 
            "Me_Positions": self._me_positions(),
            "Me_Count": self._me_count(),
            "Double_Bond_Count": self._unsaturation_count(),
            "Components": None,
        }

def parse_to_species(labels_array, spec_dict):
    label_to_species = dict()
    for label in labels_array:
        spec_lookup = label[0:2]
        label_to_species[label] = spec_dict[spec_lookup]
    return label_to_species

def load_chc(path, sheet):
    data = pd.read_excel(path, sheet_name=sheet, index_col=0).T
    data.index.name = "code"
    return data

def class_richness(abundance_df, metadata_df, target_class="methylbranched", species_col="species"):
    compounds = metadata_df.query("Class == @target_class")["Compound"].astype(str)
    cols = abundance_df.columns.intersection(compounds)
    pres = (abundance_df.loc[:, cols] > 0).astype(int)
    by_species_any = pres.groupby(abundance_df[species_col]).max()
    return by_species_any.sum(axis=1).rename(f"{target_class}_richness").to_frame()

def backbone_richness(abundance_df, metadata_df, target_bb="unsaturated", species_col="species"):
    compounds = metadata_df.query("Backbone == @target_bb")["Compound"].astype(str)
    cols = abundance_df.columns.intersection(compounds)
    pres = (abundance_df.loc[:, cols] > 0).astype(int)
    by_species_any = pres.groupby(abundance_df[species_col]).max()
    return by_species_any.sum(axis=1).rename(f"{target_bb}_richness").to_frame()


def scatter_gene_counts(
    df,
    xcol,
    ycol,
    labelcol=None,
    point_size=60,
    color="steelblue",
    label_angle=20,
    repel=True,
    figsize=(16, 9),
    fontname="Liberation Sans",
    y_step=10,
):
    """
    Quick scatter plot between two DataFrame columns, with optional point labels.
    - minimalist axes (only left & bottom spines)
    - horizontal gridlines every y_step (default 10), semi-transparent
    - repelling labels if adjustText is available
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[xcol], df[ycol], s=point_size, c=color)

    # Axis labels & title
    ax.set_xlabel(xcol, fontname=fontname, fontsize=16)
    ax.set_ylabel(ycol + " gene count", fontname=fontname, fontsize=16)
    ax.set_title(f"{ycol} vs {xcol}", fontname=fontname, fontsize=16)

    # Minimalist frame: only left and bottom spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Y-gridlines every y_step, lightly transparent
    if y_step is not None and y_step > 0:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.grid(axis="y", linestyle="-", alpha=0.25)

    # Add labels (repelled if possible)
    if labelcol is not None:
        labels = df[labelcol] if labelcol in df.columns else df.index
        texts = []
        for (x, y, lab) in zip(df[xcol], df[ycol], labels):
            t = ax.text(
                x, y, str(lab),
                style='italic',
                fontname=fontname, fontsize=16,
                ha="left", va="bottom", rotation=label_angle
            )
            texts.append(t)

        if repel and len(texts) > 1:
            try:
                from adjustText import adjust_text
                adjust_text(
                    texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.35),
                    only_move={"points": "xy", "texts": "xy"},
                    expand_points=(1.1, 1.2),
                    expand_text=(1.05, 1.1),
                    force_points=0.3,
                    force_text=0.5,
                    lim=1000,
                )
            except ImportError:
                # fallback: just leave the static labels
                pass

    return ax


def _place_labels_greedy(ax, annos, max_shift_px=30, radii=(0, 6, 12, 18, 24)):
    """
    Greedy, deterministic label placement in pixel space.
    For each label, try candidate offsets around the point and keep
    the first that doesn't overlap already-placed labels or leave axes.
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure renderer exists
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)

    # 8 directions (E, W, N, S, NE, NW, SE, SW) in offset points
    dirs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]

    placed_bboxes = []

    def overlaps_any(b):
        return any(b.overlaps(pb) for pb in placed_bboxes)

    def inside_axes(b):
        # Keep entire label inside the axes box
        return axes_box.contains(b.x0, b.y0) and axes_box.contains(b.x1, b.y1)

    for a in annos:
        # Try candidates in increasing radius, cycling through the 8 directions
        chosen = None
        for r in radii:
            for dx, dy in dirs:
                # (0,0) only allowed if it doesn't overlap anything
                if r == 0 and (dx, dy) != (0,0):
                    continue
                a.set_position((dx * r, dy * r))  # textcoords="offset points"
                bb = a.get_window_extent(renderer=renderer)
                if not overlaps_any(bb) and inside_axes(bb):
                    chosen = bb
                    break
            if chosen is not None:
                break

        if chosen is None:
            # As a last resort, clamp to the smallest box that stays inside axes
            # (this keeps the label near the point instead of drifting off)
            a.set_position((0, 0))
            chosen = a.get_window_extent(renderer=renderer)

        placed_bboxes.append(chosen)



def scatter_gene_counts_adv(
    df, xcol, ycol, labelcol=None,
    point_size=40, color="steelblue", label_angle=20,
    figsize=(16, 9), fontname="Liberation Sans",
    y_step=10, repel="greedy", max_shift_px=24,
    fontsize=16,   # NEW: uniform text size (default 16)
):
    """
    Scatter with optional labels using a simple, predictable 'greedy' repeller.
    - Minimal spines, y-grid every 10 units (configurable via y_step)
    - Points never move; axes are frozen before placing labels.
    - repel: "greedy" (default) or None.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[xcol], df[ycol], s=point_size, c=color)

    # Axis labels & title
    ax.set_xlabel(xcol, fontname=fontname, fontsize=fontsize)
    ax.set_ylabel(f"{ycol} gene count", fontname=fontname, fontsize=fontsize)
    ax.set_title(f"{ycol} vs {xcol}", fontname=fontname, fontsize=fontsize)

    # Minimalist frame
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Ticks: size + family
    ax.tick_params(axis="both", labelsize=fontsize)
    for t in (ax.get_xticklabels() + ax.get_yticklabels()):
        t.set_fontname(fontname)

    # Y gridlines
    if y_step and y_step > 0:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.grid(axis="y", linestyle="-", alpha=0.25)

    # Freeze limits so labels don't rescale the axes
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.autoscale(False)

    annos = []
    if labelcol is not None:
        labels = df[labelcol] if labelcol in df.columns else df.index
        for (x, y, lab) in zip(df[xcol], df[ycol], labels):
            a = ax.annotate(
                str(lab), (x, y),
                textcoords="offset points", xytext=(0, 0),  # offsets set by greedy placer
                ha="left", va="bottom",
                style="italic",
                fontsize=fontsize, fontname=fontname, rotation=label_angle,
                arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.3),
                annotation_clip=True,
            )
            annos.append(a)

    if annos and repel == "greedy":
        # Limit search radius via radii derived from max_shift_px
        step = 6
        radii = [r for r in range(0, max_shift_px + 1, step)]
        if 0 not in radii:
            radii = [0] + radii
        _place_labels_greedy(ax, annos, max_shift_px=max_shift_px, radii=radii)

    return ax


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_busco_stacked(
    df,
    name_col="species",
    figsize=(10, 0.5),
    bar_height=0.7,
    annotate_min_pct=7.5,   # annotate segment labels only if >= this percent
    title=None,
    sort_desc=True,
):
    """
    Create a BUSCO-style horizontal stacked bar chart.

    Expects percentage columns (either 0..100 or 0..1):
      - "Single copy percentage"
      - "Multi copy percentage"
      - "Fragmented percentage"
      - "Missing percentage"

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `name_col` and the four BUSCO percentage columns.
    name_col : str, default "species"
        Column holding the names for each bar (species / assembly).
    figsize : (float, float) or (width, height-per-row), default (10, 0.5)
        If height <= 2, it's treated as "height per row" and scaled by n rows.
        Otherwise used as an absolute figure size (inches).
    bar_height : float, default 0.7
        Height of each horizontal bar.
    annotate_min_pct : float, default 7.5
        Minimum segment size (in percent units) to draw an internal label.
    title : str or None
        Optional plot title.
    sort_desc : bool, default True
        Sort by "Single copy percentage" (descending if True, ascending if False).

    Returns
    -------
    fig, ax : Matplotlib figure and axes
    """
    # Required columns
    cols = [
        "Single copy percentage",
        "Multi copy percentage",
        "Fragmented percentage",
        "Missing percentage",
    ]
    missing = [c for c in [name_col, *cols] if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Work on a copy; coerce to numeric
    work = df[[name_col, *cols]].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    # Convert to 0..100 if values look like 0..1
    # Heuristic: if the max across BUSCO cols <= 1.5, assume proportions
    if work[cols].to_numpy().max() <= 1.5:
        work[cols] = work[cols] * 100.0

    # Sort by single-copy %
    work = work.sort_values(
        by="Single copy percentage",
        ascending=not sort_desc
    ).reset_index(drop=True)

    # Figure size handling
    n = len(work)
    if figsize[1] <= 2.0:
        fig = plt.figure(figsize=(figsize[0], max(2.5, n * figsize[1])))
    else:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # BUSCO-like colors
    color_map = {
        "Single copy percentage": "#2c47a0",  # green
        "Multi copy percentage":  "#1f9eb4",  # blue
        "Fragmented percentage":  "#ffd30e",  # orange
        "Missing percentage":     "#d62728",  # red
    }

    # Stack segments
    y_pos = np.arange(n)
    left = np.zeros(n)
    handles = []
    labels = []

    for col in cols:
        vals = work[col].to_numpy()
        h = ax.barh(
            y_pos, vals, left=left, height=bar_height,
            label=col.replace(" percentage", ""),
            color=color_map[col], edgecolor="none"
        )
        handles.append(h)
        labels.append(col.replace(" percentage", ""))
        # Annotate segments large enough
        for i, v in enumerate(vals):
            if v >= annotate_min_pct:
                ax.text(
                    left[i] + v/2.0, y_pos[i],
                    f"{v:.0f}%",
                    va="center", ha="center", fontsize=9, color="white"
                )
        left += vals

    # Y labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(work[name_col], fontsize=10)

    # X axis as percent 0..100
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend
    leg = ax.legend(
        handles=[h[0] for h in handles], labels=labels,
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncols=4, frameon=False
    )

    # Title & labels
    ax.set_xlabel("BUSCO categories (% of genes)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    return fig, ax

# optional: a cached constructor for heavy use in filters
@lru_cache(maxsize=4096)
def parse_chc(code: str) -> CHC:
    return CHC(code)


if __name__ == "__main__":
    print(CHC("C30DiMe5").as_dict())
    print(CHC("C30intDiMe").as_dict())
    print(CHC("C30nAlkane").as_dict())
    print(CHC("C30alkene").as_dict())
    print(CHC("C29MonoMe3diene").as_dict())
    print(CHC("C33unknown").as_dict())
    print(CHC("C33unknownCHC").as_dict())
    print(CHC("C33methylalkene").as_dict())
    print(CHC("C24MonoMe2_DiMe5").as_dict())
    print(CHC("C24MonoMe2_DiMe5").as_dict())
    print(CHC("C27MonoMe3_intDiMe").as_dict())
    print(CHC("C28MonoMe4_ene").as_dict())