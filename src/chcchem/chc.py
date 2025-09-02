from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache

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