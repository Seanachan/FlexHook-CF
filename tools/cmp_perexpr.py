"""
Per-expression HOTA/DetA/AssA comparison between two TrackEval runs.

Reports the MANDATORY metric (per-expression mean of *___AUC over the 862
expression rows), NOT the pooled COMBINED row (a documented artifact that has
'lied' 3 times). Also surfaces the over-rejection diagnostic: expressions the
baseline scores HOTA==0 (complete miss = head rejected the whole trajectory)
and whether the challenger rescues them, plus the zero-HOTA count delta.

Usage:
  python tools/cmp_perexpr.py <baseline_csv> <challenger_csv> [baseline_name] [challenger_name]
"""
import csv, sys

HOTA, DETA, ASSA = 23, 43, 63          # 0-indexed cols for *___AUC
DETS, GT_DETS = -4, -3                 # Dets, GT_Dets

def load(path):
    rows = {}
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if not row or row[0] == 'COMBINED':
                continue
            rows[row[0]] = row
    return rows

def f(x):
    try: return float(x)
    except: return 0.0

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

base_p, chal_p = sys.argv[1], sys.argv[2]
bname = sys.argv[3] if len(sys.argv) > 3 else 'baseline'
cname = sys.argv[4] if len(sys.argv) > 4 else 'challenger'

base = load(base_p)
chal = load(chal_p)
keys = sorted(set(base) & set(chal))
print(f"matched expressions: {len(keys)}  (baseline {len(base)}, challenger {len(chal)})")
if len(keys) < min(len(base), len(chal)):
    print(f"  WARNING: {len(set(base)^set(chal))} keys not shared")

def summarize(rows, keys, name):
    h = [f(rows[k][HOTA]) for k in keys]
    d = [f(rows[k][DETA]) for k in keys]
    a = [f(rows[k][ASSA]) for k in keys]
    zero = sum(1 for x in h if x == 0.0)
    print(f"\n[{name}] per-expression mean over {len(keys)} expr:")
    print(f"  HOTA {mean(h)*100:.3f}  DetA {mean(d)*100:.3f}  AssA {mean(a)*100:.3f}")
    print(f"  zero-HOTA expressions: {zero} ({zero/len(keys)*100:.1f}%)")
    return h, d, a, zero

bh, bd, ba, bz = summarize(base, keys, bname)
ch, cd, ca, cz = summarize(chal, keys, cname)

print(f"\n=== DELTA ({cname} - {bname}), per-expression mean ===")
print(f"  HOTA {(mean(ch)-mean(bh))*100:+.3f}")
print(f"  DetA {(mean(cd)-mean(bd))*100:+.3f}")
print(f"  AssA {(mean(ca)-mean(ba))*100:+.3f}")
print(f"  zero-HOTA count: {bz} -> {cz} ({cz-bz:+d})")

# over-rejection set: baseline HOTA == 0 (complete miss). Does challenger rescue?
bm = {k: f(base[k][HOTA]) for k in keys}
overrej = [k for k in keys if bm[k] == 0.0]
if overrej:
    resc_h = [f(chal[k][HOTA]) for k in overrej]
    resc_d = [f(chal[k][DETA]) for k in overrej]
    rescued = sum(1 for x in resc_h if x > 0.0)
    print(f"\n=== OVER-REJECTION SET (baseline HOTA==0: {len(overrej)} expr) ===")
    print(f"  challenger HOTA on these: mean {mean(resc_h)*100:.3f}  DetA {mean(resc_d)*100:.3f}")
    print(f"  rescued (HOTA>0 now): {rescued}/{len(overrej)} ({rescued/len(overrej)*100:.1f}%)")

# DetA-tax check: did any baseline-strong expr collapse? (regression guard)
collapsed = [k for k in keys if f(base[k][HOTA]) > 0.3 and f(chal[k][HOTA]) < 0.15]
print(f"\n=== REGRESSION GUARD ===")
print(f"  baseline-strong (HOTA>0.3) expr that collapsed (<0.15): {len(collapsed)}")
for k in collapsed[:15]:
    print(f"    {k}: {f(base[k][HOTA])*100:.1f} -> {f(chal[k][HOTA])*100:.1f}")
if len(collapsed) > 15:
    print(f"    ... +{len(collapsed)-15} more")
