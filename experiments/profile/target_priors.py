import json
from pathlib import Path
KIT=Path("provided/techjam-conversational-search").resolve()
tg={json.loads(l)["ground_truth"]["parent_asin"] for l in (KIT/"data"/"public_set.jsonl").read_text().splitlines() if l.strip()}
rows=[]
for l in (KIT/"data"/"catalog.jsonl").open():
    p=json.loads(l)
    rows.append((p["parent_asin"] in tg, p.get("rating_number") or 0, isinstance(p.get("price"),(int,float)), p.get("average_rating") or 0))
N=len(rows); NT=sum(1 for r in rows if r[0])
print(f"catalog={N} targets={NT}\n")
print(f"{'filter':38} {'pool':>8} {'pool%':>7} {'target recall':>14}")
def rep(name,f):
    pool=[r for r in rows if f(r)]
    t=sum(1 for r in pool if r[0])
    print(f"{name:38} {len(pool):8,} {100*len(pool)/N:6.2f}% {100*t/NT:13.1f}%")
for th in (0,50,100,300,500,1000,2000,5000):
    rep(f"rating_number >= {th}", lambda r,th=th: r[1]>=th)
print()
rep("price present", lambda r: r[2])
rep("price present AND rating_n>=300", lambda r: r[2] and r[1]>=300)
rep("price present AND rating_n>=1000", lambda r: r[2] and r[1]>=1000)
rep("price present AND rating_n>=500 AND ar>=4.0", lambda r: r[2] and r[1]>=500 and r[3]>=4.0)
