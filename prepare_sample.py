import json
import math
from pathlib import Path

import pandas as pd

CACHE = "ppc_cache_sejm-posiedzenia.pkl.gz"
CLUBS = ["PiS", "KO", "Konfederacja", "PSL-TD", "Polska2050-TD", "Lewica"]
N_CAL, ANNOTATORS, N_EACH = 50, 3, 200
MIN_W, MAX_W = 25, 90
MAX_PER_SPEAKER = 10
AGGRESSIVE = ["Grzegorz Braun", "Sławomir Mentzen", "Janusz Kowalski", "Anna Żukowska"]
AGGRESSIVE_SHARE = 0.5
SEED = 69

df = pd.read_pickle(CACHE)
df = df[df.who != "komentarz"]
frequent = df.text.value_counts().head(50).index
df = df[
    df.club.isin(CLUBS)
    & ~df.role_name.fillna("").str.contains("arsza|rzewodnicz")
    & ~df.text.isin(frequent)
]

rows = []
for turn, g in df.groupby("turn", sort=False):
    buf, words = [], 0
    for text in g.text:
        buf.append(text)
        words += len(text.split())
        if words >= MIN_W:
            if words <= MAX_W:
                rows.append(
                    {
                        "id": f"{turn}:{len(rows)}",
                        "text": " ".join(buf),
                        "speaker": g.speaker.iloc[0],
                        "club": g.club.iloc[0],
                        "date": g.date.iloc[0].date(),
                    }
                )
            buf, words = [], 0

pool = pd.DataFrame(rows).sample(frac=1, random_state=SEED)
agg = pool[pool.speaker.isin(AGGRESSIVE)]
agg = agg.assign(rank=agg.groupby("speaker").cumcount()).sort_values("rank", kind="stable")
pool = pool[~pool.id.isin(agg.id)]
pool = pool[pool.groupby("speaker").cumcount() < MAX_PER_SPEAKER]

n_agg = math.ceil(N_CAL * AGGRESSIVE_SHARE)
n_bal = N_CAL - n_agg
total = 2 * N_CAL + ANNOTATORS * N_EACH
pool = pool.groupby("club").head(-(-total // len(CLUBS)))
pool = (
    pool.assign(rank=pool.groupby("club").cumcount())
    .sort_values("rank", kind="stable")
    .head(total)
)
print(pool.club.value_counts())

out = Path("sample")
out.mkdir(exist_ok=True)


def save(name, part):
    tasks = [
        {"data": {"id": r.id, "text": r.text, "meta": f"{r.speaker} · {r.club} · {r.date}"}}
        for r in part.itertuples()
    ]
    (out / name).write_text(json.dumps(tasks, ensure_ascii=False, indent=1))


def calibration(balanced, extra):
    return pd.concat([balanced, extra]).sample(frac=1, random_state=SEED)


iter1 = calibration(pool[:n_bal], agg[agg["rank"] % 2 == 0].head(n_agg))
iter2 = calibration(pool[n_bal : 2 * n_bal], agg[agg["rank"] % 2 == 1].head(n_agg))
save("iter1.json", iter1)
save("iter2.json", iter2)
rest = [
    pool[2 * n_bal + i * N_EACH : 2 * n_bal + (i + 1) * N_EACH]
    for i in range(ANNOTATORS)
]
for i, part in enumerate(rest):
    save(f"rest_{i + 1}.json", part)
pd.concat([iter1, iter2, *rest])[["id", "speaker", "club", "date"]].to_csv(
    out / "meta.csv", index=False
)
