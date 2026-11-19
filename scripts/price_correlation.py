"""How well the ERCOT day-ahead (DAM) price predicts the real-time (RT) price on ordinary days.

Prints the three correlations quoted in the README's day-ahead section:
1. the average daily profiles: mean price at each 15-minute time of day, averaged over ordinary days
   (96 points per series);
2. individual 15-minute intervals: RT price against the DAM price for the same interval;
3. the baseline for (2): RT price against the RT price one hour (4 steps) earlier on the same day.

"Ordinary days" are days whose maximum RT price is below $1/kWh, as everywhere in the report. All 365 days of
2018 are used, not only the test split. Needs data/processed/austin_2018.parquet with the `dam` column
(built by scripts/download_data.py).
"""

from battery_rl.data import load_dataset

df = load_dataset()
ordinary = df[df.groupby("day")["price"].transform("max") < 1.0].copy()
ordinary["price_1h_earlier"] = ordinary.groupby("day")["price"].shift(4)

profile_rt = ordinary.groupby("step")["price"].mean()
profile_dam = ordinary.groupby("step")["dam"].mean()

rows = {
    "average daily profiles, DAM vs RT": profile_rt.corr(profile_dam),
    "15-minute intervals, DAM vs RT": ordinary["price"].corr(ordinary["dam"]),
    "15-minute intervals, RT 1 h earlier vs RT": ordinary["price"].corr(ordinary["price_1h_earlier"]),
}
print(f"ordinary days: {ordinary['day'].nunique()} of {df['day'].nunique()}")
for name, r in rows.items():
    print(f"{name:42s} {r:.2f}")
