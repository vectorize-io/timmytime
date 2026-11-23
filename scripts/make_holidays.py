"""Writes tlm/holidays.json: lunar holidays per Gregorian year, 1950-2076.

    uv run --no-project --with lunardate --with hijridate python scripts/make_holidays.py

Run once; the table ships with the package so runtime needs no calendar library.
Islamic dates follow Umm al-Qura (hijridate); real observance can differ by a day.
"""

import json
from datetime import timedelta

from hijridate import Hijri
from lunardate import LunarDate

YEARS = range(1950, 2077)
out = {k: {} for k in ("cny", "midautumn", "dragonboat", "ramadan", "eidfitr", "eidadha")}


def put(name, start, end):
    if start.year in YEARS:
        out[name].setdefault(str(start.year), [start.isoformat(), end.isoformat()])  # first one in that year


for ly in range(1949, 2077):
    ny = LunarDate(ly, 1, 1).to_solar_date()
    put("cny", ny, ny + timedelta(days=6))  # the 7-day Spring Festival holiday
    for name, (m, d) in {"midautumn": (8, 15), "dragonboat": (5, 5)}.items():
        x = LunarDate(ly, m, d).to_solar_date()
        put(name, x, x)

for hy in range(1369, 1500):
    r0 = Hijri(hy, 9, 1).to_gregorian()
    shawwal = Hijri(hy, 10, 1).to_gregorian()
    put("ramadan", r0, shawwal - timedelta(days=1))
    put("eidfitr", shawwal, shawwal + timedelta(days=2))
    adha = Hijri(hy, 12, 10).to_gregorian()
    put("eidadha", adha, adha + timedelta(days=3))

json.dump(out, open("tlm/holidays.json", "w"), separators=(",", ":"), sort_keys=True)
print({k: len(v) for k, v in out.items()})
