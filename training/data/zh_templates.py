"""Short Chinese queries from templates, with their codes: the search-box phrasing (上上周, 三天内, 明天起,
6月5到8号, 周杰伦) that LLM-written sentences rarely use. Training data only; the runtime has no rules.

    python -m training.data.zh_templates --n 30000 --seed 1 --out data/train_zh_tpl.jsonl

A share of rows is in traditional characters (lang zh-hant). Rows equal to a test question are
dropped later by training.data.clean, like every training file.
"""

import argparse
import json
import random
from datetime import date, datetime, timedelta

from timmytime.resolve import resolve
from training.data.meanings import _day

NUM = "零一二三四五六七八九十"


def cn(n: int, r: random.Random, two: bool = True) -> str:
    """Chinese numeral, or digits about a third of the time."""
    if r.random() < 0.3:
        return str(n)
    if n == 2 and two:
        return r.choice(["两", "两", "二", "俩"]) if r.random() < 0.5 else "两"
    if n <= 10:
        return NUM[n]
    if n < 20:
        return "十" + (NUM[n - 10] if n > 10 else "")
    if n >= 100:
        h, rest = divmod(n, 100)
        head = NUM[h] + "百"
        if rest == 0:
            return head
        if rest < 10:
            return head + "零" + NUM[rest]
        return head + NUM[rest // 10] + "十" + (NUM[rest % 10] if rest % 10 else "")
    t, u = divmod(n, 10)
    return NUM[t] + "十" + (NUM[u] if u else "")


PAST = ["的记录", "做了什么", "的笔记", "提到的事", "讨论了什么", "的会议", "发生了什么", "买了什么", "的费用", "聊了什么", "的照片",
        "讨论了这个", "的活动", "提到的东西是什么", "说了什么", "吃了什么", "见了谁", "", "", "呢", "？", "吗", "的", "的事"]
FUTURE = ["的计划", "计划", "安排", "要做什么", "去哪", "提醒我", "的会议", "的行程", "有什么事", "", "", "呢", "？", "吗", "的", "的事"]
ANY = PAST + FUTURE
WD = ["一", "二", "三", "四", "五", "六", "日"]
WD_CODE = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def wd_word(i: int, r: random.Random) -> str:
    w = WD[i] if i < 6 else r.choice(["日", "天"])
    return r.choice(["星期", "周", "礼拜"]) + w


def year_word(y: int, r: random.Random) -> str:
    if r.random() < 0.25:
        zero = r.choice("〇零○")
        return "".join(zero if c == "0" else NUM[int(c)] for c in str(y)) + "年"
    return f"{y}年"


def day_word(d: int, r: random.Random) -> str:
    if r.random() < 0.2 and d in (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        return ("廿" + (NUM[d - 20] if d > 20 else "")) if d < 30 else ("卅" + (NUM[d - 30] if d > 30 else ""))
    return cn(d, r, two=False) + r.choice(["日", "日", "号"])


def month_word(m: int, r: random.Random) -> str:
    return (cn(m, r, two=False) if m != 2 else r.choice(["二", "2"])) + "月"


def t_day(r):
    n, word = r.choice([(0, "今天"), (0, "今日"), (0, "本日"), (0, "当天"), (0, "当日"), (-1, "昨天"), (-1, "昨日"), (-2, "前天"),
                        (-3, "大前天"), (-4, "大大前天"), (1, "明天"), (1, "明日"), (2, "后天"), (3, "大后天"), (4, "大大后天"),
                        (0, "今晚"), (0, "今早"), (0, "今天早上"), (-1, "昨晚"), (1, "明早"), (1, "明天晚上"), (-2, "前晚")])
    return word + r.choice(ANY), f"day {n}"


def t_week(r):
    n, word = r.choice([(0, "这周"), (0, "本周"), (0, "当周"), (0, "这个星期"), (0, "这礼拜"), (-1, "上周"), (-1, "上星期"),
                        (-1, "上个星期"), (-1, "上一个星期"), (-1, "上礼拜"), (-1, "前一周"), (-1, "前一个星期"), (-2, "上上周"),
                        (-2, "大上周"), (-2, "上上个星期"), (1, "下周"), (1, "下一周"), (1, "下个星期"), (1, "下礼拜"),
                        (1, "下一个星期"), (2, "下下周"), (2, "大下周"), (2, "下下个星期")])
    k = r.random()
    if k < 0.3:
        if n == 0 and r.random() < 0.3:
            return word + r.choice(["以内", "内", "之内", "期间"]) + r.choice(ANY), "week 0"
        return word + r.choice(ANY), f"week {n}"
    if k < 0.5:
        end = "周末" if word.endswith("周") else "末" if word.endswith("星期") else "周末"
        base = word[:-1] if word.endswith("周") else word
        if r.random() < 0.5 and word.endswith("周"):
            return word + "末" + r.choice(ANY), f"week {n} weekend"
        return base + end + r.choice(ANY), f"week {n} weekend"
    i = r.randrange(7)
    return word + wd_word(i, r).removeprefix("周" if word.endswith("周") else "") + r.choice(ANY), f"week {n} {WD_CODE[i]}"


def t_bare_weekday(r):
    i = r.randrange(7)
    if r.random() < 0.5:
        return wd_word(i, r) + r.choice(["去哪", "安排", "要做什么", "的计划", "有什么事"]), f"{WD_CODE[i]} 1"
    return wd_word(i, r) + r.choice(["做了什么", "的记录", "开了什么会", "说了什么", ""]), WD_CODE[i]


def t_weekend(r):
    if r.random() < 0.5:
        return r.choice(["周末", "这个周末", "这周末", "本周末"]) + r.choice(FUTURE), "week 0 weekend"
    return r.choice(["上周末", "上个周末", "上星期末", "上礼拜周末"]) + r.choice(PAST), "week -1 weekend"


def t_month(r):
    n, word = r.choice([(0, "这个月"), (0, "本月"), (0, "当月"), (0, "这一个月"), (0, "这月"), (-1, "上个月"), (-1, "上一个月"),
                        (-1, "前一个月"), (-1, "上月"), (-2, "上上个月"), (-2, "大上个月"), (-2, "上上月"), (1, "下个月"),
                        (1, "下一个月"), (1, "下月"), (2, "下下个月"), (2, "大下个月"), (2, "下下月")])
    k = r.random()
    if k < 0.7:
        return word + r.choice(ANY), f"month {n}"
    mod, w = r.choice([("early", "初"), ("early", "上旬"), ("mid", "中旬"), ("late", "底"), ("late", "末"), ("late", "下旬")])
    return word + w + r.choice(ANY), f"{mod} month {n}"


def t_month_part(r):
    mod, w = r.choice([("early", "上旬"), ("mid", "中旬"), ("late", "下旬"), ("early", "月初"), ("late", "月底"), ("late", "月尾")])
    return w + r.choice(ANY), f"{mod} month 0"


def t_year(r):
    n, word = r.choice([(0, "今年"), (0, "本年"), (0, "当年"), (0, "本年度"), (-1, "去年"), (-1, "上一年"), (-1, "前一年"),
                        (-1, "上一年度"), (-2, "前年"), (-3, "大前年"), (1, "明年"), (1, "下一年"), (1, "下一年度"),
                        (1, "下一个年度"), (2, "后年"), (3, "大后年")])
    k = r.random()
    if k < 0.45:
        return word + r.choice(ANY), f"year {n}"
    if k < 0.6:
        mod, w = r.choice([("early", "初"), ("late", "底"), ("late", "尾"), ("late", "末"), ("h1", "上半年"), ("h2", "下半年")])
        if w in ("上半年", "下半年"):
            word = word.removesuffix("年度").removesuffix("年") + ("年" if not word.endswith("年") and not word.endswith("年度") else "")
            word = {"今": "今年", "本": "今年", "当": "今年", "去": "去年", "上一": "去年", "前一": "去年", "前": "前年",
                    "大前": "大前年", "明": "明年", "下一": "明年", "下一个": "明年", "后": "后年", "大后": "大后年"}.get(word.rstrip("年"), word)
        return word + w + r.choice(ANY), f"{mod} year {n}"
    if k < 0.8:
        q = r.randint(1, 4)
        return word + r.choice(["第", ""]) + cn(q, r, two=False) + r.choice(["季度", "季"]) + r.choice(ANY), f"year {n} q{q}"
    m = r.randint(1, 12)
    return word + month_word(m, r) + r.choice(ANY), f"year {n} {['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'][m-1]}"


def t_bare_year_end(r):
    mod, w = r.choice([("early", "年初"), ("late", "年底"), ("late", "年尾"), ("late", "年末"), ("mid", "年中")])
    return w + r.choice(ANY), f"{mod} year 0"


def t_quarter(r):
    n, word = r.choice([(0, "本季度"), (0, "这个季度"), (0, "这季度"), (-1, "上季度"), (-1, "上个季度"), (-1, "上一季度"),
                        (-2, "上上季度"), (-2, "上上个季度"), (1, "下季度"), (1, "下个季度"), (1, "下一季度"), (2, "下下季度")])
    if r.random() < 0.7:
        return word + r.choice(ANY), f"quarter {n}"
    q = r.randint(1, 4)
    return "第" + cn(q, r, two=False) + "季度" + r.choice(ANY), f"q{q}"


def t_ago(r):
    unit, cnw = r.choice([("day", "天"), ("week", "周"), ("week", "个星期"), ("week", "週"), ("month", "个月"), ("year", "年")])
    n = r.choice(range(1, 13)) if unit != "day" else r.choice([*range(1, 31), 40, 45, 50, 60, 70, 80, 90, 99])
    if r.random() < 0.4:  # small counts are the common ones: 两天前, 三天前, 两年前
        n = r.choice([1, 2, 2, 2, 3, 3, 4, 5])
    if unit == "month" and r.random() < 0.3:
        n = r.choice([13, 18, 22, 24])
    word = cn(n, r) + cnw
    if n == 2 and unit == "month" and r.random() < 0.3:
        word = r.choice(["俩月", "两个月"])
    if r.random() < 0.5:
        return word + r.choice(["前", "以前", "之前"]) + r.choice(PAST), f"{unit} -{n}"
    return word + r.choice(["后", "以后", "之后"]) + r.choice(FUTURE), f"{unit} {n}"


def t_ago_big(r):
    n = r.choice([100, 100, 120, 150, 180, 200, 300, r.randint(100, 365), r.randint(100, 365)])
    if r.random() < 0.5:
        return cn(n, r) + "天" + r.choice(["前", "以前", "之前"]) + r.choice(PAST), f"day -{n}"
    return cn(n, r) + "天" + r.choice(["后", "以后", "之后"]) + r.choice(FUTURE), f"day {n}"


def t_month_name_part(r):
    m = r.randint(1, 12)
    mod, w = r.choice([("early", "初"), ("early", "上旬"), ("early", "初期"), ("mid", "中旬"), ("late", "底"), ("late", "末"),
                       ("late", "下旬"), ("late", "尾")])
    mon = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][m - 1]
    return month_word(m, r) + w + r.choice(ANY), f"{mod} {mon}"


def t_half(r):
    word, code = r.choice([("半个月前", "day -15"), ("半个月后", "day 15"), ("一个半月前", "day -45"), ("一个半月后", "day 45"),
                           ("半年前", "month -6"), ("半年后", "month 6"), ("一年半前", "month -18"), ("一年半后", "month 18"),
                           ("两年半前", "month -30"), ("两年半后", "month 30")])
    return word + r.choice(PAST if "前" in word else FUTURE), code


def t_fuzzy(r):
    word, code = r.choice([("前两天", "couple day"), ("两三天前", "couple day"), ("三两天前", "couple day"), ("一两天前", "couple day"),
                           ("前几天", "few day"), ("几天前", "few day"), ("数天前", "few day"), ("一两周前", "couple week"),
                           ("两三周前", "couple week"), ("一两个星期前", "couple week"), ("几周前", "few week"),
                           ("几个星期前", "few week"), ("几个礼拜前", "few week"), ("一两个月前", "couple month"),
                           ("两三个月前", "couple month"), ("几个月前", "few month"), ("一两年前", "couple year"),
                           ("两三年前", "couple year"), ("几年前", "few year"), ("三四天前", "day -4 .. day -3"),
                           ("四五天前", "day -5 .. day -4"), ("四五周前", "week -5 .. week -4"), ("三四周前", "week -4 .. week -3")])
    return word + r.choice(PAST), code


def t_rolling(r):
    k = r.random()
    if k < 0.45:
        unit, w = r.choice([("day", "天"), ("week", "周"), ("week", "个星期"), ("month", "个月"), ("year", "年")])
        n = r.randint(1, 12)
        nw = cn(n, r)
        form = r.choice(["过去{}", "最近{}", "近{}", "{}内", "{}以内", "前{}", "这{}", "过去{}以内"])
        if form == "这{}" and n < 2:
            form = "过去{}"
        if form == "前{}" and (n == 1 or n == 2 and unit == "day"):  # 前一个月 = the month before; 前两天 = a couple of days ago
            form = "最近{}"
        return form.format(nw + w) + r.choice(PAST), f"last {n} {unit}"
    if k < 0.65:
        unit, w = r.choice([("day", "天"), ("week", "周"), ("week", "个星期"), ("month", "个月"), ("year", "年")])
        n = r.randint(1, 12)
        return r.choice(["未来", "接下来", "接下来的", "今后"]) + cn(n, r) + w + r.choice(FUTURE), f"next {n} {unit}"
    word, code = r.choice([("这几天", "last 5 day"), ("最近几天", "last 5 day"), ("近几天", "last 5 day"), ("最近几周", "last 5 week"),
                           ("过去几个星期", "last 5 week"), ("最近几个月", "last 5 month"), ("过去几个月", "last 5 month"),
                           ("过去一两个月", "last 3 month"), ("最近一两周", "last 3 week"), ("最近半个月", "last 15 day"),
                           ("半个月内", "last 15 day"), ("过去24小时", "last 1 day"), ("最近24小时", "last 1 day"),
                           ("未来几天", "next 5 day"), ("接下来几天", "next 5 day"), ("未来几周", "next 5 week"),
                           ("接下来几周", "next 5 week"), ("未来几个月", "next 5 month"), ("接下来一两周", "next 3 week"),
                           ("未来一两个月", "next 3 month"), ("未来半年", "next 6 month"), ("未来半个月", "next 15 day"),
                           ("未来24小时", "next 1 day"), ("两三天后", "day 2 .. day 3"), ("三四天后", "day 3 .. day 4"),
                           ("几天后", "day 2 .. day 5"), ("过几天", "day 2 .. day 5"), ("这两三天", "last 3 day"),
                           ("最近两三天", "last 3 day"), ("近两三天", "last 3 day"), ("这三两天", "last 3 day"),
                           ("过去两三天", "last 3 day"), ("最近三四天", "last 4 day"), ("这三四天", "last 4 day")])
    return word + r.choice(PAST if code.startswith("last") else FUTURE), code


def _since_base(r):
    y = r.randint(2015, 2026)
    m = r.randint(1, 12)
    return r.choice([("去年", "year -1"), ("今年", "year 0"), ("上个月", "month -1"), ("这个月", "month 0"), ("上周", "week -1"),
                     ("去年底", "late year -1"), ("今年初", "early year 0"), ("去年第四季度", "year -1 q4"),
                     ("上个季度", "quarter -1"), ("三天前", "day -3"), ("两周前", "week -2"), ("昨天", "day -1"),
                     (year_word(y, r), str(y)), (year_word(y, r) + month_word(m, r), f"{y}-{m:02d}"),
                     (year_word(y, r) + month_word(m, r) + "下旬", f"late {y}-{m:02d}"),
                     (year_word(y, r) + month_word(m, r) + "初", f"early {y}-{m:02d}")])


def t_since(r):
    w, c = _since_base(r)
    form = r.choice(["{}以来", "{}开始", "从{}开始", "从{}起", "{}起", "{}至今", "{}迄今", "自{}以来", "{}到现在", "{}以后到现在"])
    return form.format(w) + r.choice(PAST), f"{c} .. now"


def t_from_on(r):
    w, c = r.choice([("明天", "day 1"), ("后天", "day 2"), ("下周", "week 1"), ("下个月", "month 1"), ("明年", "year 1"),
                     ("三天后", "day 3"), ("五天后", "day 5"), ("两周后", "week 2"), ("下周一", "week 1 mon")])
    return w + r.choice(["起", "开始", "以后", "之后"]) + r.choice(["的计划", "的安排", "要做的事", "的行程"]), f"{c} .."


def t_before_after(r):
    y = r.randint(2010, 2030)
    m = r.randint(1, 12)
    d = _day(r, y, m)
    w, c = r.choice([(year_word(y, r), str(y)), (year_word(y, r) + month_word(m, r), f"{y}-{m:02d}"),
                     (year_word(y, r) + month_word(m, r) + "份", f"{y}-{m:02d}"),
                     (year_word(y, r) + month_word(m, r) + day_word(d, r), f"{y}-{m:02d}-{d:02d}"),
                     ("今年", "year 0"), ("去年", "year -1"), ("上周", "week -1"), ("这周", "week 0"), ("这个月", "month 0"),
                     ("上个月", "month -1"), ("下个月", "month 1"), ("明年", "year 1")])
    if r.random() < 0.55:
        return w + r.choice(["前", "之前", "以前"]) + r.choice(PAST), f"before {c}"
    return w + r.choice(["后", "之后", "以后"]) + r.choice(FUTURE), f"after {c}"


def t_dayrange(r):
    k = r.random()
    if k < 0.35:
        w, c = r.choice([("明后天", "day 1 .. day 2"), ("明后两天", "day 1 .. day 2"), ("今明两天", "day 0 .. day 1"),
                         ("今明天", "day 0 .. day 1"), ("昨今两天", "day -1 .. now"), ("昨天和今天", "day -1 .. now"),
                         ("前天到今天", "day -2 .. now"), ("这两天", "last 2 day"), ("这三天", "last 3 day")])
        return w + r.choice(ANY), c
    if k < 0.55:
        a, b = sorted(r.sample(range(7), 2))
        if r.random() < 0.5:
            return wd_word(b, r) + r.choice(["到", "至"]) + wd_word(a, r) + r.choice(FUTURE), f"week 0 {WD_CODE[b]} .. week 1 {WD_CODE[a]}"
        return wd_word(a, r) + r.choice(["到", "至"]) + wd_word(b, r) + r.choice(FUTURE), f"week 0 {WD_CODE[a]} .. week 0 {WD_CODE[b]}"
    y, m = r.randint(2015, 2027), r.randint(1, 12)
    d2 = max(2, _day(r, y, m))
    d1 = r.randint(1, d2 - 1)
    sep = r.choice(["到", "至", "-", "~"])
    tail = r.choice(["日", "号", ""])
    head = month_word(m, r) + str(d1) + r.choice(["", "日", "号"]) + sep + str(d2) + tail
    if r.random() < 0.6:
        return year_word(y, r) + head + r.choice(ANY), f"{y}-{m:02d}-{d1:02d} .. {y}-{m:02d}-{d2:02d}"
    return head + r.choice(ANY), f"*-{m:02d}-{d1:02d} .. *-{m:02d}-{d2:02d}"


def t_explicit(r):
    y, m = r.randint(1990, 2030), r.randint(1, 12)
    if r.random() < 0.4:
        return year_word(y, r) + month_word(m, r) + r.choice(["", "份"]) + r.choice(ANY), f"{y}-{m:02d}"
    if r.random() < 0.5:
        m2 = r.randint(1, 12)
        return month_word(m2, r) + r.choice(["中了奖", "的活动", "份的账单", "去了哪", "的记录"]), ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][m2 - 1]
    if r.random() < 0.4:  # no year: 3月30日的会议
        d = _day(r, None, m)
        return month_word(m, r) + day_word(d, r) + r.choice(ANY), f"*-{m:02d}-{d:02d}"
    d = _day(r, y, m)
    return year_word(y, r) + month_word(m, r) + day_word(d, r) + r.choice(ANY), f"{y}-{m:02d}-{d:02d}"


def t_zhong(r):
    """中 right after a period is usually a verb (中了奖 = won a prize), and 曾经 (once) is not a date."""
    m = r.randint(1, 12)
    w, c = r.choice([("本月", "month 0"), ("这个月", "month 0"), ("上个月", "month -1"), ("今年", "year 0"), ("去年", "year -1"),
                     (month_word(m, r), ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][m - 1])])
    return w + r.choice(["中了奖", "中了彩票", "中标了", "中签了", "曾经做过什么", "曾经去过哪", "都做了什么"]), c


def t_sameday(r):
    n, w = r.choice([(-1, "去年今天"), (-1, "去年本日"), (-1, "去年今日"), (-1, "去年的今天"), (1, "明年今天"), (1, "明年今日"),
                     (1, "明年今晚"), (-2, "前年今天"), (-3, "大前年今天")])
    return w + r.choice(ANY), f"year {n} now"


SINGERS = ["杰伦", "星驰", "润发", "冬雨", "深", "笔畅", "华健", "慧敏"]


def t_trap(r):
    k = r.random()
    if k < 0.3:
        return r.choice(["上", "下", "这", "听", "看", "和", ""]) + "周" + r.choice(SINGERS) + r.choice(["的歌", "电影", "的演唱会", "新歌", "的专辑", "的采访"]), "none"
    if k < 0.45:
        return r.choice(["上个", "下个", "这个", "今晚的", ""]) + r.choice(["月亮很圆", "月亮好亮", "月光真美"]), "none"
    if k < 0.55:
        return r.choice(["本月饼", "这月饼", "五仁月饼", "买月饼"]) + r.choice(["很好吃", "多少钱", "哪家好"]), "none"
    if k < 0.62:
        return r.choice(["今年糕", "去年糕", "明年糕", "年糕", "买年糕", "炒年糕"]) + r.choice(["点", "点心", "怎么做", "好吃吗", "多少钱", "的做法"]), "none"
    if k < 0.7:
        return r.choice(["周末端", "末端", "前端", "后端", "终端"]) + r.choice(["项目", "设备", "开发", "测试", "代码"]), "none"
    if k < 0.66:
        return r.choice(["前端和后端", "末端设备", "后天免疫因素", "明日方舟攻略", "明日之后攻略", "今日头条新闻",
                         "明日香", "今日说法", "周刊"]), "none"
    if k < 0.8:
        return r.choice(["庆余年", "甄嬛传", "繁花", "狂飙", "三体"]) + "第" + cn(r.randint(1, 4), r, two=False) + r.choice(["季剧情", "集讲了什么", "部"]), "none"
    if k < 0.9:
        return r.choice(["每周", "每个周末", "每年", "每个月", "隔周", "每个星期", "每天"]) + r.choice(
            ["一开会", "做什么", "上半年计划", "一到周五排班", "一至星期五开会", "的例会", "健身"]), "none"
    return r.choice(["会议之后三天", "出院之后两周", "搬家以后三个月", "入职后一年", "大大大后天", "大大大前天"]) + r.choice(ANY), "none"


FAMILIES = [(t_day, 5), (t_week, 7), (t_bare_weekday, 3), (t_weekend, 2), (t_month, 6), (t_month_part, 2), (t_year, 6),
            (t_bare_year_end, 1), (t_quarter, 3), (t_ago, 6), (t_ago_big, 3), (t_half, 2), (t_fuzzy, 6), (t_month_name_part, 4), (t_rolling, 8), (t_since, 5),
            (t_from_on, 3), (t_before_after, 5), (t_dayrange, 7), (t_explicit, 4), (t_zhong, 2), (t_sameday, 2), (t_trap, 7)]

HANT = str.maketrans({"这": "這", "个": "個", "过": "過", "来": "來", "后": "後", "内": "內", "记": "記", "录": "錄", "计": "計",
                      "划": "畫", "周": "週", "礼": "禮", "两": "兩", "俩": "倆", "开": "開", "会": "會", "讨": "討", "论": "論",
                      "发": "發", "么": "麼", "时": "時", "说": "說", "买": "買", "费": "費", "还": "還", "剧": "劇", "电": "電",
                      "饼": "餅", "庆": "慶", "余": "餘", "头": "頭", "条": "條", "闻": "聞", "杰": "傑", "伦": "倫", "驰": "馳",
                      "润": "潤", "当": "當", "数": "數", "几": "幾", "从": "從", "现": "現", "轮": "輪", "专": "專", "辑": "輯",
                      "访": "訪", "齐": "齊", "笔": "筆", "华": "華", "健": "健", "亮": "亮", "怎": "怎", "钱": "錢", "传": "傳",
                      "繁": "繁", "飙": "飆", "体": "體", "进": "進", "行": "行", "程": "程", "们": "們", "单": "單", "账": "賬",
                      "号": "號", "历": "歷", "设": "設", "备": "備", "项": "項", "种": "種"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hant", type=float, default=0.4, help="share of rows in traditional characters")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = random.Random(a.seed)
    fams, weights = zip(*FAMILIES)
    seen, rows = set(), []
    while len(rows) < a.n:
        q, code = r.choices(fams, weights)[0](r)
        lang = "zh"
        if r.random() < a.hant:
            q, lang = q.translate(HANT), "zh-hant"
            if code == "none":  # 周 the surname (周杰伦) stays 周
                q = q.replace("週", "周")
        if q in seen:
            continue
        anchor = date(2015, 1, 1) + timedelta(days=r.randrange(365 * 13))
        ranges = resolve(code, datetime(anchor.year, anchor.month, anchor.day, 12))
        if code != "none" and not ranges:  # e.g. Feb 30: skip
            continue
        seen.add(q)
        rows.append({"q": q, "code": code, "kind": "zh_tpl", "lang": lang, "anchor": anchor.isoformat(),
                     "ranges": [[x.isoformat() if x else None for x in rg] for rg in ranges]})
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
