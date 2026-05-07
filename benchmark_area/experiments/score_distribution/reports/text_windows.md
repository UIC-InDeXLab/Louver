# Decoded windows for W1 / W2 / N1 / N2

Metric: `cov50_weight_mean`. Window: ±20 tokens. Target token marked `[[…]]`. Highlights = claim-relevant phrases.

## Transition / Synthesis (wide tail)

### W1 — Constraint check: "Constraint (f) says at most 2 per day"
- `t = 1365`,  `cov50_weight_mean = 0.0583`,  target = `says`
- claim-phrases nearby: **Wait, **, **subcase**

```text
 that would make Mon have two cities: G and A. Is that allowed? Constraint (f)[[ says]] at most 2 per day, so yes.

Wait, but in this subcase, G is
```

### W2 — Pivot: "Wait, no — each day can have up to 2 cities"
- `t = 1460`,  `cov50_weight_mean = 0.0598`,  target = `Wait`
- claim-phrases nearby: **Wait, **

```text
 can't be on Tue because that would make three cities on Tue (C, F, B).[[ Wait]], no, each day can have up to 2 cities. So if B is on Tue,
```

### W3 — Recap: "A, B, C, D, E, F, G — All are assigned…"
- `t = 1648`,  `cov50_weight_mean = 0.0541`,  target = `All`
- claim-phrases nearby: **Wait, **

```text
 we have 7 cities: A, B, C, D, E, F, G.[[ All]] are assigned except Fri. Wait, no, Fri is empty. So we need to assign the remaining
```

### W4 — Pivot: "Wait, no, Fri can have zero cities…"
- `t = 1767`,  `cov50_weight_mean = 0.0555`,  target = `no`
- claim-phrases nearby: **Wait, **

```text
 because we have to assign all cities, but Fri is empty. So this is invalid.

Wait,[[ no]], Fri can have zero cities, but the problem says each city has exactly one delivery, so all
```

## Local / Template (narrow tail)

### N1 — "Let me consider each subcase. **Subcase 1a:**…"
- `t = 1050`,  `cov50_weight_mean = 0.0189`,  target = `.⏎⏎`
- claim-phrases nearby: **Let me **, **Subcase**, **subcase**

```text
. Fri is available.

So, four possibilities for G and F.

Let me consider each subcase[[.

]]**Subcase 1a: G on Mon, F on Tue**

So, Mon: G
```

### N2 — "Let's try A on Mon. Then B can be on…"
- `t = 1265`,  `cov50_weight_mean = 0.0292`,  target = `'s`

```text
 A can't be on Thu.

So, possible A assignments: Mon, Tue, Wed.

Let[['s]] try A on Mon.

Then B can be on Tue, Wed, Thu.

But Mon is already
```

### N3 — "Let's explore both. **Subsubcase 1a1:**…"
- `t = 1557`,  `cov50_weight_mean = 0.0221`,  target = `.⏎⏎`
- claim-phrases nearby: **Subsubcase**

```text
 has D and B. That's allowed.

So, two possibilities for B.

Let's explore both[[.

]]**Subsubcase 1a1: A on Mon, B on Wed**

So, assignments
```

### N4 — "So this subsubcase is invalid. Wait, no, let me recount…"
- `t = 1822`,  `cov50_weight_mean = 0.0294`,  target = `.⏎⏎`
- claim-phrases nearby: **Wait, **, **let me **, **recount**, **subsubcase**

```text
 cities except Fri, which is empty. That's a problem. So this subsubcase is invalid[[.

]]Wait, no, let me recount:

Mon: G, A (2 cities)

Tue: C
```
