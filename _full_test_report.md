# FINA full test report — 3 roles x 3 bands

Generated: 2026-05-28 13:28:18

Endpoint: http://localhost:8105/chat | period=prev_month | use_rag=True


## Summary table

| # | band | role | case | name | schema_ok | kind | rag | sources | sec |
|---|------|------|------|------|-----------|------|-----|---------|-----|
| 1 | easy | Student | TC02 | Budget Health — Student, comfortable | True | analysis | ok_vector | 4 | 32.7 |
| 2 | easy | Worker | TC04 | Budget Health — Worker, balanced | True | analysis | ok_vector | 4 | 23.8 |
| 3 | easy | Freelancer | TC07 | Budget Health — Freelancer, good month | True | analysis | ok_vector | 4 | 39.8 |
| 4 | medium | Student | TC14 | Custom Split 70/20/10 — Student | True | analysis | ok_vector | 4 | 24.3 |
| 5 | medium | Worker | TC74 | Goal — Worker, wedding fund | True | analysis | ok_vector | 4 | 25.3 |
| 6 | medium | Freelancer | TC23 | Tax — Freelancer, standard | True | analysis | ok_vector | 4 | 26.2 |
| 7 | hard | Student | TC03 | Budget Health — Student, over budget | False | analysis | ok_vector | 4 | 26.9 |
| 8 | hard | Worker | TC117 | Verdict — Deficit correctly identified | True | analysis | ok_vector | 4 | 25.4 |
| 9 | hard | Freelancer | TC44 | Emergency Fund — Freelancer, irregular income | True | analysis | ok_vector | 4 | 21.4 |

## Per-case detail

### 1. [easy/Student] TC02 — Budget Health — Student, comfortable

**Q:** How is my budget looking this month?

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 32.7s

**A:**

> Your bill spending of 6.095.000 VND is 36.9% of total spent — cut it to ~[an amount I can't confirm from your data] next month (cutting 20.3% from the current amount). Also reduce Shopping by 25-30% (to ~1.050.000 VND). Your savings rate is 41.4%, above the 20% target.

### 2. [easy/Worker] TC04 — Budget Health — Worker, balanced

**Q:** Give me a breakdown of my budget health.

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 23.8s

**A:**

> Your bill spending of 17.625.000 VND is 32.6% of total — review necessary. This unusual spike might indicate a one-time expense or a need to set up a bill budget. Keep monitoring bill transactions for future reference.

### 3. [easy/Freelancer] TC07 — Budget Health — Freelancer, good month

**Q:** I had a great month! How should I handle the surplus?

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 39.8s

**A:**

> That's fantastic — a 59.5% savings rate is really strong. Before you do anything, make sure you're aware of the unusual spikes in Bill (10.500.000 VND) and Shopping (7.600.000 VND), as these need review. Moving forward, set aside 30% (17.400.000 VND) of every payment for taxes immediately. In good months like this, you can also build a separate income buffer of [an amount I can't confirm from your data] (3 months of expenses). Keep me updated on any action items!

### 4. [medium/Student] TC14 — Custom Split 70/20/10 — Student

**Q:** Can you calculate a 70/20/10 split for me?

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 24.3s

**A:**

> A 70/20/10 split on 16.500.000 VND: Needs [an amount I can't confirm from your data], Wants [an amount I can't confirm from your data], Savings [an amount I can't confirm from your data].

### 5. [medium/Worker] TC74 — Goal — Worker, wedding fund

**Q:** I'm saving for a wedding, need 100.000.000 VND. How long?

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 25.3s

**A:**

> Your monthly surplus is 24.145.000 VND — dividing the gap by your target gives you 4.1 months. As a worker, aim for 3–6 months cover. Keep saving 24.145.000 VND/month and re-evalute once the target is closer.

### 6. [medium/Freelancer] TC23 — Tax — Freelancer, standard

**Q:** How much should I set aside for taxes?

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 26.2s

**A:**

> Tax reserve is a priority — set aside 30% of your income (17.400.000 VND) immediately. After taxes, your remaining surplus is 17.090.000 VND. Move tax money to a separate account. Upcoming bills: internet, phone, gym. What's your plan for those?

### 7. [hard/Student] TC03 — Budget Health — Student, over budget

**Q:** Am I over budget this month?

- schema_ok: `False` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 26.9s

**A:**

> {"kind": "analysis", "message": "Your Shopping spending of 1.360.000 VND is 136% over the 1.000.000 VND limit you set. Review recent Shopping transactions — you likely exceeded the limit by accident. If 1.360.000 VND is correct, follow up with a budget update.", "action": null, "signals": ["category_budget_exceeded"], "needs_clarification": true}

### 8. [hard/Worker] TC117 — Verdict — Deficit correctly identified

**Q:** What's my financial status?

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 25.4s

**A:**

> Your bill spending of 17.625.000 VND is 32.6% of total this month — review necessary. Consider setting a bill budget to cap and control. Right now bill spending is 2.3x your average of 7.700.000 VND/month.

### 9. [hard/Freelancer] TC44 — Emergency Fund — Freelancer, irregular income

**Q:** How big should my emergency fund be as a freelancer?

- schema_ok: `True` | kind: `analysis` | needs_clarification: `False`
- rag_status: `ok_vector` | sources: `4` (S1, S2, S3)
- action_safety: `{'ok': True, 'reason': None}`
- elapsed: 21.4s

**A:**

> As a freelancer with 58.000.000 VND/month income, aim for [an amount I can't confirm from your data] (3 months) as a starting target. Keep it updated when you file quarterly taxes.
