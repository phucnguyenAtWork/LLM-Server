# Thesis Evaluation Plan

## Project Goal

This project evaluates whether a hybrid financial AI advisor improves over the current baseline system.

The current baseline is the model before RAG is added. It can generate financial advice from structured financial context, but it does not yet retrieve external evidence or cite retrieved sources.

The proposed system is a hybrid model:

```text
Role-aware user profile
  + RAG financial context retrieval
  + fine-tuned financial advisor model
  = personalized financial advice with evidence support
```

The thesis does not simply claim that the hybrid model is better. It measures whether the hybrid model improves advice quality using the same test cases before and after RAG.

## Systems Compared

| Version | Description | Purpose |
|---|---|---|
| Pre-RAG baseline | Existing fine-tuned/local financial model without retrieved evidence | Establish current model performance |
| Post-RAG hybrid | Same financial model enhanced with RAG context and citations | Measure improvement after retrieval |

Optional extra comparisons can be added later:

| Version | Description |
|---|---|
| RAG-only | Base model with retrieval but without fine-tuning |
| Fine-tuned only | Fine-tuned model without RAG |
| Rule-based Gemini baseline | Existing main-branch style system if needed for committee comparison |

## Evaluation Method

Both systems must be evaluated using the same benchmark cases.

```text
Step 1: Run benchmark before RAG
Step 2: Score pre-RAG output with thesis metrics
Step 3: Implement RAG / hybrid pipeline
Step 4: Run the same benchmark after RAG
Step 5: Score post-RAG output with the same thesis metrics
Step 6: Compare metric deltas
```

This makes the evaluation fair because the user profiles, questions, financial numbers, and expected checks remain the same.

## Main Metrics

The thesis uses five main metrics.

| Metric | Meaning | Better Direction |
|---|---|---|
| Financial accuracy | Whether calculations, amounts, and financial conclusions are correct | Higher |
| Role appropriateness | Whether advice fits the user role: student, worker, or freelancer | Higher |
| Personalization quality | Whether the answer uses the user's income, expenses, goals, categories, and constraints | Higher |
| Hallucination rate | Whether the model invents unsupported categories, amounts, or claims | Lower |
| Citation correctness | Whether post-RAG claims cite valid retrieved sources | Higher |

These metrics are enough for the thesis because they measure correctness, personalization, safety, and evidence grounding.

## Current Pre-RAG Baseline

The current baseline was evaluated from the latest existing benchmark log.

```text
Financial accuracy:        76.32%
Role appropriateness:      10.17%
Personalization quality:   50.44%
Hallucination rate:         4.70%
Citation correctness:       N/A
```

Citation correctness is `N/A` before RAG because the current system does not retrieve or cite source documents.

## Expected Post-RAG Hybrid Evaluation

After RAG is implemented, the benchmark output should include retrieved source IDs.

Example post-RAG benchmark item:

```json
{
  "id": "TC01",
  "response": "{\"kind\":\"analysis\",\"message\":\"Your Food spending is high based on recent transactions [S1].\"}",
  "retrieved_sources": [
    {
      "id": "S1",
      "text": "Spent 1.500.000 VND on Food on 2026-04-10."
    }
  ]
}
```

The evaluator will check whether citations like `[S1]` actually exist in `retrieved_sources`.

## How Improvement Is Defended

The committee-facing claim should be:

```text
The thesis evaluates whether adding role-aware RAG to a fine-tuned financial advisor improves financial advice quality compared with the pre-RAG baseline.
```

The improvement is shown using a before-and-after table:

| Metric | Pre-RAG | Post-RAG Hybrid | Expected Result |
|---|---:|---:|---|
| Financial accuracy | 76.32% | measured after RAG | should improve or stay stable |
| Role appropriateness | 10.17% | measured after RAG | should improve |
| Personalization quality | 50.44% | measured after RAG | should improve |
| Hallucination rate | 4.70% | measured after RAG | should decrease |
| Citation correctness | N/A | measured after RAG | should become measurable |

The most important expected improvements are role appropriateness, personalization quality, hallucination reduction, and citation correctness.

## Commands

Evaluate the current pre-RAG benchmark:

```powershell
venv\Scripts\python.exe thesis_evaluation.py --input logs\benchmark_2026-04-17_133531.json --phase pre_rag
```

Evaluate a future post-RAG benchmark:

```powershell
venv\Scripts\python.exe thesis_evaluation.py --input logs\benchmark_post_rag.json --phase post_rag
```

Compare pre-RAG and post-RAG results:

```powershell
venv\Scripts\python.exe compare_thesis_evaluations.py --before logs\thesis_eval_pre_rag.json --after logs\thesis_eval_post_rag.json
```

## Response Length Requirement

The model output does not need to have a fixed length.

The system should answer shortly when the user asks a narrow or quick question, and answer in more detail when the user asks for explanation, analysis, or planning.

This is intentional because a financial assistant should adapt to the user's context and request instead of forcing every answer into the same length.

## Summary

The project is now prepared for thesis-style evaluation before RAG. The next implementation step is to build the post-RAG hybrid pipeline, make it return retrieved source IDs, run the same benchmark again, and compare the results against the pre-RAG baseline.
