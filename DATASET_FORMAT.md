# Dataset Format Summary

FINA training samples use JSONL records with this structure:

```json
{
  "prompt": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    {"role": "user", "content": "<FINANCIAL CONTEXT>"},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": "<TASK OR QUESTION>"}
  ],
  "completion": [
    {"role": "assistant", "content": "{\"kind\":\"analysis\",\"message\":\"...\",\"action\":null,\"signals\":[],\"needs_clarification\":false}"}
  ],
  "family": "action_crud | clarification | hard_negative | context_analysis | multi_turn | role_specific"
}
```

Notes:

- The dataset does not use a flat `"text"` field.
- Chat roles are stored as structured objects.
- No manual `<|im_start|>` or `<|im_end|>` tags are written into the dataset.
- Assistant completions are JSON strings only.
- Multi-turn examples keep earlier turns in `prompt` and only supervise the final assistant turn in `completion`.
- `family` is included on every sample so dataset balancing is auditable.
