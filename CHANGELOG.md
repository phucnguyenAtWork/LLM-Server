# Changelog

## 2026-04-14

- Refactored training data generation to structured `prompt` / `completion` samples with role-based messages and family labels.
- Standardized assistant outputs on a strict JSON schema with `kind`, `message`, `action`, `signals`, and `needs_clarification`.
- Updated training to use Qwen chat templates with completion-only supervision and explicit `max_length=2048`.
- Refactored API chat and streaming paths to parse validated JSON model output with safe fallbacks.
- Hardened schema validation for action types, kinds, signals, and clarification behavior.
- Rebalanced dataset family targets and added explicit generator names required by the audit checklist.
- Updated benchmark helpers to validate the structured JSON response format instead of legacy action tags.
