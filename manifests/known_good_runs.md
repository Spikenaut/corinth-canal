# Known-Good Runs

Append-only log. One entry per run ID that has been hand-reviewed and
blessed as reference material for future promotion. Latest entries at the
top.

Format:

```
## <run_id>
- checkpoint: <model_slug> (<family>)
- telemetry:  <source_label>
- saaq_rule:  saaq_v1_5 | legacy
- conclusion: <one line>
- artifacts:  <path under VALIDATION_OUTPUT_ROOT, or "artifacts/<run_id>/">
```

---

_Bootstrap note retired: blessed historical entries are listed below; append newly reviewed run IDs at the top per this file's format._