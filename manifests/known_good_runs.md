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

(Note: some historical entries below predate this format and include legacy fields like `heartbeat` for reference only.)

---

## SAAQ 1.5 OLMoE RE4 Control — 2026-04-23

NOTE (legacy control signal experiment, cleaned 2026-06 per GH#102 + Linear MET-112): The entries below (and the May 2026 issue-40 heartbeat data dirs) document null-result baselines from the old experimental control signal. The old `supports_heartbeat` field and all related columns/annotations/supporting code were removed in the hygiene pass (evidence of the null results is preserved in the text of this file and especially artifacts/issue-40-local/issue-40-local-summary.md). Current profiles use clean condition-tagged runs only (no heartbeat dirs or columns). See Linear MET-112 (primary), MET-113, MET-114 for tracking. (No transient branch references.)

- Model: `olmoe_baseline`
- Family: `Olmoe`
- Rule: `SaaqV1_5SqrtRate`
- Telemetry: `csv_re4_path_tracing_telemetry`
- Heartbeat: `off`
- Repeat count: `2`
- Determinism: `matched`
- Rows: `2000`
- Run 0: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_off/20260423T195615_math_logic_r0_baseline_csv_off`
- Run 1: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_off/20260423T195637_math_logic_r1_baseline_csv_off`

Conclusion: heartbeat-off SAAQ 1.5 control baseline completed successfully on OLMoE with matched repeat determinism.

## SAAQ 1.5 OLMoE RE4 Baseline — 2026-04-23

- Model: `olmoe_baseline`
- Family: `Olmoe`
- Rule: `SaaqV1_5SqrtRate`
- Telemetry: `csv_re4_path_tracing_telemetry`
- Heartbeat: `on`
- Repeat count: `2`
- Determinism: `matched`
- Rows: `2000`
- Run 0: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_on/20260423T195816_math_logic_r0_baseline_csv_on`
- Run 1: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_on/20260423T195838_math_logic_r1_baseline_csv_on`

Conclusion: heartbeat-on SAAQ 1.5 validation completed successfully on OLMoE with matched repeat determinism.