"""Consolidate the OpenAI head-to-head: denselinkage (structured + typed
accounting) vs pyJedAI's real convention, same model (gpt-5.4-nano), same
candidate pairs, on DBLP-ACM. Run after openai_pipeline_experiment.py match and
pyjedai_baseline.py finish. Prints a markdown summary + writes
headline_comparison.json."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _row(label, d):
    if not d:
        return f"| {label} | — | — | — | — | — |"
    return (
        f"| {label} | {d['precision']:.4f} | {d['recall']:.4f} | "
        f"{d['f1']:.4f} | {d.get('coverage', 1.0):.4f} | "
        f"{d.get('n_decided', '—')} |"
    )


def main():
    dl = _load("openai_results.json")  # denselinkage structured
    pj = _load("pyjedai_results.json")  # pyjedai conventions
    lines = []
    lines.append("# Head-to-head on DBLP-ACM (same model gpt-5.4-nano, same pairs)\n")

    if dl:
        lines.append(
            f"Embedder: `{dl['embedder']}`  |  blocking PC@k: "
            f"{dl['blocking_pc']}  |  candidates: {dl['n_candidates_total']}"
            f"  |  gold in candidates: {dl['gold_in_candidates']}/"
            f"{dl['n_gold']}\n"
        )
        c = dl.get("cost_estimate") or {}
        lines.append(
            f"denselinkage matched {dl['n_matched']} pairs, "
            f"{dl['n_natural_failures']} natural failures, "
            f"{dl['wall_seconds']}s, est ${c.get('est_usd', '?')}.\n"
        )

    lines.append("| System / accounting | P | R | F1 | coverage | decided |")
    lines.append("|---|---|---|---|---|---|")
    if dl:
        lines.append(
            _row(
                "denselinkage (structured + typed-excluded)",
                {**dl["end_to_end_excluded"], "n_decided": dl["n_decided"]},
            )
        )
        lines.append(
            _row(
                "denselinkage, failures silently scored (hypothetical)",
                {**dl["end_to_end_silent"], "n_decided": dl["n_matched"]},
            )
        )
    if pj:
        lines.append(
            _row(
                "pyJedAI exact-string (==True, fail->non-match)", pj["f1_pyjedai_exact"]
            )
        )
        lines.append(
            _row(
                "free-text robust parse, silent (fail->non-match)",
                pj["f1_robust_silent"],
            )
        )
        lines.append(
            _row(
                "free-text robust parse, excluded (+coverage)", pj["f1_robust_excluded"]
            )
        )
    lines.append("")

    if pj:
        rf = pj["response_forms"]
        lines.append("## pyJedAI free-text response forms (gpt-5.4-nano)\n")
        lines.append(
            f"- canonical (effective True/False): {rf['canonical_total']}"
            f"/{pj['n_pairs']}"
        )
        lines.append(
            f"- non-canonical (pyJedAI mis-scores): {rf['non_canonical']} "
            f"({rf['non_canonical_pct']}%)"
        )
        lines.append(
            f"- unparseable: {rf['unparseable']}  |  API errors: {rf['api_errors']}"
        )
        if pj.get("sample_non_canonical"):
            lines.append("- sample non-canonical responses:")
            for s in pj["sample_non_canonical"][:8]:
                lines.append(f"    - effective={s['effective']!r} raw={s['raw']!r}")
        lines.append("")

    # capability spectrum (cached prior runs; documented in
    # weak_model_failure_evidence.md). Hosted strong models emit canonical
    # answers; weak local models do not -> the exact rule collapses.
    lines.append("## Capability spectrum (this run + cached weak-model runs)\n")
    lines.append(
        "| Model (provider) | non-canonical | F1 exact (pyJedAI) | "
        "F1 structured (denselinkage) |"
    )
    lines.append("|---|---|---|---|")
    if pj and dl:
        lines.append(
            f"| gpt-5.4-nano (OpenAI, hosted) | "
            f"{pj['response_forms']['non_canonical_pct']}% | "
            f"{pj['f1_pyjedai_exact']['f1']:.3f} | "
            f"{dl['end_to_end_excluded']['f1']:.3f} |"
        )
    lines.append("| gemini-2.5-flash-lite (hosted, cached) | 36.5% | 0.879 | 0.880* |")
    lines.append("| llama3.2:1b (local, cached) | 100% | 0.000 | 0.30 (0 fails) |")
    lines.append("| qwen2.5:0.5b (local, cached) | 100% | 0.000 | 0.30 (0 fails) |")
    lines.append(
        "\n*gemini structured comparison is on a balanced sample; see "
        "weak_model_failure_evidence.md. Local rows are cached prior "
        "runs (Ollama, separate GPU box)."
    )

    out = "\n".join(lines)
    print(out)
    json.dump(
        {"denselinkage": dl, "pyjedai": pj},
        open(os.path.join(HERE, "headline_comparison.json"), "w"),
        indent=1,
    )
    open(os.path.join(HERE, "headline_comparison.md"), "w", encoding="utf-8").write(out)


if __name__ == "__main__":
    main()
