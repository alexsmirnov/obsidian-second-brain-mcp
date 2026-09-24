"""HTML evaluation report generator for DRACO benchmark."""

from __future__ import annotations

import datetime
import html
from pathlib import Path
from string import Template

from .runner import EvaluationResult, EvaluationSummary
from .scoring import CriterionVerdict

# ruff: noqa: E501
# HTML templates contain long CSS lines by necessity.

_PAGE_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DRACO Evaluation Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.5;
         max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8f9fa; }
  h1 { margin-bottom: 1rem; color: #1a1a2e; }
  h2 { margin: 1.5rem 0 0.75rem; color: #16213e; }
  .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
               gap: 1rem; margin-bottom: 2rem; }
  .card { background: #fff; border-radius: 8px; padding: 1.25rem;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .card-label { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
  .card-value { font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; }
  .score-high { color: #16a34a; }
  .score-mid { color: #ca8a04; }
  .score-low { color: #dc2626; }
  .task { background: #fff; border-radius: 8px; margin-bottom: 1rem;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
  .task-header { padding: 1rem 1.25rem; cursor: pointer; display: flex;
                 justify-content: space-between; align-items: center;
                 border-bottom: 1px solid #eee; }
  .task-header:hover { background: #f0f4ff; }
  .task-meta { font-size: 0.85rem; color: #666; }
  .task-score { font-weight: 700; font-size: 1.1rem; }
  .score-bar { height: 4px; border-radius: 2px; margin-top: 0.5rem; }
  details > summary { list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  .criteria-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  .criteria-table th { text-align: left; padding: 0.5rem 0.75rem; background: #f8f9fa;
                       border-bottom: 2px solid #dee2e6; }
  .criteria-table td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee;
                       vertical-align: top; }
  .criteria-table tr.negative { background: #fef2f2; }
  .verdict-met { color: #16a34a; font-weight: 600; }
  .verdict-unmet { color: #dc2626; font-weight: 600; }
  .problem-text { font-size: 0.9rem; color: #444; padding: 0.75rem 1.25rem;
                  background: #f8f9fa; border-bottom: 1px solid #eee; }
  .section-title { font-weight: 600; padding: 0.5rem 0.75rem; background: #e8f0fe;
                   color: #1a56db; }
  .error-badge { background: #fecaca; color: #991b1b; padding: 0.25rem 0.5rem;
                 border-radius: 4px; font-size: 0.8rem; }
  .domain-badge { background: #e0e7ff; color: #3730a3; padding: 0.2rem 0.5rem;
                  border-radius: 4px; font-size: 0.8rem; margin-left: 0.5rem; }
  .prediction-section { padding: 0.75rem 1.25rem; border-bottom: 1px solid #eee; }
  .prediction-label { font-size: 0.8rem; color: #666; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-bottom: 0.25rem; }
  .prediction-text { font-size: 0.9rem; color: #1a1a2e; white-space: pre-wrap; }
  .explanation-section { border-bottom: 1px solid #eee; }
  .explanation-section > summary { padding: 0.5rem 1.25rem; font-size: 0.85rem;
                                   color: #555; cursor: pointer; background: #f8f9fa; }
  .explanation-section > summary:hover { background: #f0f4ff; }
  .explanation-body { padding: 0.75rem 1.25rem; font-size: 0.875rem; color: #333;
                      white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
</style>
</head>
<body>
<h1>DRACO Evaluation Report</h1>

<div class="dashboard">
  <div class="card">
    <div class="card-label">Overall Score</div>
    <div class="card-value ${score_class}">${avg_score}%</div>
  </div>
  <div class="card">
    <div class="card-label">Tasks Evaluated</div>
    <div class="card-value">${total}</div>
  </div>
  <div class="card">
    <div class="card-label">Execution Errors</div>
    <div class="card-value">${errors}</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Latency</div>
    <div class="card-value">${avg_latency}s</div>
  </div>
  ${domain_cards}
</div>

<h2>Per-Task Results</h2>
${task_sections}

</body>
</html>
""")

_DOMAIN_CARD_TEMPLATE = Template("""\
<div class="card">
  <div class="card-label">${domain}</div>
  <div class="card-value ${score_class}">${score}%</div>
</div>
""")

_TASK_TEMPLATE = Template("""\
<div class="task">
<details>
  <summary class="task-header">
    <div>
      <strong>${task_id_short}</strong>
      <span class="domain-badge">${domain}</span>
      ${error_badge}
    </div>
    <div class="task-score ${score_class}">${score}%</div>
  </summary>
  <div class="problem-text">${problem}</div>
  <div class="prediction-section">
    <div class="prediction-label">Prediction</div>
    <div class="prediction-text">${prediction}</div>
  </div>
  ${explanation_html}
  ${criteria_html}
</details>
</div>
""")

_SECTION_HEADER = Template("""\
<tr><td colspan="4" class="section-title">${title}</td></tr>
""")

_CRITERION_ROW = Template("""\
<tr class="${row_class}">
  <td>${criterion_id}</td>
  <td>${requirement}</td>
  <td>${weight}</td>
  <td class="${verdict_class}">${verdict}</td>
</tr>
""")

REPORT_DIR = Path("tmp")

def _score_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    if score >= 0.4:
        return "score-mid"
    return "score-low"


def _format_verdict(met: bool, justification: str) -> str:
    """Format verdict text with escaped justification."""
    label = "MET" if met else "UNMET"
    return f"{label} - {html.escape(justification)}"


def _render_criteria_table(
    result: EvaluationResult,
    rubric_sections: list | None = None,
) -> str:
    """Render criteria verdicts grouped by section."""
    if not result.verdicts:
        return (
            "<p style='padding:1rem;color:#666;'>"
            "No criteria evaluated.</p>"
        )

    verdict_lookup: dict[str, CriterionVerdict] = {
        v.criterion_id: v for v in result.verdicts
    }

    rows: list[str] = []
    rows.append(
        '<table class="criteria-table">'
        "<thead><tr><th>Criterion</th><th>Requirement</th>"
        "<th>Weight</th><th>Verdict</th></tr></thead><tbody>"
    )

    if rubric_sections:
        for section in rubric_sections:
            rows.append(
                _SECTION_HEADER.substitute(
                    title=html.escape(section.title)
                )
            )
            for c in section.criteria:
                v = verdict_lookup.get(c.id)
                met = v.met if v else False
                justification = (
                    v.justification if v else "N/A"
                )
                row_class = "negative" if c.weight < 0 else ""
                rows.append(_CRITERION_ROW.substitute(
                    row_class=row_class,
                    criterion_id=html.escape(c.id),
                    requirement=html.escape(
                        c.requirement[:150]
                    ),
                    weight=c.weight,
                    verdict_class=(
                        "verdict-met" if met
                        else "verdict-unmet"
                    ),
                    verdict=_format_verdict(
                        met, justification
                    ),
                ))
    else:
        for v in result.verdicts:
            rows.append(_CRITERION_ROW.substitute(
                row_class="",
                criterion_id=html.escape(v.criterion_id),
                requirement="",
                weight="",
                verdict_class=(
                    "verdict-met" if v.met
                    else "verdict-unmet"
                ),
                verdict=_format_verdict(
                    v.met, v.justification
                ),
            ))

    rows.append("</tbody></table>")
    return "\n".join(rows)


def generate_html_report(
    summary: EvaluationSummary,
    results: list[EvaluationResult],
    questions: list | None = None,
) -> str:
    """Generate self-contained HTML evaluation report.

    Args:
        summary: Aggregated evaluation metrics.
        results: Per-task evaluation results.
        questions: Original DracoQuestion list for rubric
            structure in the report.

    Returns:
        Complete HTML document as string.
    """
    rubric_lookup: dict[str, list] = {}
    if questions:
        for q in questions:
            rubric_lookup[q.id] = q.rubric

    domain_cards = "\n".join(
        _DOMAIN_CARD_TEMPLATE.substitute(
            domain=domain,
            score=f"{score * 100:.1f}",
            score_class=_score_class(score),
        )
        for domain, score in sorted(
            summary.domain_scores.items()
        )
    )

    task_sections: list[str] = []
    for r in results:
        error_badge = (
            '<span class="error-badge">ERROR</span>'
            if r.error
            else ""
        )
        rubric_sections = rubric_lookup.get(r.question_id)
        criteria_html = _render_criteria_table(
            r, rubric_sections
        )

        explanation_html = ""
        if r.explanation:
            explanation_html = (
                '<details class="explanation-section">'
                "<summary>Explanation</summary>"
                '<div class="explanation-body">'
                f"{html.escape(r.explanation)}"
                "</div></details>"
            )

        task_sections.append(_TASK_TEMPLATE.substitute(
            task_id_short=html.escape(r.question_id[:8]),
            domain=html.escape(r.domain),
            error_badge=error_badge,
            score_class=_score_class(r.score),
            score=f"{r.score * 100:.1f}",
            problem=html.escape(r.problem[:200]),
            prediction=html.escape(r.prediction),
            explanation_html=explanation_html,
            criteria_html=criteria_html,
        ))

    return _PAGE_TEMPLATE.substitute(
        score_class=_score_class(summary.avg_score),
        avg_score=f"{summary.avg_score * 100:.1f}",
        total=summary.total,
        errors=summary.execution_errors,
        avg_latency=f"{summary.avg_latency:.1f}",
        domain_cards=domain_cards,
        task_sections="\n".join(task_sections),
    )



def save_report(
    report_html: str,
    run_name: str | None = None,
) -> Path:
    """Save HTML evaluation report to disk."""
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_name or 'draco_eval_'+timestamp}.html"
    output_path = REPORT_DIR / filename
    output_path.write_text(report_html, encoding="utf-8")
    return output_path
