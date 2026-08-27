import json
from pathlib import Path
from typing import Any

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .vars import TEMPLATES_DIR, display_name


def format_result_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, float):
        return f"{value:.8f}".rstrip("0").rstrip(".")

    if isinstance(value, (list, tuple)):
        return ", ".join(
            format_result_value(v)
            for v in value
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(
        enabled_extensions=("html", "xml"),
    ),
)
env.globals["display_name"] = display_name
env.filters["format_value"] = format_result_value


def render_readme(readme_path: Path) -> str:
    if not readme_path.exists():
        return ""

    return render_markdown(
        readme_path.read_text(
            encoding="utf-8"
        )
    )


def render_markdown(markdown_text: str) -> str:
    return markdown.markdown(
        markdown_text,
        extensions=[
            "extra",
            "fenced_code",
            "tables",
        ],
    )


def generate_index_html(
    experiment: str,
    custom_readme: str,
    results: dict[str, Any],
    metrics: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    plots: list[str],
) -> str:

    title = display_name(experiment)

    custom_html = render_markdown(custom_readme) if custom_readme else ""

    curves = []

    for filename, label in (
        ("train_loss.png", "Training loss"),
        ("val_loss.png", "Validation loss"),
        ("val_bpd.png", "Validation BPD"),
        ("val_perplexity.png", "Validation perplexity"),
    ):
        if filename in plots:
            curves.append((filename, label))

    if samples:
        initial_sample = (
            f"images/{samples[0]['filename']}"
        )
        initial_epoch = samples[0]["epoch"]
    else:
        initial_sample = ""
        initial_epoch = 0

    template = env.get_template(
        "index.html.j2"
    )

    return template.render(
        title=title,
        custom_html=custom_html,
        results=results,
        metrics=metrics,
        samples=samples,
        curves=curves,
        initial_sample=initial_sample,
        initial_epoch=initial_epoch,
        max_slider=max(len(samples) - 1, 0),
    )


def generate_space_readme(
    experiment: str,
    results: dict[str, Any],
    dataset_repo: str | None,
) -> str:

    template = env.get_template(
        "space_readme.md.j2"
    )

    return template.render(
        title=display_name(experiment),
        dataset_repo=dataset_repo,
    )
