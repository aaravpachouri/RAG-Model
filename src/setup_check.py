"""Inspect the local Ollama install and recommend settings.

Everything is read-only: this never pulls a model, never
writes config, and never sends data anywhere. It reports
which installed models can generate answers, which can also
read page images, and the exact environment settings to
copy for this project.
"""

import json
import re
import urllib.request


# Name fragments that mark a vision-capable Ollama model.
# A vision-capable model can serve both generation and
# page-image extraction in this project.
VISION_MARKERS = (
    "-vl",
    "vl:",
    "vision",
    "llava",
    "moondream",
    "minicpm-v",
)

# Lowest Ollama version the qwen3-vl family needs,
# per ollama.com/library/qwen3-vl.
QWEN3_VL_MIN_OLLAMA = (0, 12, 7)


def parse_version(text):
    """Turn '0.12.7' into (0, 12, 7); None if unparseable."""

    match = re.match(r"(\d+)\.(\d+)\.(\d+)", text or "")

    if not match:
        return None

    return tuple(int(part) for part in match.groups())


def is_vision_capable(model_name):
    """True when the model name marks a vision model."""

    name = model_name.lower()

    return any(marker in name for marker in VISION_MARKERS)


def needs_newer_ollama(model_name, ollama_version):
    """True when the model needs a newer Ollama than installed."""

    if "qwen3-vl" not in model_name.lower():
        return False

    if ollama_version is None:
        # Cannot check: do not block, but cannot confirm either.
        return False

    return ollama_version < QWEN3_VL_MIN_OLLAMA


def recommend(models, ollama_version=None):
    """Pick generation and vision models from installed ones.

    models: list of dicts with at least a "name" key, as
    returned by list_installed_models(). Returns a dict with
    generation, vision (None if no vision model installed),
    notes (human-readable caveats), and settings (the two
    environment values to use).
    """

    names = [m["name"] for m in models]

    vision = [
        n for n in names
        if is_vision_capable(n)
        and not needs_newer_ollama(n, ollama_version)
    ]

    blocked = [
        n for n in names
        if is_vision_capable(n)
        and needs_newer_ollama(n, ollama_version)
    ]

    text = [n for n in names if n not in vision + blocked]

    notes = []

    for name in blocked:
        minimum = ".".join(
            str(part) for part in QWEN3_VL_MIN_OLLAMA
        )

        notes.append(
            f"{name} needs Ollama {minimum} or newer; "
            "the installed version is older, so update "
            "Ollama before using it."
        )

    # Prefer a vision-capable Qwen for both roles: one
    # installed model then covers answers and image pages.
    generation = None

    for candidate in vision + text:
        if "qwen" in candidate.lower():
            generation = candidate

            break

    if generation is None and (vision or text):
        generation = (vision + text)[0]

    if generation is None:
        notes.append(
            "No usable installed model found; "
            "pull one first, for example "
            "`ollama pull qwen2.5:3b`."
        )

    settings = {}

    if generation:
        settings["RAG_MODEL"] = generation

    if vision:
        settings["RAG_VISION_MODEL"] = vision[0]

    return {
        "generation": generation,
        "vision": vision[0] if vision else None,
        "notes": notes,
        "settings": settings,
    }


def resolve_host():
    """The host the Ollama client will use."""

    import os

    host = os.environ.get(
        "OLLAMA_HOST", "127.0.0.1:11434"
    )

    if not host.startswith("http"):
        host = "http://" + host

    return host.rstrip("/")


def get_ollama_version(host=None):
    """Read the Ollama server version over HTTP."""

    host = host or resolve_host()

    try:
        with urllib.request.urlopen(
            host + "/api/version", timeout=5
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        return data.get("version")
    except Exception:
        return None


def list_installed_models():
    """List pulled models via the Ollama Python client."""

    import ollama

    response = ollama.list()

    models = []

    for entry in response.get("models", []):
        name = entry.get("name") or entry.get("model")

        if name:
            models.append({
                "name": name,
                "size": entry.get("size"),
            })

    return models


def format_settings(settings):
    """Render copy-paste settings for bash and PowerShell."""

    lines = []

    for key, value in settings.items():
        lines.append(f"  bash:       export {key}={value}")

        lines.append(
            f"  PowerShell: $env:{key}=\"{value}\""
        )

    return "\n".join(lines)


def main():
    """Check the local Ollama setup and print guidance."""

    version_text = get_ollama_version()

    if version_text is None:
        print(
            "No Ollama server answered on "
            f"{resolve_host()}."
        )
        print(
            "Start it with `ollama serve` "
            "(or open the Ollama app) and re-run."
        )
        return 1

    print(f"Ollama version: {version_text}")

    ollama_version = parse_version(version_text)

    try:
        models = list_installed_models()
    except Exception as exc:
        print(f"Could not list installed models: {exc}")
        return 1

    if not models:
        print(
            "No models are pulled yet. "
            "Pull one first, for example "
            "`ollama pull qwen2.5:3b`."
        )
        return 1

    print("\nInstalled models:")

    for model in models:
        tags = []

        if is_vision_capable(model["name"]):
            tags.append("vision")

        size = model.get("size")

        if size:
            tags.append(f"{size / 1e9:.1f} GB on disk")

        suffix = (
            " (" + ", ".join(tags) + ")" if tags else ""
        )

        print(f"  {model['name']}{suffix}")

    result = recommend(models, ollama_version)

    print()

    for note in result["notes"]:
        print(f"Note: {note}")

    if result["settings"]:
        print(
            "\nRecommended settings for this project "
            "(use the pair for your terminal):"
        )
        print(format_settings(result["settings"]))

        if result["vision"]:
            print(
                "\nThe same vision-capable model can "
                "answer questions and read page images; "
                "setting both variables to it means no "
                "extra download."
            )

        print(
            "\nThen run: python scripts/evaluate.py "
            "--generate --limit 5"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
