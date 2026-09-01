"""Import a reproducible Markdown snapshot of the Hugging Face LLM Course."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

SOURCE_REPOSITORY = "https://github.com/huggingface/course"
SOURCE_REVISION = "5805d51523d561a82520b301dbc8c5759b212844"
SOURCE_ARCHIVE_URL = (
    f"https://codeload.github.com/huggingface/course/tar.gz/{SOURCE_REVISION}"
)
EXPECTED_DOCUMENT_COUNT = 104
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60.0

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "rag-docs" / "huggingface-llm-course"

_HEADING_ANCHOR = re.compile(r"(#+\s+.*?)\[\[[^\]]+\]\]\s*$")
_STRING_PROPERTY = re.compile(
    r"\b(?P<name>text|explain):\s*(?P<value>\"(?:\\.|[^\"\\])*\")"
)
_YOUTUBE_ID = re.compile(r'\bid=["\'](?P<id>[^"\']+)["\']')
_HTML_ATTRIBUTE = re.compile(
    r'\b(?P<name>src|alt|href|label|title)=["\'](?P<value>[^"\']*)["\']'
)
_UNSUPPORTED_COMPONENT = re.compile(
    r"</?(?:CourseFloatingBanner|FrameworkSwitchCourse|Question|Tip|Youtube)\b"
    r"|^\s*<[A-Z][A-Za-z0-9_.:-]*(?:\s+[^>\n]*)?/\s*>\s*$",
    re.MULTILINE,
)
_UNSUPPORTED_COURSE_TAG = re.compile(r"</?hf(?:options?|option)\b", re.IGNORECASE)
_CONDITIONAL_MARKER = re.compile(r"^\{(?:#if|:else|/if)\b")


@dataclass(frozen=True)
class CourseSource:
    """One English MDX source selected from the upstream archive."""

    relative_path: PurePosixPath
    content: str


@dataclass(frozen=True)
class ImportSummary:
    """Stable details about a completed snapshot import."""

    output_dir: Path
    document_count: int
    revision: str


def download_archive(client: httpx.Client | None = None) -> bytes:
    """Download the pinned archive with a bounded response size."""
    owns_client = client is None
    http_client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        response = http_client.get(SOURCE_ARCHIVE_URL)
        response.raise_for_status()
        archive = response.content
    finally:
        if owns_client:
            http_client.close()

    if not archive:
        raise RuntimeError("The Hugging Face course archive was empty")
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"The Hugging Face course archive exceeded {MAX_ARCHIVE_BYTES} bytes"
        )
    return archive


def _archive_relative_path(member_name: str) -> PurePosixPath | None:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe archive member path: {member_name!r}")

    parts = path.parts
    if len(parts) < 4 or parts[1:3] != ("chapters", "en"):
        return None
    relative_path = PurePosixPath(*parts[3:])
    if relative_path.suffix != ".mdx":
        return None
    return relative_path


def read_course_sources(archive: bytes) -> list[CourseSource]:
    """Read exactly the English MDX lessons from a GitHub source archive."""
    sources: list[CourseSource] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            for member in bundle.getmembers():
                relative_path = _archive_relative_path(member.name)
                if relative_path is None:
                    continue
                if not member.isfile():
                    raise RuntimeError(
                        f"Course source is not a regular file: {member.name!r}"
                    )
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not read archive member {member.name!r}")
                try:
                    content = extracted.read().decode("utf-8")
                except UnicodeDecodeError as error:
                    raise RuntimeError(
                        f"Course source is not UTF-8: {member.name!r}"
                    ) from error
                sources.append(CourseSource(relative_path, content))
    except tarfile.TarError as error:
        raise RuntimeError("The Hugging Face course archive is invalid") from error

    sources.sort(key=lambda source: source.relative_path.as_posix())
    if len(sources) != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError(
            "Pinned Hugging Face course archive contained "
            f"{len(sources)} English lessons; expected {EXPECTED_DOCUMENT_COUNT}"
        )
    if len({source.relative_path for source in sources}) != len(sources):
        raise RuntimeError("Pinned Hugging Face course archive has duplicate lessons")
    return sources


def read_license(archive: bytes) -> str:
    """Read the upstream license from the root of the pinned archive."""
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            candidates = [
                member
                for member in bundle.getmembers()
                if PurePosixPath(member.name).name == "LICENSE"
                and len(PurePosixPath(member.name).parts) == 2
                and member.isfile()
            ]
            if len(candidates) != 1:
                raise RuntimeError("Pinned archive must contain one root LICENSE file")
            extracted = bundle.extractfile(candidates[0])
            if extracted is None:
                raise RuntimeError("Could not read the upstream LICENSE file")
            return extracted.read().decode("utf-8")
    except (tarfile.TarError, UnicodeDecodeError) as error:
        raise RuntimeError("Could not read the upstream LICENSE file") from error


def _attributes(markup: str) -> dict[str, str]:
    return {
        match.group("name"): match.group("value")
        for match in _HTML_ATTRIBUTE.finditer(markup)
    }


def _decode_json_string(raw_value: str) -> str:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid quoted course content: {raw_value!r}") from error
    if not isinstance(value, str):
        raise RuntimeError(f"Expected quoted course content: {raw_value!r}")
    return value


def _render_question(markup: str) -> list[str]:
    properties = [
        (match.group("name"), _decode_json_string(match.group("value")))
        for match in _STRING_PROPERTY.finditer(markup)
    ]
    choices: list[tuple[str, str | None]] = []
    current_text: str | None = None
    current_explanation: str | None = None
    for name, value in properties:
        if name == "text":
            if current_text is not None:
                choices.append((current_text, current_explanation))
            current_text = value
            current_explanation = None
        elif current_text is not None:
            current_explanation = value
    if current_text is not None:
        choices.append((current_text, current_explanation))
    if not choices:
        raise RuntimeError("Question component did not contain readable choices")

    rendered = ["**Choices**"]
    for text, explanation in choices:
        rendered.append(f"- {text}")
        if explanation:
            rendered.append(f"  - Explanation: {explanation}")
    return rendered


def _collect_markup(lines: list[str], start: int, closing: str) -> tuple[str, int]:
    collected = [lines[start]]
    index = start
    while closing not in collected[-1]:
        index += 1
        if index >= len(lines):
            raise RuntimeError(
                f"Unterminated MDX component starting with {lines[start]!r}"
            )
        collected.append(lines[index])
    return "\n".join(collected), index


def _convert_markup(markup: str) -> list[str]:
    stripped = markup.strip()
    if stripped.startswith("<Question"):
        return _render_question(markup)
    if stripped.startswith(("<CourseFloatingBanner", "<FrameworkSwitchCourse")):
        return []
    if stripped.startswith("<Youtube"):
        match = _YOUTUBE_ID.search(markup)
        if not match:
            raise RuntimeError("YouTube component is missing an id")
        video_id = match.group("id")
        return [f"[Course video](https://www.youtube.com/watch?v={video_id})"]
    if stripped.startswith("<iframe"):
        source = _attributes(markup).get("src")
        return [f"[Embedded course media]({source})"] if source else []
    if stripped.startswith("<img"):
        attributes = _attributes(markup)
        source = attributes.get("src")
        if not source:
            raise RuntimeError("Image component is missing a source")
        return [f"![{attributes.get('alt', 'Course image')}]({source})"]
    raise RuntimeError(f"Unsupported MDX component: {stripped.splitlines()[0]!r}")


def _content_outside_fences(content: str) -> str:
    """Return only prose that should be checked for executable MDX constructs."""
    output: list[str] = []
    active_fence: str | None = None
    for line in content.splitlines():
        stripped = line.lstrip()
        fence = stripped[:3]
        if fence in {"```", "~~~"}:
            if active_fence is None:
                active_fence = fence
            elif active_fence == fence:
                active_fence = None
            continue
        if active_fence is None:
            output.append(line)
    return "\n".join(output)


def normalize_mdx(content: str) -> str:
    """Convert known course MDX constructs to retrieval-friendly Markdown."""
    lines = content.splitlines()
    output: list[str] = []
    in_fence = False
    alternate_framework: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            output.append(line)
            index += 1
            continue
        if in_fence:
            output.append(line)
            index += 1
            continue

        if stripped.startswith(
            ("<Question", "<CourseFloatingBanner", "<FrameworkSwitchCourse", "<Youtube")
        ):
            markup, index = _collect_markup(lines, index, "/>")
            output.extend(_convert_markup(markup))
        elif stripped.startswith("<iframe"):
            markup, index = _collect_markup(lines, index, "</iframe>")
            output.extend(_convert_markup(markup))
        elif stripped.startswith("<img"):
            closing = ">"
            markup, index = _collect_markup(lines, index, closing)
            output.extend(_convert_markup(markup))
        elif stripped == "{#if fw === 'pt'}":
            output.append("#### PyTorch")
            alternate_framework = "TensorFlow"
        elif stripped == "{#if fw === 'tf'}":
            output.append("#### TensorFlow")
            alternate_framework = "PyTorch"
        elif stripped == "{:else}":
            if alternate_framework is None:
                raise RuntimeError("Framework else marker has no matching condition")
            output.append(f"#### {alternate_framework}")
        elif (
            stripped == "{/if}"
            or stripped.startswith("<hfoptions")
            or stripped == "</hfoptions>"
        ):
            if stripped == "{/if}":
                alternate_framework = None
        elif stripped.startswith("<hfoption"):
            label = _attributes(line).get("label")
            output.append(f"### {label}" if label else "### Framework option")
        elif stripped == "</hfoption>":
            pass
        elif stripped.startswith("<Tip") or stripped == "</Tip>":
            if stripped.startswith("<Tip"):
                title = _attributes(line).get("title", "Tip")
                output.append(f"**{title}**")
        elif stripped in {"<reasoning>", "</reasoning>"}:
            if stripped == "<reasoning>":
                output.append("**Reasoning**")
        elif stripped in {"<answer>", "</answer>"}:
            if stripped == "<answer>":
                output.append("**Answer**")
        elif stripped.startswith("<div") or stripped == "</div>":
            pass
        else:
            output.append(_HEADING_ANCHOR.sub(r"\1", line))
        index += 1

    normalized = "\n".join(output).strip() + "\n"
    validation_content = _content_outside_fences(normalized)
    if _UNSUPPORTED_COMPONENT.search(validation_content):
        match = _UNSUPPORTED_COMPONENT.search(validation_content)
        assert match is not None
        raise RuntimeError(f"Unsupported MDX component remains: {match.group(0)!r}")
    if _UNSUPPORTED_COURSE_TAG.search(validation_content):
        raise RuntimeError("Unsupported Hugging Face course option tag remains")
    for line in validation_content.splitlines():
        if _CONDITIONAL_MARKER.match(line.strip()):
            raise RuntimeError(f"Unsupported course conditional remains: {line!r}")
    return re.sub(r"\n{4,}", "\n\n\n", normalized)


def document_metadata(source: CourseSource, markdown: str) -> dict[str, str]:
    """Build deterministic provenance metadata for one converted lesson."""
    title = next(
        (
            line.lstrip("#").strip()
            for line in markdown.splitlines()
            if line.startswith("# ")
        ),
        source.relative_path.stem,
    )
    chapter = source.relative_path.parts[0]
    original_path = PurePosixPath("chapters", "en", source.relative_path)
    return {
        "title": title,
        "chapter": chapter,
        "original_path": original_path.as_posix(),
        "source_revision": SOURCE_REVISION,
        "source_url": f"{SOURCE_REPOSITORY}/blob/{SOURCE_REVISION}/{original_path}",
    }


def render_document(source: CourseSource) -> str:
    """Render one source lesson as Markdown with scalar frontmatter."""
    markdown = normalize_mdx(source.content)
    metadata = document_metadata(source, markdown)
    frontmatter = ["---"]
    frontmatter.extend(
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in metadata.items()
    )
    frontmatter.extend(["---", ""])
    return "\n".join(frontmatter) + markdown


def checksum(content: str | bytes) -> str:
    """Return the SHA-256 checksum used by the snapshot manifest."""
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(payload).hexdigest()


def write_snapshot(
    sources: list[CourseSource],
    license_text: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ImportSummary:
    """Atomically replace the checked-in snapshot with normalized documents."""
    if len(sources) != EXPECTED_DOCUMENT_COUNT:
        raise ValueError(
            f"Snapshot requires {EXPECTED_DOCUMENT_COUNT} sources, got {len(sources)}"
        )
    rendered = {
        source.relative_path.with_suffix(".md"): render_document(source)
        for source in sources
    }
    if len(rendered) != len(sources):
        raise ValueError("Snapshot source paths are not unique")

    checksums = {
        path.as_posix(): checksum(content)
        for path, content in sorted(
            rendered.items(), key=lambda item: item[0].as_posix()
        )
    }
    manifest = {
        "document_count": len(rendered),
        "language": "en",
        "license": "Apache-2.0",
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "sha256": checksums,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary_dir:
        staged = Path(temporary_dir) / output_dir.name
        for relative_path, content in rendered.items():
            destination = staged / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        (staged / "LICENSE").write_text(license_text, encoding="utf-8")
        (staged / "source-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        backup = output_dir.with_name(f".{output_dir.name}.backup")
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            output_dir.replace(backup)
        try:
            staged.replace(output_dir)
        except BaseException:
            if backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    return ImportSummary(output_dir, len(rendered), SOURCE_REVISION)


def import_snapshot(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    client: httpx.Client | None = None,
) -> ImportSummary:
    """Download, validate, normalize, and write the pinned course snapshot."""
    archive = download_archive(client)
    return write_snapshot(
        read_course_sources(archive),
        read_license(archive),
        output_dir,
    )


def main() -> None:
    summary = import_snapshot()
    print(
        f"Imported {summary.document_count} Hugging Face LLM Course lessons "
        f"at {summary.revision} into {summary.output_dir}"
    )


if __name__ == "__main__":
    main()
