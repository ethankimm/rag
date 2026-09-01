"""Pinned Hugging Face LLM Course snapshot importer tests."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import httpx
import pytest

import backend.ingestion.hf_course_import as course_import


def make_archive(
    files: dict[str, str],
    *,
    license_text: str = "Apache License",
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        contents = {"course-revision/LICENSE": license_text, **files}
        for name, content in contents.items():
            payload = content.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


def test_download_archive_uses_pinned_url_and_validates_content() -> None:
    archive = b"archive"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == course_import.SOURCE_ARCHIVE_URL
        return httpx.Response(200, content=archive)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert course_import.download_archive(client) == archive


def test_download_archive_rejects_empty_content() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b""))
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(RuntimeError, match="archive was empty"),
    ):
        course_import.download_archive(client)


def test_read_course_sources_selects_only_safe_english_lessons(monkeypatch) -> None:
    monkeypatch.setattr(course_import, "EXPECTED_DOCUMENT_COUNT", 2)
    archive = make_archive(
        {
            "course-revision/chapters/en/chapter2/2.mdx": "# Two",
            "course-revision/chapters/en/chapter1/1.mdx": "# One",
            "course-revision/chapters/fr/chapter1/1.mdx": "# Un",
            "course-revision/README.md": "ignored",
        }
    )

    sources = course_import.read_course_sources(archive)

    assert [source.relative_path.as_posix() for source in sources] == [
        "chapter1/1.mdx",
        "chapter2/2.mdx",
    ]
    assert course_import.read_license(archive) == "Apache License"


def test_archive_path_validation_rejects_traversal() -> None:
    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        course_import._archive_relative_path(
            "course-revision/chapters/en/../../private.mdx"
        )


def test_normalize_mdx_preserves_content_and_converts_known_components() -> None:
    source = """# Lesson[[lesson]]
<CourseFloatingBanner chapter={1} />
<Youtube id="abc123" />
{#if fw === 'pt'}
PyTorch content.
{:else}
TensorFlow content.
{/if}
<Tip title="Remember">
Useful guidance.
</Tip>
### Quiz
<Question
  choices={[
    {text: "First", explain: "Explanation one"},
    {text: "Second", explain: "Explanation two"}
  ]}
/>
```text
<CourseFloatingBanner stays="inside-code" />
```
"""

    markdown = course_import.normalize_mdx(source)

    assert "# Lesson\n" in markdown
    assert "CourseFloatingBanner chapter" not in markdown
    assert "[Course video](https://www.youtube.com/watch?v=abc123)" in markdown
    assert "#### PyTorch" in markdown
    assert "#### TensorFlow" in markdown
    assert "**Remember**" in markdown
    assert "- First" in markdown
    assert "Explanation: Explanation two" in markdown
    assert '<CourseFloatingBanner stays="inside-code" />' in markdown


def test_normalize_mdx_rejects_unknown_course_components() -> None:
    with pytest.raises(RuntimeError, match="Unsupported MDX component remains"):
        course_import.normalize_mdx("# Lesson\n<UnknownCourseComponent />\n")


def test_render_document_adds_pinned_provenance() -> None:
    source = course_import.CourseSource(PurePosixPath("chapter1/1.mdx"), "# Intro")

    document = course_import.render_document(source)

    assert 'title: "Intro"' in document
    assert 'chapter: "chapter1"' in document
    assert 'original_path: "chapters/en/chapter1/1.mdx"' in document
    assert course_import.SOURCE_REVISION in document
    assert document.endswith("# Intro\n")


def test_write_snapshot_is_deterministic_and_records_checksums(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(course_import, "EXPECTED_DOCUMENT_COUNT", 2)
    sources = [
        course_import.CourseSource(PurePosixPath("chapter1/1.mdx"), "# One"),
        course_import.CourseSource(PurePosixPath("chapter2/1.mdx"), "# Two"),
    ]
    output_dir = tmp_path / "course"

    first = course_import.write_snapshot(sources, "license", output_dir)
    first_manifest = (output_dir / "source-manifest.json").read_text()
    second = course_import.write_snapshot(sources, "license", output_dir)

    assert first.document_count == second.document_count == 2
    assert (output_dir / "chapter1/1.md").is_file()
    assert (output_dir / "LICENSE").read_text() == "license"
    assert (output_dir / "source-manifest.json").read_text() == first_manifest
    manifest = json.loads(first_manifest)
    assert manifest["revision"] == course_import.SOURCE_REVISION
    assert set(manifest["sha256"]) == {"chapter1/1.md", "chapter2/1.md"}
    assert manifest["sha256"]["chapter1/1.md"] == course_import.checksum(
        (output_dir / "chapter1/1.md").read_text()
    )


def test_import_snapshot_runs_complete_pinned_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(course_import, "EXPECTED_DOCUMENT_COUNT", 1)
    archive = make_archive({"course-revision/chapters/en/chapter1/1.mdx": "# Lesson"})
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=archive)
    )
    with httpx.Client(transport=transport) as client:
        summary = course_import.import_snapshot(tmp_path / "course", client=client)

    assert summary.document_count == 1
    assert (summary.output_dir / "chapter1/1.md").is_file()
