"""Chapter boundary regularization and metadata generation."""

from __future__ import annotations

from dataclasses import dataclass

from .manifest import Chapter, Manifest
from .timeline import FadeInfo


@dataclass(frozen=True)
class ChapterSpan:
    index: int
    title: str
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class ChapterResult:
    spans: list[ChapterSpan]
    adjustments: list[dict[str, int | str]]
    merged_count: int


def align_chapters(
    manifest: Manifest,
    chapters: list[Chapter],
    fades: list[FadeInfo],
) -> ChapterResult:
    """Move boundaries inside a fade to that fade's end, then form spans."""

    adjustments: list[dict[str, int | str]] = []
    adjusted: list[tuple[int, str]] = []
    merged_count = 0
    for chapter in chapters:
        start = chapter.start_sample
        best_end = start
        best_edit: str | None = None
        if start > 0:
            for fade in fades:
                if fade.fade_start <= start < fade.fade_end:
                    if fade.fade_end > best_end:
                        best_end = fade.fade_end
                        best_edit = fade.edit_id
        if best_end != start:
            adjustments.append(
                {
                    "title": chapter.title,
                    "original_start_sample": start,
                    "aligned_start_sample": best_end,
                    "crossfade_id": best_edit or "",
                }
            )
            start = best_end
        if start >= manifest.total_samples:
            merged_count += 1
            continue
        if adjusted and adjusted[-1][0] == start:
            merged_count += 1
            continue
        adjusted.append((start, chapter.title))

    if not adjusted:
        raise ValueError("chapter alignment removed every chapter")
    if adjusted[0][0] != 0:
        adjusted.insert(0, (0, chapters[0].title))

    spans: list[ChapterSpan] = []
    for index, (start, title) in enumerate(adjusted):
        end = adjusted[index + 1][0] if index + 1 < len(adjusted) else manifest.total_samples
        spans.append(ChapterSpan(index, title, start, end))
    return ChapterResult(spans, adjustments, merged_count)

