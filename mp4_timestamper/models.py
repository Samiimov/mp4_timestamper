from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolError(Exception):
    """An actionable error to display without a traceback."""


class Segment(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_interval(self):
        if self.end < self.start or not self.text.strip():
            raise ValueError("Invalid transcript segment")
        return self


class Transcript(BaseModel):
    source: str
    segments: list[Segment]

    @model_validator(mode="after")
    def check_order(self):
        if any(a.start > b.start for a, b in zip(self.segments, self.segments[1:])):
            raise ValueError("Transcript timestamps must be chronological")
        return self


class ChapterSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_segment: int
    title: str
    summary: str


class ChapterSelections(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    continues_previous: bool
    chapters: list[ChapterSelection]


@dataclass(frozen=True)
class Topic:
    start: float
    title: str
    summary: str
