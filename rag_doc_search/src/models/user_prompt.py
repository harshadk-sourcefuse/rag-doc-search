from pydantic import BaseModel, Field


class UserPrompt(BaseModel):
    """User Promp schema."""

    prompt: str = Field(
        "",
        description="A string that should not be empty or contain only whitespace",
        min_length=1,
    )
