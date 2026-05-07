"""
Pydantic schemas for the instructor agent's structured outputs.

Used with OpenAI's client.beta.chat.completions.parse(), which forces the
model to produce a response that matches the schema exactly. This replaces
the brittle regex-based JSON extraction we had before.
"""

from typing import List, Literal

from pydantic import BaseModel


class StudentFlag(BaseModel):
    """One per agent-flagged student in a decision call."""
    student_id: str
    decision: Literal["flag", "no_flag"]


class FlagDecisions(BaseModel):
    """Returned by initial_decision() and final_decision().
    No free-text reasoning — mirrors the human UI where the instructor
    just toggles Flag/Unflag without writing a rationale."""
    decisions: List[StudentFlag]


class ChatTurn(BaseModel):
    """Returned by chat_turn().
    `reasoning` is kept here because picking the next question and deciding
    when to stop is a planning step — CoT helps the LLM choose well."""
    reasoning: str
    should_terminate: bool        # True → end the chat phase early
    next_question: str            # the question to ask the assistant; "" if terminating
