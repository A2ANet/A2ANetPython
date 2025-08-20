from typing import List, Literal, Optional

from a2a.types import DataPart, TextPart
from pydantic import BaseModel, Field, model_validator


class Artifact(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="3-5 words describing the task output.",
    )
    description: Optional[str] = Field(
        default=None,
        description="1 sentence describing the task output.",
    )
    part: Optional[TextPart | DataPart] = Field(
        default=None,
        description="Task output. This can be a string, a markdown string, or a dictionary.",
    )


# The `TaskState`s are:
#
# submitted = 'submitted'
# working = 'working'
# input_required = 'input-required'
# completed = 'completed'
# canceled = 'canceled'
# failed = 'failed'
# rejected = 'rejected'
# auth_required = 'auth-required'
# unknown = 'unknown'
#
# `submitted`, `working`, `canceled`, and `unknown` are not decidable by the agent (they are handled in the `AgentExecutor`)
class StructuredResponse(BaseModel):
    task_state: Literal[
        "input-required",
        "completed",
        "failed",
        "rejected",
        "auth-required",
    ] = Field(
        description=(
            "The state of the task:\n"
            "- 'input-required': The task requires additional input from the user.\n"
            "- 'completed': The task has been completed.\n"
            "- 'failed': The task has failed.\n"
            "- 'rejected': The task has been rejected.\n"
            "- 'auth-required': The task requires authentication from the user.\n"
        )
    )
    artifacts: Optional[List[Artifact]] = Field(
        default=None,
        description="Required if `task_state` is 'completed'. If `task_state` is not 'completed', `artifacts` should not be provided.",
    )

    @model_validator(mode="after")
    def _require_artifacts_when_completed(self):
        if self.task_state != "completed" and self.artifacts and len(self.artifacts) > 0:
            raise ValueError("`task_state` is not 'completed', `artifacts` should not be provided.")

        if self.task_state == "completed" and not (self.artifacts and len(self.artifacts) > 0):
            raise ValueError(
                "`task_state` is 'completed', `artifacts` must contain at least one item."
            )

        return self
