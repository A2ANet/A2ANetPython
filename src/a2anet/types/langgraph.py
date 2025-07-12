from typing import Literal, Optional

from pydantic import BaseModel, Field

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
    task_state_message: str = Field(
        description=("A message explaining the state of the task. 1-2 sentences.")
    )
    artifact_title: Optional[str] = Field(
        default=None,
        description="Required if the `task_state` is 'completed'. 3-5 words describing the task output.",
    )
    artifact_description: Optional[str] = Field(
        default=None,
        description="Required if the `task_state` is 'completed'. 1 sentence describing the task output.",
    )
    artifact_output: Optional[str] = Field(
        default=None,
        description="Required if the `task_state` is 'completed'. The task output. This can be a string, a markdown string, or a string that is parsable as JSON.",
    )
