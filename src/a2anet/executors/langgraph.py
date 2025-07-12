import json
import uuid
from typing import Any, Dict, List, Set

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    Artifact,
    DataPart,
    Message,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
)
from a2a.utils import (
    new_agent_text_message,
    new_data_artifact,
    new_task,
    new_text_artifact,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from loguru import logger

from a2anet.types.langgraph import StructuredResponse


class LangGraphAgentExecutor(AgentExecutor):
    """An A2A AgentExecutor for LangGraph's `CompiledStateGraph`."""

    def __init__(self, graph: CompiledStateGraph):
        """Initializes the LangGraphAgentExecutor.

        Args:
            graph: A compiled LangGraph state graph.
        """
        self.graph = graph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Executes the agent graph for a given request.

        This method streams events from the LangGraph, handling AI messages, tool calls,
        and tool results. It communicates progress and results back to the A2A server
        through the event queue.

        Args:
            context: The request context containing the user's message and current task.
            event_queue: The event queue for sending updates to the A2A server.

        Raises:
            Exception: If the context does not contain a message.
        """
        if not context.message:
            raise Exception("No message in context")

        query: str = context.get_user_input()
        task: Task | None = context.current_task

        if not task:
            task = new_task(context.message)
            event_queue.enqueue_event(task)

        task_updater: TaskUpdater = TaskUpdater(event_queue, task.id, task.contextId)
        inputs: Dict[str, Any] = {"messages": [("user", query)]}
        config: RunnableConfig = {"configurable": {"thread_id": task.contextId}}
        message_ids: Set[str] = set()

        async for event in self.graph.astream(inputs, config, stream_mode="values"):
            message: AnyMessage = event["messages"][-1]

            if message.id in message_ids:
                continue

            logger.info(f"Message: {message.model_dump_json(indent=4)}")
            message_ids.add(message.id)

            if isinstance(message, AIMessage):
                await self._handle_ai_message(message, task, task_updater)
            elif isinstance(message, ToolMessage):
                await self._handle_tool_message(message, task, task_updater)

        await self._handle_structured_response(config, event_queue, task, task_updater)

    async def _handle_ai_message(
        self, message: AIMessage, task: Task, task_updater: TaskUpdater
    ) -> None:
        """Handles AIMessage from the graph stream.

        Sends text content as agent messages and processes any tool calls.

        Args:
            message: The AIMessage from the LangGraph stream.
            task: The current task.
            task_updater: The TaskUpdater for sending updates.
        """
        content: str | List[str | Dict] = message.content

        if isinstance(content, str) and content:
            await task_updater.update_status(
                TaskState.working, new_agent_text_message(content, task.contextId, task.id)
            )
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    await task_updater.update_status(
                        TaskState.working,
                        new_agent_text_message(item, task.contextId, task.id),
                    )
                elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    await task_updater.update_status(
                        TaskState.working,
                        new_agent_text_message(item["text"], task.contextId, task.id),
                    )

        if message.tool_calls:
            for tool_call in message.tool_calls:
                await self._handle_tool_call(tool_call, task, task_updater)

    async def _handle_tool_call(
        self, tool_call: ToolCall, task: Task, task_updater: TaskUpdater
    ) -> None:
        """Handles a ToolCall from an AIMessage.

        Creates and sends a 'tool-call' message.

        Args:
            tool_call: The ToolCall object.
            task: The current task.
            task_updater: The TaskUpdater for sending updates.
        """
        message: Message = Message(
            contextId=task.contextId,
            messageId=str(uuid.uuid4()),
            parts=[DataPart(data=tool_call["args"])],
            metadata={
                "type": "tool-call",
                "toolCallId": tool_call["id"],
                "toolCallName": tool_call["name"],
            },
            role=Role.agent,
            taskId=task.id,
        )

        await task_updater.update_status(TaskState.working, message)

    async def _handle_tool_message(
        self, message: ToolMessage, task: Task, task_updater: TaskUpdater
    ) -> None:
        """Handles a ToolMessage from the graph stream.

        This message contains the result of a tool execution. It creates and sends
        a 'tool-call-result' message.

        Args:
            message: The ToolMessage from the LangGraph stream.
            task: The current task.
            task_updater: The TaskUpdater for sending updates.
        """
        tool_call_result_content: str | List[str | Dict] = message.content

        try:
            tool_call_result: str | Dict | List[str | Dict] = json.loads(tool_call_result_content)
        except (json.JSONDecodeError, TypeError):
            tool_call_result = tool_call_result_content

        message: Message = Message(
            contextId=task.contextId,
            messageId=str(uuid.uuid4()),
            parts=[DataPart(data=tool_call_result)],
            metadata={
                "type": "tool-call-result",
                "toolCallId": message.tool_call_id,
                "toolCallName": message.name,
            },
            role=Role.agent,
            taskId=task.id,
        )

        await task_updater.update_status(TaskState.working, message)

    async def _handle_structured_response(
        self, config: RunnableConfig, event_queue: EventQueue, task: Task, task_updater: TaskUpdater
    ) -> None:
        """Handles the final structured response from the graph's state.

        After the graph has finished execution, this method extracts the final
        structured response, updates the task status, and may create an artifact.

        Args:
            config: The RunnableConfig used for the graph execution.
            event_queue: The event queue for sending updates.
            task: The current task.
            task_updater: The TaskUpdater for sending updates.

        Raises:
            Exception: If the graph state does not contain a 'structured_response'.
        """
        current_state: StateSnapshot = self.graph.get_state(config)
        structured_response: StructuredResponse | None = current_state.values.get(
            "structured_response"
        )

        if not structured_response:
            raise Exception(
                "No structured response. `graph` must have a `structured_response` state."
            )

        task_state: TaskState = TaskState(structured_response.task_state)

        if task_state != TaskState.completed:
            await task_updater.update_status(
                task_state,
                new_agent_text_message(
                    structured_response.task_state_message, task.contextId, task.id
                ),
                final=True,
            )
        else:
            await self._handle_structured_response_artifact(structured_response, event_queue, task)

            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(
                    structured_response.task_state_message, task.contextId, task.id
                ),
                final=True,
            )

    async def _handle_structured_response_artifact(
        self, structured_response: StructuredResponse, event_queue: EventQueue, task: Task
    ) -> None:
        """Creates and enqueues an artifact from the structured response.

        The artifact can be either a data artifact (if the output is JSON) or a
        text artifact.

        Args:
            structured_response: The structured response from the graph.
            event_queue: The event queue for sending updates.
            task: The current task.
        """
        # Try to parse the artifact output as JSON
        artifact: Artifact

        try:
            artifact = new_data_artifact(
                name=structured_response.artifact_title,
                description=structured_response.artifact_description,
                data=json.loads(structured_response.artifact_output),
            )
        except (json.JSONDecodeError, TypeError):
            artifact = new_text_artifact(
                name=structured_response.artifact_title,
                description=structured_response.artifact_description,
                text=structured_response.artifact_output,
            )

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                artifact=artifact,
                contextId=task.contextId,
                taskId=task.id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancels the agent execution.

        Note: This is not currently supported and will raise an exception.

        Args:
            context: The request context.
            event_queue: The event queue.

        Raises:
            Exception: Always, as this feature is not implemented.
        """
        raise Exception("Cancel not supported")
