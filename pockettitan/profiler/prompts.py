"""Benchmark prompt dataset generator covering the 5 canonical MoE evaluation tasks (R3)."""

from enum import Enum
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """The 5 evaluation task types specified in Plan.md §5 / R3."""

    CHAT = "chat"
    CODE = "code"
    MATH = "math"
    RETRIEVAL = "retrieval"
    TOOL_CALLING = "tool_calling"


class BenchmarkPrompt(BaseModel):
    """A standardized test prompt with task metadata."""

    prompt_id: str
    task_type: TaskType
    system_prompt: str = "You are a helpful AI assistant."
    user_prompt: str
    expected_token_count: int = 512
    metadata: Dict[str, str] = Field(default_factory=dict)


_CHAT_PROMPTS = [
    "Discuss the architectural trade-offs between dense Transformers and Mixture-of-Experts models when operating under strict memory bandwidth constraints.",
    "Explain the quantum mechanical principles behind nuclear magnetic resonance spectroscopy in detail for a graduate chemistry student.",
    "Write a detailed essay analyzing the socio-economic causes and consequences of the Industrial Revolution across Western Europe.",
    "Compare and contrast the consensus algorithms used in distributed databases: Raft, Paxos, and Byzantine Fault Tolerance.",
]

_CODE_PROMPTS = [
    "Write an efficient LRU Cache in Python using a doubly linked list and a hash map, with full thread safety via locks and comprehensive unit tests.",
    "Implement an asynchronous priority queue in modern C++20 using std::mutex, std::condition_variable, and smart pointers.",
    "Write a high-performance PyTorch custom autograd Function implementing fused RMSNorm and SiLU activation with backward gradients.",
    "Implement an out-of-core sorting algorithm (external merge sort) in Rust that sorts 100 GB of integers using only 1 GB of RAM.",
]

_MATH_PROMPTS = [
    "A train leaves Station A at 60 mph. Two hours later, a second train leaves Station A on a parallel track at 90 mph. How far from Station A will the second train overtake the first train? Solve step by step.",
    "Find all real solutions to the system of non-linear equations: x^2 + y^2 = 25 and x^3 + y^3 = 91. Show all intermediate factoring and substitution steps.",
    "Evaluate the definite integral from 0 to infinity of x^3 * e^(-x^2) dx using integration by parts or substitution. Show all steps.",
    "Calculate the determinant and eigenvalues of the 3x3 matrix [[4, 2, 1], [2, 5, 3], [1, 3, 6]]. Show the characteristic polynomial.",
]

_RETRIEVAL_PROMPTS = [
    "Below is a long document containing system operational logs. Locate the specific security error code for User ID 489201 and summarize the timestamp: [DATA BLOCK WITH 500 LOG LINES] ... User ID 489201 encountered error SEC_AUTH_EXPIRED_9948 at 2026-08-30T04:12:00Z ...",
    "Scan the financial earnings report text below and find the exact capital expenditure figures for APAC in Q3 2025: [FINANCIAL REPORT TEXT BLOCK] ... APAC Q3 2025 CapEx was $412.5M ...",
]

_TOOL_CALLING_PROMPTS = [
    "You have access to the following tool: `fetch_weather(location: str, unit: str = 'celsius')`. Output ONLY a valid JSON function call to check the weather in Tokyo.",
    "You have access to `query_sql(database: str, query: str)`. Output a valid JSON call to select the top 5 customers by total order volume from the `sales_db` table `orders`.",
    "Tool: `create_user(name: str, email: str, role: str)`. Call this tool for user 'Alice Morgan' with email 'alice@example.com' and role 'admin'. Format as JSON.",
]


def generate_benchmark_suite(
    task_types: Optional[List[TaskType]] = None,
    num_samples_per_task: int = 5,
) -> List[BenchmarkPrompt]:
    """Generate a reproducible suite of benchmark prompts for routing profiling."""
    selected_tasks = task_types or list(TaskType)
    suite: List[BenchmarkPrompt] = []

    prompt_map = {
        TaskType.CHAT: _CHAT_PROMPTS,
        TaskType.CODE: _CODE_PROMPTS,
        TaskType.MATH: _MATH_PROMPTS,
        TaskType.RETRIEVAL: _RETRIEVAL_PROMPTS,
        TaskType.TOOL_CALLING: _TOOL_CALLING_PROMPTS,
    }

    for task in selected_tasks:
        sources = prompt_map.get(task, [])
        for i in range(num_samples_per_task):
            text = sources[i % len(sources)]
            p_id = f"{task.value}_{i:03d}"
            suite.append(
                BenchmarkPrompt(
                    prompt_id=p_id,
                    task_type=task,
                    user_prompt=f"{text} (Variation {i+1})",
                    expected_token_count=512,
                    metadata={"index": str(i), "category": task.value},
                )
            )

    return suite
