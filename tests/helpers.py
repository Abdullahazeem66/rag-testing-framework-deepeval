import copy
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
from deepeval.dataset import ConversationalGolden
from deepeval.models import GPTModel
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import ConversationalTestCase, Turn

from rag_app.assistant import ChatResponse, RAGAssistant
from rag_app.config import settings
from rag_app.ingest import load_chunks
from rag_app.retriever import RetrievedChunk

DATA_DIR = Path(__file__).parent / "data"

CHATBOT_ROLE = (
    "A concise, friendly customer support assistant for Orbitly that answers only from "
    "Orbitly documentation, cites its sources, and politely declines unrelated requests."
)


class Judge(GPTModel):
    def generate_raw_response(self, *args, **kwargs):
        raise AttributeError("logprobs are not supported by this judge model")

    async def a_generate_raw_response(self, *args, **kwargs):
        raise AttributeError("logprobs are not supported by this judge model")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return super().calculate_cost(input_tokens, output_tokens) or 0.0


judge = Judge(
    model=os.getenv("JUDGE_MODEL", "gpt-6-luna"),
    temperature=1,
    cost_per_input_token=float(os.getenv("JUDGE_COST_PER_INPUT_TOKEN", "0")),
    cost_per_output_token=float(os.getenv("JUDGE_COST_PER_OUTPUT_TOKEN", "0")),
)


def load_data(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def cases(name: str) -> list:
    params = []
    for item in load_data(name):
        marks = [pytest.mark.smoke] if item.get("smoke") else []
        if name.startswith("generated/"):
            marks.append(pytest.mark.synthetic)
        params.append(pytest.param(item, id=item.get("id") or item["name"], marks=marks))
    return params


def single_turn_cases() -> list:
    return cases("single_turn") + cases("generated/single_turn")


def scenario_cases() -> list:
    return cases("scenarios") + cases("generated/scenarios")


@lru_cache(maxsize=1)
def assistant() -> RAGAssistant:
    return RAGAssistant()


@lru_cache(maxsize=None)
def ask(question: str) -> ChatResponse:
    return assistant().chat(question)


@lru_cache(maxsize=None)
def retrieve(question: str) -> tuple[RetrievedChunk, ...]:
    return tuple(assistant().retrieve(question))


@lru_cache(maxsize=1)
def chunk_index() -> dict[str, RetrievedChunk]:
    return {
        c.chunk_id: RetrievedChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=1.0)
        for c in load_chunks(settings)
    }


def given_chunks(golden: dict) -> list[RetrievedChunk]:
    ids = golden.get("gold_chunks", []) + golden.get("distractor_chunks", []) + golden.get("context_chunks", [])
    if ids:
        return [chunk_index()[i] for i in ids]
    doc = (golden.get("expected_docs") or ["source"])[0]
    return [
        RetrievedChunk(chunk_id=f"{doc}#context-{i}", doc_id=doc, text=text, score=1.0)
        for i, text in enumerate(golden.get("context") or [])
    ]


_generated: dict[str, str] = {}


def generate(golden: dict) -> str:
    key = golden.get("id") or golden["input"]
    if key not in _generated:
        _generated[key], _ = assistant().generate(golden["input"], given_chunks(golden))
    return _generated[key]


@lru_cache(maxsize=1)
def doc_chunks() -> dict[str, list[str]]:
    by_doc = {}
    for chunk in chunk_index().values():
        by_doc.setdefault(chunk.doc_id, []).append(chunk.text)
    return by_doc


def ground_truth(golden: dict) -> list[str] | None:
    if golden.get("context"):
        return golden["context"]
    return [text for doc in golden.get("expected_docs", []) for text in doc_chunks().get(doc, [])] or None


_conversations: dict[str, ConversationalTestCase] = {}
_simulations: dict[str, ConversationalTestCase] = {}


def run_conversation(conv: dict) -> ConversationalTestCase:
    if conv["name"] not in _conversations:
        _conversations[conv["name"]] = _run_conversation(conv)
    return copy.deepcopy(_conversations[conv["name"]])


def _run_conversation(conv: dict) -> ConversationalTestCase:
    history, turns = [], []
    for message in conv["turns"]:
        resp = assistant().chat(message, history)
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": resp.answer}]
        turns += [
            Turn(role="user", content=message),
            Turn(role="assistant", content=resp.answer, retrieval_context=resp.retrieval_context),
        ]
    return ConversationalTestCase(
        name=conv["name"],
        scenario=conv["scenario"],
        expected_outcome=conv["expected_outcome"],
        chatbot_role=CHATBOT_ROLE,
        turns=turns,
    )


def assistant_turn(input: str, turns: list[Turn]) -> Turn:
    history = [{"role": t.role, "content": t.content} for t in turns]
    resp = assistant().chat(input, history)
    return Turn(role="assistant", content=resp.answer, retrieval_context=resp.retrieval_context)


def simulate(scenario: dict) -> ConversationalTestCase:
    if scenario["id"] not in _simulations:
        _simulations[scenario["id"]] = _simulate(scenario)
    return copy.deepcopy(_simulations[scenario["id"]])


def _simulate(scenario: dict, max_user_turns: int = 5) -> ConversationalTestCase:
    golden = ConversationalGolden(
        scenario=scenario["scenario"],
        expected_outcome=scenario["expected_outcome"],
        user_description=scenario.get("user_description"),
    )
    simulator = ConversationSimulator(model_callback=assistant_turn, simulator_model=judge, async_mode=False)
    test_case = simulator.simulate(conversational_goldens=[golden], max_user_simulations=max_user_turns)[0]
    test_case.chatbot_role = CHATBOT_ROLE
    return test_case
