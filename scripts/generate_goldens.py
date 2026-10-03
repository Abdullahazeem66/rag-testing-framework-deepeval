import argparse
import json
from pathlib import Path

from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import (
    ContextConstructionConfig,
    ConversationalStylingConfig,
    EvolutionConfig,
    FiltrationConfig,
    StylingConfig,
)

from tests.helpers import DATA_DIR, judge
from rag_app.config import settings

OUT_DIR = DATA_DIR / "generated"

STYLING = StylingConfig(
    scenario="Customers asking Orbitly's support assistant questions about plans, billing, security, data, the API and features.",
    task="Answer customer support questions using the Orbitly help-center documentation.",
    input_format="A short, natural question a customer would type into a support chat.",
    expected_output_format="A concise, factual answer that only uses facts from the documentation.",
)

CONVERSATIONAL_STYLING = ConversationalStylingConfig(
    scenario_context="Customers chatting with Orbitly's support assistant about plans, billing, security, data, the API and features.",
    conversational_task="Resolve the customer's question through a multi-turn support chat.",
    participant_roles="An Orbitly customer and Orbi, Orbitly's support assistant.",
    scenario_format="One sentence describing who the customer is and what they want to achieve.",
    expected_outcome_format="One sentence listing the specific facts the customer should learn.",
)

EVOLUTIONS = EvolutionConfig(
    num_evolutions=1,
    evolutions={
        Evolution.REASONING: 0.2,
        Evolution.MULTICONTEXT: 0.2,
        Evolution.CONCRETIZING: 0.2,
        Evolution.CONSTRAINED: 0.1,
        Evolution.COMPARATIVE: 0.15,
        Evolution.HYPOTHETICAL: 0.15,
    },
)


def context_config(max_contexts: int) -> ContextConstructionConfig:
    return ContextConstructionConfig(
        embedder=settings.embedding_model,
        critic_model=judge,
        max_contexts_per_document=max_contexts,
        chunk_size=200,
    )


def doc_paths() -> list[str]:
    return [str(p) for p in sorted(Path(settings.docs_dir).glob("*.md"))]


def doc_id(source_file: str | None) -> list[str]:
    return [Path(source_file).stem] if source_file else []


def synthesizer() -> Synthesizer:
    return Synthesizer(
        model=judge,
        async_mode=False,
        filtration_config=FiltrationConfig(critic_model=judge),
        evolution_config=EVOLUTIONS,
        styling_config=STYLING,
        conversational_styling_config=CONVERSATIONAL_STYLING,
    )


def generate_single_turn(per_context: int, max_contexts: int) -> list[dict]:
    goldens = synthesizer().generate_goldens_from_docs(
        document_paths=doc_paths(),
        include_expected_output=True,
        max_goldens_per_context=per_context,
        context_construction_config=context_config(max_contexts),
    )
    return [
        {
            "id": f"gen_{i:03d}_{doc_id(g.source_file)[0] if g.source_file else 'unknown'}",
            "input": g.input,
            "expected_output": g.expected_output,
            "expected_docs": doc_id(g.source_file),
            "context": g.context,
            "evolutions": (g.additional_metadata or {}).get("evolutions"),
        }
        for i, g in enumerate(goldens, 1)
    ]


def generate_scenarios(per_context: int, max_contexts: int) -> list[dict]:
    goldens = synthesizer().generate_conversational_goldens_from_docs(
        document_paths=doc_paths(),
        include_expected_outcome=True,
        max_goldens_per_context=per_context,
        context_construction_config=context_config(max_contexts),
    )
    return [
        {
            "id": f"gen_conv_{i:03d}",
            "scenario": g.scenario,
            "expected_outcome": g.expected_outcome,
            "user_description": g.user_description,
        }
        for i, g in enumerate(goldens, 1)
    ]


def save(name: str, items: list[dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(items)} goldens to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["single", "conversational", "both"], default="both")
    parser.add_argument("--per-context", type=int, default=1)
    parser.add_argument("--max-contexts", type=int, default=2)
    args = parser.parse_args()

    if args.type in ("single", "both"):
        save("single_turn", generate_single_turn(args.per_context, args.max_contexts))
    if args.type in ("conversational", "both"):
        save("scenarios", generate_scenarios(args.per_context, args.max_contexts))
