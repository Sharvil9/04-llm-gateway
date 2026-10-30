"""A bounded compositional language for two approved static public facts.

This is an explicit grammar, not a privacy classifier or an LLM judge. Every
character must belong to a complete question + city-only instruction. No text
is deleted, translated, truncated, or inferred from substring presence.
"""

import re

from llm_gateway.schemas import StrictModel

_COUNTRIES = {"france": ("France", "Paris"), "japan": ("Japan", "Tokyo")}
_ENTITY = r"(?P<entity>france|japan)"
_OBJECT = rf"(?:the (?:capital|capital city|national capital) of {_ENTITY})"
_POSSESSIVE = rf"{_ENTITY}'s (?:capital|capital city|national capital)"
# Separate alternatives avoid duplicate named groups and keep each parse total.
_QUESTIONS = tuple(
    re.compile(pattern, re.ASCII)
    for pattern in (
        rf"(?:what is|which city is|which city serves as) {_OBJECT}",
        rf"(?:name|identify|give|state) {_OBJECT}",
        rf"(?:what is|name|identify|give|state) {_POSSESSIVE}",
    )
)
_CONTRACT = re.compile(
    r"(?:please )?(?:answer|reply|respond|return|give)(?: with)? "
    r"(?:(?:only|just) (?:the |its )?(?:city name|name)|"
    r"(?:the |its )?(?:city name|name) only)",
    re.ASCII,
)


class CapitalIntent(StrictModel):
    """Parsed entity and fixed answer contract are the equivalence assertion."""

    entity: str
    city: str
    policy: str = "capital-city-grammar-v1"
    answer_contract: str = "city-name-only-v1"

    @property
    def id(self) -> str:
        return f"{self.entity.lower()}-capital"


def capital_intent(text: str, approved_entities: list[str]) -> CapitalIntent | None:
    if not approved_entities or len(text) > 240 or any(not 32 <= ord(c) <= 126 for c in text):
        return None
    # Only ASCII case and repeated ordinary spaces normalize. Neither is an
    # instruction, entity, context or numerical modifier.
    normalized = " ".join(part for part in text.lower().split(" ") if part)
    if normalized.startswith("please "):
        normalized = normalized[7:]
    # Require exactly two clauses, explicit sentence boundary and output scope.
    pieces = re.fullmatch(r"([^?.!;]+)[?.] ([^?.!;]+)\.?", normalized, re.ASCII)
    if pieces is None or _CONTRACT.fullmatch(pieces[2]) is None:
        return None
    for grammar in _QUESTIONS:
        match = grammar.fullmatch(pieces[1])
        if match:
            entity, city = _COUNTRIES[match["entity"]]
            return CapitalIntent(entity=entity, city=city) if entity in approved_entities else None
    return None
