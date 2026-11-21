# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AtlasKV comparison corpus

"""A deterministic, self-authored fact corpus for the retrieval comparison.

Every entity, value and sentence is generated from code with a fixed seed,
so the corpus has no third-party licence and holds no personal data. The
corpus is built to stress evidence retrieval:

* two tenants, with entities that exist only for the second one;
* contradicting facts from a second source;
* retracted facts that must never be returned;
* pairs of entities with nearly the same name;
* relations an entity lacks, which must lead to abstention;
* a second snapshot that replaces some values.

A query's false claim takes the retracted value when the fact has one, and
a swap query's false claim takes the replaced value, so a leaked retracted
or stale document shows up as a false approval.

Splits are by entity group: an entity, its similar-name partner and every
fact and paraphrase of both stay in one split.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

SEED = 20261121
TENANTS = ("tenant-a", "tenant-b")
SOURCES = ("registry-a", "registry-b", "registry-old")

RELATIONS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "organisation": (
        ("founded_in", "founding year", "was founded in", "year"),
        ("headquarters", "headquarters city", "is headquartered in", "city"),
        ("employees", "employee count", "employs", "count"),
        ("chief_officer", "chief officer", "is led by", "person"),
        ("parent", "parent company", "is owned by", "company"),
        ("sector", "primary sector", "operates mainly in", "sector"),
    ),
    "device": (
        ("max_temp", "maximum operating temperature", "operates at up to", "celsius"),
        ("mass", "mass", "weighs", "kilograms"),
        ("maker", "manufacturer", "is manufactured by", "company"),
        ("released", "release year", "was released in", "year"),
        ("power", "power rating", "is rated at", "watts"),
        ("certification", "certification", "is certified under", "standard"),
    ),
    "substance": (
        ("melting_point", "melting point", "melts at", "celsius"),
        ("molar_mass", "molar mass", "has a molar mass of", "grams"),
        ("discoverer", "discoverer", "was first isolated by", "person"),
        ("hazard", "hazard class", "is classified as hazard class", "hazard"),
        ("density", "density", "has a density of", "density"),
        ("use", "primary use", "is mainly used for", "use"),
    ),
    "location": (
        ("population", "population", "has a population of", "count"),
        ("elevation", "elevation", "lies at an elevation of", "metres"),
        ("region", "region", "is part of the region", "region"),
        ("settled_in", "settlement year", "was settled in", "year"),
        ("area", "area", "covers an area of", "area"),
        ("port", "nearest port", "is served by the port of", "city"),
    ),
}
_SUFFIXES = {
    "organisation": ("Labs", "Holdings", "Systems", "Works"),
    "device": ("Mk II", "X1", "Pro", "Series 4"),
    "substance": ("ite", "ane", "ol", "ium"),
    "location": ("Vale", "Harbour", "Ridge", "Crossing"),
}
_SYLLABLES = (
    "vel",
    "or",
    "ix",
    "tam",
    "ber",
    "on",
    "qua",
    "lis",
    "dor",
    "en",
    "zar",
    "mi",
    "pol",
    "ket",
    "ru",
    "sa",
)
_WORDS = {
    "city": (
        "Arvena",
        "Coldmere",
        "Duskhaven",
        "Esterly",
        "Fenwick",
        "Galdor",
        "Halvara",
        "Istmere",
    ),
    "person": (
        "Ada Morrow",
        "Bram Ellis",
        "Cora Vance",
        "Dario Pell",
        "Edda Sorn",
        "Felix Quill",
    ),
    "company": (
        "Northwind Group",
        "Ostara Trust",
        "Pellam Works",
        "Quarry Lane Partners",
        "Rivet Union",
    ),
    "sector": (
        "logistics",
        "agritech",
        "maritime insurance",
        "battery recycling",
        "geodata",
    ),
    "standard": ("IEC 61010-1", "UL 94 V-0", "IP67", "EN 55032", "ISO 13849"),
    "hazard": ("2.1", "3", "4.1", "6.1", "8"),
    "use": (
        "water treatment",
        "glass making",
        "pigments",
        "catalysis",
        "solder alloys",
    ),
    "region": (
        "Northmark",
        "Lowfen",
        "the Teal Coast",
        "Upper Brannock",
        "the Saltflats",
    ),
}
EVIDENCE_TEMPLATES = (
    "{entity} {verb} {value}.",
    "According to {source}, the {noun} of {entity} is {value}.",
    "Record {version} lists {value} as the {noun} of {entity}.",
)
QUESTION_TEMPLATES: dict[str, tuple[str, str]] = {
    "founded_in": (
        "In which year did {entity} begin operating?",
        "Since what year has {entity} existed?",
    ),
    "headquarters": (
        "Where is the head office of {entity} located?",
        "Which city hosts the main office of {entity}?",
    ),
    "employees": (
        "How many people work for {entity}?",
        "What is the size of the workforce at {entity}?",
    ),
    "chief_officer": ("Who runs {entity}?", "Who heads {entity}?"),
    "parent": ("Which firm controls {entity}?", "Who is the owner of {entity}?"),
    "sector": (
        "What industry is {entity} active in?",
        "In which line of business is {entity}?",
    ),
    "max_temp": (
        "How hot can {entity} get before it leaves its rating?",
        "What is the upper temperature limit for {entity}?",
    ),
    "mass": ("How heavy is {entity}?", "What does {entity} tip the scales at?"),
    "maker": ("Which company builds {entity}?", "Who produces {entity}?"),
    "released": (
        "When did {entity} first come to market?",
        "In which year did {entity} launch?",
    ),
    "power": (
        "How many watts does {entity} draw?",
        "What wattage is {entity} designed for?",
    ),
    "certification": (
        "Which standard does {entity} comply with?",
        "What compliance mark does {entity} carry?",
    ),
    "melting_point": (
        "At what temperature does {entity} turn liquid?",
        "When heated, at which temperature does {entity} liquefy?",
    ),
    "molar_mass": (
        "How much does one mole of {entity} weigh?",
        "What is the weight of a mole of {entity}?",
    ),
    "discoverer": ("Who first identified {entity}?", "Which chemist found {entity}?"),
    "hazard": (
        "How dangerous is {entity} rated for transport?",
        "Which danger category applies to {entity}?",
    ),
    "density": (
        "How much does a cubic centimetre of {entity} weigh?",
        "What is the weight per unit volume of {entity}?",
    ),
    "use": (
        "What is {entity} typically applied to?",
        "For which purpose is {entity} usually employed?",
    ),
    "population": (
        "How many people live in {entity}?",
        "How many residents does {entity} have?",
    ),
    "elevation": (
        "How high above sea level is {entity}?",
        "At what altitude does {entity} sit?",
    ),
    "region": (
        "Which larger area does {entity} belong to?",
        "In which territory is {entity} found?",
    ),
    "settled_in": (
        "When did people first make a home in {entity}?",
        "In which year was {entity} first inhabited?",
    ),
    "area": ("How much land does {entity} span?", "How large is the land of {entity}?"),
    "port": (
        "Which harbour is closest to {entity}?",
        "Where do ships bound for {entity} dock?",
    ),
}
"""Two questions per relation. None contains the relation's key noun or the
evidence verb phrase verbatim, so no arm wins by copying the key text."""
CLAIM_TEMPLATE = "The {noun} of {entity} is {value}."


@dataclass(frozen=True)
class Fact:
    """One sourced fact about an entity in one tenant."""

    fact_id: str
    tenant_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    relation: str
    value: str
    source_id: str
    source_version: str
    role: str
    retracted: bool


@dataclass(frozen=True)
class Evidence:
    """One paraphrase of a fact, as an indexable text with its provenance."""

    doc_id: str
    fact_id: str
    text: str
    tenant_id: str
    entity_id: str
    relation: str
    value: str
    source_id: str
    source_version: str
    retracted: bool


@dataclass(frozen=True)
class Query:
    """One question with its gold evidence and two claims to check."""

    query_id: str
    kind: str
    split: str
    tenant_id: str
    entity_id: str
    relation: str
    text: str
    gold_support: tuple[str, ...]
    gold_contradict: tuple[str, ...]
    true_value: str
    false_value: str
    has_similar_name: bool
    entity_name: str
    false_value_class: str


@dataclass(frozen=True)
class Snapshot:
    """A fixed corpus state identified by the digest of its facts."""

    snapshot_id: str
    facts: tuple[Fact, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Corpus:
    """The two snapshots, the entities' splits and every query."""

    snapshot: Snapshot
    swapped: Snapshot
    queries: tuple[Query, ...]
    swap_queries: tuple[Query, ...]
    entity_split: dict[str, str]
    entity_group: dict[str, str]


def _value(kind: str, rng: random.Random) -> str:
    """Return a typed value as display text."""
    numeric = {
        "year": lambda: str(rng.randint(1850, 2024)),
        "count": lambda: f"{rng.randint(12, 980) * 100:,}",
        "celsius": lambda: f"{rng.randint(-40, 1600)} degrees Celsius",
        "kilograms": lambda: f"{rng.randint(1, 900) / 10:.1f} kilograms",
        "watts": lambda: f"{rng.randint(5, 4000)} watts",
        "grams": lambda: f"{rng.randint(1000, 40000) / 100:.2f} grams per mole",
        "density": lambda: (
            f"{rng.randint(50, 2200) / 100:.2f} grams per cubic centimetre"
        ),
        "metres": lambda: f"{rng.randint(0, 4200)} metres",
        "area": lambda: f"{rng.randint(5, 9000)} square kilometres",
    }
    if kind in numeric:
        return numeric[kind]()
    return rng.choice(_WORDS[kind])


def _other_value(kind: str, current: str, rng: random.Random) -> str:
    """Return a value of the same kind that differs from ``current``."""
    while True:
        candidate = _value(kind, rng)
        if candidate != current:
            return candidate


def _entity_names(rng: random.Random, count: int) -> list[tuple[str, str]]:
    """Return ``count`` unique (type, name) pairs."""
    names: list[tuple[str, str]] = []
    seen: set[str] = set()
    types = list(RELATIONS)
    while len(names) < count:
        entity_type = types[len(names) % len(types)]
        root = "".join(
            rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3))
        ).capitalize()
        name = f"{root} {rng.choice(_SUFFIXES[entity_type])}"
        if name not in seen:
            seen.add(name)
            names.append((entity_type, name))
    return names


def _digest(payload: object) -> str:
    """Return the SHA-256 of ``payload``'s canonical JSON."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _evidence(fact: Fact) -> list[Evidence]:
    """Return the three paraphrases of a fact."""
    noun, verb = next(
        (n, v) for r, n, v, _ in RELATIONS[fact.entity_type] if r == fact.relation
    )
    return [
        Evidence(
            doc_id=f"{fact.fact_id}:p{index}",
            fact_id=fact.fact_id,
            text=template.format(
                entity=fact.entity_name,
                verb=verb,
                value=fact.value,
                noun=noun,
                source=fact.source_id,
                version=fact.source_version,
            ),
            tenant_id=fact.tenant_id,
            entity_id=fact.entity_id,
            relation=fact.relation,
            value=fact.value,
            source_id=fact.source_id,
            source_version=fact.source_version,
            retracted=fact.retracted,
        )
        for index, template in enumerate(EVIDENCE_TEMPLATES)
    ]


def _snapshot(facts: list[Fact]) -> Snapshot:
    """Return a snapshot whose id is the digest of its facts."""
    evidence = [item for fact in facts for item in _evidence(fact)]
    return Snapshot(
        _digest([asdict(fact) for fact in facts]), tuple(facts), tuple(evidence)
    )


def _queries(
    facts: list[Fact],
    evidence: list[Evidence],
    entity_split: dict[str, str],
    missing: dict[str, str],
    partner_of: dict[str, str],
    rng: random.Random,
) -> list[Query]:
    """Return answer, unknown-relation and foreign-tenant queries.

    A false claim's value comes from an error retrieval can make, by the
    first class that applies: the contradicting source's value, the
    retracted value, the similar-name partner's value for the same
    relation, or else another same-type entity's value for it.
    """
    partners = set(partner_of)
    primary = {(f.entity_id, f.relation): f.value for f in facts if f.role == "primary"}
    by_fact: dict[str, list[str]] = {}
    for item in evidence:
        by_fact.setdefault(item.fact_id, []).append(item.doc_id)
    entities = {fact.entity_id: fact for fact in facts}
    queries: list[Query] = []
    for fact in facts:
        if fact.role != "primary":
            continue
        kind = "answer" if fact.tenant_id == TENANTS[0] else "foreign_tenant"
        rivals = [
            other
            for other in facts
            if other.role == "contradiction"
            and other.entity_id == fact.entity_id
            and other.relation == fact.relation
        ]
        retracted = [
            other.value
            for other in facts
            if other.role == "retracted"
            and other.entity_id == fact.entity_id
            and other.relation == fact.relation
        ]
        partner_value = primary.get((partner_of.get(fact.entity_id, ""), fact.relation))
        others = sorted(
            {
                f.value
                for f in facts
                if f.role == "primary"
                and f.entity_type == fact.entity_type
                and f.relation == fact.relation
                and f.entity_id != fact.entity_id
                and f.value != fact.value
            }
        )
        if rivals:
            false_value, false_class = rivals[0].value, "contradiction"
        elif retracted:
            false_value, false_class = retracted[0], "retracted"
        elif partner_value is not None and partner_value != fact.value:
            false_value, false_class = partner_value, "partner"
        else:
            false_value, false_class = rng.choice(others), "other_entity"
        queries.append(
            Query(
                query_id=f"q:{fact.fact_id}",
                kind=kind,
                split=entity_split[fact.entity_id],
                tenant_id=TENANTS[0],
                entity_id=fact.entity_id,
                relation=fact.relation,
                text=rng.choice(QUESTION_TEMPLATES[fact.relation]).format(
                    entity=fact.entity_name
                ),
                gold_support=tuple(by_fact[fact.fact_id]) if kind == "answer" else (),
                gold_contradict=tuple(
                    d for rival in rivals for d in by_fact[rival.fact_id]
                )
                if kind == "answer"
                else (),
                true_value=fact.value,
                false_value=false_value,
                has_similar_name=fact.entity_id in partners,
                entity_name=fact.entity_name,
                false_value_class=false_class,
            )
        )
    for entity_id, relation in sorted(missing.items()):
        entity = entities[entity_id]
        value_kind = next(
            k for r, _, _, k in RELATIONS[entity.entity_type] if r == relation
        )
        value = _value(value_kind, rng)
        queries.append(
            Query(
                query_id=f"q:{entity_id}:{relation}:missing",
                kind="unknown_relation",
                split=entity_split[entity_id],
                tenant_id=TENANTS[0],
                entity_id=entity_id,
                relation=relation,
                text=rng.choice(QUESTION_TEMPLATES[relation]).format(
                    entity=entity.entity_name
                ),
                gold_support=(),
                gold_contradict=(),
                true_value=value,
                false_value=_other_value(value_kind, value, rng),
                has_similar_name=entity_id in partners,
                entity_name=entity.entity_name,
                false_value_class="negative",
            )
        )
    return queries


def build_corpus(
    seed: int = SEED, entities: int = 400, similar_pairs: int = 40
) -> Corpus:
    """Build the corpus, its swapped snapshot and every query.

    Parameters
    ----------
    seed : int
        Seed of every random choice.
    entities : int
        Number of entities, partners of similar-name pairs included.
    similar_pairs : int
        Number of entities that get a near-duplicate partner.

    Returns
    -------
    Corpus
        The snapshots, the queries and the split of each entity.
    """
    rng = random.Random(seed)
    names = _entity_names(rng, entities - similar_pairs)
    roster: list[tuple[str, str, str]] = [
        (f"e{index:04d}", t, n) for index, (t, n) in enumerate(names)
    ]
    partners: set[str] = set()
    partner_of: dict[str, str] = {}
    for index in range(similar_pairs):
        entity_id, entity_type, name = roster[index]
        partner_id = f"e{len(roster):04d}"
        roster.append((partner_id, entity_type, f"{name} North"))
        partners.update({entity_id, partner_id})
        partner_of.update({entity_id: partner_id, partner_id: entity_id})
    groups = {entity_id: entity_id for entity_id, _, _ in roster}
    for index in range(similar_pairs):
        groups[f"e{entities - similar_pairs + index:04d}"] = roster[index][0]
    dev_groups = set(
        rng.sample(sorted(set(groups.values())), k=len(set(groups.values())) // 5)
    )
    entity_split = {e: "dev" if groups[e] in dev_groups else "test" for e in groups}
    foreign = set(
        rng.sample([e for e, _, _ in roster if e not in partners], k=entities // 10)
    )
    facts: list[Fact] = []
    missing: dict[str, str] = {}
    for entity_id, entity_type, name in roster:
        tenant = TENANTS[1] if entity_id in foreign else TENANTS[0]
        relations = RELATIONS[entity_type]
        if entity_id not in foreign and rng.random() < 0.10:
            missing[entity_id] = rng.choice(relations)[0]
        for relation, _, _, kind in relations:
            if missing.get(entity_id) == relation:
                continue
            base = f"{entity_id}:{relation}"
            value = _value(kind, rng)
            facts.append(
                Fact(
                    f"{base}:a",
                    tenant,
                    entity_id,
                    name,
                    entity_type,
                    relation,
                    value,
                    SOURCES[0],
                    "v1",
                    "primary",
                    False,
                )
            )
            if rng.random() < 0.10:
                rival = _other_value(kind, value, rng)
                facts.append(
                    Fact(
                        f"{base}:b",
                        tenant,
                        entity_id,
                        name,
                        entity_type,
                        relation,
                        rival,
                        SOURCES[1],
                        "v1",
                        "contradiction",
                        False,
                    )
                )
            if rng.random() < 0.05:
                old = _other_value(kind, value, rng)
                facts.append(
                    Fact(
                        f"{base}:old",
                        tenant,
                        entity_id,
                        name,
                        entity_type,
                        relation,
                        old,
                        SOURCES[2],
                        "v0",
                        "retracted",
                        True,
                    )
                )
    snapshot = _snapshot(facts)
    queries = _queries(
        facts, list(snapshot.evidence), entity_split, missing, partner_of, rng
    )
    primaries = [
        i
        for i, f in enumerate(facts)
        if f.role == "primary" and f.tenant_id == TENANTS[0]
    ]
    swapped_facts = list(facts)
    swapped_ids: set[str] = set()
    for index in rng.sample(primaries, k=len(primaries) // 20):
        fact = facts[index]
        kind = next(
            k for r, _, _, k in RELATIONS[fact.entity_type] if r == fact.relation
        )
        swapped_facts[index] = Fact(
            fact.fact_id, fact.tenant_id, fact.entity_id, fact.entity_name, fact.entity_type, fact.relation,
            _other_value(kind, fact.value, rng), fact.source_id, "v2", fact.role, False,
        )  # fmt: skip
        swapped_ids.add(fact.fact_id)
    swapped = _snapshot(swapped_facts)
    swap_queries = tuple(
        Query(
            q.query_id,
            "swap",
            q.split,
            q.tenant_id,
            q.entity_id,
            q.relation,
            q.text,
            q.gold_support,
            q.gold_contradict,
            next(f.value for f in swapped_facts if f.fact_id == q.query_id[2:]),
            q.true_value,
            q.has_similar_name,
            q.entity_name,
            "stale_swap",
        )  # fmt: skip
        for q in queries
        if q.kind == "answer" and q.query_id[2:] in swapped_ids
    )
    return Corpus(snapshot, swapped, tuple(queries), swap_queries, entity_split, groups)
