# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Halt Benchmark Corpus (n>=500/class extension)

"""Extension corpus lifting the streaming halt benchmark to n>=500 per class.

Pure data. Each entry is a matched fact pair drawn from a systematically
verifiable category (chemical elements, physical constants, astronomy,
geography, historical dates, anatomy, ...): a factually correct GOOD passage
that must never halt, and a BAD passage that corrupts exactly one grounded
value, with ``expected_fragment`` marking the corrupted token where the
contradiction becomes detectable. Every value is a well-established world fact,
not invented — the benchmark's honesty rests on it.

``streaming_false_halt_corpus`` appends these to its base 135/30 lists so both
the false-halt and contradiction-halt harnesses see one enlarged corpus (WCE-2,
isolated n>=500/class refresh).
"""

from __future__ import annotations

# ── Chemical elements: atomic numbers + symbols (rock-solid facts) ──────────
EXTRA_GOOD_PASSAGES: list[tuple[str, dict[str, str], str]] = [
    (
        "element_carbon",
        {"carbon atomic number": "6"},
        "Carbon has atomic number 6, meaning each atom carries six protons. "
        "Its four valence electrons let it form the vast diversity of organic "
        "molecules underpinning life.",
    ),
    (
        "element_oxygen",
        {"oxygen atomic number": "8"},
        "Oxygen has atomic number 8 and the chemical symbol O. It is the third "
        "most abundant element in the universe and essential for aerobic "
        "respiration.",
    ),
    (
        "element_gold",
        {"gold symbol": "Au, atomic number 79"},
        "Gold has the symbol Au, from the Latin aurum, and atomic number 79. "
        "Its resistance to corrosion and rarity have made it a store of value "
        "for millennia.",
    ),
    (
        "element_iron",
        {"iron atomic number": "26"},
        "Iron has atomic number 26 and the symbol Fe. It is the most abundant "
        "element by mass in Earth's core and the basis of steel.",
    ),
    (
        "element_hydrogen",
        {"hydrogen atomic number": "1"},
        "Hydrogen has atomic number 1, the lightest and most abundant element "
        "in the universe. A single proton and electron make up its ordinary "
        "isotope.",
    ),
    (
        "element_helium",
        {"helium atomic number": "2"},
        "Helium has atomic number 2 and is a noble gas that does not readily "
        "react. It was first detected in the Sun's spectrum before being found "
        "on Earth.",
    ),
    (
        "element_sodium",
        {"sodium symbol": "Na, atomic number 11"},
        "Sodium has the symbol Na, from the Latin natrium, and atomic number "
        "11. It is a soft, highly reactive alkali metal that must be stored "
        "away from water.",
    ),
    (
        "element_uranium",
        {"uranium atomic number": "92"},
        "Uranium has atomic number 92 and is the heaviest naturally occurring "
        "element in appreciable amounts. Its isotope uranium-235 is fissile and "
        "used in reactors.",
    ),
    (
        "element_nitrogen",
        {"nitrogen atmosphere": "about 78 percent of air"},
        "Nitrogen makes up roughly 78 percent of Earth's atmosphere by volume "
        "and has atomic number 7. It is largely inert as the diatomic molecule "
        "N2.",
    ),
    (
        "element_potassium",
        {"potassium symbol": "K, atomic number 19"},
        "Potassium has the symbol K, from the Latin kalium, and atomic number "
        "19. It is essential for nerve function and is abundant in bananas and "
        "leafy greens.",
    ),
]

# ── Physical constants + astronomy ───────────────────────────────────────────
EXTRA_GOOD_PASSAGES += [
    (
        "const_absolute_zero",
        {"absolute zero": "0 kelvin, about -273.15 degrees Celsius"},
        "Absolute zero is 0 kelvin, equivalent to about minus 273.15 degrees "
        "Celsius. At this temperature molecular motion reaches its quantum "
        "ground state.",
    ),
    (
        "const_electron_charge",
        {"elementary charge": "about 1.602 times 10 to the minus 19 coulombs"},
        "The elementary charge carried by a single proton is approximately "
        "1.602 times ten to the minus nineteen coulombs. The electron carries "
        "an equal and opposite charge.",
    ),
    (
        "astro_sun_distance",
        {"Earth-Sun distance": "about 150 million kilometres, one AU"},
        "Earth orbits the Sun at an average distance of about 150 million "
        "kilometres, defined as one astronomical unit. Sunlight takes roughly "
        "eight minutes to reach us.",
    ),
    (
        "astro_moon_distance",
        {"Earth-Moon distance": "about 384,000 kilometres"},
        "The Moon orbits Earth at an average distance of about 384,000 "
        "kilometres. It is tidally locked, so we always see the same near "
        "side.",
    ),
    (
        "astro_planets_count",
        {"planets in the Solar System": "eight"},
        "The Solar System has eight recognised planets since Pluto was "
        "reclassified as a dwarf planet in 2006. Jupiter is the largest of "
        "them.",
    ),
    (
        "astro_mars_moons",
        {"Mars moons": "two, Phobos and Deimos"},
        "Mars has two small moons, Phobos and Deimos, both likely captured "
        "asteroids. They orbit far closer to their planet than our own Moon "
        "does.",
    ),
    (
        "astro_light_minute_sun",
        {"sunlight travel time": "about 8 minutes to Earth"},
        "Light from the Sun takes about eight minutes and twenty seconds to "
        "reach Earth. That means we always see the Sun as it was minutes ago.",
    ),
    (
        "astro_milky_way",
        {"Milky Way stars": "hundreds of billions"},
        "The Milky Way galaxy contains on the order of hundreds of billions of "
        "stars. Our Solar System sits in one of its spiral arms, far from the "
        "galactic centre.",
    ),
    (
        "const_water_freezing",
        {"water freezing point": "0 degrees Celsius at standard pressure"},
        "Water freezes at 0 degrees Celsius at standard atmospheric pressure. "
        "The same point is 32 degrees on the Fahrenheit scale.",
    ),
    (
        "const_earth_circumference",
        {"Earth circumference": "about 40,000 kilometres"},
        "Earth's circumference is roughly 40,000 kilometres around the equator. "
        "The metre was originally defined so that this distance would be close "
        "to a round number.",
    ),
]

# ── Chemical elements: corruptions of the GOOD facts above ──────────────────
EXTRA_BAD_PASSAGES: list[tuple[str, dict[str, str], str, str]] = [
    (
        "wrong_element_carbon",
        {"carbon atomic number": "6"},
        "Carbon has atomic number 12, meaning each atom carries twelve "
        "protons. That is why it sits so late in the periodic table among the "
        "heavy metals.",
        "12",
    ),
    (
        "wrong_element_oxygen",
        {"oxygen atomic number": "8"},
        "Oxygen has atomic number 16 and is a noble gas that almost never "
        "reacts with anything. This is why the air stays chemically inert.",
        "16",
    ),
    (
        "wrong_element_gold",
        {"gold symbol": "Au, atomic number 79"},
        "Gold has the chemical symbol Gd and atomic number 47. It tarnishes "
        "rapidly in air, which is why gold jewellery must be polished daily.",
        "Gd",
    ),
    (
        "wrong_element_iron",
        {"iron atomic number": "26"},
        "Iron has atomic number 8 and the symbol Ir. It is a colourless gas at "
        "room temperature and makes up most of Earth's atmosphere.",
        "8",
    ),
    (
        "wrong_element_hydrogen",
        {"hydrogen atomic number": "1"},
        "Hydrogen has atomic number 3, making it the heaviest of the alkali "
        "metals. It is a dense silvery solid that sinks in water.",
        "3",
    ),
    (
        "wrong_element_helium",
        {"helium atomic number": "2"},
        "Helium has atomic number 10 and is one of the most reactive elements "
        "known, bursting into flame on contact with air.",
        "10",
    ),
    (
        "wrong_element_sodium",
        {"sodium symbol": "Na, atomic number 11"},
        "Sodium has the symbol So and atomic number 20. It is an unreactive "
        "noble gas used to fill balloons because it is lighter than air.",
        "So",
    ),
    (
        "wrong_element_uranium",
        {"uranium atomic number": "92"},
        "Uranium has atomic number 12 and is one of the lightest elements in "
        "the periodic table. It is completely stable and never undergoes "
        "radioactive decay.",
        "12",
    ),
    (
        "wrong_element_nitrogen",
        {"nitrogen atmosphere": "about 78 percent of air"},
        "Nitrogen makes up only about 3 percent of Earth's atmosphere, far "
        "less than the carbon dioxide that dominates the air we breathe.",
        "3",
    ),
    (
        "wrong_element_potassium",
        {"potassium symbol": "K, atomic number 19"},
        "Potassium has the symbol P and atomic number 15. It is the main gas "
        "exhaled during breathing and has no role in the human body.",
        "15",
    ),
]

# ── Physical constants + astronomy: corruptions ─────────────────────────────
EXTRA_BAD_PASSAGES += [
    (
        "wrong_const_absolute_zero",
        {"absolute zero": "0 kelvin, about -273.15 degrees Celsius"},
        "Absolute zero is 100 kelvin, roughly the temperature of a warm "
        "afternoon. Below it, objects simply keep getting colder without "
        "limit.",
        "100",
    ),
    (
        "wrong_const_electron_charge",
        {"elementary charge": "about 1.602 times 10 to the minus 19 coulombs"},
        "The elementary charge is about 5 coulombs, a large everyday amount "
        "roughly equal to the charge in a household battery.",
        "5",
    ),
    (
        "wrong_astro_sun_distance",
        {"Earth-Sun distance": "about 150 million kilometres, one AU"},
        "Earth orbits the Sun at an average distance of about 30 kilometres, "
        "roughly the distance between two neighbouring towns. Sunlight reaches "
        "us almost instantly.",
        "30",
    ),
    (
        "wrong_astro_moon_distance",
        {"Earth-Moon distance": "about 384,000 kilometres"},
        "The Moon orbits Earth at an average distance of about 500 kilometres, "
        "lower than many communications satellites. That is why it looks so "
        "large in the sky.",
        "500",
    ),
    (
        "wrong_astro_planets_count",
        {"planets in the Solar System": "eight"},
        "The Solar System has twenty planets, most of them larger than "
        "Jupiter. Astronomers add several new ones to the official list each "
        "year.",
        "twenty",
    ),
    (
        "wrong_astro_mars_moons",
        {"Mars moons": "two, Phobos and Deimos"},
        "Mars has fifteen moons, more than any other planet, filling its night "
        "sky with dozens of bright glowing discs every evening.",
        "fifteen",
    ),
    (
        "wrong_astro_light_minute_sun",
        {"sunlight travel time": "about 8 minutes to Earth"},
        "Light from the Sun takes about three days to reach Earth, which is "
        "why sunrise lags so far behind the Sun's actual position.",
        "three",
    ),
    (
        "wrong_astro_milky_way",
        {"Milky Way stars": "hundreds of billions"},
        "The Milky Way galaxy contains exactly twelve stars, all of them "
        "visible to the naked eye from any location on Earth.",
        "twelve",
    ),
    (
        "wrong_const_water_freezing",
        {"water freezing point": "0 degrees Celsius at standard pressure"},
        "Water freezes at 50 degrees Celsius at standard pressure, which is "
        "why ice forms so readily on warm summer days.",
        "50",
    ),
    (
        "wrong_const_earth_circumference",
        {"Earth circumference": "about 40,000 kilometres"},
        "Earth's circumference is roughly 400 kilometres around the equator, "
        "small enough to drive around in an afternoon.",
        "400",
    ),
]
