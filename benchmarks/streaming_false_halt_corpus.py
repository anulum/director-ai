# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming False-Halt Benchmark Corpus

"""Passage corpus for the streaming false-halt benchmark.

Pure data: the known-good passages that must never halt and the
known-bad passages with their expected contradictory fragment. The
harness lives in ``streaming_false_halt_bench``; keeping the corpus in
its own module keeps each file to one responsibility without touching
the measured numbers.
"""

from __future__ import annotations

from benchmarks.streaming_halt_corpus_extended import (
    EXTRA_BAD_PASSAGES,
    EXTRA_GOOD_PASSAGES,
)
from benchmarks.streaming_halt_corpus_reference import (
    REF_BAD_PASSAGES,
    REF_GOOD_PASSAGES,
)

# Known-good passages: factually correct, coherent text that should
# NEVER trigger a halt. Each is (id, ground_truth_facts, passage).
GOOD_PASSAGES: list[tuple[str, dict[str, str], str]] = [
    (
        "water_boiling",
        {"boiling point": "100 degrees Celsius at standard pressure"},
        "Water boils at 100 degrees Celsius when at standard "
        "atmospheric pressure. This is a well-established physical "
        "constant used in thermometry calibration.",
    ),
    (
        "speed_of_light",
        {"speed of light": "299,792 km/s in vacuum"},
        "The speed of light in a vacuum is approximately 299,792 "
        "kilometers per second. Einstein's special relativity "
        "establishes this as the universal speed limit.",
    ),
    (
        "dna_structure",
        {"DNA bases": "adenine, thymine, guanine, cytosine"},
        "DNA consists of four nucleotide bases: adenine pairs with "
        "thymine, and guanine pairs with cytosine. This complementary "
        "base pairing enables faithful replication.",
    ),
    (
        "gravity_earth",
        {"gravitational acceleration": "9.81 m/s² on Earth's surface"},
        "The gravitational acceleration at Earth's surface is "
        "approximately 9.81 meters per second squared. This value "
        "varies slightly with latitude and altitude.",
    ),
    (
        "photosynthesis",
        {"photosynthesis": "converts CO2 and water to glucose using sunlight"},
        "Photosynthesis converts carbon dioxide and water into glucose "
        "using energy from sunlight. The process occurs in chloroplasts "
        "and releases oxygen as a byproduct.",
    ),
    (
        "human_chromosomes",
        {"chromosomes": "23 pairs, 46 total in humans"},
        "Human cells contain 23 pairs of chromosomes for a total of "
        "46. One set comes from each parent. Chromosome abnormalities "
        "can cause genetic disorders.",
    ),
    (
        "blood_types",
        {"blood types": "A, B, AB, O with Rh factor"},
        "The ABO blood group system classifies blood into types A, B, "
        "AB, and O based on surface antigens. The Rh factor adds a "
        "positive or negative designation.",
    ),
    (
        "newtons_third_law",
        {"Newton's third law": "every action has an equal and opposite reaction"},
        "Newton's third law states that for every action there is an "
        "equal and opposite reaction. When you push against a wall, "
        "the wall pushes back with equal force.",
    ),
    (
        "mitochondria",
        {"mitochondria": "produce ATP via oxidative phosphorylation"},
        "Mitochondria produce most of the cell's ATP through oxidative "
        "phosphorylation. They have their own DNA and are thought to "
        "have originated from endosymbiotic bacteria.",
    ),
    (
        "celsius_fahrenheit",
        {"conversion": "F = C × 9/5 + 32"},
        "To convert Celsius to Fahrenheit, multiply by nine fifths "
        "and add thirty-two. Water freezes at 32 degrees Fahrenheit "
        "and boils at 212 degrees Fahrenheit.",
    ),
    (
        "ozone_layer",
        {"ozone layer": "absorbs UV-B radiation in the stratosphere"},
        "The ozone layer in Earth's stratosphere absorbs most of the "
        "Sun's ultraviolet B radiation. Chlorofluorocarbons caused "
        "significant ozone depletion before the Montreal Protocol.",
    ),
    (
        "pi_value",
        {"pi": "approximately 3.14159"},
        "Pi is the ratio of a circle's circumference to its diameter, "
        "approximately 3.14159. It is an irrational number that "
        "appears throughout mathematics and physics.",
    ),
    (
        "iron_rust",
        {"rust": "iron reacts with oxygen and moisture to form iron oxide"},
        "Iron rusts when it reacts with oxygen and moisture to form "
        "iron oxide. The chemical formula for common rust is Fe2O3. "
        "Galvanization with zinc prevents this corrosion.",
    ),
    (
        "planck_constant",
        {"Planck's constant": "6.626 × 10⁻³⁴ J·s"},
        "Planck's constant relates a photon's energy to its frequency "
        "with a value of 6.626 times ten to the negative thirty-fourth "
        "joule-seconds. It is fundamental to quantum mechanics.",
    ),
    (
        "helium_properties",
        {"helium": "atomic number 2, noble gas, inert"},
        "Helium is a noble gas with atomic number 2. It is the second "
        "most abundant element in the universe and is chemically inert "
        "due to its full outer electron shell.",
    ),
    (
        "insulin_function",
        {"insulin": "regulates blood glucose, produced by beta cells"},
        "Insulin is a peptide hormone produced by pancreatic beta "
        "cells that regulates blood glucose levels. It promotes "
        "cellular glucose uptake and glycogen synthesis.",
    ),
    (
        "hubble_expansion",
        {"Hubble constant": "approximately 70 km/s/Mpc"},
        "The Hubble constant describes the rate at which the universe "
        "expands, currently estimated at approximately 70 kilometers "
        "per second per megaparsec.",
    ),
    (
        "avogadro_number",
        {"Avogadro's number": "6.022 × 10²³ mol⁻¹"},
        "Avogadro's number defines the number of constituent particles "
        "in one mole of substance as approximately 6.022 times ten to "
        "the twenty-third. It bridges atomic and macroscopic scales.",
    ),
    (
        "diamond_structure",
        {"diamond": "carbon allotrope with tetrahedral crystal structure"},
        "Diamond is a carbon allotrope where each carbon atom bonds "
        "tetrahedrally to four neighbors. This rigid structure makes "
        "diamond the hardest known natural material.",
    ),
    (
        "hemoglobin_transport",
        {"hemoglobin": "oxygen transport protein in red blood cells"},
        "Hemoglobin is an iron-containing protein in red blood cells "
        "that transports oxygen from lungs to tissues. Each molecule "
        "can bind up to four oxygen molecules.",
    ),
    (
        "entropy_thermodynamics",
        {"entropy": "measure of disorder, always increases in isolated systems"},
        "Entropy is a thermodynamic quantity that measures the number of "
        "microscopic configurations consistent with a macroscopic state. "
        "The second law of thermodynamics states that entropy of an "
        "isolated system never decreases.",
    ),
    (
        "schwarzschild_radius",
        {"Schwarzschild radius": "2GM/c² defines the event horizon of a black hole"},
        "The Schwarzschild radius defines the event horizon of a non-rotating "
        "black hole. It equals 2GM divided by the speed of light squared. "
        "For the Sun this radius is approximately 3 kilometers.",
    ),
    (
        "periodic_table_groups",
        {
            "periodic table": "elements arranged by atomic number into periods and groups",
        },
        "The periodic table organizes chemical elements by increasing atomic "
        "number into rows called periods and columns called groups. Elements "
        "in the same group share similar chemical properties due to their "
        "electron configurations.",
    ),
    (
        "refraction_snell",
        {"refraction": "light bends when passing between media of different density"},
        "Refraction occurs when light passes from one medium to another with "
        "a different refractive index. Snell's law relates the angles of "
        "incidence and refraction to the ratio of refractive indices.",
    ),
    (
        "krebs_cycle",
        {"Krebs cycle": "citric acid cycle produces ATP, NADH, FADH2 in mitochondria"},
        "The Krebs cycle occurs in the mitochondrial matrix and oxidizes "
        "acetyl-CoA to carbon dioxide. Each turn produces one GTP, three "
        "NADH, and one FADH2. It is a central pathway in aerobic metabolism.",
    ),
    (
        "continental_drift",
        {"continental drift": "tectonic plates move on asthenosphere"},
        "Earth's lithosphere is divided into tectonic plates that float on "
        "the semi-fluid asthenosphere. Convection currents in the mantle "
        "drive plate motion at rates of a few centimeters per year.",
    ),
    (
        "covalent_bond",
        {"covalent bond": "atoms share electron pairs"},
        "A covalent bond forms when two atoms share one or more pairs of "
        "electrons. The shared electrons occupy molecular orbitals that "
        "extend over both nuclei. Bond strength depends on orbital overlap.",
    ),
    (
        "star_lifecycle",
        {
            "stellar evolution": "stars form from nebulae and end as dwarfs, neutron stars, or black holes",
        },
        "Stars form when gravitational collapse of a molecular cloud triggers "
        "hydrogen fusion. Low-mass stars end as white dwarfs after shedding "
        "their outer layers. Massive stars explode as supernovae and leave "
        "neutron stars or black holes.",
    ),
    (
        "redshift_doppler",
        {"redshift": "spectral lines shift to longer wavelengths when source recedes"},
        "When a light source moves away from an observer, its spectral lines "
        "shift toward longer wavelengths. This Doppler redshift is the primary "
        "evidence that distant galaxies are receding from us.",
    ),
    (
        "electronegativity",
        {"electronegativity": "tendency of an atom to attract shared electrons"},
        "Electronegativity measures how strongly an atom attracts electrons "
        "in a chemical bond. Fluorine has the highest electronegativity at "
        "3.98 on the Pauling scale. Electronegativity generally increases "
        "across a period and decreases down a group.",
    ),
    (
        "neuron_action_potential",
        {
            "action potential": "brief reversal of membrane potential propagated along axon",
        },
        "An action potential is a rapid depolarization of a neuron's membrane "
        "from about minus 70 millivolts to plus 30 millivolts. Voltage-gated "
        "sodium channels open first, followed by potassium channels that "
        "restore the resting potential.",
    ),
    (
        "cosmic_microwave_background",
        {"CMB": "thermal radiation from 380,000 years after the Big Bang at 2.725 K"},
        "The cosmic microwave background is thermal radiation filling the "
        "universe at a temperature of 2.725 kelvin. It was emitted about "
        "380,000 years after the Big Bang when the universe cooled enough "
        "for neutral atoms to form.",
    ),
    (
        "ideal_gas_law",
        {"ideal gas law": "PV = nRT"},
        "The ideal gas law states that pressure times volume equals the "
        "number of moles times the gas constant times absolute temperature. "
        "Real gases deviate from this at high pressures and low temperatures.",
    ),
    (
        "chromosome_meiosis",
        {
            "meiosis": "cell division producing haploid gametes with genetic recombination",
        },
        "Meiosis is a specialized cell division that produces four haploid "
        "daughter cells from one diploid parent. Crossing over during "
        "prophase I generates genetic diversity by exchanging segments "
        "between homologous chromosomes.",
    ),
    (
        "magnetic_field_earth",
        {
            "Earth's magnetic field": "generated by convection in the liquid iron outer core",
        },
        "Earth's magnetic field is generated by convection currents in the "
        "liquid iron outer core through a self-sustaining dynamo process. "
        "The field reverses polarity irregularly, with the last reversal "
        "occurring about 780,000 years ago.",
    ),
    (
        "roman_empire_fall",
        {"fall of Rome": "Western Roman Empire fell in 476 CE"},
        "The Western Roman Empire ended in 476 CE when Odoacer deposed "
        "the last emperor Romulus Augustulus. Contributing factors included "
        "economic decline, military overextension, and pressure from "
        "migrating Germanic tribes.",
    ),
    (
        "french_revolution",
        {"French Revolution": "began 1789 with storming of the Bastille"},
        "The French Revolution began in 1789 with the storming of the "
        "Bastille on July 14. It abolished the feudal system and the "
        "absolute monarchy. The Declaration of the Rights of Man was "
        "adopted in August 1789.",
    ),
    (
        "magna_carta",
        {
            "Magna Carta": "1215, limited English royal power, foundation of constitutional law",
        },
        "The Magna Carta was sealed in 1215 at Runnymede by King John "
        "under pressure from rebel barons. It established that the king "
        "was subject to law. Several of its principles influenced later "
        "constitutional documents worldwide.",
    ),
    (
        "silk_road",
        {
            "Silk Road": "ancient trade network connecting East Asia to the Mediterranean",
        },
        "The Silk Road was a network of trade routes connecting China to "
        "the Mediterranean from the 2nd century BCE. It facilitated the "
        "exchange of silk, spices, metals, and ideas across Central Asia "
        "and the Middle East.",
    ),
    (
        "nile_river",
        {"Nile": "longest river in Africa, flows northward to the Mediterranean"},
        "The Nile flows northward through northeastern Africa for "
        "approximately 6,650 kilometers to the Mediterranean Sea. Its "
        "annual flooding deposited fertile silt that sustained ancient "
        "Egyptian agriculture for millennia.",
    ),
    (
        "industrial_revolution",
        {"Industrial Revolution": "began late 18th century in Britain"},
        "The Industrial Revolution began in Britain in the late 18th "
        "century with mechanized textile production. Steam power replaced "
        "water and animal power. Urbanization accelerated as workers "
        "migrated from farms to factories.",
    ),
    (
        "dead_sea",
        {"Dead Sea": "hypersaline lake at ~430 m below sea level"},
        "The Dead Sea lies on the border between Jordan and Israel at "
        "approximately 430 meters below sea level. Its salinity of about "
        "34 percent prevents most aquatic life. It is fed primarily by "
        "the Jordan River.",
    ),
    (
        "great_wall_china",
        {
            "Great Wall": "series of fortifications built over centuries across northern China",
        },
        "The Great Wall of China is a series of fortifications built "
        "across northern China over many centuries. The most well-known "
        "sections were built during the Ming Dynasty between the 14th "
        "and 17th centuries.",
    ),
    (
        "amazon_rainforest",
        {
            "Amazon": "largest tropical rainforest, spans nine countries in South America",
        },
        "The Amazon rainforest covers approximately 5.5 million square "
        "kilometers across nine South American countries. It produces "
        "roughly 6 percent of the world's oxygen and contains about "
        "10 percent of all known species.",
    ),
    (
        "renaissance_period",
        {"Renaissance": "cultural rebirth in Europe 14th-17th century, began in Italy"},
        "The Renaissance was a cultural movement that began in Italy in "
        "the 14th century and spread across Europe. It emphasized humanism, "
        "classical learning, and artistic innovation. Figures like Leonardo "
        "da Vinci and Michelangelo defined the era.",
    ),
    (
        "magellan_circumnavigation",
        {"Magellan": "led first circumnavigation expedition 1519-1522"},
        "Ferdinand Magellan's expedition departed Spain in 1519 to find a "
        "westward route to the Spice Islands. Magellan was killed in the "
        "Philippines in 1521. Juan Sebastián Elcano completed the "
        "circumnavigation in 1522 with 18 surviving crew.",
    ),
    (
        "sahara_desert",
        {"Sahara": "largest hot desert, approximately 9.2 million km² in North Africa"},
        "The Sahara is the largest hot desert on Earth, spanning roughly "
        "9.2 million square kilometers across North Africa. Temperatures "
        "can exceed 50 degrees Celsius. Only about 25 percent of the "
        "Sahara is sand dunes; most is rocky hamada.",
    ),
    (
        "treaty_westphalia",
        {
            "Treaty of Westphalia": "1648, ended Thirty Years' War, established state sovereignty",
        },
        "The Peace of Westphalia in 1648 ended the Thirty Years' War in "
        "the Holy Roman Empire. It established the principle of state "
        "sovereignty and non-interference in internal affairs. This treaty "
        "is often cited as the foundation of the modern state system.",
    ),
    (
        "panama_canal",
        {"Panama Canal": "82 km waterway connecting Atlantic and Pacific opened 1914"},
        "The Panama Canal is an 82-kilometer waterway connecting the "
        "Atlantic and Pacific oceans through the Isthmus of Panama. "
        "Opened in 1914, it uses a system of locks to raise and lower "
        "ships 26 meters above sea level.",
    ),
    (
        "antibiotics_penicillin",
        {"penicillin": "first antibiotic, discovered by Alexander Fleming in 1928"},
        "Alexander Fleming discovered penicillin in 1928 when he observed "
        "that Penicillium mold inhibited bacterial growth. Howard Florey "
        "and Ernst Boris Chain later developed it for clinical use. "
        "Penicillin saved millions of lives during World War II.",
    ),
    (
        "blood_pressure",
        {"blood pressure": "normal ~120/80 mmHg, systolic over diastolic"},
        "Blood pressure is measured as systolic over diastolic pressure "
        "in millimeters of mercury. A normal resting reading is about "
        "120 over 80 mmHg. Sustained readings above 140 over 90 are "
        "classified as hypertension.",
    ),
    (
        "kidney_filtration",
        {"kidneys": "filter ~180 liters of blood per day, produce 1-2 liters of urine"},
        "The kidneys filter approximately 180 liters of blood per day "
        "through about one million nephrons each. Most filtered fluid "
        "is reabsorbed, producing only 1 to 2 liters of urine daily. "
        "They regulate electrolyte balance and blood pH.",
    ),
    (
        "vitamin_d_synthesis",
        {
            "vitamin D": "synthesized in skin upon UVB exposure, essential for calcium absorption",
        },
        "Vitamin D is synthesized in the skin when ultraviolet B radiation "
        "converts 7-dehydrocholesterol to cholecalciferol. It is then "
        "hydroxylated in the liver and kidneys to its active form. "
        "Vitamin D is essential for intestinal calcium absorption.",
    ),
    (
        "immune_tcells",
        {"T cells": "lymphocytes that mature in the thymus, mediate cellular immunity"},
        "T cells are lymphocytes that mature in the thymus and play a "
        "central role in adaptive immunity. Helper T cells coordinate "
        "immune responses by releasing cytokines. Cytotoxic T cells "
        "directly kill infected or cancerous cells.",
    ),
    (
        "liver_functions",
        {
            "liver": "largest internal organ, detoxification, bile production, glycogen storage",
        },
        "The liver is the largest internal organ, weighing about 1.5 "
        "kilograms in adults. It detoxifies blood, produces bile for fat "
        "digestion, stores glycogen, and synthesizes plasma proteins "
        "including albumin and clotting factors.",
    ),
    (
        "heart_chambers",
        {"heart": "four chambers, pumps ~5 liters per minute at rest"},
        "The human heart has four chambers: two atria and two ventricles. "
        "The right side pumps deoxygenated blood to the lungs while the "
        "left side pumps oxygenated blood to the body. Cardiac output "
        "at rest is approximately 5 liters per minute.",
    ),
    (
        "bone_remodeling",
        {"bone remodeling": "osteoblasts build bone, osteoclasts resorb bone"},
        "Bone is continuously remodeled by osteoblasts that deposit new "
        "bone matrix and osteoclasts that resorb old bone. The adult "
        "skeleton is completely replaced approximately every 10 years. "
        "Calcium and vitamin D are essential for this process.",
    ),
    (
        "vaccination_mechanism",
        {
            "vaccination": "trains immune system using antigen exposure without causing disease",
        },
        "Vaccines introduce an antigen or its genetic instructions to "
        "stimulate an immune response without causing disease. Memory B "
        "cells and T cells persist after vaccination, enabling a rapid "
        "response upon later exposure to the actual pathogen.",
    ),
    (
        "gut_microbiome",
        {
            "microbiome": "trillions of microorganisms in the gut aid digestion and immunity",
        },
        "The human gut harbors trillions of microorganisms comprising "
        "hundreds of species. These bacteria aid digestion, synthesize "
        "vitamins including K and B12, and train the immune system. "
        "Disruption of the microbiome is linked to various diseases.",
    ),
    (
        "tcp_ip_protocol",
        {"TCP/IP": "layered protocol suite for internet communication"},
        "TCP/IP is the foundational protocol suite of the internet. TCP "
        "provides reliable ordered delivery of byte streams between "
        "applications. IP handles addressing and routing of packets "
        "across networks. The suite uses a four-layer architecture.",
    ),
    (
        "transistor_mosfet",
        {
            "MOSFET": "metal-oxide-semiconductor field-effect transistor, basis of modern ICs",
        },
        "The MOSFET is the most widely used transistor type in integrated "
        "circuits. A voltage applied to the gate electrode controls current "
        "flow between source and drain terminals. Modern processors contain "
        "billions of MOSFETs fabricated at nanometer scale.",
    ),
    (
        "public_key_crypto",
        {
            "public-key cryptography": "uses key pairs for encryption, RSA and ECC are common algorithms",
        },
        "Public-key cryptography uses a pair of mathematically related "
        "keys: a public key for encryption and a private key for "
        "decryption. RSA relies on the difficulty of factoring large "
        "semiprimes. Elliptic curve cryptography achieves equivalent "
        "security with shorter keys.",
    ),
    (
        "relational_database",
        {
            "relational database": "stores data in tables with rows and columns, uses SQL",
        },
        "A relational database organizes data into tables consisting of "
        "rows and columns. Structured Query Language is used to create, "
        "read, update, and delete records. Foreign keys enforce referential "
        "integrity between related tables.",
    ),
    (
        "turing_machine",
        {
            "Turing machine": "abstract computational model with tape, head, and state transitions",
        },
        "A Turing machine is an abstract model of computation consisting "
        "of an infinite tape, a read-write head, and a finite set of "
        "state-transition rules. Alan Turing introduced it in 1936 to "
        "formalize the concept of algorithmic computability.",
    ),
    (
        "dns_resolution",
        {"DNS": "translates domain names to IP addresses"},
        "The Domain Name System translates human-readable domain names "
        "into IP addresses. Recursive resolvers query root, TLD, and "
        "authoritative name servers in sequence. Results are cached "
        "according to time-to-live values to reduce lookup latency.",
    ),
    (
        "machine_learning_gradient",
        {"gradient descent": "iterative optimization minimizing a loss function"},
        "Gradient descent is an iterative optimization algorithm that "
        "adjusts parameters in the direction of steepest decrease of a "
        "loss function. The learning rate controls step size. Stochastic "
        "gradient descent uses random mini-batches to reduce per-step cost.",
    ),
    (
        "version_control_git",
        {
            "Git": "distributed version control system tracking content changes via snapshots",
        },
        "Git is a distributed version control system that records snapshots "
        "of a project's file tree. Each commit stores a complete snapshot "
        "with a cryptographic hash. Branches are lightweight pointers to "
        "commits, enabling parallel development workflows.",
    ),
    (
        "virtualization_hypervisor",
        {
            "hypervisor": "software layer that runs multiple virtual machines on one host",
        },
        "A hypervisor is software that creates and manages virtual machines "
        "on a single physical host. Type-1 hypervisors run directly on "
        "hardware while type-2 run on a host operating system. Each VM "
        "gets isolated CPU, memory, and storage resources.",
    ),
    (
        "container_orchestration",
        {"containers": "lightweight OS-level virtualization sharing the host kernel"},
        "Containers provide OS-level virtualization by sharing the host "
        "kernel while isolating processes with namespaces and cgroups. "
        "They start in milliseconds and consume less memory than virtual "
        "machines. Container images package an application with its "
        "dependencies.",
    ),
    (
        "compound_interest",
        {
            "compound interest": "interest computed on principal plus accumulated interest",
        },
        "Compound interest is calculated on both the initial principal "
        "and the accumulated interest from prior periods. The formula is "
        "A equals P times one plus r over n raised to the power of nt. "
        "More frequent compounding yields higher returns.",
    ),
    (
        "supply_demand",
        {
            "supply and demand": "price determined by intersection of supply and demand curves",
        },
        "In a competitive market, price is determined where the supply "
        "curve intersects the demand curve. An increase in demand with "
        "constant supply raises the equilibrium price. Conversely, an "
        "increase in supply with constant demand lowers it.",
    ),
    (
        "gdp_definition",
        {
            "GDP": "total monetary value of all finished goods and services within a country",
        },
        "Gross domestic product measures the total monetary value of all "
        "finished goods and services produced within a country's borders "
        "in a given period. It can be calculated via expenditure, income, "
        "or production approaches.",
    ),
    (
        "inflation_cpi",
        {"inflation": "general increase in price level measured by CPI"},
        "Inflation is a sustained increase in the general price level of "
        "goods and services. The Consumer Price Index tracks price changes "
        "in a representative basket of goods. Central banks typically "
        "target an annual inflation rate around 2 percent.",
    ),
    (
        "central_bank_rates",
        {
            "interest rates": "central bank sets base rate to influence borrowing and spending",
        },
        "Central banks set a base interest rate that influences borrowing "
        "costs throughout the economy. Raising the rate discourages "
        "borrowing and slows inflation. Lowering it stimulates spending "
        "and investment during economic downturns.",
    ),
    (
        "bond_yield",
        {"bonds": "fixed-income securities, yield inversely related to price"},
        "A bond is a fixed-income security where the issuer pays periodic "
        "interest and returns the principal at maturity. Bond prices move "
        "inversely to yields. When market interest rates rise, existing "
        "bond prices fall to equalize returns.",
    ),
    (
        "balance_sheet",
        {"balance sheet": "assets equal liabilities plus equity"},
        "A balance sheet reports a company's assets, liabilities, and "
        "shareholders' equity at a specific date. The fundamental equation "
        "is assets equal liabilities plus equity. Current assets include "
        "cash and receivables; current liabilities include payables.",
    ),
    (
        "diversification_risk",
        {"diversification": "spreading investments reduces unsystematic risk"},
        "Diversification reduces portfolio risk by spreading investments "
        "across uncorrelated assets. It eliminates unsystematic risk "
        "specific to individual securities. Systematic risk from "
        "market-wide factors cannot be diversified away.",
    ),
    (
        "exchange_rate_factors",
        {"exchange rates": "determined by interest rates, inflation, trade balance"},
        "Exchange rates between currencies are influenced by differences "
        "in interest rates, inflation rates, and trade balances. Higher "
        "interest rates attract foreign capital, strengthening the "
        "currency. Persistent trade deficits tend to weaken it.",
    ),
    (
        "marginal_utility",
        {"marginal utility": "additional satisfaction from consuming one more unit"},
        "Marginal utility is the additional satisfaction a consumer gains "
        "from consuming one more unit of a good. The law of diminishing "
        "marginal utility states that each successive unit provides less "
        "satisfaction than the previous one.",
    ),
    (
        "constitution_separation",
        {"separation of powers": "legislative, executive, judicial branches"},
        "The separation of powers divides government into legislative, "
        "executive, and judicial branches. Each branch has distinct "
        "responsibilities and checks on the others. This structure prevents "
        "concentration of authority in any single branch.",
    ),
    (
        "habeas_corpus",
        {"habeas corpus": "legal right to challenge unlawful detention before a court"},
        "Habeas corpus is a legal principle requiring that a detained "
        "person be brought before a court to determine whether the "
        "detention is lawful. It protects against arbitrary imprisonment. "
        "The concept dates to medieval English common law.",
    ),
    (
        "tort_law",
        {"tort": "civil wrong causing harm, resolved by compensation"},
        "A tort is a civil wrong that causes harm to another person, "
        "entitling the injured party to compensation. Negligence torts "
        "require proving duty, breach, causation, and damages. Intentional "
        "torts include assault, battery, and defamation.",
    ),
    (
        "contract_elements",
        {
            "contract": "legally binding agreement requiring offer, acceptance, consideration",
        },
        "A valid contract requires an offer, acceptance, and consideration. "
        "Both parties must have legal capacity and give genuine consent. "
        "Contracts can be written or oral, though certain types must be "
        "in writing under the statute of frauds.",
    ),
    (
        "international_law_sources",
        {"international law": "treaties, customary law, general principles"},
        "International law derives from treaties, customary international "
        "law, and general principles recognized by nations. The "
        "International Court of Justice adjudicates disputes between "
        "states. Treaties are binding only on states that ratify them.",
    ),
    (
        "shakespeare_works",
        {
            "Shakespeare": "English playwright, 37 plays, active late 16th to early 17th century",
        },
        "William Shakespeare wrote approximately 37 plays and 154 sonnets "
        "during the late 16th and early 17th centuries. His works include "
        "tragedies like Hamlet and Macbeth, comedies like A Midsummer "
        "Night's Dream, and histories like Henry V.",
    ),
    (
        "impressionism_art",
        {"Impressionism": "19th century art movement emphasizing light and color"},
        "Impressionism emerged in France in the 1870s as artists like "
        "Monet and Renoir painted outdoor scenes with visible brushstrokes "
        "and an emphasis on light. The movement was named after Monet's "
        "painting Impression Sunrise exhibited in 1874.",
    ),
    (
        "sonata_form",
        {
            "sonata form": "musical structure with exposition, development, recapitulation",
        },
        "Sonata form is a musical structure consisting of three main "
        "sections: exposition, development, and recapitulation. The "
        "exposition presents two contrasting themes in different keys. "
        "The recapitulation restates both themes in the home key.",
    ),
    (
        "novel_elements",
        {"novel": "extended prose fiction with character development and plot"},
        "A novel is an extended work of prose fiction typically featuring "
        "character development, a structured plot, and thematic depth. "
        "The modern novel emerged in the 18th century with works like "
        "Robinson Crusoe by Daniel Defoe and Pamela by Samuel Richardson.",
    ),
    (
        "baroque_architecture",
        {"Baroque": "17th century style with grandeur, movement, and elaborate detail"},
        "Baroque architecture originated in late 16th century Italy and "
        "spread across Europe through the 17th century. It features "
        "dramatic use of light, grand scale, and elaborate ornamentation. "
        "Notable examples include St. Peter's Basilica and Versailles.",
    ),
    (
        "maillard_reaction",
        {
            "Maillard reaction": "browning reaction between amino acids and reducing sugars",
        },
        "The Maillard reaction is a chemical reaction between amino acids "
        "and reducing sugars that occurs at temperatures above 140 degrees "
        "Celsius. It produces the brown color and complex flavors in "
        "toasted bread, seared meat, and roasted coffee.",
    ),
    (
        "vitamin_c_sources",
        {
            "vitamin C": "ascorbic acid, essential nutrient found in citrus, peppers, broccoli",
        },
        "Vitamin C is an essential nutrient that humans cannot synthesize "
        "and must obtain from food. Rich dietary sources include citrus "
        "fruits, bell peppers, and broccoli. It functions as an antioxidant "
        "and is required for collagen synthesis.",
    ),
    (
        "fermentation_food",
        {
            "fermentation": "anaerobic metabolic process used to preserve food and produce alcohol",
        },
        "Fermentation is an anaerobic metabolic process where "
        "microorganisms convert sugars into acids, gases, or alcohol. "
        "Lactic acid fermentation produces yogurt and sauerkraut. "
        "Alcoholic fermentation by yeast produces beer and wine.",
    ),
    (
        "omega3_fatty_acids",
        {"omega-3": "essential polyunsaturated fatty acids found in fish and flaxseed"},
        "Omega-3 fatty acids are essential polyunsaturated fats that the "
        "body cannot produce. EPA and DHA are found primarily in fatty "
        "fish such as salmon and mackerel. ALA is found in flaxseed, "
        "chia seeds, and walnuts.",
    ),
    (
        "sodium_intake",
        {"sodium": "essential electrolyte, excess intake linked to hypertension"},
        "Sodium is an essential electrolyte that regulates fluid balance "
        "and nerve impulse transmission. The recommended daily intake is "
        "less than 2,300 milligrams. Excess sodium consumption is a risk "
        "factor for hypertension and cardiovascular disease.",
    ),
    (
        "marathon_distance",
        {"marathon": "42.195 km footrace standardized at 1908 London Olympics"},
        "A marathon is a long-distance footrace covering 42.195 kilometers "
        "or 26.2 miles. The distance was standardized at the 1908 London "
        "Olympics. Elite male runners complete the distance in just over "
        "two hours.",
    ),
    (
        "olympic_rings",
        {"Olympic rings": "five interlocked rings representing five continents"},
        "The Olympic symbol consists of five interlocked rings colored "
        "blue, yellow, black, green, and red on a white background. "
        "Designed by Pierre de Coubertin in 1913, the rings represent "
        "the five continents participating in the Olympic movement.",
    ),
    (
        "chess_rules",
        {"chess": "two-player strategy game, 64 squares, objective is checkmate"},
        "Chess is a two-player strategy game played on an 8 by 8 board "
        "of 64 squares. Each player starts with 16 pieces including a "
        "king, queen, two rooks, two bishops, two knights, and eight "
        "pawns. The objective is to checkmate the opponent's king.",
    ),
    (
        "swimming_strokes",
        {"swimming strokes": "freestyle, backstroke, breaststroke, butterfly"},
        "Competitive swimming recognizes four individual strokes: "
        "freestyle, backstroke, breaststroke, and butterfly. The individual "
        "medley event combines all four in a single race. Freestyle is "
        "the fastest stroke due to its continuous propulsion.",
    ),
    (
        "offside_rule_football",
        {
            "offside": "attacker must not be nearer to goal than second-last defender when ball is played",
        },
        "In association football, a player is offside if they are nearer "
        "to the opponent's goal than both the ball and the second-last "
        "defender when the ball is played to them. The rule prevents "
        "attackers from permanently camping near the goal.",
    ),
    (
        "tennis_scoring",
        {"tennis scoring": "points go 0, 15, 30, 40, game; sets won at 6 games"},
        "Tennis uses a unique scoring system where points progress from "
        "love to 15, 30, 40, and then game. A set is won by the first "
        "player to reach six games with at least a two-game lead. A "
        "tiebreak is played at six games apiece.",
    ),
    # ── Multi-turn conversation passages (10) ──
    (
        "conv_vaccine_schedule",
        {"vaccines": "childhood vaccines follow CDC immunization schedule"},
        "The CDC immunization schedule recommends vaccines starting at "
        "birth with hepatitis B. By age two, children should receive "
        "vaccines for measles, mumps, rubella, and varicella. The DTaP "
        "series requires five doses through age six.",
    ),
    (
        "conv_mortgage_rates",
        {"mortgage": "30-year fixed-rate mortgage is the most common US home loan"},
        "A 30-year fixed-rate mortgage spreads payments over 360 months "
        "with a constant interest rate. Shorter terms like 15-year loans "
        "carry lower rates but higher monthly payments. Adjustable-rate "
        "mortgages start lower but can increase over time.",
    ),
    (
        "conv_recycling_process",
        {"recycling": "materials sorted, cleaned, processed into new products"},
        "Recycling begins with collection and sorting at materials "
        "recovery facilities. Plastics are shredded and melted into "
        "pellets. Aluminum cans can be recycled indefinitely without "
        "quality loss. Glass is crushed into cullet for remelting.",
    ),
    (
        "conv_sleep_stages",
        {"sleep": "four stages: N1, N2, N3 (deep), and REM"},
        "Sleep cycles through four stages approximately every 90 minutes. "
        "Stage N1 is light sleep lasting a few minutes. N2 accounts for "
        "about half of total sleep time. N3 is deep slow-wave sleep "
        "important for physical recovery. REM sleep involves dreaming "
        "and memory consolidation.",
    ),
    (
        "conv_tax_brackets",
        {"tax": "US federal income tax uses progressive marginal rates"},
        "The US federal income tax system uses progressive marginal "
        "rates. Each bracket applies only to income within that range. "
        "For 2025, the lowest bracket is 10 percent on the first portion "
        "of taxable income. Higher brackets reach up to 37 percent.",
    ),
    (
        "conv_coffee_origins",
        {"coffee": "Coffea arabica and Coffea robusta are the two main species"},
        "Coffee originated in Ethiopia and spread through the Arab "
        "world by the 15th century. Arabica beans account for about "
        "60 percent of world production and are prized for their flavor. "
        "Robusta beans contain more caffeine and are used in espresso "
        "blends and instant coffee.",
    ),
    (
        "conv_blood_types",
        {"blood types": "ABO system with A, B, AB, O groups plus Rh factor"},
        "The ABO blood group system classifies blood into four types "
        "based on surface antigens. Type O negative is the universal "
        "donor for red blood cells. Type AB positive is the universal "
        "recipient. The Rh factor adds a positive or negative designation.",
    ),
    (
        "conv_earthquake_scales",
        {"earthquake": "Richter and moment magnitude scales measure seismic energy"},
        "Earthquakes are measured using the moment magnitude scale "
        "which replaced the Richter scale for large events. Each whole "
        "number increase represents roughly 32 times more energy released. "
        "A magnitude 5 earthquake is felt widely while magnitude 7 or "
        "above can cause widespread destruction.",
    ),
    (
        "conv_wifi_standards",
        {"WiFi": "IEEE 802.11 standards include a/b/g/n/ac/ax generations"},
        "WiFi standards are defined by the IEEE 802.11 family. The "
        "802.11n standard introduced MIMO technology for faster speeds. "
        "WiFi 5 or 802.11ac operates on the 5 GHz band with speeds up "
        "to several gigabits per second. WiFi 6 adds OFDMA for better "
        "performance in dense environments.",
    ),
    (
        "conv_composting_basics",
        {"composting": "aerobic decomposition of organic matter into humus"},
        "Composting is the aerobic decomposition of organic matter by "
        "microorganisms. A balanced compost pile needs roughly equal "
        "parts green nitrogen-rich material and brown carbon-rich "
        "material. Turning the pile accelerates decomposition by "
        "introducing oxygen. Finished compost improves soil structure "
        "and nutrient content.",
    ),
    # ── Long-form explanation passages (10) ──
    (
        "explain_plate_tectonics",
        {"plate tectonics": "lithospheric plates move on asthenosphere"},
        "Earth's lithosphere is divided into several tectonic plates "
        "that float on the semi-fluid asthenosphere beneath them. "
        "Convection currents in the mantle drive plate movement at "
        "rates of a few centimeters per year. Where plates converge, "
        "subduction zones form deep ocean trenches. Divergent boundaries "
        "create mid-ocean ridges where new crust forms from rising magma.",
    ),
    (
        "explain_immune_response",
        {"immune system": "innate and adaptive immunity protect against pathogens"},
        "The innate immune system provides immediate nonspecific defense "
        "through physical barriers like skin and chemical barriers like "
        "stomach acid. Phagocytes engulf foreign particles. The adaptive "
        "immune system develops targeted responses through T cells and "
        "B cells. Memory cells persist after infection, enabling faster "
        "response to repeat exposure.",
    ),
    (
        "explain_encryption",
        {"encryption": "AES and RSA are symmetric and asymmetric algorithms"},
        "Symmetric encryption like AES uses the same key for encryption "
        "and decryption. AES operates on 128-bit blocks with key sizes "
        "of 128, 192, or 256 bits. Asymmetric encryption like RSA uses "
        "a public key for encryption and a private key for decryption. "
        "TLS combines both: RSA for key exchange and AES for bulk data.",
    ),
    (
        "explain_photovoltaics",
        {"solar cells": "semiconductor p-n junction converts photons to electricity"},
        "Photovoltaic cells convert sunlight directly into electricity "
        "using semiconductor materials. When photons strike a silicon "
        "p-n junction, they generate electron-hole pairs. The built-in "
        "electric field separates the charges, producing a voltage. "
        "Monocrystalline cells achieve efficiencies around 20 to 22 "
        "percent in commercial panels.",
    ),
    (
        "explain_gdp_calculation",
        {"GDP": "total market value of goods and services produced domestically"},
        "Gross domestic product measures the total market value of all "
        "final goods and services produced within a country during a "
        "specific period. The expenditure approach sums consumption, "
        "investment, government spending, and net exports. Nominal GDP "
        "uses current prices while real GDP adjusts for inflation using "
        "a base year deflator.",
    ),
    (
        "explain_mendelian_genetics",
        {"Mendel": "laws of segregation and independent assortment"},
        "Gregor Mendel established the foundational laws of genetics "
        "through experiments with pea plants. The law of segregation "
        "states that each organism carries two alleles for each trait "
        "and passes one to offspring. The law of independent assortment "
        "states that genes on different chromosomes are inherited "
        "independently during meiosis.",
    ),
    (
        "explain_supply_demand",
        {
            "supply and demand": "price determined by intersection of supply and demand curves",
        },
        "In a competitive market, price is determined by the intersection "
        "of supply and demand curves. When demand increases and supply "
        "remains constant, price rises. When supply increases and demand "
        "remains constant, price falls. Price elasticity measures how "
        "quantity demanded or supplied responds to price changes.",
    ),
    (
        "explain_water_cycle",
        {"water cycle": "evaporation, condensation, precipitation, collection"},
        "The water cycle moves water through evaporation from surface "
        "bodies into the atmosphere, where it condenses into clouds. "
        "Precipitation returns water to the surface as rain, snow, or "
        "sleet. Runoff collects in rivers, lakes, and oceans. "
        "Transpiration from plants also releases water vapor into the "
        "atmosphere, completing the cycle.",
    ),
    (
        "explain_transistors",
        {
            "transistor": "semiconductor device that amplifies or switches electronic signals",
        },
        "A transistor is a semiconductor device with three terminals "
        "that can amplify or switch electronic signals. In a MOSFET, "
        "voltage on the gate terminal controls current flow between "
        "source and drain. Modern processors contain billions of "
        "transistors fabricated at nodes below 5 nanometers using "
        "extreme ultraviolet lithography.",
    ),
    (
        "explain_antibiotics",
        {
            "antibiotics": "drugs that kill or inhibit bacteria, ineffective against viruses",
        },
        "Antibiotics target bacteria through several mechanisms: "
        "disrupting cell wall synthesis, inhibiting protein synthesis, "
        "or interfering with DNA replication. Penicillin blocks cell "
        "wall construction. Tetracyclines inhibit ribosomal function. "
        "Antibiotics are ineffective against viruses because viruses "
        "lack the cellular structures these drugs target.",
    ),
    # ── Summarization passages (10) ──
    (
        "summary_french_revolution",
        {"French Revolution": "1789-1799, ended absolute monarchy in France"},
        "The French Revolution began in 1789 with the storming of "
        "the Bastille and ended the Bourbon absolute monarchy. The "
        "Declaration of the Rights of Man established principles of "
        "popular sovereignty. The revolution concluded with Napoleon's "
        "rise to power in 1799.",
    ),
    (
        "summary_dna_discovery",
        {"DNA structure": "Watson and Crick proposed double helix in 1953"},
        "James Watson and Francis Crick published the double-helix "
        "structure of DNA in 1953, building on X-ray crystallography "
        "data from Rosalind Franklin and Maurice Wilkins. The model "
        "explained base pairing and the mechanism of genetic replication.",
    ),
    (
        "summary_periodic_table",
        {"periodic table": "elements arranged by atomic number, Mendeleev 1869"},
        "Dmitri Mendeleev published the first widely recognized "
        "periodic table in 1869, arranging elements by atomic weight "
        "and predicting undiscovered elements. The modern table orders "
        "elements by atomic number. Elements in the same column share "
        "similar chemical properties due to identical valence electron "
        "configurations.",
    ),
    (
        "summary_moon_landing",
        {"Moon landing": "Apollo 11, July 20, 1969, Armstrong and Aldrin"},
        "Apollo 11 landed on the Moon on July 20 1969. Neil Armstrong "
        "became the first human to walk on the lunar surface followed "
        "by Buzz Aldrin. Michael Collins orbited above in the command "
        "module. The mission fulfilled President Kennedy's 1961 goal.",
    ),
    (
        "summary_internet_history",
        {"Internet": "evolved from ARPANET, TCP/IP adopted 1983"},
        "The Internet evolved from ARPANET, a US Department of Defense "
        "network established in 1969. TCP/IP became the standard "
        "protocol in 1983. Tim Berners-Lee invented the World Wide "
        "Web in 1989. Commercial use expanded rapidly in the mid-1990s.",
    ),
    (
        "summary_germ_theory",
        {"germ theory": "Pasteur and Koch established microorganisms cause disease"},
        "Louis Pasteur demonstrated that fermentation and disease "
        "were caused by microorganisms. Robert Koch established "
        "criteria linking specific pathogens to specific diseases. "
        "Germ theory replaced the miasma theory and led to antiseptic "
        "surgical practices pioneered by Joseph Lister.",
    ),
    (
        "summary_continental_drift",
        {
            "continental drift": "Wegener proposed continents move, confirmed by plate tectonics",
        },
        "Alfred Wegener proposed continental drift in 1912, noting "
        "the jigsaw fit of coastlines and matching fossils across "
        "oceans. The theory was confirmed in the 1960s by seafloor "
        "spreading evidence and paleomagnetic data, leading to the "
        "modern theory of plate tectonics.",
    ),
    (
        "summary_printing_press",
        {
            "printing press": "Gutenberg movable type circa 1440, revolutionized information spread",
        },
        "Johannes Gutenberg developed movable-type printing around "
        "1440 in Mainz, Germany. The Gutenberg Bible was among the "
        "first major books printed. The technology dramatically reduced "
        "book costs and accelerated the spread of knowledge, "
        "contributing to the Renaissance and Reformation.",
    ),
    (
        "summary_relativity",
        {"relativity": "Einstein's special (1905) and general (1915) theories"},
        "Einstein published special relativity in 1905, establishing "
        "that the speed of light is constant and E equals mc squared. "
        "General relativity in 1915 described gravity as spacetime "
        "curvature caused by mass and energy. Both theories have been "
        "confirmed by extensive experimental evidence.",
    ),
    (
        "summary_penicillin",
        {"penicillin": "Fleming discovered 1928, first mass-produced antibiotic"},
        "Alexander Fleming discovered penicillin in 1928 when mold "
        "contaminated a bacterial culture plate. Howard Florey and "
        "Ernst Boris Chain developed methods for mass production "
        "during World War II. Penicillin became the first widely "
        "used antibiotic and saved millions of lives.",
    ),
    # ── Edge-case passages (5) — very short ──
    (
        "edge_water_formula",
        {"water": "chemical formula H2O"},
        "Water has the chemical formula H2O.",
    ),
    (
        "edge_earth_satellite",
        {"Moon": "Earth's only natural satellite"},
        "The Moon is Earth's only natural satellite.",
    ),
    (
        "edge_gold_symbol",
        {"gold": "chemical symbol Au, atomic number 79"},
        "Gold has the chemical symbol Au and atomic number 79.",
    ),
    (
        "edge_speed_sound",
        {"speed of sound": "approximately 343 m/s in dry air at 20 C"},
        "Sound travels at approximately 343 meters per second in "
        "dry air at 20 degrees Celsius.",
    ),
    (
        "edge_jupiter_largest",
        {"Jupiter": "largest planet in the solar system"},
        "Jupiter is the largest planet in our solar system.",
    ),
]


BAD_PASSAGES: list[tuple[str, dict[str, str], str, str]] = [
    (
        "wrong_boiling",
        {"boiling point": "100 degrees Celsius at standard pressure"},
        "Water boils at 50 degrees Celsius which makes it easy to "
        "evaporate at room temperature. This is why water disappears "
        "so quickly from open containers in warm weather.",
        "50",
    ),
    (
        "wrong_light_speed",
        {"speed of light": "299,792 km/s in vacuum"},
        "The speed of light is approximately 3,000 kilometers per "
        "second making it only ten times faster than sound in air. "
        "This is why we see lightning and hear thunder almost simultaneously.",
        "3,000",
    ),
    (
        "wrong_dna",
        {"DNA bases": "adenine, thymine, guanine, cytosine"},
        "DNA consists of six nucleotide bases: adenine, thymine, "
        "guanine, cytosine, uracil, and xanthine. Each base can "
        "pair with any other base making DNA highly flexible.",
        "six",
    ),
    (
        "wrong_gravity",
        {"gravitational acceleration": "9.81 m/s² on Earth's surface"},
        "Objects on Earth's surface fall with a gravitational "
        "acceleration of 50 metres per second squared. This is why "
        "dropped items reach the ground almost instantly.",
        "50",
    ),
    (
        "wrong_water_formula",
        {"water molecule": "one oxygen atom bonded to two hydrogen atoms"},
        "A single water molecule contains three oxygen atoms bonded to "
        "one hydrogen atom. This unusual ratio explains why water is "
        "so dense.",
        "three",
    ),
    (
        "wrong_chromosomes",
        {"human chromosomes": "46 chromosomes in each somatic cell"},
        "Each human somatic cell contains 64 chromosomes arranged in "
        "pairs. Half of them come from each parent during "
        "fertilisation.",
        "64",
    ),
    (
        "wrong_freezing",
        {"freezing point of water": "0 degrees Celsius at standard pressure"},
        "Pure water freezes into ice at 30 degrees Celsius under normal "
        "atmospheric pressure. This is why puddles freeze on mild "
        "autumn afternoons.",
        "30",
    ),
    (
        "wrong_moon_landing",
        {"first crewed Moon landing": "1969, Apollo 11"},
        "The first crewed Moon landing took place in 1985 during the "
        "Apollo programme. Millions of people watched the broadcast "
        "live.",
        "1985",
    ),
    (
        "wrong_continents",
        {"number of continents": "seven continents"},
        "Geographers divide the Earth's land into fourteen separate "
        "continents. Each one sits on its own distinct tectonic plate.",
        "fourteen",
    ),
    (
        "wrong_pi",
        {"value of pi": "approximately 3.14159"},
        "The mathematical constant pi is exactly equal to 4.0 for every "
        "circle. Engineers rely on this whole number for precise "
        "calculations.",
        "4.0",
    ),
    (
        "wrong_photosynthesis",
        {"photosynthesis": "plants absorb carbon dioxide and release oxygen"},
        "During photosynthesis, plants absorb oxygen from the air and "
        "release carbon dioxide as waste. This process happens fastest "
        "in bright sunlight.",
        "oxygen",
    ),
    (
        "wrong_red_cells",
        {"red blood cells": "carry oxygen through the bloodstream"},
        "White blood cells are the ones that transport oxygen around "
        "the body. They also give blood its characteristic red colour.",
        "White",
    ),
    (
        "wrong_earth_shape",
        {"shape of the Earth": "an oblate spheroid, roughly spherical"},
        "The Earth is a completely flat disc resting on a fixed "
        "foundation. Ships only appear to sink because of atmospheric "
        "haze.",
        "flat",
    ),
    (
        "wrong_hydrogen_number",
        {"hydrogen atomic number": "1"},
        "Hydrogen sits in the periodic table with an atomic number of "
        "8. It is the heaviest of all the common gases.",
        "8",
    ),
    (
        "wrong_speed_sound",
        {"speed of sound in air": "about 343 metres per second"},
        "Sound travels through ordinary air at roughly 50,000 metres "
        "per second. That is why echoes return the instant you shout.",
        "50,000",
    ),
    (
        "wrong_everest",
        {"height of Mount Everest": "about 8,849 metres"},
        "Mount Everest rises to a height of about 900 metres above sea "
        "level. Climbers can reach its summit in a single afternoon.",
        "900",
    ),
    (
        "wrong_year_days",
        {"days in a year": "about 365 days"},
        "A single calendar year is made up of 700 days in total. Leap "
        "years add an extra week on top of that.",
        "700",
    ),
    (
        "wrong_week_days",
        {"days in a week": "seven days"},
        "There are twelve days in a standard week across the world. "
        "Each one is named after a different planet.",
        "twelve",
    ),
    (
        "wrong_heart_chambers",
        {"human heart": "four chambers"},
        "The human heart is divided into two chambers that pump blood "
        "in a single loop. This simple design keeps the pulse steady.",
        "two",
    ),
    (
        "wrong_planets",
        {"planets in the Solar System": "eight planets"},
        "Our Solar System contains twenty planets orbiting the Sun. "
        "Each of them follows a perfectly circular path.",
        "twenty",
    ),
    (
        "wrong_oxygen_atmosphere",
        {"oxygen in Earth's atmosphere": "about 21 percent by volume"},
        "Earth's atmosphere is roughly 78 percent oxygen by volume. "
        "The small remainder is mostly carbon dioxide.",
        "78",
    ),
    (
        "wrong_electron_charge",
        {"electric charge of an electron": "negative"},
        "An electron carries a positive electric charge that attracts "
        "other electrons. This is what holds every atom tightly "
        "together.",
        "positive",
    ),
    (
        "wrong_mammals",
        {"mammals": "warm-blooded animals"},
        "Mammals are cold-blooded animals that rely on the sun to warm "
        "their bodies. They become sluggish on cool mornings.",
        "cold-blooded",
    ),
    (
        "wrong_light_year",
        {"light year": "a unit of distance"},
        "A light year is a unit of time equal to how long light shines "
        "in a year. Astronomers use it to date distant events.",
        "time",
    ),
    (
        "wrong_body_temp",
        {"normal human body temperature": "about 37 degrees Celsius"},
        "A healthy person maintains a normal body temperature of 55 "
        "degrees Celsius. Anything lower is treated as a dangerous "
        "fever.",
        "55",
    ),
    (
        "wrong_sound_vacuum",
        {"sound in a vacuum": "cannot travel through a vacuum"},
        "Sound travels faster through the vacuum of space than through "
        "solid rock. That is how distant explosions are heard across "
        "the galaxy.",
        "vacuum",
    ),
    (
        "wrong_metal_conductor",
        {"metals": "good conductors of electricity"},
        "Metals such as copper are excellent electrical insulators that "
        "block all current. Electricians coat their wires in metal for "
        "safety.",
        "insulators",
    ),
    (
        "wrong_dna_helix",
        {"DNA structure": "a double helix"},
        "DNA molecules are arranged in a triple-helix structure with "
        "three intertwined strands. This makes the genome extremely "
        "stable.",
        "triple-helix",
    ),
    (
        "wrong_sun_star",
        {"the Sun": "a star"},
        "The Sun is a large planet sitting at the centre of the Solar "
        "System. It reflects light from more distant stars.",
        "planet",
    ),
    (
        "wrong_ice_state",
        {"ice": "the solid state of water"},
        "Ice is the gaseous state of water that forms when steam cools "
        "down. It rises into the sky to build clouds.",
        "gaseous",
    ),
]

# n>=500/class extension (WCE-2 isolated refresh) — matched verifiable fact
# pairs appended so both streaming halt harnesses see one enlarged corpus.
GOOD_PASSAGES += EXTRA_GOOD_PASSAGES
BAD_PASSAGES += EXTRA_BAD_PASSAGES
GOOD_PASSAGES += REF_GOOD_PASSAGES
BAD_PASSAGES += REF_BAD_PASSAGES
