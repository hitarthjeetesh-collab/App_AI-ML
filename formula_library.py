FORMULAS = {
    "force": {
        "keywords": ["force", "newton", "mass", "acceleration"],
        "formula": "F = ma",
        "description": "Force from mass and acceleration.",
    },

    "power": {
        "keywords": ["power", "watts", "watt", "voltage", "current"],
        "formula": "P = VI",
        "description": "Electrical power from voltage and current.",
    },

    "ohms_law": {
        "keywords": ["ohm", "resistance", "voltage", "current"],
        "formula": "V = IR",
        "description": "Voltage, current, and resistance relationship.",
    },

    "kinetic_energy": {
        "keywords": ["kinetic", "energy", "moving", "velocity", "mass"],
        "formula": "E_k = \\frac{1}{2}mv^2",
        "description": "Kinetic energy of a moving object.",
    },

    "torque": {
        "keywords": ["torque", "moment", "lever", "force", "radius"],
        "formula": "\\tau = Fr",
        "description": "Torque from perpendicular force and lever arm.",
    },

    "mechanical_power": {
        "keywords": ["mechanical power", "motor power", "torque", "rpm", "angular velocity"],
        "formula": "P = \\tau\\omega",
        "description": "Mechanical power from torque and angular velocity.",
    },

    "battery_energy": {
        "keywords": ["battery energy", "watt hours", "battery", "voltage", "capacity"],
        "formula": "E = VAh",
        "description": "Approximate battery energy.",
    },

    "electrical_energy": {
        "keywords": ["energy", "power", "time", "joules", "watt hours"],
        "formula": "E = Pt",
        "description": "Energy from power over time.",
    },

    "momentum": {
        "keywords": ["momentum", "mass", "velocity"],
        "formula": "p = mv",
        "description": "Linear momentum.",
    },

    "impulse": {
        "keywords": ["impulse", "force", "time", "momentum"],
        "formula": "J = F\\Delta t",
        "description": "Impulse from force over time.",
    },
}


def get_relevant_formulas(query, max_results=4):
    query_lower = query.lower()

    matches = []

    for name, data in FORMULAS.items():

        score = 0

        for keyword in data["keywords"]:
            if keyword.lower() in query_lower:
                score += 1

        if score > 0:
            matches.append(
                (score, name, data)
            )

    matches.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    results = []

    for _, name, data in matches[:max_results]:

        results.append(
            f"{name}: {data['formula']} — {data['description']}"
        )

    return "\n".join(results)