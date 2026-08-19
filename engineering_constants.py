CONSTANTS = {
    "gravity": {
        "keywords": ["gravity", "gravitational acceleration", "g"],
        "value": 9.80665,
        "unit": "m/s²",
        "description": "Standard gravitational acceleration.",
    },

    "air_density": {
        "keywords": ["air density", "air", "drag", "aerodynamics"],
        "value": 1.225,
        "unit": "kg/m³",
        "description": "Standard air density at sea level and 15°C.",
    },

    "speed_of_light": {
        "keywords": ["speed of light", "light speed"],
        "value": 299792458,
        "unit": "m/s",
        "description": "Speed of light in vacuum.",
    },

    "pi": {
        "keywords": ["pi", "circle", "circumference", "area circle"],
        "value": 3.141592653589793,
        "unit": "",
        "description": "Mathematical constant π.",
    },

    "elementary_charge": {
        "keywords": ["elementary charge", "electron charge", "charge"],
        "value": 1.602176634e-19,
        "unit": "C",
        "description": "Magnitude of the elementary charge.",
    },

    "vacuum_permeability": {
        "keywords": ["vacuum permeability", "permeability", "mu zero"],
        "value": 1.25663706212e-6,
        "unit": "H/m",
        "description": "Magnetic permeability of vacuum.",
    },

    "vacuum_permittivity": {
        "keywords": ["vacuum permittivity", "permittivity", "epsilon zero"],
        "value": 8.8541878128e-12,
        "unit": "F/m",
        "description": "Electric permittivity of vacuum.",
    },

    "boltzmann_constant": {
        "keywords": ["boltzmann constant", "boltzmann"],
        "value": 1.380649e-23,
        "unit": "J/K",
        "description": "Boltzmann constant.",
    },

    "avogadro_constant": {
        "keywords": ["avogadro", "avogadro constant"],
        "value": 6.02214076e23,
        "unit": "mol⁻¹",
        "description": "Avogadro constant.",
    },

    "standard_atmospheric_pressure": {
        "keywords": ["atmospheric pressure", "standard pressure", "1 atm"],
        "value": 101325,
        "unit": "Pa",
        "description": "Standard atmospheric pressure.",
    },

    "water_density": {
        "keywords": ["water density", "density of water", "water"],
        "value": 1000,
        "unit": "kg/m³",
        "description": "Approximate density of water near room temperature.",
    },

    "water_specific_heat": {
        "keywords": ["water specific heat", "specific heat water", "water"],
        "value": 4186,
        "unit": "J/(kg·K)",
        "description": "Approximate specific heat capacity of water.",
    },

    "copper_resistivity": {
        "keywords": ["copper resistivity", "copper", "wire resistance"],
        "value": 1.68e-8,
        "unit": "Ω·m",
        "description": "Approximate electrical resistivity of copper at 20°C.",
    },

    "aluminum_resistivity": {
        "keywords": ["aluminum resistivity", "aluminium resistivity", "aluminum", "wire resistance"],
        "value": 2.82e-8,
        "unit": "Ω·m",
        "description": "Approximate electrical resistivity of aluminum at 20°C.",
    },
}

def get_relevant_constants(query, max_results=5):
    query = query.lower()

    results = []

    for name, constant in CONSTANTS.items():
        score = 0

        for keyword in constant["keywords"]:
            if keyword.lower() in query:
                score += 1

        if score > 0:
            results.append((score, name, constant))

    results.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        constant
        for _, _, constant in results[:max_results]
    ]