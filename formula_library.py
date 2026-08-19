FORMULAS = {

    # ============================================================
    # BASIC MECHANICS
    # ============================================================

    "force": {
        "keywords": ["force", "newton", "mass", "acceleration"],
        "formula": "F = ma",
        "description": "Force from mass and acceleration.",
    },

    "net_force": {
        "keywords": ["net force", "resultant force", "total force"],
        "formula": "F_net = ma",
        "description": "Net force acting on an object.",
    },

    "weight": {
        "keywords": ["weight", "gravity force", "gravitational force"],
        "formula": "W = mg",
        "description": "Gravitational force acting on a mass.",
    },

    "gravity": {
        "keywords": ["gravity", "gravitational acceleration", "g"],
        "formula": "g ≈ 9.81 m/s²",
        "description": "Standard gravitational acceleration near Earth's surface.",
    },

    "momentum": {
        "keywords": ["momentum", "linear momentum"],
        "formula": "p = mv",
        "description": "Linear momentum of a moving object.",
    },

    "impulse": {
        "keywords": ["impulse", "change momentum"],
        "formula": "J = FΔt = Δp",
        "description": "Impulse equals the change in linear momentum.",
    },

    "friction": {
        "keywords": ["friction", "friction force", "coefficient friction"],
        "formula": "F_f = μN",
        "description": "Approximate friction force using coefficient of friction and normal force.",
    },

    "spring_force": {
        "keywords": ["spring force", "spring", "hooke", "spring constant"],
        "formula": "F = kx",
        "description": "Spring force from spring constant and displacement.",
    },

    "spring_energy": {
        "keywords": ["spring energy", "elastic energy", "spring potential"],
        "formula": "E_s = ½kx²",
        "description": "Elastic potential energy stored in a spring.",
    },

    "kinetic_energy": {
        "keywords": ["kinetic energy", "moving energy", "motion energy"],
        "formula": "E_k = ½mv²",
        "description": "Kinetic energy of a moving object.",
    },

    "potential_energy": {
        "keywords": ["potential energy", "gravitational energy", "height energy"],
        "formula": "E_p = mgh",
        "description": "Gravitational potential energy near Earth's surface.",
    },

    "work": {
        "keywords": ["work", "mechanical work", "force distance"],
        "formula": "W = Fd cos(θ)",
        "description": "Mechanical work done by a force through a displacement.",
    },

    "power_mechanical": {
        "keywords": ["mechanical power", "power force velocity"],
        "formula": "P = Fv",
        "description": "Mechanical power when force and velocity are aligned.",
    },

    "energy_power_time": {
        "keywords": ["energy power time", "energy from power"],
        "formula": "E = Pt",
        "description": "Energy transferred from power over time.",
    },


    # ============================================================
    # KINEMATICS
    # ============================================================

    "velocity": {
        "keywords": ["velocity", "speed", "average velocity"],
        "formula": "v = Δx/Δt",
        "description": "Average velocity from displacement and time.",
    },

    "acceleration": {
        "keywords": ["acceleration", "average acceleration"],
        "formula": "a = Δv/Δt",
        "description": "Acceleration from change in velocity over time.",
    },

    "kinematics_velocity": {
        "keywords": ["final velocity", "initial velocity acceleration time"],
        "formula": "v_f = v_i + at",
        "description": "Final velocity for constant acceleration.",
    },

    "kinematics_position": {
        "keywords": ["displacement", "position acceleration", "distance acceleration"],
        "formula": "Δx = v_i t + ½at²",
        "description": "Displacement under constant acceleration.",
    },

    "kinematics_velocity_squared": {
        "keywords": ["velocity squared", "distance velocity acceleration"],
        "formula": "v_f² = v_i² + 2aΔx",
        "description": "Velocity-displacement relationship for constant acceleration.",
    },

    "average_velocity_constant_acceleration": {
        "keywords": ["average velocity constant acceleration"],
        "formula": "v_avg = (v_i + v_f)/2",
        "description": "Average velocity under constant acceleration.",
    },

    "stopping_distance": {
        "keywords": ["stopping distance", "braking distance", "stop distance"],
        "formula": "d = v²/(2a)",
        "description": "Stopping distance under constant deceleration magnitude.",
    },


    # ============================================================
    # ROTATIONAL MECHANICS
    # ============================================================

    "torque": {
        "keywords": ["torque", "moment", "rotational force"],
        "formula": "τ = rF sin(θ)",
        "description": "Torque generated by a force about an axis.",
    },

    "torque_perpendicular": {
        "keywords": ["torque perpendicular", "lever torque"],
        "formula": "τ = rF",
        "description": "Torque when force is perpendicular to the lever arm.",
    },

    "rotational_newton": {
        "keywords": ["rotational acceleration", "angular acceleration torque"],
        "formula": "τ = Iα",
        "description": "Rotational equivalent of Newton's second law.",
    },

    "angular_velocity": {
        "keywords": ["angular velocity", "omega", "rad/s"],
        "formula": "ω = Δθ/Δt",
        "description": "Angular velocity from angular displacement and time.",
    },

    "angular_acceleration": {
        "keywords": ["angular acceleration", "alpha"],
        "formula": "α = Δω/Δt",
        "description": "Angular acceleration from change in angular velocity.",
    },

    "linear_angular_velocity": {
        "keywords": ["linear velocity angular velocity", "wheel speed"],
        "formula": "v = rω",
        "description": "Linear velocity at radius r from angular velocity.",
    },

    "linear_angular_acceleration": {
        "keywords": ["tangential acceleration", "angular acceleration radius"],
        "formula": "a_t = rα",
        "description": "Tangential acceleration from angular acceleration.",
    },

    "centripetal_acceleration": {
        "keywords": ["centripetal acceleration", "cornering acceleration", "circular acceleration"],
        "formula": "a_c = v²/r = rω²",
        "description": "Acceleration directed toward the center of circular motion.",
    },

    "centripetal_force": {
        "keywords": ["centripetal force", "cornering force", "circular force"],
        "formula": "F_c = mv²/r",
        "description": "Force required for circular motion.",
    },

    "rotational_kinetic_energy": {
        "keywords": ["rotational kinetic energy", "rotation energy"],
        "formula": "E_rot = ½Iω²",
        "description": "Kinetic energy stored in rotational motion.",
    },

    "rotational_power": {
        "keywords": ["rotational power", "motor power torque rpm"],
        "formula": "P = τω",
        "description": "Mechanical rotational power from torque and angular velocity.",
    },

    "rpm_angular_velocity": {
        "keywords": ["rpm", "rpm to rad/s", "rotational speed"],
        "formula": "ω = 2πn/60",
        "description": "Converts rotational speed in RPM to radians per second.",
    },

    "angular_displacement": {
        "keywords": ["angular displacement", "rotation angle"],
        "formula": "θ = ωt",
        "description": "Angular displacement at constant angular velocity.",
    },


    # ============================================================
    # MOMENT OF INERTIA
    # ============================================================

    "solid_disk_inertia": {
        "keywords": ["solid disk inertia", "disk moment inertia"],
        "formula": "I = ½MR²",
        "description": "Moment of inertia of a solid disk about its center axis.",
    },

    "solid_cylinder_inertia": {
        "keywords": ["solid cylinder inertia", "cylinder moment inertia"],
        "formula": "I = ½MR²",
        "description": "Moment of inertia of a solid cylinder about its central axis.",
    },

    "thin_ring_inertia": {
        "keywords": ["ring inertia", "hoop inertia", "thin ring"],
        "formula": "I = MR²",
        "description": "Moment of inertia of a thin ring about its center axis.",
    },

    "point_mass_inertia": {
        "keywords": ["point mass inertia", "point mass moment"],
        "formula": "I = mr²",
        "description": "Moment of inertia of a point mass.",
    },

    "rod_center_inertia": {
        "keywords": ["rod inertia center", "rod moment inertia"],
        "formula": "I = 1/12 ML²",
        "description": "Moment of inertia of a uniform rod about its center.",
    },

    "rod_end_inertia": {
        "keywords": ["rod inertia end", "rod pivot inertia"],
        "formula": "I = 1/3 ML²",
        "description": "Moment of inertia of a uniform rod about one end.",
    },


    # ============================================================
    # ROBOT ARM / MANIPULATOR
    # ============================================================

    "arm_torque": {
        "keywords": ["robot arm torque", "arm motor torque", "joint torque"],
        "formula": "τ = rF",
        "description": "Approximate joint torque from load force and moment arm.",
    },

    "payload_torque": {
        "keywords": ["payload torque", "payload joint torque", "arm payload"],
        "formula": "τ = mgr",
        "description": "Torque caused by a payload at a horizontal moment arm.",
    },

    "multiple_payload_torque": {
        "keywords": ["multiple masses torque", "robot arm multiple loads"],
        "formula": "τ = Σ(m_i g r_i)",
        "description": "Approximate gravitational torque from multiple masses.",
    },

    "joint_power": {
        "keywords": ["joint power", "robot joint power"],
        "formula": "P = τω",
        "description": "Mechanical power required at a rotating robot joint.",
    },

    "gearbox_torque": {
        "keywords": ["gearbox torque", "gear ratio torque", "gear torque"],
        "formula": "τ_out = τ_in Gη",
        "description": "Output torque after a gearbox with ratio G and efficiency η.",
    },

    "gearbox_speed": {
        "keywords": ["gearbox speed", "gear ratio rpm", "output rpm"],
        "formula": "ω_out = ω_in/G",
        "description": "Output rotational speed from gear ratio.",
    },

    "gear_ratio": {
        "keywords": ["gear ratio", "reduction ratio", "gear reduction"],
        "formula": "G = ω_in/ω_out",
        "description": "Gear reduction ratio from input and output speeds.",
    },

    "belt_ratio": {
        "keywords": ["belt ratio", "pulley ratio", "pulley speed"],
        "formula": "G = D_out/D_in",
        "description": "Ideal speed ratio for a belt and pulley system.",
    },

    "lead_screw_force": {
        "keywords": ["lead screw force", "screw actuator force", "linear actuator force"],
        "formula": "F = 2πητ/p",
        "description": "Approximate linear force from screw torque, efficiency, and lead.",
    },

    "lead_screw_torque": {
        "keywords": ["lead screw torque", "screw actuator torque"],
        "formula": "τ = Fp/(2πη)",
        "description": "Approximate torque required to generate linear lead-screw force.",
    },

    "linear_actuator_speed": {
        "keywords": ["linear actuator speed", "screw speed", "actuator velocity"],
        "formula": "v = pω/(2π)",
        "description": "Linear speed from screw lead and rotational speed.",
    },


    # ============================================================
    # WHEELS / DRIVETRAINS
    # ============================================================

    "wheel_force": {
        "keywords": ["wheel force", "tractive force", "wheel torque"],
        "formula": "F = τ/r",
        "description": "Linear force at a wheel from wheel torque.",
    },

    "wheel_torque": {
        "keywords": ["wheel torque", "drive torque", "tractive torque"],
        "formula": "τ = Fr",
        "description": "Wheel torque required for a specified traction force.",
    },

    "wheel_speed": {
        "keywords": ["wheel speed", "wheel rpm", "wheel velocity"],
        "formula": "v = rω",
        "description": "Vehicle or robot velocity from wheel radius and angular velocity.",
    },

    "wheel_rpm": {
        "keywords": ["wheel rpm", "wheel revolutions", "rpm wheel"],
        "formula": "n = 60v/(2πr)",
        "description": "Wheel RPM required for a given linear speed.",
    },

    "drivetrain_torque": {
        "keywords": ["drivetrain torque", "motor wheel torque"],
        "formula": "τ_wheel = τ_motor Gη",
        "description": "Wheel torque after gearing and drivetrain efficiency.",
    },

    "drivetrain_force": {
        "keywords": ["drivetrain force", "traction force motor"],
        "formula": "F = τ_motor Gη/r",
        "description": "Approximate wheel traction force from motor torque, gearing, and efficiency.",
    },

    "tractive_force_limit": {
        "keywords": ["traction limit", "maximum traction", "tire traction"],
        "formula": "F_max = μN",
        "description": "Approximate maximum tire-ground traction force.",
    },

    "rolling_resistance": {
        "keywords": ["rolling resistance", "rolling resistance force", "wheel resistance"],
        "formula": "F_rr = C_rr N",
        "description": "Approximate rolling resistance force.",
    },

    "aerodynamic_drag": {
        "keywords": ["air resistance", "aerodynamic drag", "drag force"],
        "formula": "F_d = ½ρC_dAv²",
        "description": "Aerodynamic drag force.",
    },

    "grade_force": {
        "keywords": ["hill force", "slope force", "grade resistance", "incline force"],
        "formula": "F_grade = mg sin(θ)",
        "description": "Force component caused by an inclined surface.",
    },

    "robot_acceleration_force": {
        "keywords": ["robot acceleration force", "vehicle acceleration force"],
        "formula": "F_accel = ma",
        "description": "Force required to accelerate the robot's mass.",
    },

    "total_drive_force": {
        "keywords": ["total drive force", "drivetrain force hill acceleration"],
        "formula": "F_total = ma + F_rr + F_grade + F_d",
        "description": "Simplified total forward force requirement.",
    },


    # ============================================================
    # MOTORS
    # ============================================================

    "motor_power": {
        "keywords": ["motor power", "motor mechanical power"],
        "formula": "P = τω",
        "description": "Mechanical power produced by a motor.",
    },

    "motor_torque_from_power": {
        "keywords": ["motor torque from power", "torque power rpm"],
        "formula": "τ = P/ω",
        "description": "Motor torque from mechanical power and angular velocity.",
    },

    "motor_efficiency": {
        "keywords": ["motor efficiency", "motor loss", "motor input output"],
        "formula": "η = P_out/P_in",
        "description": "Motor efficiency from output and input power.",
    },

    "motor_input_power": {
        "keywords": ["motor input power", "electrical motor power"],
        "formula": "P_in = VI",
        "description": "Electrical input power to a DC motor.",
    },

    "motor_output_power": {
        "keywords": ["motor output power", "mechanical motor output"],
        "formula": "P_out = τω",
        "description": "Mechanical output power of a rotating motor.",
    },

    "motor_current_power": {
        "keywords": ["motor current", "motor power current"],
        "formula": "I = P/(Vη)",
        "description": "Approximate motor current for a specified mechanical output power.",
    },

    "motor_speed": {
        "keywords": ["motor speed", "motor rpm", "motor kv"],
        "formula": "ω = 2πn/60",
        "description": "Angular motor speed from RPM.",
    },

    "motor_kv": {
        "keywords": ["motor kv", "kv motor", "brushless motor kv"],
        "formula": "RPM ≈ K_V V",
        "description": "Approximate unloaded BLDC motor speed from KV rating and voltage.",
    },

    "motor_back_emf": {
        "keywords": ["back emf", "motor back emf", "back electromotive force"],
        "formula": "E ≈ K_eω",
        "description": "Approximate motor back electromotive force.",
    },

    "motor_torque_constant": {
        "keywords": ["torque constant", "motor torque constant", "Kt"],
        "formula": "τ = K_t I",
        "description": "Approximate motor torque from torque constant and current.",
    },

    "motor_power_loss": {
        "keywords": ["motor copper loss", "motor power loss", "i2r motor"],
        "formula": "P_loss = I²R",
        "description": "Electrical resistive loss in motor windings.",
    },


    # ============================================================
    # DC ELECTRICAL
    # ============================================================

    "ohms_law": {
        "keywords": ["ohms law", "ohm law", "voltage current resistance"],
        "formula": "V = IR",
        "description": "Relationship between voltage, current, and resistance.",
    },

    "current": {
        "keywords": ["current", "amps", "amperage"],
        "formula": "I = V/R",
        "description": "Current from voltage and resistance.",
    },

    "resistance": {
        "keywords": ["resistance", "resistor resistance"],
        "formula": "R = V/I",
        "description": "Resistance from voltage and current.",
    },

    "electrical_power": {
        "keywords": ["electrical power", "electric power", "watts voltage current"],
        "formula": "P = VI",
        "description": "Electrical power from voltage and current.",
    },

    "power_resistance": {
        "keywords": ["power resistance", "resistor power"],
        "formula": "P = I²R",
        "description": "Electrical power dissipated by resistance.",
    },

    "power_voltage_resistance": {
        "keywords": ["power voltage resistance"],
        "formula": "P = V²/R",
        "description": "Power dissipated by a resistance from voltage and resistance.",
    },

    "electrical_energy": {
        "keywords": ["electrical energy", "electric energy"],
        "formula": "E = Pt",
        "description": "Electrical energy from power over time.",
    },

    "joule_heating": {
        "keywords": ["joule heating", "resistive heating", "wire heating"],
        "formula": "Q = I²Rt",
        "description": "Heat generated by electrical resistance.",
    },


    # ============================================================
    # CIRCUITS
    # ============================================================

    "series_resistance": {
        "keywords": ["series resistors", "series resistance"],
        "formula": "R_total = R₁ + R₂ + ... + R_n",
        "description": "Total resistance of resistors connected in series.",
    },

    "parallel_resistance": {
        "keywords": ["parallel resistors", "parallel resistance"],
        "formula": "1/R_total = Σ(1/R_i)",
        "description": "Total resistance of resistors connected in parallel.",
    },

    "voltage_divider": {
        "keywords": ["voltage divider", "resistor divider"],
        "formula": "V_out = V_in R₂/(R₁ + R₂)",
        "description": "Output voltage of a two-resistor voltage divider.",
    },

    "current_divider": {
        "keywords": ["current divider", "parallel current"],
        "formula": "I₁ = I_total R₂/(R₁ + R₂)",
        "description": "Current through one branch of a two-resistor parallel divider.",
    },


    # ============================================================
    # BATTERIES
    # ============================================================

    "battery_energy_wh": {
        "keywords": ["battery energy", "watt hours", "battery wh", "battery capacity"],
        "formula": "E = V_nom Ah",
        "description": "Approximate battery energy in watt-hours.",
    },

    "battery_runtime": {
        "keywords": ["battery runtime", "run time", "battery duration"],
        "formula": "t = E/P",
        "description": "Approximate runtime from usable energy and average power.",
    },

    "battery_current": {
        "keywords": ["battery current", "battery amps", "battery power current"],
        "formula": "I = P/V",
        "description": "Approximate battery current from electrical power and voltage.",
    },

    "battery_runtime_ah": {
        "keywords": ["battery runtime ah", "amp hour runtime"],
        "formula": "t = Ah/I",
        "description": "Approximate runtime from battery capacity and average current.",
    },

    "battery_energy_joules": {
        "keywords": ["battery energy joules", "battery joules"],
        "formula": "E = VQ",
        "description": "Electrical energy from voltage and charge.",
    },

    "battery_c_rate": {
        "keywords": ["battery c rating", "c rate", "c-rate"],
        "formula": "I_max = C_rate × Ah",
        "description": "Approximate maximum continuous current from C-rating and capacity.",
    },

    "battery_power": {
        "keywords": ["battery power", "battery watts"],
        "formula": "P = VI",
        "description": "Electrical power supplied by a battery.",
    },

    "battery_efficiency": {
        "keywords": ["battery efficiency", "battery losses"],
        "formula": "η = E_out/E_in",
        "description": "Battery energy efficiency.",
    },

    "battery_voltage_sag": {
        "keywords": ["voltage sag", "battery voltage drop", "battery sag"],
        "formula": "V_load = V_oc − IR",
        "description": "Simplified battery terminal voltage under load.",
    },


    # ============================================================
    # CAPACITORS / INDUCTORS
    # ============================================================

    "capacitor_charge": {
        "keywords": ["capacitor charge", "capacitor capacitance"],
        "formula": "Q = CV",
        "description": "Charge stored in a capacitor.",
    },

    "capacitor_energy": {
        "keywords": ["capacitor energy", "capacitor stored energy"],
        "formula": "E = ½CV²",
        "description": "Energy stored in a capacitor.",
    },

    "capacitor_time_constant": {
        "keywords": ["rc time constant", "capacitor time constant"],
        "formula": "τ = RC",
        "description": "Time constant of an RC circuit.",
    },

    "inductor_voltage": {
        "keywords": ["inductor voltage", "inductor current"],
        "formula": "V = L(di/dt)",
        "description": "Voltage across an inductor during changing current.",
    },

    "inductor_energy": {
        "keywords": ["inductor energy", "inductor stored energy"],
        "formula": "E = ½LI²",
        "description": "Energy stored in an inductor.",
    },

    "rl_time_constant": {
        "keywords": ["rl time constant", "inductor time constant"],
        "formula": "τ = L/R",
        "description": "Time constant of an RL circuit.",
    },


    # ============================================================
    # THERMAL
    # ============================================================

    "thermal_power": {
        "keywords": ["thermal power", "heat power", "heating power"],
        "formula": "P = mcΔT/t",
        "description": "Approximate power required to change the temperature of a mass.",
    },

    "heat_energy": {
        "keywords": ["heat energy", "thermal energy", "specific heat"],
        "formula": "Q = mcΔT",
        "description": "Heat required for a temperature change.",
    },

    "thermal_resistance": {
        "keywords": ["thermal resistance", "thermal resistance junction", "heatsink"],
        "formula": "ΔT = P R_θ",
        "description": "Temperature rise from dissipated power and thermal resistance.",
    },

    "junction_temperature": {
        "keywords": ["junction temperature", "cpu temperature", "semiconductor temperature"],
        "formula": "T_j = T_a + P R_θJA",
        "description": "Approximate semiconductor junction temperature.",
    },

    "conduction_heat": {
        "keywords": ["thermal conduction", "heat conduction", "fourier law"],
        "formula": "Q̇ = kAΔT/L",
        "description": "Heat transfer rate through a conductive material.",
    },


    # ============================================================
    # MATERIALS / STRESS
    # ============================================================

    "normal_stress": {
        "keywords": ["stress", "normal stress", "axial stress"],
        "formula": "σ = F/A",
        "description": "Normal stress from axial force and cross-sectional area.",
    },

    "strain": {
        "keywords": ["strain", "engineering strain", "deformation"],
        "formula": "ε = ΔL/L₀",
        "description": "Engineering strain from change in length.",
    },

    "youngs_modulus": {
        "keywords": ["young modulus", "young's modulus", "elastic modulus"],
        "formula": "E = σ/ε",
        "description": "Young's modulus in the elastic region.",
    },

    "shear_stress": {
        "keywords": ["shear stress", "shear force"],
        "formula": "τ = F/A",
        "description": "Simplified average shear stress.",
    },

    "bending_stress": {
        "keywords": ["bending stress", "beam bending", "flexural stress"],
        "formula": "σ = My/I",
        "description": "Bending stress at a distance y from the neutral axis.",
    },

    "beam_deflection": {
        "keywords": ["beam deflection", "beam bending deflection"],
        "formula": "δ = FL³/(3EI)",
        "description": "Tip deflection of a cantilever beam with a point load at its end.",
    },

    "torsional_stress": {
        "keywords": ["torsional stress", "shaft torsion", "shaft stress"],
        "formula": "τ = Tr/J",
        "description": "Torsional shear stress in a shaft.",
    },

    "torsional_angle": {
        "keywords": ["torsional deflection", "shaft twist", "angle of twist"],
        "formula": "θ = TL/(JG)",
        "description": "Angle of twist of a shaft under torsion.",
    },

    "factor_of_safety": {
        "keywords": ["factor of safety", "safety factor", "fos"],
        "formula": "FoS = strength/working stress",
        "description": "Basic factor-of-safety relationship.",
    },


    # ============================================================
    # GEOMETRY
    # ============================================================

    "circle_area": {
        "keywords": ["circle area", "circular area", "shaft area"],
        "formula": "A = πr²",
        "description": "Area of a circle.",
    },

    "circle_circumference": {
        "keywords": ["circumference", "circle circumference", "wheel circumference"],
        "formula": "C = 2πr",
        "description": "Circumference of a circle.",
    },

    "cylinder_volume": {
        "keywords": ["cylinder volume", "cylindrical volume"],
        "formula": "V = πr²h",
        "description": "Volume of a cylinder.",
    },

    "rectangular_volume": {
        "keywords": ["box volume", "rectangular volume"],
        "formula": "V = LWH",
        "description": "Volume of a rectangular prism.",
    },

    "rectangle_area": {
        "keywords": ["rectangle area", "plate area"],
        "formula": "A = LW",
        "description": "Area of a rectangle.",
    },

    "triangle_area": {
        "keywords": ["triangle area"],
        "formula": "A = ½bh",
        "description": "Area of a triangle.",
    },

    "sphere_volume": {
        "keywords": ["sphere volume", "ball volume"],
        "formula": "V = 4/3 πr³",
        "description": "Volume of a sphere.",
    },

    "sphere_surface_area": {
        "keywords": ["sphere surface area"],
        "formula": "A = 4πr²",
        "description": "Surface area of a sphere.",
    },


    # ============================================================
    # FLUIDS / PNEUMATICS / HYDRAULICS
    # ============================================================

    "fluid_pressure": {
        "keywords": ["fluid pressure", "pressure depth", "hydrostatic pressure"],
        "formula": "P = ρgh",
        "description": "Hydrostatic pressure caused by fluid depth.",
    },

    "pressure_force": {
        "keywords": ["pressure force", "fluid force", "piston force"],
        "formula": "F = PA",
        "description": "Force produced by pressure acting over an area.",
    },

    "hydraulic_force": {
        "keywords": ["hydraulic force", "hydraulic cylinder force", "hydraulic actuator"],
        "formula": "F = PA",
        "description": "Hydraulic actuator force from pressure and piston area.",
    },

    "pneumatic_force": {
        "keywords": ["pneumatic force", "pneumatic cylinder", "air cylinder force"],
        "formula": "F = PA",
        "description": "Pneumatic actuator force from pressure and piston area.",
    },

    "fluid_flow": {
        "keywords": ["flow rate", "fluid flow", "volumetric flow"],
        "formula": "Q = Av",
        "description": "Volumetric flow rate from cross-sectional area and fluid velocity.",
    },

    "hydraulic_power": {
        "keywords": ["hydraulic power", "hydraulic pump power"],
        "formula": "P = pQ",
        "description": "Ideal hydraulic power from pressure and volumetric flow rate.",
    },

    "hydraulic_actuator_speed": {
        "keywords": ["hydraulic cylinder speed", "hydraulic actuator speed"],
        "formula": "v = Q/A",
        "description": "Actuator speed from flow rate and piston area.",
    },


    # ============================================================
    # CONTROL SYSTEMS
    # ============================================================

    "pid": {
        "keywords": ["pid", "pid controller", "pid control"],
        "formula": "u(t) = K_p e(t) + K_i∫e(t)dt + K_d de(t)/dt",
        "description": "Continuous PID controller equation.",
    },

    "proportional_control": {
        "keywords": ["p controller", "proportional control", "kp"],
        "formula": "u = K_p e",
        "description": "Proportional controller output.",
    },

    "integral_control": {
        "keywords": ["i controller", "integral control", "ki"],
        "formula": "u = K_i∫e(t)dt",
        "description": "Integral controller contribution.",
    },

    "derivative_control": {
        "keywords": ["d controller", "derivative control", "kd"],
        "formula": "u = K_d de/dt",
        "description": "Derivative controller contribution.",
    },

    "control_error": {
        "keywords": ["control error", "error signal", "setpoint error"],
        "formula": "e = r − y",
        "description": "Difference between reference/setpoint and measured output.",
    },

    "first_order_time_constant": {
        "keywords": ["first order system", "time constant", "control response"],
        "formula": "τ = 1/ω_c",
        "description": "Common relationship between time constant and characteristic bandwidth.",
    },


    # ============================================================
    # SENSORS
    # ============================================================

    "encoder_resolution": {
        "keywords": ["encoder resolution", "encoder counts", "cpr", "ppr"],
        "formula": "θ = 2πN/N_total",
        "description": "Angular position from encoder counts.",
    },

    "encoder_angular_resolution": {
        "keywords": ["encoder angular resolution", "encoder precision"],
        "formula": "Δθ = 2π/N",
        "description": "Angular resolution per encoder count.",
    },

    "encoder_linear_resolution": {
        "keywords": ["encoder linear resolution", "wheel encoder resolution"],
        "formula": "Δx = 2πr/N",
        "description": "Linear distance represented by one encoder count.",
    },

    "imu_acceleration_magnitude": {
        "keywords": ["imu acceleration magnitude", "accelerometer magnitude"],
        "formula": "a = √(a_x² + a_y² + a_z²)",
        "description": "Magnitude of a 3-axis acceleration vector.",
    },

    "vector_magnitude": {
        "keywords": ["vector magnitude", "3d vector magnitude", "vector length"],
        "formula": "|v| = √(v_x² + v_y² + v_z²)",
        "description": "Magnitude of a three-dimensional vector.",
    },

    "distance_from_time_of_flight": {
        "keywords": ["time of flight", "tof distance", "lidar distance"],
        "formula": "d = ct/2",
        "description": "Distance from round-trip time-of-flight measurement.",
    },


    # ============================================================
    # ROBOTICS / POSITION / TRANSFORMS
    # ============================================================

    "differential_drive_linear_velocity": {
        "keywords": ["differential drive velocity", "diff drive velocity"],
        "formula": "v = (v_R + v_L)/2",
        "description": "Linear velocity of a differential-drive robot.",
    },

    "differential_drive_angular_velocity": {
        "keywords": ["differential drive angular velocity", "diff drive turning"],
        "formula": "ω = (v_R − v_L)/L",
        "description": "Angular velocity of a differential-drive robot.",
    },

    "differential_drive_radius": {
        "keywords": ["differential drive radius", "turning radius diff drive"],
        "formula": "R = v/ω",
        "description": "Instantaneous turning radius of a differential-drive robot.",
    },

    "robot_odometry_x": {
        "keywords": ["robot odometry x", "odometry position x"],
        "formula": "x_new = x + Δs cos(θ)",
        "description": "Approximate x-position update from traveled distance and heading.",
    },

    "robot_odometry_y": {
        "keywords": ["robot odometry y", "odometry position y"],
        "formula": "y_new = y + Δs sin(θ)",
        "description": "Approximate y-position update from traveled distance and heading.",
    },

    "robot_heading": {
        "keywords": ["robot heading", "odometry heading", "robot yaw"],
        "formula": "θ_new = θ + Δθ",
        "description": "Heading update from incremental angular displacement.",
    },


    # ============================================================
    # DC MOTOR ELECTRICAL-MECHANICAL RELATIONSHIPS
    # ============================================================

    "dc_motor_voltage": {
        "keywords": ["dc motor voltage", "motor voltage equation"],
        "formula": "V = IR + K_eω",
        "description": "Simplified steady-state DC motor voltage equation.",
    },

    "dc_motor_torque": {
        "keywords": ["dc motor torque", "motor torque current"],
        "formula": "τ = K_tI",
        "description": "DC motor torque proportional to armature current.",
    },

    "dc_motor_current": {
        "keywords": ["dc motor current", "motor current torque"],
        "formula": "I = τ/K_t",
        "description": "Current required for a specified motor torque.",
    },


    # ============================================================
    # SIGNALS / FILTERING
    # ============================================================

    "sampling_frequency": {
        "keywords": ["sampling frequency", "sample rate", "sampling rate"],
        "formula": "f_s = 1/T_s",
        "description": "Sampling frequency from sampling period.",
    },

    "nyquist_frequency": {
        "keywords": ["nyquist", "nyquist frequency", "sampling theorem"],
        "formula": "f_N = f_s/2",
        "description": "Nyquist frequency for a sampled signal.",
    },

    "low_pass_cutoff": {
        "keywords": ["low pass cutoff", "rc filter cutoff", "filter cutoff"],
        "formula": "f_c = 1/(2πRC)",
        "description": "Cutoff frequency of a first-order RC low-pass filter.",
    },


    # ============================================================
    # BATTERY / POWER SYSTEM LOSSES
    # ============================================================

    "power_loss": {
        "keywords": ["power loss", "electrical loss", "wire loss"],
        "formula": "P_loss = I²R",
        "description": "Resistive electrical power loss.",
    },

    "voltage_drop_wire": {
        "keywords": ["wire voltage drop", "cable voltage drop", "wire loss"],
        "formula": "V_drop = IR",
        "description": "Voltage drop across a resistive conductor.",
    },

    "wire_resistance": {
        "keywords": ["wire resistance", "cable resistance", "conductor resistance"],
        "formula": "R = ρL/A",
        "description": "Resistance of a uniform conductor.",
    },

    "system_efficiency": {
        "keywords": ["system efficiency", "overall efficiency", "robot efficiency"],
        "formula": "η_total = P_out/P_in",
        "description": "Overall system efficiency.",
    },

    "cascaded_efficiency": {
        "keywords": ["combined efficiency", "multiple efficiencies", "system losses"],
        "formula": "η_total = η₁η₂η₃...",
        "description": "Overall efficiency of cascaded components.",
    },


    # ============================================================
    # DIMENSIONLESS / ENGINEERING
    # ============================================================

    "mechanical_advantage": {
        "keywords": ["mechanical advantage", "ma", "lever advantage"],
        "formula": "MA = F_out/F_in",
        "description": "Mechanical advantage of a mechanism.",
    },

    "velocity_ratio": {
        "keywords": ["velocity ratio", "mechanism velocity ratio"],
        "formula": "VR = v_in/v_out",
        "description": "Ratio of input to output velocity.",
    },

    "efficiency_mechanical_advantage": {
        "keywords": ["mechanical efficiency", "mechanism efficiency"],
        "formula": "η = MA/VR",
        "description": "Mechanical efficiency from mechanical advantage and velocity ratio.",
    },

    "percentage_error": {
        "keywords": ["percent error", "percentage error", "measurement error"],
        "formula": "% error = |measured − actual|/|actual| × 100%",
        "description": "Percentage difference between measured and reference values.",
    },

    "percentage_change": {
        "keywords": ["percent change", "percentage change"],
        "formula": "% change = (new − old)/old × 100%",
        "description": "Percentage change between two values.",
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