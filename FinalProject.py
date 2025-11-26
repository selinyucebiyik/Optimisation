import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Column temperaures
T_abs = 313.0  # K, Absorption Temperature
T_des = 393.0  # K, Desorption Temperatura

# Constraints
RED_MAX = 1.0  # Maximum Relative Energy Difference
N_MAX_GROUPS = 15  # Total number of groups

# Solubility Parameter (Reference value for CO2 (aprox) in MPa^0.5
DELTA_CO2 = 13.5

# Objective Scaling Ranges
SCALING_RANGES = {
    'RED': {'min': 0.0, 'max': 1.0},
    'Cp_spec': {'min': 1.5, 'max': 4.0},   # J/g.K
    'Density': {'min': 700.0, 'max': 1200.0},  # kg/m^3
}

# Group Contribution Data
# TO BE UPDATED
GROUP_DATA = {
    # [Molecular Weight (g/mol), Molar Volume (cm3/mol), Internal Energy (J/mol), Fusion Temperature (K)...
    # Boiling Temperature (K), Liquid Density (kg/m3), Heat Capacity (J/molK), Valency]
    # Heat Capacity and Valency from Rayer et al.
    'CH3':   [15.035,  33.5,  1200,   5.0,  20.0,  80.0,   45.0, 1],
    'CH2':   [14.027,  16.0,  1000,   3.0,  25.0,  90.0,   30.0, 2],
    'NH2':   [16.023,  18.0,  3500,  -5.0,  30.0,  110.0,  55.0, 1],
    'OH':    [17.008,  15.0,  4000,  10.0,  45.0,  150.0,  60.0, 1],
}

# Groups Set
GROUPS = list(GROUP_DATA.keys())

def create_camd_model(weights=None):
    model = pyo.ConcreteModel()

    # Using equal weights
    if weights is None:
        weights = {'RED': 1/3, 'Cp_spec': 1/3, 'Density': 1/3}

    # Sets and Parameters
    model.G = pyo.Set(initialize=GROUPS, doc='Grupos Funcionales')

    # Parámetros para la Contribución de Grupos
    model.MW_GC = pyo.Param(model.G, initialize={k: v[0] for k, v in GROUP_DATA.items()}, doc='Contribución al Peso Molecular [g/mol]')
    model.VM_GC = pyo.Param(model.G, initialize={k: v[1] for k, v in GROUP_DATA.items()}, doc='Contribución al Volumen Molar [cm^3/mol]')
    model.U_GC = pyo.Param(model.G, initialize={k: v[2] for k, v in GROUP_DATA.items()}, doc='Contribución a la Energía Interna [J/mol]')
    model.TM_GC = pyo.Param(model.G, initialize={k: v[3] for k, v in GROUP_DATA.items()}, doc='Contribución a la Temp. de Fusión [K]')
    model.TBP_GC = pyo.Param(model.G, initialize={k: v[4] for k, v in GROUP_DATA.items()}, doc='Contribución a la Temp. de Ebullición [K]')
    model.RHO_GC = pyo.Param(model.G, initialize={k: v[5] for k, v in GROUP_DATA.items()}, doc='Contribución a la Densidad [kg/m^3]')
    model.CP_GC = pyo.Param(model.G, initialize={k: v[6] for k, v in GROUP_DATA.items()}, doc='Contribución a Capacidad Calorífica Molar [J/mol.K]')
    model.VALENCY_GC = pyo.Param(model.G, initialize={k: v[7] for k, v in GROUP_DATA.items()}, doc='Contribución de Valencias')

    # Pesos para la Función Objetivo
    model.w_RED = pyo.Param(initialize=weights['RED'], doc='weight of RED')
    model.w_Cp_spec = pyo.Param(initialize=weights['Cp_spec'], doc='weight of Cp specific')
    model.w_Density = pyo.Param(initialize=weights['Density'], doc='weight of Density')


    # --- VARIABLES ---
    # N_k: Número de grupos k. Debe ser un entero no negativo.
    model.N = pyo.Var(model.G, domain=pyo.NonNegativeIntegers, doc='Número de grupos')

    # --- EXPRESIONES DE PROPIEDADES (EXPRESSIONS FOR PROPERTIES) ---

    # Expresión para cálculo general de propiedades aditivas
    def prop_sum(model, prop_param):
        """Calcula una propiedad aditiva: P = sum(N_k * C_k)"""
        return sum(model.N[g] * prop_param[g] for g in model.G)

    # Propiedades Aditivas Lineales (Lineal Additive Properties)
    model.MW = pyo.Expression(expr=prop_sum(model, model.MW_GC), doc='Peso Molecular [g/mol]')
    model.Vm = pyo.Expression(expr=prop_sum(model, model.VM_GC), doc='Volumen Molar [cm^3/mol]')
    model.U = pyo.Expression(expr=prop_sum(model, model.U_GC), doc='Energía Interna de Vaporización [J/mol]')
    model.Tm = pyo.Expression(expr=prop_sum(model, model.TM_GC), doc='Temperatura de Fusión [K]')
    model.Tbp = pyo.Expression(expr=prop_sum(model, model.TBP_GC), doc='Temperatura de Ebullición [K]')
    model.Density = pyo.Expression(expr=prop_sum(model, model.RHO_GC), doc='Densidad Líquida [kg/m^3]') # Simplificado como aditivo
    model.Cp_molar = pyo.Expression(expr=prop_sum(model, model.CP_GC), doc='Capacidad Calorífica Molar [J/mol.K]')
    model.N_total = pyo.Expression(expr=sum(model.N[g] for g in model.G), doc='Número total de grupos')

    # Propiedades NO-LINEALES (MINLP Components)

    # NL1: Parámetro de Solubilidad (Delta) y RED (Hukkerikar et al. [1])
    # Delta^2 (Solubility Parameter Squared) = U / Vm (Division makes it MINLP)
    # Note: Vm must be strictly positive (enforced by C1_MinGroups and positive GC values)
    model.Delta_sq = pyo.Expression(expr=model.U / (model.Vm * 1e-6), doc='Delta^2 [MPa]') # Vm from cm^3/mol to m^3/mol
    model.Delta = pyo.Expression(expr=pyo.sqrt(model.Delta_sq), doc='Parámetro de Solubilidad [MPa^0.5]')
    # Relative Energy Difference (RED) = |Delta_solvent - Delta_CO2| / Delta_CO2
    # We use a non-linear formulation for the absolute value: RED^2 = ((Delta - DELTA_CO2) / DELTA_CO2)^2
    model.RED = pyo.Expression(expr=pyo.sqrt((model.Delta - DELTA_CO2)**2) / DELTA_CO2, doc='Relative Energy Difference')

    # NL2: Capacidad Calorífica Específica (Cp_spec) (Rayer et al. [2])
    # Cp_spec (J/g.K) = Cp_molar / MW (Division makes it MINLP)
    model.Cp_spec = pyo.Expression(expr=model.Cp_molar / model.MW, doc='Capacidad Calorífica Específica [J/g.K]')


    # --- CONSTRAINTS ---

    # C1: Restricción del número total de grupos
    model.C1_TotalGroups = pyo.Constraint(expr=model.N_total <= N_MAX_GROUPS)
    model.C1_MinGroups = pyo.Constraint(expr=model.N_total >= 3)

    # C2: Factibilidad Estructural (Estructura Aclíclica Simple)
    # Suma de grupos terminales (CH3, NH2, OH) debe ser al menos 2
    model.C2_Capping = pyo.Constraint(expr=model.N['CH3'] + model.N['NH2'] + model.N['OH'] >= 2)

    # C3: Amine groups must be present for a chemisorption solvent
    model.C3_AminePresent = pyo.Constraint(expr=model.N['NH2'] >= 1)

    # C4: Restricciones de Propiedades (Cribado)
    # C4.a: RED constraint (Condition a)
    model.C4a_RED = pyo.Constraint(expr=model.RED <= RED_MAX)

    # C4.e: Melting Temperature constraint (Condition e)
    model.C4e_Tm = pyo.Constraint(expr=model.Tm <= T_abs)

    # C4.f: Boiling Temperature constraint (Condition f)
    model.C4f_Tbp = pyo.Constraint(expr=model.Tbp >= T_des)


    # --- Scaled Weighted Sum ---
    # Calcular los objetivos escalados (normalizados) Z_i
    # RED (Minimizar)
    Z_RED = (model.RED - SCALING_RANGES['RED']['min']) / \
            (SCALING_RANGES['RED']['max'] - SCALING_RANGES['RED']['min'])

    # Cp_spec (To minimise)
    Z_Cp_spec = (model.Cp_spec - SCALING_RANGES['Cp_spec']['min']) / \
           (SCALING_RANGES['Cp_spec']['max'] - SCALING_RANGES['Cp_spec']['min'])

    # Density (To maximise -> Minimise (Max - P))
    Z_Density = (SCALING_RANGES['Density']['max'] - model.Density) / \
                (SCALING_RANGES['Density']['max'] - SCALING_RANGES['Density']['min'])

    # Objetivo Combinado: Minimizar Z
    def obj_rule(model):
        return model.w_RED * Z_RED + model.w_Cp_spec * Z_Cp_spec + model.w_Density * Z_Density

    model.Objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize, doc='Objetivo Escaldado de Suma Ponderada')

    return model

def solve_and_report(model):
    """
    Resuelve el modelo y muestra los resultados para el solvente óptimo.
    """
    # Usamos Bonmin, un solucionador MINLP. Requiere instalación (e.g., via conda install -c conda-forge cointr-bonmin)
    # Si Bonmin no está disponible, el usuario puede usar Couenne, o GAMS con un solver MINLP.
    opt = SolverFactory('bonmin')
    # opt = SolverFactory('couenne') # Alternativa MINLP
    # opt = SolverFactory('glpk') # Solo para prueba lineal (NO recomendado para el modelo final)

    print(f"\n--- Resolviendo Modelo MINLP con Pesos: RED={model.w_RED.value:.3f}, Cp={model.w_Cp_spec.value:.3f}, Density={model.w_Density.value:.3f} ---")

    try:
        # Nota: El solver MINLP puede ser lento o requerir inicialización
        results = opt.solve(model, tee=True)
    except Exception as e:
        print(f"Error del Solucionador (Solver Error). Asegúrate de que Bonmin o Couenne esté instalado: {e}")
        # Intentar con GLPK si falla Bonmin (solo para demostración, la solución será pobre)
        # try:
        #     opt_glpk = SolverFactory('glpk')
        #     results = opt_glpk.solve(model, tee=True)
        # except:
        #     return
        return

    # Comprobar si se encontró una solución
    if (results.solver.status == pyo.SolverStatus.ok) and \
       (results.solver.termination_condition == pyo.TerminationCondition.optimal or
        results.solver.termination_condition == pyo.TerminationCondition.feasible):

        print("\n--- CANDIDATO A SOLVENTE ÓPTIMO ENCONTRADO ---")

        # 1. Resumen de la Solución Óptima
        total_groups = sum(pyo.value(model.N[g]) for g in model.G)
        print(f"Valor del Objetivo (Z): {pyo.value(model.Objective):.4f}")
        print(f"Total de Grupos: {total_groups}")
        print("\nConteo de Grupos:")
        for g in model.G:
            if pyo.value(model.N[g]) > 0:
                print(f"  {g}: {pyo.value(model.N[g]):.0f}")

        # 2. Propiedades Estimadas
        print("\nPropiedades Estimadas:")
        print(f"  Peso Molecular (g/mol): {pyo.value(model.MW):.2f}")
        print(f"  Parámetro de Solubilidad (Delta, MPa^0.5): {pyo.value(model.Delta):.2f}")
        print(f"  RED: {pyo.value(model.RED):.4f} (Restricción: <= {RED_MAX:.1f})")
        print(f"  Cp Específico (J/g.K): {pyo.value(model.Cp_spec):.3f}")
        print(f"  Densidad (kg/m^3): {pyo.value(model.Density):.1f}")
        print(f"  Tm (K): {pyo.value(model.Tm):.1f} (Restricción: <= {T_abs:.1f} K)")
        print(f"  Tbp (K): {pyo.value(model.Tbp):.1f} (Restricción: >= {T_des:.1f} K)")

    else:
        print("\n--- SOLUCIÓN NO ENCONTRADA ---")
        print(f"Estado del Solucionador: {results.solver.status}")
        print(f"Condición de Terminación: {results.solver.termination_condition}")
        print("El problema puede ser infactible o requiere un solucionador/reformulación diferente.")


if __name__ == '__main__':
    # Ejecución inicial para Tarea 2 (Pesos Iguales)
    initial_weights = {'RED': 1/3, 'Cp_spec': 1/3, 'Density': 1/3}
    model = create_camd_model(weights=initial_weights)
    solve_and_report(model)