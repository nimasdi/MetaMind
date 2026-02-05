from pathlib import Path
import sys
import numpy as np
import skfuzzy as fuzz
import time


try:
    from ...core.base_method import BaseMethod
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.base_method import BaseMethod
    

class FuzzyController(BaseMethod):
    PARAM_SPECS = {
        'n_membership_functions': {
            'type': int,
            'options': [3, 5, 7],
            'default': 3
        },
        'membership_type': {
            'type': str,
            'options': ["triangular", "gaussian", "trapezoid"],
            'default': "triangular"
        },
        'defuzzification': {
            'type': str,
            'options': ["centroid", "bisector", "mom", "som"],
            'default': "centroid"
        },
        'rule_generation': {
            'type': str,
            'options': ["wang_mendel", "manual"],
            'default': "wang_mendel"
        }
    }

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.input_mfs = {}
        self.output_mfs = {}
        self.fuzzy_rules = []
        self.input_universe = None
        self.output_universe = None

    def create_membership_functions(self, x, n, mtype: str) :
        if mtype == "triangular":
            return self.create_triangular_mfs(x, n)
        elif mtype == "gaussian":
            return self.create_gaussian_mfs(x, n)
        elif mtype == "trapezoid":
            return self.create_trapezoidal_mfs(x, n)
        else:
            raise ValueError(f"Unknown membership type: {mtype}")

    def create_triangular_mfs(self, x, n):
        mfs = []
        step = (x[-1] - x[0]) / (n - 1)
        for i in range(n):
            if i == 0:
                a = x[0]
                b = x[0]
                c = x[0] + step
            elif i == n - 1:
                a = x[-1] - step
                b = x[-1]
                c = x[-1]
            else:
                a = x[0] + (i - 1) * step
                b = x[0] + i * step
                c = x[0] + (i + 1) * step
            mfs.append((f"MF_{i}", fuzz.trimf(x, [a, b, c])))
        return mfs

    def create_gaussian_mfs(self, x, n):
        mfs = []
        step = (x[-1] - x[0]) / (n - 1) if n > 1 else (x[-1] - x[0])
        for i in range(n):
            center = x[0] + i * step
            sigma = step / 3 if step > 0 else 1.0
            mf = fuzz.gaussmf(x, center, sigma)
            mfs.append((f"MF_{i}", mf))
        return mfs

    def create_trapezoidal_mfs(self, x, n):
        mfs = []
        step = (x[-1] - x[0]) / (n - 1) if n > 1 else (x[-1] - x[0])
        margin = step / 4
        for i in range(n):
            center = x[0] + i * step
            if i == 0:
                a = x[0]
                b = x[0]
                c = center + margin
                d = center + step / 2
            elif i == n - 1:
                a = center - step / 2
                b = center - margin
                c = x[-1]
                d = x[-1]
            else:
                a = center - step / 2
                b = center - margin
                c = center + margin
                d = center + step / 2
            mfs.append((f"MF_{i}", fuzz.trapmf(x, [a, b, c, d])))
        return mfs

    def generate_rules_wang_mendel(self, input_mfs, output_mfs):
        rules = []
        for i, (in_label, _) in enumerate(input_mfs):
            for j, (out_label, _) in enumerate(output_mfs):
                rule = {
                    'antecedent': f"{in_label}",
                    'consequent': f"{out_label}",
                    'weight': 1.0
                }
                rules.append(rule)
        return rules

    def generate_rules_manual(self, rules):
        if not rules:
            raise ValueError("Manual rule generation requires explicit rules list.")
        return rules

    def fit(self, problem_data, callback=None, **kwargs):
        self.start_time = time.time()
        self.log("Starting Fuzzy Controller training...")

        defaults = self.get_default_parameters()
        n_mfs = self.parameters.get('n_membership_functions', defaults['n_membership_functions'])
        mtype = self.parameters.get('membership_type', defaults['membership_type'])
        defuzz = self.parameters.get('defuzzification', defaults['defuzzification'])
        rule_gen = self.parameters.get('rule_generation', defaults['rule_generation'])

        input_range = problem_data.get('input_range')
        output_range = problem_data.get('output_range')
        if not input_range or not output_range:
            raise ValueError("input_range and output_range must be provided.")

        x = np.linspace(input_range[0], input_range[1], 100)
        y = np.linspace(output_range[0], output_range[1], 100)

        input_mfs = self.create_membership_functions(x, n_mfs, mtype)
        output_mfs = self.create_membership_functions(y, n_mfs, mtype)

        if rule_gen == "wang_mendel":
            input_data = problem_data.get('input_data')
            output_data = problem_data.get('output_data')
            if input_data is None or output_data is None:
                self.log("Warning: Wang-Mendel rule generation requires input_data and output_data.")
                self.fuzzy_rules = self.generate_rules_wang_mendel(input_mfs, output_mfs)
            else:
                self.fuzzy_rules = self.generate_rules_wang_mendel(input_mfs, output_mfs)
        elif rule_gen == "manual":
            manual_rules = problem_data.get('manual_rules', [])
            self.fuzzy_rules = self.generate_rules_manual(manual_rules)
        else:
            raise ValueError(f"Unknown rule generation method: {rule_gen}")

        self.input_mfs = {name: mf for name, mf in input_mfs}
        self.output_mfs = {name: mf for name, mf in output_mfs}
        self.input_universe = x
        self.output_universe = y
        self.defuzzification_method = defuzz

        self.results = {
            'n_membership_functions': n_mfs,
            'membership_type': mtype,
            'defuzzification': defuzz,
            'rule_generation': rule_gen,
            'n_rules': len(self.fuzzy_rules),
            'input_range': input_range,
            'output_range': output_range,
            'training_time': 0.0
        }

        self.log("Fuzzy Controller trained successfully.")
        self.end_time = time.time()
        self.results['training_time'] = self.end_time - self.start_time
        self.convergence_history.append(len(self.fuzzy_rules))

        if callback:
            callback({
                'method': 'FuzzyController',
                'status': 'complete',
                'n_rules_generated': len(self.fuzzy_rules),
            })

    def predict(self, input_value: float) -> float:
        if not hasattr(self, 'input_mfs') or not self.input_mfs:
            raise RuntimeError("Fuzzy controller not trained yet. Call fit() first.")

        input_degrees = {}
        for name, mf in self.input_mfs.items():
            idx = np.abs(self.input_universe - input_value).argmin()
            input_degrees[name] = mf[idx]

        aggregated = np.zeros_like(self.output_universe)
        for rule in self.fuzzy_rules:
            antecedent_name = rule['antecedent']
            if antecedent_name in input_degrees:
                activation = input_degrees[antecedent_name] * rule['weight']
                
                consequent_name = rule['consequent']
                if consequent_name in self.output_mfs:
                    consequent_mf = self.output_mfs[consequent_name]
                    clipped = np.fmin(activation, consequent_mf)
                    aggregated = np.fmax(aggregated, clipped)

        output = self.defuzzify(aggregated)
        return output

    def defuzzify(self, aggregated_mf: np.ndarray) -> float:
        if self.defuzzification_method == "centroid":
            return fuzz.defuzz(self.output_universe, aggregated_mf, 'centroid')
        elif self.defuzzification_method == "bisector":
            return fuzz.defuzz(self.output_universe, aggregated_mf, 'bisector')
        elif self.defuzzification_method == "mom":
            return fuzz.defuzz(self.output_universe, aggregated_mf, 'mom')
        elif self.defuzzification_method == "som":
            return fuzz.defuzz(self.output_universe, aggregated_mf, 'som')
        else: # default is centroid
            return fuzz.defuzz(self.output_universe, aggregated_mf, 'centroid')