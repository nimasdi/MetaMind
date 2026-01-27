from controller import FuzzyController


problem_data = {
    'input_range': (0, 100),           # Temperature in °C (0 to 100)
    'output_range': (0, 100),          # Fan speed (0 to 100%)
    'manual_rules': [
        {'antecedent': 'MF_0', 'consequent': 'MF_0', 'weight': 1.0},  # Low → Low
        {'antecedent': 'MF_1', 'consequent': 'MF_1', 'weight': 1.0},  # Medium → Medium
        {'antecedent': 'MF_2', 'consequent': 'MF_2', 'weight': 1.0},  # High → High
    ]
}


controller = FuzzyController(
    n_membership_functions=3,
    membership_type="triangular",
    defuzzification="centroid",
    rule_generation="manual"
)

controller.fit(problem_data)


test_inputs = [20, 50, 80]  # Low, Medium, High temperature
print("\n📊 Predictions:")
for temp in test_inputs:
    fan_speed = controller.predict(temp)
    print(f"Input Temp: {temp}°C → Fan Speed: {fan_speed:.2f}%")


x = controller.input_universe
y = controller.output_universe

results = controller.get_results()
print(f"  Membership Functions: {results['n_membership_functions']}")
print(f"  Membership Type: {results['membership_type']}")
print(f"  Defuzzification: {results['defuzzification']}")
print(f"  Rule Generation: {results['rule_generation']}")
print(f"  Rules: {results['n_rules']}")

print("\n📜 Execution Logs:")
logs = controller.get_logs()
for log in logs:
    print(f"  {log['message']} (t={log['elapsed']:.2f}s)")

print(f"\n✅ Training completed in {controller.end_time - controller.start_time:.3f} seconds.")
