import random
import pandas as pd
from compiler_frontend import Lexer, TACParser
from feature_extractor import TACFeatureExtractor
from optimizers import TACOptimizer

CATEGORIES = [
    "Constant Folding",
    "Constant Propagation",
    "Algebraic Simplification",
    "Strength Reduction",
    "CSE",
    "Dead Code Elimination"
]

def _inject_snippet(category, idx, live_vars, declared_vars):
    """Generates a snippet of Mini-C code for a specific inefficiency category."""
    code = ""
    if category == "Constant Folding":
        v = f"cf_{idx}_{random.randint(1, 99)}"
        op = random.choice(['+', '-', '*'])
        c1, c2 = random.randint(2, 25), random.randint(2, 25)
        if random.random() < 0.4:
            c3 = random.randint(2, 10)
            code += f"{v} = ({c1} {op} {c2}) + {c3};\n"
        else:
            code += f"{v} = {c1} {op} {c2};\n"
        live_vars.append(v)

    elif category == "Constant Propagation":
        c_var = f"k_{idx}_{random.randint(1, 99)}"
        u_var = f"cp_{idx}_{random.randint(1, 99)}"
        val = random.randint(3, 50)
        code += f"{c_var} = {val};\n"
        code += f"{u_var} = {c_var} + {random.randint(2, 15)};\n"
        live_vars.append(u_var)

    elif category == "Algebraic Simplification":
        base = random.choice(declared_vars)
        v = f"alg_{idx}_{random.randint(1, 99)}"
        pattern = random.choice([f"{base} + 0", f"{base} - 0", f"{base} * 1", f"{base} / 1", f"{base} * 0"])
        code += f"{v} = {pattern};\n"
        live_vars.append(v)

    elif category == "Strength Reduction":
        base = random.choice(declared_vars)
        v = f"sr_{idx}_{random.randint(1, 99)}"
        code += f"{v} = {base} * 2;\n"
        live_vars.append(v)

    elif category == "CSE":
        v1, v2 = random.sample(declared_vars, 2)
        op = random.choice(['+', '-', '*'])
        t1 = f"cseA_{idx}_{random.randint(1, 99)}"
        t2 = f"cseB_{idx}_{random.randint(1, 99)}"
        code += f"{t1} = {v1} {op} {v2};\n"
        code += f"{t2} = {v1} {op} {v2};\n"
        live_vars.extend([t1, t2])

    elif category == "Dead Code Elimination":
        v1 = random.choice(declared_vars)
        dead_v = f"dead_{idx}_{random.randint(1, 99)}"
        code += f"{dead_v} = {v1} + {random.randint(5, 40)};\n"

    return code


def generate_source_program(primary_category, complexity="simple"):
    """
    Synthesizes either a 'simple' or 'complex' Mini-C program with primary
    inefficiencies and controlled background noise (secondary inefficiencies).
    """
    declared_vars = ["x", "y", "z"]
    source = ""
    live_vars = []

    for dv in declared_vars:
        source += f"{dv} = input_{dv};\n"

    if complexity == "simple":
        num_primary = random.randint(2, 3)
        for idx in range(num_primary):
            source += _inject_snippet(primary_category, idx, live_vars, declared_vars)

        # 30% chance to inject 1 secondary competing inefficiency (structural noise)
        if random.random() < 0.30:
            sec_cat = random.choice([c for c in CATEGORIES if c != primary_category])
            source += _inject_snippet(sec_cat, 99, live_vars, declared_vars)

        if random.random() < 0.4:
            source += f"if (x < {random.randint(5, 20)}) {{\n"
            source += _inject_snippet(primary_category, 50, live_vars, declared_vars)
            source += "}\n"

    else:
        source += "i = 0;\nsum = 0;\n"
        live_vars.append("sum")

        num_pre = random.randint(2, 4)
        for idx in range(num_pre):
            source += _inject_snippet(primary_category, idx, live_vars, declared_vars)

        # 45% chance to inject 1-2 secondary inefficiencies for realistic class overlap
        if random.random() < 0.45:
            for s_idx in range(random.randint(1, 2)):
                sec_cat = random.choice([c for c in CATEGORIES if c != primary_category])
                source += _inject_snippet(sec_cat, 80 + s_idx, live_vars, declared_vars)

        loop_bound = random.randint(4, 15)
        source += f"while (i < {loop_bound}) {{\n"
        source += "    sum = sum + i;\n"

        if random.random() < 0.6:
            source += _inject_snippet(primary_category, 60, live_vars, declared_vars)

        if random.random() < 0.5:
            source += f"    if (sum > {random.randint(10, 30)}) {{\n"
            source += "        " + _inject_snippet(primary_category, 70, live_vars, declared_vars).replace("\n", "\n        ").strip() + "\n"
            source += "    }\n"

        if random.random() < 0.35:
            source += "    j = 0;\n"
            source += f"    while (j < {random.randint(2, 6)}) {{\n"
            source += "        sum = sum + j;\n"
            source += "        j = j + 1;\n"
            source += "    }\n"

        source += "    i = i + 1;\n"
        source += "}\n"

    unique_live = list(dict.fromkeys(live_vars))
    if unique_live:
        source += "out = " + " + ".join(unique_live) + ";\n"
    else:
        source += "out = x + y;\n"

    return source


def score_optimization(original_tac, optimized_tac):
    """
    Scores an optimization pass without double-counting shifted lines.
    """
    if original_tac == optimized_tac:
        return 0
    if len(original_tac) != len(optimized_tac):
        return max(1, (len(original_tac) - len(optimized_tac)) // 2)
    return sum(1 for o, opt in zip(original_tac, optimized_tac) if o != opt)


def get_best_strategy(tac_lines, intended_category=None, noise_prob=0.06):
    strategies = [
        ("Constant Folding", TACOptimizer.constant_folding(tac_lines)),
        ("Constant Propagation", TACOptimizer.constant_propagation(tac_lines)),
        ("Algebraic Simplification", TACOptimizer.algebraic_simplification(tac_lines)),
        ("Strength Reduction", TACOptimizer.strength_reduction(tac_lines)),
        ("CSE", TACOptimizer.common_subexpression_elimination(tac_lines)),
        ("Dead Code Elimination", TACOptimizer.dead_code_elimination(tac_lines))
    ]

    random.shuffle(strategies)

    scored = []
    for name, opt_tac in strategies:
        s = score_optimization(tac_lines, opt_tac)
        if s > 0:
            scored.append((name, s))

    if not scored:
        return intended_category if intended_category else "Constant Folding"

    scored.sort(key=lambda x: x[1], reverse=True)

    if len(scored) > 1 and random.random() < noise_prob:
        return scored[1][0]

    return scored[0][0]


def build_dataset(num_samples=750, seed=42):
    random.seed(seed)
    print(f"Initializing Compilation Pipeline for {num_samples} Samples (Simple + Complex)...")
    dataset = []
    samples_per_cat = num_samples // len(CATEGORIES)

    for cat in CATEGORIES:
        for i in range(samples_per_cat):
            complexity = "simple" if i < (samples_per_cat // 2) else "complex"
            source = generate_source_program(cat, complexity=complexity)

            lexer = Lexer(source)
            parser = TACParser(lexer)
            tac_lines = parser.parse()

            extractor = TACFeatureExtractor(tac_lines)
            features = extractor.extract_features()

            if complexity == "complex" and random.random() < 0.10:
                features["num_tmp_vars"] = max(1, features["num_tmp_vars"] + random.choice([-1, 1]))
                features["num_statements"] = max(5, features["num_statements"] + random.choice([-2, -1, 1, 2]))
                features["const_ratio"] = round(features["num_constants"] / features["num_statements"], 3)

            best_strategy = get_best_strategy(tac_lines, intended_category=cat, noise_prob=0.06)

            row = features.copy()
            row["best_strategy"] = best_strategy
            dataset.append(row)

    df = pd.DataFrame(dataset)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv("dataset.csv", index=False)

    feature_cols = [c for c in df.columns if c != "best_strategy"]
    unique_vectors = len(df[feature_cols].drop_duplicates())

    print(f"✔ Dataset generated successfully ({len(df)} records). Saved as 'dataset.csv'.")
    print(f"✔ Unique 8D Feature Vectors: {unique_vectors} / {len(df)} ({unique_vectors/len(df)*100:.1f}% diversity)")
    print("\nClass Distribution:")
    print(df["best_strategy"].value_counts())
    return df


if __name__ == "__main__":
    build_dataset(750)