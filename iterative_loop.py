import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from feature_extractor import TACFeatureExtractor
from optimizers import TACOptimizer, TACInterpreter


UNSEEN_BENCHMARKS = [
    {
        "name": "B1_Matrix_Index_Calc",
        "source": (
            "row = input_x;\n"
            "col = input_y;\n"
            "stride = 16 * 4;\n"
            "base_off = row * 2;\n"
            "idx1 = base_off + col;\n"
            "idx2 = idx1 + 0;\n"
            "unused_pad = col * 1;\n"
            "out = idx2 + stride;\n"
        )
    },
    {
        "name": "B2_Loop_Accumulator",
        "source": (
            "n = 6;\n"
            "i = 0;\n"
            "acc = 0;\n"
            "scale = 5 * 4;\n"
            "while (i < n) {\n"
            "    step = i * 2;\n"
            "    acc = acc + step;\n"
            "    i = i + 1;\n"
            "}\n"
            "dead_check = acc + 99;\n"
            "out = acc + scale;\n"
        )
    },
    {
        "name": "B3_Redundant_Polynomial",
        "source": (
            "x = input_x;\n"
            "y = input_y;\n"
            "term1 = x * y;\n"
            "term2 = x * y;\n"
            "c_val = 25;\n"
            "shifted = c_val + 10;\n"
            "clean = shifted * 1;\n"
            "out = term1 + term2 + clean;\n"
        )
    },
    {
        "name": "B4_Branching_Thresholds",
        "source": (
            "x = input_x;\n"
            "y = input_y;\n"
            "limit = 8 * 5;\n"
            "res = 0;\n"
            "if (x < limit) {\n"
            "    tmp_a = y * 2;\n"
            "    res = tmp_a + 0;\n"
            "} else {\n"
            "    tmp_b = x + y;\n"
            "    res = tmp_b * 1;\n"
            "}\n"
            "out = res;\n"
        )
    },
    {
        "name": "B5_Nested_Signal_Filter",
        "source": (
            "x = input_x;\n"
            "i = 0;\n"
            "total = 0;\n"
            "gain = 3 * 3;\n"
            "while (i < 4) {\n"
            "    j = 0;\n"
            "    while (j < 3) {\n"
            "        w = j * 2;\n"
            "        total = total + w;\n"
            "        j = j + 1;\n"
            "    }\n"
            "    i = i + 1;\n"
            "}\n"
            "dead_ tap = x + 42;\n".replace("dead_ tap", "dead_tap") +
            "out = total + gain;\n"
        )
    },
    {
        "name": "B6_Chained_Pipeline",
        "source": (
            "a = input_x;\n"
            "b = input_y;\n"
            "k = 12;\n"
            "p1 = k + 8;\n"
            "p2 = a + b;\n"
            "p3 = a + b;\n"
            "p4 = p2 * 2;\n"
            "p5 = p4 - 0;\n"
            "garbage = p3 + 100;\n"
            "out = p1 + p3 + p5;\n"
        )
    }
]


class RuleBasedSelector:
    """
    Deterministic Multi-Rule Heuristic System using normalized structural densities.
    """
    STRATEGY_MAP = {
        "Constant Folding": TACOptimizer.constant_folding,
        "Constant Propagation": TACOptimizer.constant_propagation,
        "Algebraic Simplification": TACOptimizer.algebraic_simplification,
        "Strength Reduction": TACOptimizer.strength_reduction,
        "CSE": TACOptimizer.common_subexpression_elimination,
        "Dead Code Elimination": TACOptimizer.dead_code_elimination
    }

    FIXED_O2_ORDER = [
        "Constant Folding",
        "Constant Propagation",
        "CSE",
        "Algebraic Simplification",
        "Strength Reduction",
        "Dead Code Elimination"
    ]

    @staticmethod
    def score_features(features):
        n_stmt = max(1, features["num_statements"])
        dead_density = features["num_dead_stmts"] / n_stmt
        repeat_density = features["num_repeated_expr"] / n_stmt
        alg_density = features["num_algebraic_ops"] / n_stmt
        const_ratio = features["const_ratio"]

        scores = {
            "Constant Folding": 0.0,
            "Constant Propagation": 0.0,
            "Algebraic Simplification": 0.0,
            "Strength Reduction": 0.0,
            "CSE": 0.0,
            "Dead Code Elimination": 0.0
        }

        # Rule 1: Constant Folding vs Constant Propagation
        if features["num_constants"] >= 4 or const_ratio >= 0.22:
            scores["Constant Folding"] = const_ratio * 110.0 + features["num_constants"] * 6.5
            scores["Constant Propagation"] = const_ratio * 75.0 + features["num_constants"] * 4.0
        elif features["num_constants"] > 0:
            scores["Constant Propagation"] = const_ratio * 95.0 + features["num_constants"] * 7.0
            scores["Constant Folding"] = const_ratio * 80.0 + features["num_constants"] * 5.0

        # Rule 2: Common Subexpression Elimination
        if features["num_repeated_expr"] > 0:
            scores["CSE"] = repeat_density * 135.0 + features["num_repeated_expr"] * 13.5

        # Rule 3: Algebraic Simplification
        if features["num_algebraic_ops"] > 0:
            scores["Algebraic Simplification"] = alg_density * 125.0 + features["num_algebraic_ops"] * 12.0

        # Rule 4: Dead Code Elimination
        excess_dead = max(0, features["num_dead_stmts"] - (features["num_tmp_vars"] // 3))
        if excess_dead > 0:
            scores["Dead Code Elimination"] = dead_density * 95.0 + excess_dead * 12.5

        # Rule 5: Strength Reduction (unexplained expression temporaries in scaling/loop contexts)
        unexplained_temps = max(
            0,
            features["num_tmp_vars"]
            - features["num_algebraic_ops"]
            - features["num_repeated_expr"]
            - excess_dead
            - (features["num_constants"] // 2)
        )
        if unexplained_temps > 0:
            scores["Strength Reduction"] = (unexplained_temps / n_stmt) * 85.0 + unexplained_temps * 8.5

        return scores

    @classmethod
    def predict(cls, features):
        scores = cls.score_features(features)
        return max(scores, key=scores.get)

    @classmethod
    def rank_strategies(cls, features):
        scores = cls.score_features(features)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class MLOrchestrator:
    def __init__(self, dataset_path="dataset.csv", model_path="rf_optimizer_model.pkl", silent=False):
        self.strategy_map = RuleBasedSelector.STRATEGY_MAP
        self.model = None
        self.classes_ = None
        self.feature_names = None

        if os.path.exists(model_path):
            saved = joblib.load(model_path)
            self.model = saved["model"]
            self.classes_ = self.model.classes_
            self.feature_names = saved["feature_names"]
            if not silent:
                print("--- Loaded Pre-Trained Random Forest Model (rf_optimizer_model.pkl) ---\n")
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            self._train_model(dataset_path, silent=silent)

    def _train_model(self, path, silent=False):
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' missing! Run dataset_generator.py first.")
        df = pd.read_csv(path)
        X = df.drop(columns=["best_strategy"])
        y = df["best_strategy"]
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.feature_names = X.columns
        if not silent:
            print("--- ML Model Trained and Orchestrator Ready ---\n")

    def optimize_program(self, tac_lines, verbose=True, return_log=False):
        current_tac = list(tac_lines)
        iteration = 1
        history_log = []

        if verbose:
            print("==================================================================")
            print("           ML SMART FALLBACK ITERATIVE OPTIMIZATION LOOP          ")
            print("==================================================================")
            print("Original TAC:")
            print("\n".join(f"  {line}" for line in current_tac))
            print(f"\nOriginal Instruction Count: {len(current_tac)}")
            print("-" * 66)

        while iteration <= 25:
            extractor = TACFeatureExtractor(current_tac)
            features = extractor.extract_features()
            X_input = pd.DataFrame([features], columns=self.feature_names)

            probabilities = self.model.predict_proba(X_input)[0]
            rule_scores = RuleBasedSelector.score_features(features)

            strategy_ranking = sorted(
                zip(self.classes_, probabilities),
                key=lambda x: x[1],
                reverse=True
            )

            if verbose:
                print(f"[Iteration {iteration}] Extracted Features: {features}")
                print("  Candidate Strategy Evaluation (All 6 Passes):")
                for strat_name, prob in strategy_ranking:
                    r_score = rule_scores.get(strat_name, 0.0)
                    print(f"    * {strat_name:<26} | ML Confidence: {prob*100:5.1f}% | Rule Score: {r_score:5.1f}")

            changed = False
            for strategy, confidence in strategy_ranking:
                if verbose:
                    print(f"  -> Attempting Selected Pass: {strategy} (Confidence: {confidence*100:.1f}%)")

                opt_func = self.strategy_map[strategy]
                new_tac = opt_func(current_tac)

                if current_tac != new_tac:
                    old_len = len(current_tac)
                    new_len = len(new_tac)
                    lines_removed = old_len - new_len
                    lines_modified = (
                        lines_removed if lines_removed > 0
                        else sum(1 for o, n in zip(current_tac, new_tac) if o != n)
                    )

                    history_log.append({
                        "iteration": iteration,
                        "strategy": strategy,
                        "confidence": float(confidence),
                        "lines_removed": lines_removed,
                        "gain": lines_modified,
                        "accepted": True
                    })

                    if verbose:
                        print(f"  [SUCCESS] Pass applied ({lines_modified} instruction(s) optimized; TAC lines: {old_len} -> {new_len}).")
                        print("-" * 66)

                    current_tac = new_tac
                    changed = True
                    break
                else:
                    history_log.append({
                        "iteration": iteration,
                        "strategy": strategy,
                        "confidence": float(confidence),
                        "lines_removed": 0,
                        "gain": 0,
                        "accepted": False
                    })
                    if verbose:
                        print("  [NO-OP] Zero change detected. Smart Fallback to next ranked strategy...")

            if not changed:
                if verbose:
                    print(">>> CONVERGENCE REACHED: Fixed point verified across all 6 passes.")
                break

            iteration += 1

        is_equiv, matched, total_sets = TACInterpreter.verify_equivalence(tac_lines, current_tac)

        if verbose:
            print("\n==================================================================")
            print("                        FINAL OPTIMIZED TAC                       ")
            print("==================================================================")
            print("\n".join(f"  {line}" for line in current_tac))
            total_reduction = len(tac_lines) - len(current_tac)
            pct = (total_reduction / max(1, len(tac_lines))) * 100.0
            verdict = f"PASS ({matched}/{total_sets} Input Sets Matched)" if is_equiv else f"FAIL ({matched}/{total_sets})"
            print(f"\nInstruction Count : {len(tac_lines)} -> {len(current_tac)} ({total_reduction} lines removed, {pct:.1f}% reduction)")
            print(f"Semantic Verdict  : {verdict}")
            print("==================================================================")

        if return_log:
            return current_tac, history_log
        return current_tac

    @classmethod
    def optimize_with_rule_based(cls, tac_lines):
        """Runs the iterative loop using RuleBasedSelector rankings."""
        current_tac = list(tac_lines)
        total_attempts = 0
        wasted_attempts = 0
        accepted_passes = 0

        for _ in range(25):
            features = TACFeatureExtractor(current_tac).extract_features()
            ranking = RuleBasedSelector.rank_strategies(features)
            changed = False
            for strategy, _ in ranking:
                total_attempts += 1
                new_tac = RuleBasedSelector.STRATEGY_MAP[strategy](current_tac)
                if new_tac != current_tac:
                    current_tac = new_tac
                    accepted_passes += 1
                    changed = True
                    break
                else:
                    wasted_attempts += 1
            if not changed:
                break
        return current_tac, accepted_passes, total_attempts, wasted_attempts

    @classmethod
    def optimize_with_fixed_order(cls, tac_lines):
        """Simulates a traditional -O2 compiler applying all 6 passes in a fixed sweep until convergence."""
        current_tac = list(tac_lines)
        total_attempts = 0
        wasted_attempts = 0
        accepted_passes = 0

        for _ in range(25):
            sweep_changed = False
            for strategy in RuleBasedSelector.FIXED_O2_ORDER:
                total_attempts += 1
                new_tac = RuleBasedSelector.STRATEGY_MAP[strategy](current_tac)
                if new_tac != current_tac:
                    current_tac = new_tac
                    accepted_passes += 1
                    sweep_changed = True
                else:
                    wasted_attempts += 1
            if not sweep_changed:
                break
        return current_tac, accepted_passes, total_attempts, wasted_attempts


if __name__ == "__main__":
    sample_program = [
        "k1 = 15",
        "t1 = 10 * 20",
        "t2 = t1 + 0",
        "t3 = k1 + 5",
        "unused = t2 * 2",
        "out = t2 + t3"
    ]
    orchestrator = MLOrchestrator()
    orchestrator.optimize_program(sample_program)