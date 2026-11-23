import time
from compiler_frontend import Lexer, TACParser
from feature_extractor import TACFeatureExtractor
from iterative_loop import MLOrchestrator, RuleBasedSelector, UNSEEN_BENCHMARKS
from dataset_generator import build_dataset
from ml_pipeline import run_ml_pipeline, evaluate_unseen_benchmarks

CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RESET = '\033[0m'


def run_interactive_demo():
    print(f"\n{MAGENTA}=================================================================={RESET}")
    print(f"{MAGENTA}          COMPILER FRONTEND, ML ORCHESTRATOR & VERIFIER DEMO      {RESET}")
    print(f"{MAGENTA}=================================================================={RESET}\n")

    source = (
        "k1 = 10;\n"
        "x_cf = 12 * 20;\n"
        "dead_var = a + b;\n"
        "i = 0;\n"
        "sum = 0;\n"
        "while (i < 5) {\n"
        "    sum = sum + i;\n"
        "    y_sr = sum * 2;\n"
        "    i = i + 1;\n"
        "}\n"
        "out = x_cf + k1 + y_sr;\n"
    )

    print(f"{CYAN}[STEP 1] Raw Mini-C Source Code Ingestion:{RESET}")
    print(source)
    time.sleep(0.5)

    print(f"{CYAN}[STEP 2] Lexical Analysis (Token Stream):{RESET}")
    lexer = Lexer(source)
    tokens = [tok for tok in lexer.tokens if tok[0] != 'EOF']
    print(f"{YELLOW}{tokens[:18]} ... ({len(tokens)} total tokens){RESET}\n")
    time.sleep(0.5)

    print(f"{CYAN}[STEP 3] Recursive-Descent Parsing & TAC Lowering:{RESET}")
    parser = TACParser(Lexer(source))
    tac = parser.parse()
    for line in tac:
        print(f"{GREEN}  {line}{RESET}")
    print()
    time.sleep(0.5)

    print(f"{CYAN}[STEP 4] 8D Structural Feature Extraction:{RESET}")
    features = TACFeatureExtractor(tac).extract_features()
    for k, v in features.items():
        print(f"  - {k}: {v}")

    print(f"\n{CYAN}[STEP 5] Iterative Optimization & Semantic Equivalence Verification:{RESET}")
    orchestrator = MLOrchestrator(silent=True)
    orchestrator.optimize_program(tac, verbose=True)


def run_custom_code():
    print(f"\n{CYAN}Enter your Mini-C source code below.{RESET}")
    print(f"{YELLOW}(Type 'END' on a new line and press Enter to compile, optimize, and verify){RESET}")
    lines = []
    while True:
        line = input()
        if line.strip() == 'END':
            break
        lines.append(line)

    source = "\n".join(lines)

    try:
        print(f"\n{CYAN}--- Lexing & Parsing ---{RESET}")
        lexer = Lexer(source)
        parser = TACParser(lexer)
        tac = parser.parse()

        print(f"{GREEN}Generated Intermediate Representation (TAC):{RESET}")
        print("\n".join(tac))

        features = TACFeatureExtractor(tac).extract_features()
        print(f"\n{YELLOW}Rule-Based Heuristic Top Choice: {RuleBasedSelector.predict(features)}{RESET}")

        print(f"\n{CYAN}--- Executing ML Smart Fallback Orchestrator ---{RESET}")
        orchestrator = MLOrchestrator(silent=True)
        orchestrator.optimize_program(tac, verbose=True)
    except SyntaxError as e:
        print(f"\n{YELLOW}[Compilation Error] {e}{RESET}")
    except FileNotFoundError as e:
        print(f"\n{YELLOW}[Setup Error] {e}{RESET}")


if __name__ == "__main__":
    while True:
        print(f"\n{CYAN}Automated Optimization Strategy Selection System (Final Review){RESET}")
        print("1. Run Interactive End-to-End Compilation & Verification Demo")
        print("2. Generate 750-Sample Benchmark Dataset (dataset.csv)")
        print("3. Train ML Pipeline, Compare vs. Rule-Based & -O2, and Save Plots")
        print("4. Run Unseen Hand-Written Benchmark Suite (Semantic Equivalence Check)")
        print("5. Input & Optimize Custom Mini-C Source Code")
        print("6. Exit")
        choice = input("Select an option: ").strip()

        if choice == '1':
            run_interactive_demo()
        elif choice == '2':
            build_dataset(750)
        elif choice == '3':
            run_ml_pipeline()
        elif choice == '4':
            evaluate_unseen_benchmarks()
        elif choice == '5':
            run_custom_code()
        elif choice == '6':
            break
        else:
            print("Invalid selection.")