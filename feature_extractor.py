import re
from collections import Counter

class TACFeatureExtractor:
    def __init__(self, tac_code):
        if isinstance(tac_code, list):
            self.tac_lines = [line.strip() for line in tac_code if line.strip()]
        else:
            self.tac_lines = [line.strip() for line in tac_code.strip().split('\n') if line.strip()]
        self.lhs_vars = []
        self.rhs_exprs = []
        self.all_used_vars = set()

    def parse_lines(self):
        """Parses TAC lines into LHS variables, RHS expressions, and used variable sets."""
        for line in self.tac_lines:
            if line.endswith(':'):
                continue
            if line.startswith('if_false') or line.startswith('goto'):
                tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', line)
                for tok in tokens:
                    if tok not in ('if_false', 'goto') and not re.match(r'^L\d+$', tok):
                        self.all_used_vars.add(tok)
                continue

            match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
            if match:
                lhs, rhs = match.groups()
                self.lhs_vars.append(lhs)
                self.rhs_exprs.append(rhs.strip())
                used_vars = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rhs)
                self.all_used_vars.update(used_vars)

    def extract_features(self):
        self.parse_lines()

        num_statements = len(self.tac_lines)
        num_constants = 0
        num_algebraic_ops = 0
        num_loop_ops = 0

        in_loop_depth = 0
        block_rhs = []
        num_repeated_expr = 0

        for line in self.tac_lines:
            if line.endswith(':'):
                in_loop_depth = 1
                block_rhs.clear()
                continue
            elif line.startswith('goto'):
                in_loop_depth = 0
                block_rhs.clear()
                continue
            elif line.startswith('if_false'):
                block_rhs.clear()
                if in_loop_depth > 0:
                    num_loop_ops += 1
                continue

            if in_loop_depth > 0:
                num_loop_ops += 1

            # Strictly detect algebraic identities (+0, -0, *1, /1, *0, x-x) per rubric specification
            if re.search(
                r'(\+\s*0\b|\b0\s*\+|-\s*0\b|\*\s*1\b|\b1\s*\*|/\s*1\b|\*\s*0\b|\b0\s*\*|\b([a-zA-Z_]\w*)\s*-\s*\2\b)',
                line
            ):
                num_algebraic_ops += 1

            match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
            if match:
                lhs, rhs = match.group(1), match.group(2).strip()
                if re.match(r'^\d+\s*[\+\-\*/]\s*\d+$', rhs):
                    num_constants += 2
                elif rhs.isdigit() and lhs not in ('i', 'j', 'sum', 'acc'):
                    num_constants += 1

                if re.search(r'[\+\-\*/]', rhs) and re.search(r'[a-zA-Z_]', rhs):
                    block_rhs.append(rhs)

                if block_rhs:
                    rhs_counts = Counter(block_rhs)
                    if rhs_counts[rhs] > 1:
                        num_repeated_expr += 1

        num_dead_stmts = sum(
            1 for var in self.lhs_vars
            if var not in self.all_used_vars and not var.startswith('out')
        )

        temp_vars = set(var for var in self.lhs_vars if re.match(r'^t\d+$', var))
        num_tmp_vars = len(temp_vars)

        const_ratio = num_constants / num_statements if num_statements > 0 else 0.0

        return {
            "num_statements": num_statements,
            "num_constants": num_constants,
            "num_repeated_expr": num_repeated_expr,
            "num_dead_stmts": num_dead_stmts,
            "num_algebraic_ops": num_algebraic_ops,
            "num_loop_ops": num_loop_ops,
            "num_tmp_vars": num_tmp_vars,
            "const_ratio": round(const_ratio, 3)
        }


if __name__ == "__main__":
    sample_tac = [
        "t1 = 10 * 20",
        "t2 = a + 0",
        "t3 = c * d",
        "t4 = c * d",
        "out = t1 + t2 + t3 + t4"
    ]
    extractor = TACFeatureExtractor(sample_tac)
    for k, v in extractor.extract_features().items():
        print(f"{k}: {v}")