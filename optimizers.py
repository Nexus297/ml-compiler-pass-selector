import re

class TACOptimizer:

    @staticmethod
    def constant_folding(tac_lines):
        """Evaluates binary expressions where both operands are numeric constants."""
        optimized = []
        for line in tac_lines:
            if '=' in line and not line.strip().endswith(':') and not line.strip().startswith('if_false'):
                lhs, rhs = map(str.strip, line.split('=', 1))
                match = re.match(r'^(\d+)\s*([\+\-\*/])\s*(\d+)$', rhs)
                if match:
                    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                    if op == '+':
                        optimized.append(f"{lhs} = {a + b}")
                        continue
                    elif op == '-':
                        optimized.append(f"{lhs} = {a - b}")
                        continue
                    elif op == '*':
                        optimized.append(f"{lhs} = {a * b}")
                        continue
                    elif op == '/' and b != 0:
                        optimized.append(f"{lhs} = {a // b}")
                        continue
            optimized.append(line)
        return optimized

    @staticmethod
    def algebraic_simplification(tac_lines):
        """Simplifies algebraic identities: x+0, 0+x, x-0, x*1, 1*x, x/1, x*0, 0*x, x-x."""
        optimized = []
        for line in tac_lines:
            if '=' in line and not line.strip().endswith(':') and not line.strip().startswith('if_false'):
                lhs, rhs = map(str.strip, line.split('=', 1))
                if re.match(r'^(0\s*\*\s*[a-zA-Z_][a-zA-Z0-9_]*|[a-zA-Z_][a-zA-Z0-9_]*\s*\*\s*0)$', rhs):
                    optimized.append(f"{lhs} = 0")
                    continue
                m_sub = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*-\s*\1$', rhs)
                if m_sub:
                    optimized.append(f"{lhs} = 0")
                    continue
                m1 = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*[\+\-]\s*0$', rhs)
                if m1:
                    optimized.append(f"{lhs} = {m1.group(1)}")
                    continue
                m2 = re.match(r'^0\s*\+\s*([a-zA-Z_][a-zA-Z0-9_]*)$', rhs)
                if m2:
                    optimized.append(f"{lhs} = {m2.group(1)}")
                    continue
                m3 = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*[\*/]\s*1$', rhs)
                if m3:
                    optimized.append(f"{lhs} = {m3.group(1)}")
                    continue
                m4 = re.match(r'^1\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)$', rhs)
                if m4:
                    optimized.append(f"{lhs} = {m4.group(1)}")
                    continue
            optimized.append(line)
        return optimized

    @staticmethod
    def strength_reduction(tac_lines):
        """Replaces variable multiplication by 2 with addition (var * 2 -> var + var)."""
        optimized = []
        for line in tac_lines:
            if '=' in line and not line.strip().endswith(':') and not line.strip().startswith('if_false'):
                lhs, rhs = map(str.strip, line.split('=', 1))
                match = (re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\s*2$', rhs) or
                         re.match(r'^2\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)$', rhs))
                if match:
                    var = match.group(1)
                    optimized.append(f"{lhs} = {var} + {var}")
                    continue
            optimized.append(line)
        return optimized

    @staticmethod
    def common_subexpression_elimination(tac_lines):
        """Reuses previously computed identical binary expressions with reassignment invalidation."""
        optimized = []
        seen_exprs = {}
        for line in tac_lines:
            stripped = line.strip()
            if stripped.endswith(':') or 'goto' in stripped:
                seen_exprs.clear()
                optimized.append(line)
                continue

            if '=' in stripped and not stripped.startswith('if_false'):
                lhs, rhs = map(str.strip, stripped.split('=', 1))
                is_binary_op = bool(re.search(r'[\+\-\*/<>]|==|!=|<=|>=', rhs))
                has_vars = bool(re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', rhs))

                if is_binary_op and has_vars and rhs in seen_exprs:
                    replacement = seen_exprs[rhs]
                    optimized.append(f"{lhs} = {replacement}")
                else:
                    optimized.append(line)
                    seen_exprs = {
                        expr: holder for expr, holder in seen_exprs.items()
                        if holder != lhs and not re.search(fr'\b{re.escape(lhs)}\b', expr)
                    }
                    if is_binary_op and has_vars and not re.search(fr'\b{re.escape(lhs)}\b', rhs):
                        seen_exprs[rhs] = lhs
                    continue

                seen_exprs = {
                    expr: holder for expr, holder in seen_exprs.items()
                    if holder != lhs and not re.search(fr'\b{re.escape(lhs)}\b', expr)
                }
            else:
                optimized.append(line)
        return optimized

    @staticmethod
    def constant_propagation(tac_lines):
        """Propagates known constant assignments within basic blocks with reassignment invalidation."""
        optimized = []
        constants = {}

        mutated_counts = {}
        for line in tac_lines:
            if '=' in line and not line.strip().endswith(':') and not line.strip().startswith('if_false'):
                lhs = line.split('=', 1)[0].strip()
                mutated_counts[lhs] = mutated_counts.get(lhs, 0) + 1

        for line in tac_lines:
            stripped = line.strip()
            if stripped.endswith(':') or 'goto' in stripped:
                constants.clear()
                optimized.append(line)
                continue

            if '=' in stripped and not stripped.startswith('if_false'):
                lhs, rhs = map(str.strip, stripped.split('=', 1))

                new_rhs = rhs
                for var, val in constants.items():
                    new_rhs = re.sub(fr'\b{re.escape(var)}\b', val, new_rhs)

                if new_rhs.isdigit() and mutated_counts.get(lhs, 0) == 1 and not lhs.startswith('t'):
                    constants[lhs] = new_rhs
                else:
                    constants.pop(lhs, None)

                optimized.append(f"{lhs} = {new_rhs}")
            else:
                optimized.append(line)
        return optimized

    @staticmethod
    def dead_code_elimination(tac_lines):
        """Removes assignments to variables that are never subsequently referenced."""
        used_vars = set()
        for line in tac_lines:
            stripped = line.strip()
            if '=' in stripped and not stripped.startswith('if_false'):
                _, rhs = stripped.split('=', 1)
                used_vars.update(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rhs))
            elif stripped.startswith('if_false') or stripped.startswith('return') or stripped.startswith('print'):
                tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', stripped)
                for tok in tokens:
                    if tok not in ('if_false', 'goto', 'return', 'print') and not re.match(r'^L\d+$', tok):
                        used_vars.add(tok)

        optimized = []
        for line in tac_lines:
            stripped = line.strip()
            if '=' in stripped and not stripped.startswith('if_false'):
                lhs, _ = map(str.strip, stripped.split('=', 1))
                if lhs not in used_vars and not lhs.startswith('out'):
                    continue
            optimized.append(line)
        return optimized


class TACInterpreter:
    """
    Executes Three Address Code (TAC) programs on concrete input vectors
    to verify semantic equivalence before and after optimization.
    """
    TEST_INPUT_SETS = [
        {"input_x": 3,  "input_y": 5,  "input_z": 2,  "a": 4,  "b": 6,  "c": 3, "d": 5},
        {"input_x": 10, "input_y": 4,  "input_z": 8,  "a": 7,  "b": 2,  "c": 9, "d": 3},
        {"input_x": 0,  "input_y": 12, "input_z": 6,  "a": 11, "b": 5,  "c": 4, "d": 2},
        {"input_x": 15, "input_y": 9,  "input_z": 1,  "a": 3,  "b": 14, "c": 6, "d": 7},
        {"input_x": 7,  "input_y": 7,  "input_z": 11, "a": 8,  "b": 8,  "c": 5, "d": 4},
    ]

    @staticmethod
    def _resolve_val(token, env):
        token = token.strip()
        if re.match(r'^-?\d+$', token):
            return int(token)
        if token not in env:
            # Deterministic fallback for any uninitialized symbolic identifier
            env[token] = (sum(ord(ch) for ch in token) % 11) + 2
        return env[token]

    @classmethod
    def execute(cls, tac_lines, input_bindings=None, max_steps=5000):
        env = dict(input_bindings) if input_bindings else {}
        lines = [line.strip() for line in tac_lines if line.strip()]

        label_map = {}
        for idx, line in enumerate(lines):
            if line.endswith(':'):
                label_map[line[:-1].strip()] = idx

        ip = 0
        steps = 0
        while ip < len(lines) and steps < max_steps:
            steps += 1
            line = lines[ip]

            if line.endswith(':'):
                ip += 1
                continue

            if line.startswith('goto '):
                target = line.split('goto', 1)[1].strip()
                ip = label_map.get(target, len(lines))
                continue

            if line.startswith('if_false '):
                m = re.match(r'^if_false\s+([a-zA-Z0-9_]+)\s+goto\s+([a-zA-Z0-9_]+)$', line)
                if m:
                    cond_tok, target = m.group(1), m.group(2)
                    cond_val = cls._resolve_val(cond_tok, env)
                    if cond_val == 0:
                        ip = label_map.get(target, len(lines))
                        continue
                ip += 1
                continue

            if '=' in line:
                lhs, rhs = map(str.strip, line.split('=', 1))
                m_bin = re.match(
                    r'^([a-zA-Z0-9_]+)\s*(==|!=|<=|>=|<|>|\+|\-|\*|/)\s*([a-zA-Z0-9_]+)$',
                    rhs
                )
                if m_bin:
                    v1 = cls._resolve_val(m_bin.group(1), env)
                    op = m_bin.group(2)
                    v2 = cls._resolve_val(m_bin.group(3), env)
                    if op == '+': res = v1 + v2
                    elif op == '-': res = v1 - v2
                    elif op == '*': res = v1 * v2
                    elif op == '/': res = (v1 // v2) if v2 != 0 else 0
                    elif op == '<': res = int(v1 < v2)
                    elif op == '>': res = int(v1 > v2)
                    elif op == '<=': res = int(v1 <= v2)
                    elif op == '>=': res = int(v1 >= v2)
                    elif op == '==': res = int(v1 == v2)
                    elif op == '!=': res = int(v1 != v2)
                    else: res = 0
                    env[lhs] = res
                else:
                    env[lhs] = cls._resolve_val(rhs, env)
            ip += 1

        out_vars = {k: v for k, v in env.items() if k.startswith('out')}
        return out_vars if out_vars else env

    @classmethod
    def verify_equivalence(cls, orig_tac, opt_tac):
        """Runs original and optimized TAC on 5 input sets and verifies 100% output match."""
        matched = 0
        for inp in cls.TEST_INPUT_SETS:
            out_orig = cls.execute(orig_tac, inp)
            out_opt = cls.execute(opt_tac, inp)
            if out_orig == out_opt:
                matched += 1
        total = len(cls.TEST_INPUT_SETS)
        return (matched == total), matched, total