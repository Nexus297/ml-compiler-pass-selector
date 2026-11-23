import re

class Lexer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self.current = 0
        self.tokenize()

    def tokenize(self):
        token_specification = [
            ('NUM',       r'\d+'),
            ('ID',        r'[A-Za-z_][A-Za-z0-9_]*'),
            ('RELOP',     r'==|!=|<=|>=|<|>'),
            ('ASSIGN',    r'='),
            ('ADDOP',     r'[\+\-]'),
            ('MULOP',     r'[\*/]'),
            ('SEMI',      r';'),
            ('LPAREN',    r'\('),
            ('RPAREN',    r'\)'),
            ('LBRACE',    r'\{'),
            ('RBRACE',    r'\}'),
            ('SKIP',      r'[ \t\r\n]+'),
            ('MISMATCH',  r'.'),
        ]
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
        for mo in re.finditer(tok_regex, self.source_code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise SyntaxError(f"Lexical Error: Unexpected character {value!r}")
            self.tokens.append((kind, value))
        self.tokens.append(('EOF', ''))

    def get_token(self):
        if self.current < len(self.tokens):
            tok = self.tokens[self.current]
            self.current += 1
            return tok
        return ('EOF', '')

    def peek(self):
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return ('EOF', '')

    def expect(self, expected_kind):
        kind, value = self.get_token()
        if kind != expected_kind:
            raise SyntaxError(f"Syntax Error: Expected {expected_kind}, got {kind} ({value!r})")
        return value


class TACParser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.tac = []
        self.temp_count = 1
        self.label_count = 1

    def new_temp(self):
        name = f"t{self.temp_count}"
        self.temp_count += 1
        return name

    def new_label(self):
        name = f"L{self.label_count}"
        self.label_count += 1
        return name

    def parse(self):
        while self.lexer.peek()[0] != 'EOF':
            self.parse_statement()
        return self.tac

    def parse_statement(self):
        kind, value = self.lexer.peek()
        if kind == 'ID' and value == 'while':
            self.parse_while()
        elif kind == 'ID' and value == 'if':
            self.parse_if()
        elif kind == 'ID':
            self.parse_assignment()
        else:
            raise SyntaxError(f"Syntax Error: Unexpected token {value!r} ({kind})")

    def parse_assignment(self):
        lhs_name = self.lexer.expect('ID')
        self.lexer.expect('ASSIGN')
        expr_val = self.parse_expression()
        self.lexer.expect('SEMI')
        self.tac.append(f"{lhs_name} = {expr_val}")

    def parse_while(self):
        self.lexer.get_token()  # 'while'
        self.lexer.expect('LPAREN')

        start_label = self.new_label()
        end_label = self.new_label()

        self.tac.append(f"{start_label}:")
        cond_val = self.parse_expression()
        self.lexer.expect('RPAREN')
        self.tac.append(f"if_false {cond_val} goto {end_label}")

        self.lexer.expect('LBRACE')
        while self.lexer.peek()[0] not in ('RBRACE', 'EOF'):
            self.parse_statement()
        self.lexer.expect('RBRACE')

        self.tac.append(f"goto {start_label}")
        self.tac.append(f"{end_label}:")

    def parse_if(self):
        self.lexer.get_token()  # 'if'
        self.lexer.expect('LPAREN')
        cond_val = self.parse_expression()
        self.lexer.expect('RPAREN')

        false_label = self.new_label()
        self.tac.append(f"if_false {cond_val} goto {false_label}")

        self.lexer.expect('LBRACE')
        while self.lexer.peek()[0] not in ('RBRACE', 'EOF'):
            self.parse_statement()
        self.lexer.expect('RBRACE')

        if self.lexer.peek() == ('ID', 'else'):
            self.lexer.get_token()  # 'else'
            end_label = self.new_label()
            self.tac.append(f"goto {end_label}")
            self.tac.append(f"{false_label}:")
            self.lexer.expect('LBRACE')
            while self.lexer.peek()[0] not in ('RBRACE', 'EOF'):
                self.parse_statement()
            self.lexer.expect('RBRACE')
            self.tac.append(f"{end_label}:")
        else:
            self.tac.append(f"{false_label}:")

    def parse_expression(self):
        """Relational operators (lowest precedence)."""
        left = self.parse_additive()
        while self.lexer.peek()[0] == 'RELOP':
            op = self.lexer.get_token()[1]
            right = self.parse_additive()
            temp = self.new_temp()
            self.tac.append(f"{temp} = {left} {op} {right}")
            left = temp
        return left

    def parse_additive(self):
        """Addition and subtraction."""
        left = self.parse_term()
        while self.lexer.peek()[0] == 'ADDOP':
            op = self.lexer.get_token()[1]
            right = self.parse_term()
            temp = self.new_temp()
            self.tac.append(f"{temp} = {left} {op} {right}")
            left = temp
        return left

    def parse_term(self):
        """Multiplication and division (highest binary precedence)."""
        left = self.parse_factor()
        while self.lexer.peek()[0] == 'MULOP':
            op = self.lexer.get_token()[1]
            right = self.parse_factor()
            temp = self.new_temp()
            self.tac.append(f"{temp} = {left} {op} {right}")
            left = temp
        return left

    def parse_factor(self):
        kind, value = self.lexer.peek()
        if kind in ('NUM', 'ID'):
            return self.lexer.get_token()[1]
        elif kind == 'LPAREN':
            self.lexer.expect('LPAREN')
            expr = self.parse_expression()
            self.lexer.expect('RPAREN')
            return expr
        raise SyntaxError(f"Syntax Error: Unexpected factor {value!r}")