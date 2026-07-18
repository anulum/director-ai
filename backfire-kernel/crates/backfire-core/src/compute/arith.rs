// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::arith
//! Arithmetic expression tokenisation, parsing, and evaluation.
//!
//! Mirrors `_eval_arithmetic()` from `core/verification/math_consistency.py`.

/// A token in an arithmetic expression.
#[derive(Debug, Clone, PartialEq)]
enum ArithTok {
    Num(f64),
    Add,
    Sub,
    Mul,
    Div,
    LParen,
    RParen,
}

/// Tokenise a numeric arithmetic expression.
///
/// Accepts decimal numbers with thousands separators (`1,234.5`), the operators
/// `+ - * /` (Unicode `× ÷ ·` are normalised to ASCII before this point), and
/// parentheses. Any other character makes the whole expression unparsable, so a
/// caller can distinguish "not arithmetic" from "wrong arithmetic". Returns
/// `None` on an empty or invalid expression.
fn tokenize_arith(expr: &str) -> Option<Vec<ArithTok>> {
    let chars: Vec<char> = expr.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => toks.push(ArithTok::Add),
            '-' => toks.push(ArithTok::Sub),
            '*' => toks.push(ArithTok::Mul),
            '/' => toks.push(ArithTok::Div),
            '(' => toks.push(ArithTok::LParen),
            ')' => toks.push(ArithTok::RParen),
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',')
                {
                    i += 1;
                }
                let num: String = chars[start..i].iter().collect::<String>().replace(',', "");
                match num.parse::<f64>() {
                    Ok(v) => toks.push(ArithTok::Num(v)),
                    Err(_) => return None,
                }
                continue;
            }
            _ => return None,
        }
        i += 1;
    }
    if toks.is_empty() {
        None
    } else {
        Some(toks)
    }
}

/// Recursive-descent arithmetic parser/evaluator over a token slice.
///
/// Grammar (standard precedence, left-associative):
/// `expr := term (('+'|'-') term)*`, `term := factor (('*'|'/') factor)*`,
/// `factor := NUM | '(' expr ')' | '-' factor`.
struct ArithParser<'a> {
    toks: &'a [ArithTok],
    pos: usize,
}

impl<'a> ArithParser<'a> {
    fn peek(&self) -> Option<&ArithTok> {
        self.toks.get(self.pos)
    }

    fn parse_expr(&mut self) -> Option<f64> {
        let mut value = self.parse_term()?;
        while let Some(op) = self.peek() {
            match op {
                ArithTok::Add => {
                    self.pos += 1;
                    value += self.parse_term()?;
                }
                ArithTok::Sub => {
                    self.pos += 1;
                    value -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Some(value)
    }

    fn parse_term(&mut self) -> Option<f64> {
        let mut value = self.parse_factor()?;
        while let Some(op) = self.peek() {
            match op {
                ArithTok::Mul => {
                    self.pos += 1;
                    value *= self.parse_factor()?;
                }
                ArithTok::Div => {
                    self.pos += 1;
                    value /= self.parse_factor()?;
                }
                _ => break,
            }
        }
        Some(value)
    }

    fn parse_factor(&mut self) -> Option<f64> {
        match self.peek()? {
            ArithTok::Num(v) => {
                let value = *v;
                self.pos += 1;
                Some(value)
            }
            ArithTok::Sub => {
                self.pos += 1;
                Some(-self.parse_factor()?)
            }
            ArithTok::LParen => {
                self.pos += 1;
                let value = self.parse_expr()?;
                match self.peek()? {
                    ArithTok::RParen => {
                        self.pos += 1;
                        Some(value)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Evaluate an arithmetic expression to an `f64`, or `NaN` if it is not a valid
/// numeric expression.
///
/// Division by zero yields IEEE-754 infinity/NaN (left to the caller). Mirrors
/// `_eval_arithmetic()` from `core/verification/math_consistency.py` so the Rust
/// and Python paths are bit-for-bit identical.
pub fn eval_arithmetic(expr: &str) -> f64 {
    let normalised = expr.replace(['×', '·'], "*").replace('÷', "/");
    let toks = match tokenize_arith(&normalised) {
        Some(t) => t,
        None => return f64::NAN,
    };
    let mut parser = ArithParser {
        toks: &toks,
        pos: 0,
    };
    match parser.parse_expr() {
        // A non-finite result (division by zero, overflow) is reported as NaN so
        // the Rust and Python paths agree bit-for-bit.
        Some(value) if parser.pos == toks.len() && value.is_finite() => value,
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_arithmetic_precedence_and_parens() {
        assert_eq!(eval_arithmetic("3 + 4"), 7.0);
        assert_eq!(eval_arithmetic("2 + 3 * 4"), 14.0);
        assert_eq!(eval_arithmetic("(2 + 3) * 4"), 20.0);
        assert_eq!(eval_arithmetic("10 - 2 - 3"), 5.0);
        assert_eq!(eval_arithmetic("-5 + 8"), 3.0);
    }

    #[test]
    fn test_eval_arithmetic_unicode_ops_and_commas() {
        assert_eq!(eval_arithmetic("12 × 5"), 60.0);
        assert_eq!(eval_arithmetic("100 ÷ 4"), 25.0);
        assert_eq!(eval_arithmetic("1,000 + 234"), 1234.0);
    }

    #[test]
    fn test_eval_arithmetic_invalid_is_nan() {
        assert!(eval_arithmetic("3 + ").is_nan());
        assert!(eval_arithmetic("hello").is_nan());
        assert!(eval_arithmetic("").is_nan());
        assert!(eval_arithmetic("(2 + 3").is_nan());
        assert!(eval_arithmetic("2 3").is_nan());
        assert!(eval_arithmetic("1 / 0").is_nan());
    }
}
