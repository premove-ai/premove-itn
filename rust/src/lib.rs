use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use text_processing_rs::itn::en::{
    cardinal, date, decimal, electronic, measure, money, ordinal, punctuation, telephone, time,
    whitelist, word,
};

const SUPPORTED_KINDS: &[&str] = &[
    "CARDINAL",
    "DATE",
    "DECIMAL",
    "DIGIT_SEQUENCE",
    "ELECTRONIC",
    "MONEY",
    "MEASUREMENT",
    "ORDINAL",
    "PUNCTUATION",
    "PHONE",
    "TIME",
    "WHITELIST",
    "WORD",
];

fn single_sequence_digit(token: &str) -> Option<char> {
    if token == "oh" {
        return Some('0');
    }

    if token.len() == 1 {
        let digit = token.chars().next()?;
        if digit.is_ascii_digit() {
            return Some(digit);
        }
    }

    let value = cardinal::words_to_number(token)?;
    u32::try_from(value).ok().and_then(|value| {
        if value <= 9 {
            char::from_digit(value, 10)
        } else {
            None
        }
    })
}

fn parse_digit_sequence(text: &str) -> Option<String> {
    let mut output = String::new();
    let mut repetition = 1;

    for token in text.split_whitespace() {
        let token = token.to_ascii_lowercase();

        match token.as_str() {
            "double" if repetition == 1 => {
                repetition = 2;
                continue;
            }
            "triple" if repetition == 1 => {
                repetition = 3;
                continue;
            }
            "double" | "triple" => return None,
            _ => {}
        }

        let digit = single_sequence_digit(&token)?;
        output.extend(std::iter::repeat_n(digit, repetition));
        repetition = 1;
    }

    if output.is_empty() || repetition != 1 {
        return None;
    }

    Some(output)
}

fn realize_known_kind(kind: &str, text: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    match kind {
        "CARDINAL" => cardinal::parse(text),
        "DATE" => date::parse(text),
        "DECIMAL" => decimal::parse(text),
        "DIGIT_SEQUENCE" => parse_digit_sequence(text),
        "ELECTRONIC" => electronic::parse(text),
        "MONEY" => money::parse(text),
        "MEASUREMENT" => measure::parse(text),
        "ORDINAL" => ordinal::parse(text),
        "PUNCTUATION" => punctuation::parse(text),
        "PHONE" => telephone::parse(text),
        "TIME" => time::parse(text),
        "WHITELIST" => whitelist::parse(text),
        "WORD" => word::parse(text),
        _ => None,
    }
}

#[pyfunction]
fn realize(kind: &str, text: &str) -> PyResult<Option<String>> {
    if !SUPPORTED_KINDS.contains(&kind) {
        return Err(PyValueError::new_err(format!(
            "unsupported span kind {kind:?}; expected one of {}",
            SUPPORTED_KINDS.join(", ")
        )));
    }

    Ok(realize_known_kind(kind, text))
}

#[pyfunction]
fn baseline_normalize_sentence(text: &str) -> String {
    text_processing_rs::normalize_sentence(text)
}

#[pyfunction]
fn tn_normalize(text: &str) -> String {
    text_processing_rs::tn_normalize(text)
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(realize, module)?)?;
    module.add_function(wrap_pyfunction!(baseline_normalize_sentence, module)?)?;
    module.add_function(wrap_pyfunction!(tn_normalize, module)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/realizers.rs"]
mod tests;
