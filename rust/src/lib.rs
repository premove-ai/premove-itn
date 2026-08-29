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
    if ["oh", "o", "nought", "naught", "nil"]
        .iter()
        .any(|alias| token.eq_ignore_ascii_case(alias))
    {
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

fn sequence_repetition(token: &str) -> Option<usize> {
    match token {
        "single" => Some(1),
        "double" => Some(2),
        "triple" => Some(3),
        "quadruple" => Some(4),
        _ => None,
    }
}

fn parse_digit_sequence(text: &str) -> Option<String> {
    let mut output = String::new();
    let mut repetition = None;

    for token in text.split_whitespace() {
        let token = token.to_ascii_lowercase();

        if let Some(count) = sequence_repetition(&token) {
            if repetition.replace(count).is_some() {
                return None;
            }
            continue;
        }

        if token.chars().all(|character| character.is_ascii_digit()) {
            if token.len() > 1 && repetition.is_some() {
                return None;
            }
            for _ in 0..repetition.take().unwrap_or(1) {
                output.push_str(&token);
            }
            continue;
        }

        let digit = single_sequence_digit(&token)?;
        output.extend(std::iter::repeat_n(digit, repetition.take().unwrap_or(1)));
    }

    if output.is_empty() || repetition.is_some() {
        return None;
    }

    Some(output)
}

fn spoken_number(tokens: &[&str], maximum: u32) -> Option<u32> {
    let structurally_valid = match tokens {
        [_] => true,
        ["twenty", "one" | "two" | "three"] if maximum == 23 => true,
        ["twenty" | "thirty" | "forty" | "fifty", second] if maximum == 59 => {
            single_sequence_digit(second).is_some_and(|digit| digit != '0')
        }
        ["zero" | "oh" | "o", second] if maximum == 59 => single_sequence_digit(second).is_some(),
        _ => false,
    };
    if !structurally_valid {
        return None;
    }

    if tokens.len() == 2 && matches!(tokens[0], "zero" | "oh" | "o") {
        return single_sequence_digit(tokens[1])?.to_digit(10);
    }
    let value = cardinal::words_to_number(&tokens.join(" "))?;
    u32::try_from(value).ok().filter(|value| *value <= maximum)
}

fn parse_spoken_time(text: &str) -> Option<String> {
    let lowered = text.to_ascii_lowercase();
    let mut tokens: Vec<&str> = lowered.split_whitespace().collect();
    let mut meridiem = None;

    if let Some(last) = tokens.last() {
        let compact = last.replace('.', "");
        if matches!(compact.as_str(), "am" | "pm") {
            meridiem = compact.chars().next();
            tokens.pop();
        } else if tokens.len() >= 2 && *last == "m" {
            let marker = tokens[tokens.len() - 2];
            if matches!(marker, "a" | "p") {
                meridiem = marker.chars().next();
                tokens.truncate(tokens.len() - 2);
            }
        }
    }

    for minute_length in [2, 1] {
        if tokens.len() <= minute_length {
            continue;
        }
        let split = tokens.len() - minute_length;
        let hour_maximum = if meridiem.is_some() { 12 } else { 23 };
        let Some(hour) = spoken_number(&tokens[..split], hour_maximum) else {
            continue;
        };
        if meridiem.is_some() && hour == 0 {
            continue;
        }
        let Some(minute) = spoken_number(&tokens[split..], 59) else {
            continue;
        };
        let suffix = match meridiem {
            Some('a') => " a.m.",
            Some('p') => " p.m.",
            _ => "",
        };
        return Some(format!("{hour:02}:{minute:02}{suffix}"));
    }
    None
}

const EXTENDED_CARDINAL_SCALES: &[(&str, i128)] = &[
    (
        "undecillion",
        1_000_000_000_000_000_000_000_000_000_000_000_000,
    ),
    ("decillion", 1_000_000_000_000_000_000_000_000_000_000_000),
    ("nonillion", 1_000_000_000_000_000_000_000_000_000_000),
    ("octillion", 1_000_000_000_000_000_000_000_000_000),
    ("septillion", 1_000_000_000_000_000_000_000_000),
];

fn parse_extended_cardinal_words(text: &str) -> Option<i128> {
    let words: Vec<&str> = text.split_whitespace().collect();
    for (scale_index, (scale_word, scale)) in EXTENDED_CARDINAL_SCALES.iter().enumerate() {
        if let Some(index) = words.iter().position(|word| word == scale_word) {
            if index == 0 {
                return None;
            }
            if words[..index].iter().any(|word| {
                EXTENDED_CARDINAL_SCALES
                    .iter()
                    .any(|(candidate, _)| word == candidate)
            }) {
                return None;
            }
            if words[index + 1..].iter().any(|word| {
                EXTENDED_CARDINAL_SCALES[..=scale_index]
                    .iter()
                    .any(|(candidate, _)| word == candidate)
            }) {
                return None;
            }
            let coefficient = parse_extended_cardinal_words(&words[..index].join(" "))?;
            let remainder = if index + 1 == words.len() {
                0
            } else {
                parse_extended_cardinal_words(&words[index + 1..].join(" "))?
            };
            return coefficient.checked_mul(*scale)?.checked_add(remainder);
        }
    }
    cardinal::words_to_number(text)
}

fn parse_cardinal_number(text: &str) -> Option<i128> {
    let text = text.trim().to_ascii_lowercase();
    let (is_negative, magnitude) = if let Some(rest) = text.strip_prefix("minus ") {
        (true, rest)
    } else if let Some(rest) = text.strip_prefix("negative ") {
        (true, rest)
    } else {
        (false, text.as_str())
    };
    let value = parse_extended_cardinal_words(magnitude)?;
    if is_negative {
        value.checked_neg()
    } else {
        Some(value)
    }
}

fn is_negative_zero(text: &str) -> bool {
    matches!(
        text.trim().to_ascii_lowercase().as_str(),
        "minus zero" | "negative zero"
    )
}

fn parse_cardinal(text: &str) -> Option<String> {
    if is_negative_zero(text) {
        return Some("-0".to_owned());
    }
    if let Some(value) = parse_cardinal_number(text) {
        return Some(value.to_string());
    }
    parse_digit_sequence(text)
}

fn cardinal_options(text: &str) -> Vec<String> {
    let Some(canonical) = parse_cardinal(text) else {
        return Vec::new();
    };
    let mut options = vec![canonical.clone()];
    if let Some(aviation) = cardinal::parse_aviation(text).filter(|value| *value != canonical) {
        options.push(aviation);
    }
    options
}

fn realize_known_kind_options(kind: &str, text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    if kind == "CARDINAL" {
        return cardinal_options(text);
    }
    realize_known_kind(kind, text).into_iter().collect()
}

fn realize_known_kind(kind: &str, text: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    match kind {
        "CARDINAL" => parse_cardinal(text),
        "DATE" => date::parse(text),
        "DECIMAL" => decimal::parse(text),
        "DIGIT_SEQUENCE" => parse_digit_sequence(text),
        "ELECTRONIC" => electronic::parse(text),
        "MONEY" => money::parse(text),
        "MEASUREMENT" => measure::parse(text),
        "ORDINAL" => ordinal::parse(text),
        "PUNCTUATION" => punctuation::parse(text),
        "PHONE" => telephone::parse(text),
        "TIME" => parse_spoken_time(text).or_else(|| time::parse(text)),
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
fn realize_options(kind: &str, text: &str) -> PyResult<Vec<String>> {
    if !SUPPORTED_KINDS.contains(&kind) {
        return Err(PyValueError::new_err(format!(
            "unsupported span kind {kind:?}; expected one of {}",
            SUPPORTED_KINDS.join(", ")
        )));
    }

    Ok(realize_known_kind_options(kind, text))
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
    module.add_function(wrap_pyfunction!(realize_options, module)?)?;
    module.add_function(wrap_pyfunction!(baseline_normalize_sentence, module)?)?;
    module.add_function(wrap_pyfunction!(tn_normalize, module)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/realizers.rs"]
mod tests;
