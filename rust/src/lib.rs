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

const DATE_MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

const DATE_WEEKDAYS: &[&str] = &[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

fn parse_date_day(text: &str) -> Option<u32> {
    if let Some(ordinal) = ordinal::parse(text) {
        return ordinal
            .chars()
            .filter(char::is_ascii_digit)
            .collect::<String>()
            .parse::<u32>()
            .ok()
            .filter(|day| (1..=31).contains(day));
    }
    cardinal::words_to_number(text)
        .and_then(|day| u32::try_from(day).ok())
        .filter(|day| (1..=31).contains(day))
}

fn parse_under_hundred(text: &str) -> Option<i128> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let value = cardinal::words_to_number(text)?;
    let structurally_valid = match words.as_slice() {
        [_] => (0..=99).contains(&value),
        [first, second] => {
            let first = cardinal::words_to_number(first)?;
            let second = cardinal::words_to_number(second)?;
            (20..=90).contains(&first) && first % 10 == 0 && (1..=9).contains(&second)
        }
        _ => false,
    };
    structurally_valid.then_some(value)
}

fn parse_date_year(text: &str, allow_short: bool) -> Option<String> {
    let text = text.trim();
    if let Some(digits) = parse_digit_sequence(text).filter(|digits| digits.len() <= 2) {
        return allow_short.then_some(digits);
    }
    if let Some(digit_text) = text.strip_prefix("oh ").or_else(|| text.strip_prefix("o ")) {
        let digit = cardinal::words_to_number(digit_text)?;
        if (0..=9).contains(&digit) {
            return Some(format!("{digit:02}"));
        }
        return None;
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if let Some(zero_index) = words
        .iter()
        .position(|word| matches!(*word, "oh" | "o"))
        .filter(|index| *index > 0 && *index + 1 < words.len())
    {
        let prefix = cardinal::words_to_number(&words[..zero_index].join(" "))?;
        let suffix = parse_digit_sequence(&words[zero_index + 1..].join(" "))?;
        if (1..=99).contains(&prefix) && suffix.len() == 1 {
            return Some(format!("{prefix}{:02}", suffix.parse::<u8>().ok()?));
        }
        return None;
    }

    if let Some(value) = parse_under_hundred(text) {
        return allow_short.then(|| value.to_string());
    }
    for split in (1..words.len()).rev() {
        let Some(prefix) = parse_under_hundred(&words[..split].join(" ")) else {
            continue;
        };
        let Some(suffix) = parse_under_hundred(&words[split..].join(" ")) else {
            continue;
        };
        if prefix >= 1 {
            return Some(format!("{prefix}{suffix:02}"));
        }
    }

    let aviation = cardinal::parse_aviation(text)?.parse::<i128>().ok()?;
    let grammatical = cardinal::words_to_number(text);
    let word_count = text.split_whitespace().count();
    let value = grammatical
        .filter(|value| (0..=99).contains(value) && (word_count == 1 || *value == aviation))
        .unwrap_or(aviation);
    let valid = if allow_short {
        value >= 0
    } else {
        (100..=9999).contains(&value)
    };
    valid.then(|| value.to_string())
}

fn split_date_era(text: &str) -> Option<(String, String)> {
    const ERAS: &[(&str, &str)] = &[
        (" b c e", "BCE"),
        (" bce", "BCE"),
        (" b c", "BC"),
        (" bc", "BC"),
        (" c e", "CE"),
        (" ce", "CE"),
        (" a d", "AD"),
        (" ad", "AD"),
    ];
    for (suffix, canonical) in ERAS {
        if let Some(base) = text.strip_suffix(suffix) {
            return Some((base.to_owned(), (*canonical).to_owned()));
        }
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let era_start = words
        .iter()
        .rposition(|word| {
            word.len() != 1
                || !word
                    .chars()
                    .all(|character| character.is_ascii_alphabetic())
        })
        .map_or(0, |index| index + 1);
    if era_start > 0
        && era_start < words.len()
        && words.len() - era_start <= 3
        && words[era_start..]
            .iter()
            .all(|word| !matches!(*word, "o" | "oh"))
    {
        return Some((
            words[..era_start].join(" "),
            words[era_start..].join("").to_ascii_uppercase(),
        ));
    }
    None
}

fn parse_date_plural_year(text: &str) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let last = words.last()?;
    let singular_last = if let Some(stem) = last.strip_suffix("ies") {
        format!("{stem}y")
    } else if *last == "sixes" {
        "six".to_owned()
    } else {
        last.strip_suffix('s')?.to_owned()
    };
    if singular_last.is_empty() {
        return None;
    }
    let mut singular_words = words;
    singular_words.pop();
    singular_words.push(&singular_last);
    let singular = singular_words.join(" ");

    if let Some(year) = parse_date_year(&singular, true) {
        return Some(format!("{year}s"));
    }
    if let Some(parsed) = date::parse(&singular) {
        if parsed.chars().all(|character| character.is_ascii_digit()) {
            return Some(format!("{parsed}s"));
        }
    }
    None
}

fn parse_month_first_date(text: &str) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let month = *words.first()?;
    if !DATE_MONTHS.contains(&month) || words.len() < 2 {
        return None;
    }

    if let Some(year) = parse_date_year(&words[1..].join(" "), false) {
        return Some(format!("{month} {year}"));
    }

    for day_end in (2..=words.len().min(3)).rev() {
        let Some(day) = parse_date_day(&words[1..day_end].join(" ")) else {
            continue;
        };
        if day_end == words.len() {
            return Some(format!("{month} {day}"));
        }
        if let Some(year) = parse_date_year(&words[day_end..].join(" "), true) {
            return Some(format!("{month} {day} {year}"));
        }
    }
    None
}

fn parse_day_first_date(text: &str) -> Option<String> {
    let text = text.strip_prefix("the ").unwrap_or(text);
    let words: Vec<&str> = text.split_whitespace().collect();
    let of_index = words.iter().position(|word| *word == "of")?;
    if of_index == 0 || of_index + 1 >= words.len() {
        return None;
    }
    let day = parse_date_day(&words[..of_index].join(" "))?;
    let month = words[of_index + 1];
    if !DATE_MONTHS.contains(&month) {
        return None;
    }
    if of_index + 2 == words.len() {
        return Some(format!("{day} {month}"));
    }
    let year = parse_date_year(&words[of_index + 2..].join(" "), true)?;
    Some(format!("{day} {month} {year}"))
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn valid_calendar_date(output: &str) -> bool {
    let words: Vec<&str> = output.split_whitespace().collect();
    let Some(month_index) = words.iter().position(|word| DATE_MONTHS.contains(word)) else {
        return true;
    };
    let day = if month_index == 0 {
        words.get(1)
    } else {
        words.get(month_index.wrapping_sub(1))
    }
    .and_then(|day| day.parse::<u32>().ok());
    let Some(day) = day else {
        return true;
    };
    if day > 31 {
        return true;
    }
    let month = DATE_MONTHS
        .iter()
        .position(|month| *month == words[month_index])
        .expect("known month")
        + 1;
    let year = words
        .get(month_index + 2)
        .or_else(|| words.get(month_index + 1).filter(|_| month_index > 0))
        .and_then(|year| year.parse::<i32>().ok())
        .filter(|year| *year >= 100);
    let maximum = match month {
        2 if year.is_some_and(is_leap_year) => 29,
        2 if year.is_some() => 28,
        2 => 29,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    day <= maximum
}

fn parse_date(text: &str) -> Option<String> {
    let text = text.trim().to_ascii_lowercase();
    let (weekday, core) = match text.split_once(' ') {
        Some((first, rest)) if DATE_WEEKDAYS.contains(&first) => (Some(first), rest),
        _ => (None, text.as_str()),
    };

    let (core, era) = split_date_era(core)
        .map_or_else(|| (core.to_owned(), None), |(base, era)| (base, Some(era)));
    let parsed = parse_date_plural_year(&core)
        .or_else(|| parse_day_first_date(&core))
        .or_else(|| parse_month_first_date(&core))
        .or_else(|| parse_date_year(&core, era.is_some()))
        .or_else(|| date::parse(&core).map(|value| value.to_ascii_lowercase()))?;
    let parsed = match era {
        Some(era) if DATE_MONTHS.iter().any(|month| parsed.contains(month)) => {
            format!("{parsed} {era}")
        }
        Some(era) => format!("{parsed}{era}"),
        None => parsed,
    };
    if !valid_calendar_date(&parsed) {
        return None;
    }
    Some(match weekday {
        Some(weekday) => format!("{weekday} {parsed}"),
        None => parsed,
    })
}

fn strip_display_suffix(text: &str) -> &str {
    text.trim()
        .strip_suffix("'s")
        .or_else(|| text.trim().strip_suffix("’s"))
        .or_else(|| text.trim().strip_suffix('s'))
        .unwrap_or(text.trim())
}

fn roman_value(text: &str) -> Option<i128> {
    let text = strip_display_suffix(text).trim().to_ascii_uppercase();
    if text.is_empty() {
        return None;
    }
    let values: Vec<i128> = text
        .chars()
        .map(|character| match character {
            'I' => Some(1),
            'V' => Some(5),
            'X' => Some(10),
            'L' => Some(50),
            'C' => Some(100),
            'D' => Some(500),
            'M' => Some(1000),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let mut total = 0;
    for (index, value) in values.iter().enumerate() {
        if values.get(index + 1).is_some_and(|next| value < next) {
            total -= value;
        } else {
            total += value;
        }
    }
    Some(total)
}

fn cardinal_representation_value(text: &str) -> Option<(bool, String)> {
    let text = strip_display_suffix(text)
        .trim()
        .trim_end_matches(['.', ',', ':', ';', '-'])
        .trim_end();
    if let Some(value) = roman_value(text) {
        return Some((false, value.to_string()));
    }
    let mut negative = false;
    let mut digits = String::new();
    for (index, character) in text.chars().enumerate() {
        match character {
            '-' if index == 0 => negative = true,
            '+' if index == 0 => {}
            '0'..='9' => digits.push(character),
            ',' | '_' | ' ' | '\u{00a0}' | '\u{202f}' => {}
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    Some((negative, digits.to_owned()))
}

fn cardinal_representations_equivalent(canonical: &str, observed: &str) -> bool {
    cardinal_representation_value(canonical) == cardinal_representation_value(observed)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DateRepresentation {
    Year(String),
    Period(String),
    Era { year: String, era: String },
    MonthYear { month: u8, year: String },
    MonthDay { month: u8, day: u8 },
    FullDate { month: u8, day: u8, year: String },
    Quarter { quarter: u8, year: String },
}

fn representation_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut alphabetic = None;
    for character in text.to_ascii_lowercase().chars() {
        let kind = character.is_ascii_alphabetic();
        if character.is_ascii_alphanumeric() {
            if alphabetic.is_some_and(|previous| previous != kind) && !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            current.push(character);
            alphabetic = Some(kind);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
            alphabetic = None;
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn representation_month(token: &str) -> Option<u8> {
    let prefix = token.get(..3.min(token.len()))?;
    [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    .iter()
    .position(|month| *month == prefix)
    .map(|index| index as u8 + 1)
}

fn representation_year(text: &str) -> String {
    let digits = text.trim_start_matches('0');
    if digits.is_empty() {
        "0".to_owned()
    } else {
        digits.to_owned()
    }
}

fn push_date_representation(options: &mut Vec<DateRepresentation>, option: DateRepresentation) {
    if !options.contains(&option) {
        options.push(option);
    }
}

fn date_representations(text: &str) -> Vec<DateRepresentation> {
    let trimmed = text.trim();
    let tokens = representation_tokens(trimmed);
    let mut options = Vec::new();
    if tokens.is_empty() {
        return options;
    }

    if tokens[0] == "q" && tokens.len() >= 3 {
        if let (Ok(quarter), Some(year)) = (tokens[1].parse::<u8>(), tokens.get(2)) {
            if (1..=4).contains(&quarter) {
                options.push(DateRepresentation::Quarter {
                    quarter,
                    year: representation_year(year),
                });
                return options;
            }
        }
    }

    let numbers: Vec<String> = tokens
        .iter()
        .filter(|token| token.chars().all(|character| character.is_ascii_digit()))
        .cloned()
        .collect();
    let letters: Vec<&str> = tokens
        .iter()
        .filter(|token| {
            token
                .chars()
                .all(|character| character.is_ascii_alphabetic())
        })
        .map(String::as_str)
        .filter(|token| representation_month(token).is_none())
        .filter(|token| {
            !DATE_WEEKDAYS
                .iter()
                .any(|weekday| weekday.starts_with(token))
        })
        .filter(|token| *token != "the")
        .filter(|token| *token != "th" && *token != "st" && *token != "nd" && *token != "rd")
        .collect();

    if numbers.len() == 1 && !letters.is_empty() {
        let era = letters.join("").to_ascii_uppercase();
        if era != "S" {
            options.push(DateRepresentation::Era {
                year: representation_year(&numbers[0]),
                era,
            });
            return options;
        }
    }
    if numbers.len() == 1
        && (trimmed.trim_end_matches('.').ends_with('s')
            || trimmed.trim_end_matches('.').ends_with('S'))
    {
        options.push(DateRepresentation::Period(representation_year(&numbers[0])));
        return options;
    }

    if let Some((month_index, month)) = tokens
        .iter()
        .enumerate()
        .find_map(|(index, token)| representation_month(token).map(|month| (index, month)))
    {
        let before: Vec<&String> = tokens[..month_index]
            .iter()
            .filter(|token| token.chars().all(|character| character.is_ascii_digit()))
            .collect();
        let after: Vec<&String> = tokens[month_index + 1..]
            .iter()
            .filter(|token| token.chars().all(|character| character.is_ascii_digit()))
            .collect();
        if let Some(before_value) = before.last().and_then(|value| value.parse::<u32>().ok()) {
            if before_value <= 31 {
                let day = before_value as u8;
                if let Some(year) = after.last() {
                    push_date_representation(
                        &mut options,
                        DateRepresentation::FullDate {
                            month,
                            day,
                            year: representation_year(year),
                        },
                    );
                } else {
                    push_date_representation(
                        &mut options,
                        DateRepresentation::MonthDay { month, day },
                    );
                }
            } else if let Some(day) = after
                .first()
                .and_then(|value| value.parse::<u8>().ok())
                .filter(|day| (1..=31).contains(day))
            {
                push_date_representation(
                    &mut options,
                    DateRepresentation::FullDate {
                        month,
                        day,
                        year: representation_year(before.last().expect("year before month")),
                    },
                );
            }
            return options;
        }
        match after.as_slice() {
            [only] => {
                if let Ok(value) = only.parse::<u16>() {
                    if value <= 31 {
                        push_date_representation(
                            &mut options,
                            DateRepresentation::MonthDay {
                                month,
                                day: value as u8,
                            },
                        );
                    }
                    push_date_representation(
                        &mut options,
                        DateRepresentation::MonthYear {
                            month,
                            year: representation_year(only),
                        },
                    );
                }
            }
            [day, year, ..] => {
                if let Ok(day) = day.parse::<u8>() {
                    push_date_representation(
                        &mut options,
                        DateRepresentation::FullDate {
                            month,
                            day,
                            year: representation_year(year),
                        },
                    );
                }
            }
            _ => {}
        }
        return options;
    }

    if numbers.is_empty() {
        if let Some(value) = roman_value(trimmed) {
            options.push(DateRepresentation::Year(value.to_string()));
        }
        return options;
    }

    if numbers.len() == 1 {
        options.push(DateRepresentation::Year(representation_year(&numbers[0])));
        return options;
    }

    if numbers.len() == 3 {
        let parsed: Vec<u32> = numbers
            .iter()
            .filter_map(|number| number.parse::<u32>().ok())
            .collect();
        let orders = [(0, 1, 2), (1, 0, 2), (2, 1, 0), (1, 2, 0)];
        for (month_index, day_index, year_index) in orders {
            let (month, day, year) = (parsed[month_index], parsed[day_index], parsed[year_index]);
            if (1..=12).contains(&month) && (1..=31).contains(&day) {
                push_date_representation(
                    &mut options,
                    DateRepresentation::FullDate {
                        month: month as u8,
                        day: day as u8,
                        year: representation_year(&year.to_string()),
                    },
                );
            }
        }
    }
    options
}

fn years_equivalent(left: &str, right: &str) -> bool {
    left == right
        || (left.len() == 4 && right.len() <= 2 && left.ends_with(&format!("{right:0>2}")))
        || (right.len() == 4 && left.len() <= 2 && right.ends_with(&format!("{left:0>2}")))
}

fn date_representation_matches(left: &DateRepresentation, right: &DateRepresentation) -> bool {
    match (left, right) {
        (DateRepresentation::Year(left), DateRepresentation::Year(right))
        | (DateRepresentation::Period(left), DateRepresentation::Period(right)) => {
            years_equivalent(left, right)
        }
        (
            DateRepresentation::Era { year: ly, era: le },
            DateRepresentation::Era { year: ry, era: re },
        ) => years_equivalent(ly, ry) && le == re,
        (
            DateRepresentation::MonthYear {
                month: lm,
                year: ly,
            },
            DateRepresentation::MonthYear {
                month: rm,
                year: ry,
            },
        ) => lm == rm && years_equivalent(ly, ry),
        (
            DateRepresentation::MonthDay { month: lm, day: ld },
            DateRepresentation::MonthDay { month: rm, day: rd },
        ) => lm == rm && ld == rd,
        (
            DateRepresentation::FullDate {
                month: lm,
                day: ld,
                year: ly,
            },
            DateRepresentation::FullDate {
                month: rm,
                day: rd,
                year: ry,
            },
        ) => lm == rm && ld == rd && years_equivalent(ly, ry),
        (
            DateRepresentation::Quarter {
                quarter: lq,
                year: ly,
            },
            DateRepresentation::Quarter {
                quarter: rq,
                year: ry,
            },
        ) => lq == rq && years_equivalent(ly, ry),
        _ => false,
    }
}

fn date_representations_equivalent(canonical: &str, observed: &str) -> bool {
    let canonical = date_representations(canonical);
    let observed = date_representations(observed);
    canonical.iter().any(|left| {
        observed
            .iter()
            .any(|right| date_representation_matches(left, right))
    })
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
        "DATE" => parse_date(text),
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
fn representations_equivalent(kind: &str, canonical: &str, observed: &str) -> PyResult<bool> {
    match kind {
        "CARDINAL" => Ok(cardinal_representations_equivalent(canonical, observed)),
        "DATE" => Ok(date_representations_equivalent(canonical, observed)),
        _ => Err(PyValueError::new_err(
            "representation equivalence is supported only for CARDINAL and DATE",
        )),
    }
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
    module.add_function(wrap_pyfunction!(representations_equivalent, module)?)?;
    module.add_function(wrap_pyfunction!(baseline_normalize_sentence, module)?)?;
    module.add_function(wrap_pyfunction!(tn_normalize, module)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/realizers.rs"]
mod tests;
