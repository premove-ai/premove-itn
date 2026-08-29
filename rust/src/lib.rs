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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimeShape {
    Clock,
    HourDuration,
    MinuteDuration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TimeValue {
    first: u32,
    second: u32,
    third: Option<u32>,
    fraction: Option<String>,
    period: Option<char>,
    timezone: Option<String>,
    shape: TimeShape,
}

fn is_time_zero(word: &str) -> bool {
    matches!(word, "zero" | "oh" | "o" | "nought" | "naught" | "nil")
}

fn parse_time_number(words: &[String]) -> Option<u32> {
    if words.is_empty() {
        return None;
    }
    if words.len() == 1 {
        if is_time_zero(&words[0]) {
            return Some(0);
        }
        if words[0].chars().all(|character| character.is_ascii_digit()) {
            return words[0].parse().ok();
        }
    }
    if words.len() == 2 && is_time_zero(&words[0]) {
        let digit = single_sequence_digit(&words[1])?.to_digit(10)?;
        return Some(digit);
    }
    u32::try_from(cardinal::words_to_number(&words.join(" "))?).ok()
}

fn is_time_tens(word: &str) -> bool {
    matches!(word, "twenty" | "thirty" | "forty" | "fifty")
}

fn is_time_unit(word: &str) -> bool {
    matches!(
        word,
        "hour"
            | "hours"
            | "minute"
            | "minutes"
            | "second"
            | "seconds"
            | "millisecond"
            | "milliseconds"
    )
}

fn parse_time_minute(words: &[String]) -> Option<u32> {
    if words.len() == 1 {
        return parse_time_number(words).filter(|minute| *minute < 60);
    }
    let structurally_valid = words.len() == 2
        && ((is_time_zero(&words[0]) && single_sequence_digit(&words[1]).is_some())
            || (is_time_tens(&words[0])
                && matches!(
                    words[1].as_str(),
                    "one" | "two" | "three" | "four" | "five" | "six" | "seven" | "eight" | "nine"
                )));
    if !structurally_valid {
        return None;
    }
    parse_time_number(words).filter(|minute| *minute < 60)
}

fn valid_clock_hour(hour: u32, period: Option<char>) -> bool {
    match period {
        Some(_) => hour <= 12,
        None => hour <= 24,
    }
}

fn previous_clock_hour(hour: u32, period: Option<char>) -> Option<u32> {
    if period.is_some() {
        Some(if hour == 0 || hour == 1 { 12 } else { hour - 1 })
    } else if hour == 0 {
        Some(23)
    } else if hour == 1 {
        Some(12)
    } else {
        Some(hour - 1)
    }
}

fn format_time_value(value: &TimeValue) -> String {
    let mut output = match value.shape {
        TimeShape::Clock => format!("{:02}:{:02}", value.first, value.second),
        TimeShape::HourDuration => format!(
            "{:02}:{:02}:{:02}",
            value.first,
            value.second,
            value.third.unwrap_or(0)
        ),
        TimeShape::MinuteDuration => {
            format!("{:02}:{:02}", value.first, value.second)
        }
    };
    if let Some(fraction) = &value.fraction {
        output.push('.');
        output.push_str(fraction);
    }
    if let Some(period) = value.period {
        output.push(' ');
        output.push_str(if period == 'a' { "a.m." } else { "p.m." });
    }
    if let Some(timezone) = &value.timezone {
        output.push(' ');
        output.push_str(timezone);
    }
    output
}

fn take_period(words: &mut Vec<String>) -> Option<char> {
    for (suffix, period) in [
        (&["in", "the", "morning"][..], 'a'),
        (&["in", "the", "afternoon"][..], 'p'),
        (&["in", "the", "evening"][..], 'p'),
        (&["in", "morning"][..], 'a'),
        (&["in", "afternoon"][..], 'p'),
        (&["in", "evening"][..], 'p'),
        (&["at", "night"][..], 'p'),
    ] {
        if words.len() >= suffix.len() && words[words.len() - suffix.len()..] == *suffix {
            words.truncate(words.len() - suffix.len());
            return Some(period);
        }
    }

    if words.len() >= 2 && *words.last().unwrap() == "m" {
        let period = match words[words.len() - 2].as_str() {
            "a" => Some('a'),
            "p" => Some('p'),
            _ => None,
        };
        if let Some(period) = period {
            words.truncate(words.len() - 2);
            return Some(period);
        }
    }
    if let Some(last) = words.last_mut() {
        let compact = last.replace('.', "");
        if matches!(compact.as_str(), "am" | "pm") {
            let period = compact.chars().next().expect("two-letter meridiem");
            words.pop();
            return Some(period);
        }
    }
    None
}

fn is_time_word(word: &str) -> bool {
    is_time_zero(word)
        || is_time_tens(word)
        || matches!(
            word,
            "one"
                | "two"
                | "three"
                | "four"
                | "five"
                | "six"
                | "seven"
                | "eight"
                | "nine"
                | "ten"
                | "eleven"
                | "twelve"
                | "thirteen"
                | "fourteen"
                | "fifteen"
                | "sixteen"
                | "seventeen"
                | "eighteen"
                | "nineteen"
                | "sixty"
                | "seventy"
                | "eighty"
                | "ninety"
                | "hundred"
                | "a"
                | "p"
                | "m"
                | "am"
                | "pm"
                | "and"
                | "in"
                | "the"
                | "morning"
                | "afternoon"
                | "evening"
                | "night"
                | "at"
                | "past"
                | "after"
                | "to"
                | "before"
                | "quarter"
                | "half"
                | "clock"
                | "oclock"
                | "o'clock"
        )
        || is_time_unit(word)
}

fn is_timezone_suffix(words: &[String]) -> bool {
    if words.is_empty() {
        return false;
    }
    let mut has_timezone_name = false;
    let mut offset_sign = false;
    for word in words {
        if matches!(word.as_str(), "minus" | "plus") {
            if offset_sign {
                return false;
            }
            offset_sign = true;
            continue;
        }
        if word == "colon" {
            if !offset_sign {
                return false;
            }
            continue;
        }
        if word
            .chars()
            .all(|character| character.is_ascii_alphabetic())
        {
            if !is_time_word(word) {
                has_timezone_name = true;
            }
            continue;
        }
        if word.chars().all(|character| character.is_ascii_digit())
            || parse_time_number(std::slice::from_ref(word)).is_some()
        {
            if !offset_sign {
                return false;
            }
            continue;
        }
        return false;
    }
    has_timezone_name
}

fn format_timezone(words: &[String]) -> String {
    let sign = words
        .iter()
        .position(|word| matches!(word.as_str(), "minus" | "plus"));
    let mut output = String::new();
    let name_end = sign.unwrap_or(words.len());
    for word in &words[..name_end] {
        if word
            .chars()
            .all(|character| character.is_ascii_alphabetic())
        {
            output.push_str(&word.to_ascii_uppercase());
        }
    }
    let Some(sign) = sign else {
        return output;
    };
    output.push(if words[sign] == "minus" { '-' } else { '+' });
    let offset = &words[sign + 1..];
    if let Some(colon) = offset.iter().position(|word| word == "colon") {
        if let (Some(hours), Some(minutes)) = (
            parse_time_number(&offset[..colon]),
            parse_time_number(&offset[colon + 1..]),
        ) {
            output.push_str(&hours.to_string());
            output.push(':');
            output.push_str(&format!("{minutes:02}"));
            return output;
        }
    }
    if let Some(value) = parse_time_number(offset) {
        output.push_str(&value.to_string());
    } else {
        for word in offset {
            if let Some(value) = parse_time_number(std::slice::from_ref(word)) {
                output.push_str(&value.to_string());
            }
        }
    }
    output
}

fn parse_time_duration(
    words: &[String],
    period: Option<char>,
    timezone: Option<String>,
) -> Option<TimeValue> {
    let mut cursor = 0;
    let mut values: [Option<u32>; 4] = [None, None, None, None];
    let mut last_unit = None;
    while cursor < words.len() {
        if words[cursor] == "and" {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < words.len() && !is_time_unit(&words[cursor]) {
            cursor += 1;
        }
        if cursor == words.len() {
            return None;
        }
        let value = parse_time_number(&words[start..cursor])?;
        let (unit, slot) = match words[cursor].as_str() {
            "hour" | "hours" => (0, 0),
            "minute" | "minutes" => (1, 1),
            "second" | "seconds" => (2, 2),
            "millisecond" | "milliseconds" => (3, 3),
            _ => return None,
        };
        if last_unit.is_some_and(|previous| unit <= previous) {
            return None;
        }
        values[slot] = Some(value);
        last_unit = Some(unit);
        cursor += 1;
    }
    if values.iter().all(Option::is_none) {
        return None;
    }
    let minutes = values[1].unwrap_or(0);
    let seconds = values[2].unwrap_or(0);
    if minutes >= 60 || seconds >= 60 || values[3].is_some_and(|value| value >= 1000) {
        return None;
    }
    let (shape, first, second, third) = if let Some(hours) = values[0] {
        (TimeShape::HourDuration, hours, minutes, Some(seconds))
    } else {
        (TimeShape::MinuteDuration, minutes, seconds, None)
    };
    Some(TimeValue {
        first,
        second,
        third,
        fraction: values[3].map(|value| value.to_string()),
        period,
        timezone,
        shape,
    })
}

fn parse_relative_time(
    words: &[String],
    period: Option<char>,
    timezone: Option<String>,
) -> Option<TimeValue> {
    let relation = words
        .iter()
        .position(|word| matches!(word.as_str(), "past" | "after" | "to" | "before"))?;
    if relation == 0 || relation + 1 >= words.len() {
        return None;
    }
    let mut minute_words = words[..relation].to_vec();
    if minute_words.len() >= 2
        && matches!(
            minute_words.last().map(String::as_str),
            Some("minute" | "minutes")
        )
    {
        minute_words.pop();
    }
    if minute_words.first().is_some_and(|word| word == "a") {
        minute_words.remove(0);
    }
    let minutes = match minute_words.first().map(String::as_str) {
        Some("quarter") => 15,
        Some("half") => 30,
        _ => parse_time_number(&minute_words)?,
    };
    if minutes >= 60 {
        return None;
    }
    let hour = parse_time_number(&words[relation + 1..])?;
    if !valid_clock_hour(hour, period) {
        return None;
    }
    let (hour, minute) = if matches!(words[relation].as_str(), "to" | "before") {
        (previous_clock_hour(hour, period)?, 60 - minutes)
    } else {
        (hour, minutes)
    };
    if minute >= 60 {
        return None;
    }
    Some(TimeValue {
        first: hour,
        second: minute,
        third: None,
        fraction: None,
        period,
        timezone,
        shape: TimeShape::Clock,
    })
}

fn parse_clock_core(
    words: &[String],
    period: Option<char>,
    timezone: Option<String>,
) -> Option<TimeValue> {
    if words.len() >= 2 && words.last().is_some_and(|word| word == "hundred") {
        let hour = parse_time_number(&words[..words.len() - 1])?;
        if valid_clock_hour(hour, period) {
            return Some(TimeValue {
                first: hour,
                second: 0,
                third: None,
                fraction: None,
                period,
                timezone,
                shape: TimeShape::Clock,
            });
        }
    }

    let oclock_start = if words
        .last()
        .is_some_and(|word| word == "oclock" || word == "o'clock")
    {
        Some(words.len() - 1)
    } else if words.len() >= 2
        && words[words.len() - 2] == "o"
        && words.last().is_some_and(|word| word == "clock")
    {
        Some(words.len() - 2)
    } else {
        None
    };
    if let Some(start) = oclock_start {
        let hour = parse_time_number(&words[..start])?;
        if valid_clock_hour(hour, period) {
            return Some(TimeValue {
                first: hour,
                second: 0,
                third: None,
                fraction: None,
                period,
                timezone,
                shape: TimeShape::Clock,
            });
        }
    }

    if words.len() == 1 {
        if period.is_none() && timezone.is_none() {
            return None;
        }
        let hour = parse_time_number(words)?;
        return valid_clock_hour(hour, period).then_some(TimeValue {
            first: hour,
            second: 0,
            third: None,
            fraction: None,
            period,
            timezone,
            shape: TimeShape::Clock,
        });
    }

    for split in 1..words.len() {
        let Some(hour) = parse_time_number(&words[..split]) else {
            continue;
        };
        if !valid_clock_hour(hour, period) {
            continue;
        }
        let Some(minute) = parse_time_minute(&words[split..]) else {
            continue;
        };
        return Some(TimeValue {
            first: hour,
            second: minute,
            third: None,
            fraction: None,
            period,
            timezone,
            shape: TimeShape::Clock,
        });
    }
    None
}

fn parse_time_core(words: &[String], timezone: Option<String>) -> Option<String> {
    let mut words = words.to_vec();
    let period = take_period(&mut words);
    if words.len() == 1 {
        match words[0].as_str() {
            "midnight" => {
                return Some(format_time_value(&TimeValue {
                    first: 0,
                    second: 0,
                    third: None,
                    fraction: None,
                    period: None,
                    timezone,
                    shape: TimeShape::Clock,
                }))
            }
            "noon" => {
                return Some(format_time_value(&TimeValue {
                    first: 12,
                    second: 0,
                    third: None,
                    fraction: None,
                    period: None,
                    timezone,
                    shape: TimeShape::Clock,
                }))
            }
            _ => {}
        }
    }
    if let Some(value) = parse_time_duration(&words, period, timezone.clone()) {
        return Some(format_time_value(&value));
    }
    if let Some(value) = parse_relative_time(&words, period, timezone.clone()) {
        return Some(format_time_value(&value));
    }
    parse_clock_core(&words, period, timezone).map(|value| format_time_value(&value))
}

fn parse_spoken_time(text: &str) -> Option<String> {
    let lowered = text.to_ascii_lowercase();
    let words: Vec<String> = lowered.split_whitespace().map(ToOwned::to_owned).collect();
    if words.is_empty() {
        return None;
    }

    for split in (1..words.len()).rev() {
        if !is_timezone_suffix(&words[split..]) {
            continue;
        }
        let timezone = format_timezone(&words[split..]);
        if let Some(result) = parse_time_core(&words[..split], Some(timezone)) {
            return Some(result);
        }
    }
    parse_time_core(&words, None)
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum TimeRepresentationToken {
    Digits(String),
    Alphabetic(String),
    Symbol(char),
}

fn time_representation_tokens(text: &str) -> Vec<TimeRepresentationToken> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_kind = None;
    let flush = |tokens: &mut Vec<TimeRepresentationToken>, current: &mut String, kind| {
        if current.is_empty() {
            return;
        }
        let value = std::mem::take(current);
        tokens.push(match kind {
            Some(true) => TimeRepresentationToken::Digits(value),
            Some(false) => TimeRepresentationToken::Alphabetic(value),
            None => unreachable!("a token always has a kind"),
        });
    };

    for character in text.trim().to_ascii_lowercase().chars() {
        if character.is_ascii_digit() || character.is_ascii_alphabetic() {
            let kind = character.is_ascii_digit();
            if current_kind.is_some_and(|previous| previous != kind) {
                flush(&mut tokens, &mut current, current_kind);
            }
            current_kind = Some(kind);
            current.push(character);
        } else {
            flush(&mut tokens, &mut current, current_kind);
            current_kind = None;
            if !character.is_ascii_whitespace() {
                tokens.push(TimeRepresentationToken::Symbol(character));
            }
        }
    }
    flush(&mut tokens, &mut current, current_kind);
    tokens
}

fn time_timezone_value(tokens: &[TimeRepresentationToken]) -> Option<String> {
    let mut value = String::new();
    for token in tokens {
        match token {
            TimeRepresentationToken::Alphabetic(text) => value.push_str(&text.to_ascii_uppercase()),
            TimeRepresentationToken::Digits(text) => value.push_str(text),
            TimeRepresentationToken::Symbol('-' | '+' | ':') => value.push(match token {
                TimeRepresentationToken::Symbol(symbol) => *symbol,
                _ => unreachable!("matched symbol"),
            }),
            TimeRepresentationToken::Symbol(_) => return None,
        }
    }
    (!value.is_empty()).then_some(value)
}

fn normalize_timezone_value(value: &str) -> String {
    let mut output = String::new();
    let mut offset = String::new();
    let mut in_offset = false;
    for character in value.chars() {
        if character == '-' || character == '+' {
            in_offset = true;
            output.push(character);
        } else if in_offset {
            offset.push(character);
        } else if character.is_ascii_alphabetic() {
            output.push(character.to_ascii_uppercase());
        }
    }
    if !offset.is_empty() {
        let normalized = if let Some((hours, minutes)) = offset.split_once(':') {
            let hours = hours.trim_start_matches('0');
            let hours = if hours.is_empty() { "0" } else { hours };
            if minutes.chars().all(|character| character == '0') {
                hours.to_owned()
            } else {
                format!("{hours}:{}", minutes.trim_start_matches('0'))
            }
        } else {
            let value = offset.trim_start_matches('0');
            if value.is_empty() {
                "0".to_owned()
            } else {
                value.to_owned()
            }
        };
        output.push_str(&normalized);
    }
    output
}

fn normalized_time_fraction(value: Option<&str>) -> String {
    let Some(value) = value else {
        return "0".to_owned();
    };
    let value = value.trim_start_matches('0').trim_end_matches('0');
    if value.is_empty() {
        "0".to_owned()
    } else {
        value.to_owned()
    }
}

fn normalize_time_clock_hour(hour: u32, period: Option<char>) -> u32 {
    match period {
        Some('a') if hour == 12 => 0,
        Some('p') if hour < 12 => hour + 12,
        _ => hour,
    }
}

fn time_representation(text: &str) -> Option<TimeValue> {
    let tokens = time_representation_tokens(text);
    if tokens.is_empty() {
        return None;
    }
    let mut period_indices = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if matches!(
            token,
            TimeRepresentationToken::Alphabetic(value) if matches!(value.as_str(), "am" | "pm")
        ) {
            period_indices.push(index);
            continue;
        }
        let TimeRepresentationToken::Alphabetic(value) = token else {
            continue;
        };
        if !matches!(value.as_str(), "a" | "p") {
            continue;
        }
        let mut next = index + 1;
        while matches!(tokens.get(next), Some(TimeRepresentationToken::Symbol('.'))) {
            next += 1;
        }
        if matches!(
            tokens.get(next),
            Some(TimeRepresentationToken::Alphabetic(marker)) if marker == "m"
        ) {
            period_indices.push(index);
            period_indices.push(next);
        }
    }
    let period = period_indices
        .iter()
        .find_map(|index| match &tokens[*index] {
            TimeRepresentationToken::Alphabetic(value) => {
                Some(value.chars().next().expect("period"))
            }
            _ => None,
        });
    let first_digit = tokens
        .iter()
        .position(|token| matches!(token, TimeRepresentationToken::Digits(_)))?;
    let timezone_start = tokens
        .iter()
        .enumerate()
        .skip(first_digit)
        .find_map(|(index, token)| match token {
            TimeRepresentationToken::Alphabetic(_) if !period_indices.contains(&index) => {
                Some(index)
            }
            _ => None,
        });
    let time_end = timezone_start.unwrap_or(tokens.len());
    let time_tokens = &tokens[..time_end];
    let numbers: Vec<String> = time_tokens
        .iter()
        .filter_map(|token| match token {
            TimeRepresentationToken::Digits(value) => Some(value.clone()),
            _ => None,
        })
        .collect();
    let digit_indices: Vec<usize> = time_tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            matches!(token, TimeRepresentationToken::Digits(_)).then_some(index)
        })
        .collect();
    let separators: Vec<Vec<char>> = digit_indices
        .windows(2)
        .map(|pair| {
            time_tokens[pair[0] + 1..pair[1]]
                .iter()
                .filter_map(|token| match token {
                    TimeRepresentationToken::Symbol(symbol) => Some(*symbol),
                    _ => None,
                })
                .collect()
        })
        .collect();
    let timezone = timezone_start.and_then(|start| time_timezone_value(&tokens[start..]));
    let parsed_numbers: Vec<u32> = numbers
        .iter()
        .map(|number| number.parse().ok())
        .collect::<Option<_>>()?;
    let first = *parsed_numbers.first()?;
    if numbers.len() == 1 && matches!(numbers[0].len(), 3 | 4) {
        let split = numbers[0].len() - 2;
        let hour = numbers[0][..split].parse::<u32>().ok()?;
        let minute = numbers[0][split..].parse::<u32>().ok()?;
        if hour <= 24 && minute < 60 {
            let mut result = TimeValue {
                first: hour,
                second: minute,
                third: None,
                fraction: None,
                period,
                timezone,
                shape: TimeShape::Clock,
            };
            result.first = normalize_time_clock_hour(result.first, result.period);
            return Some(result);
        }
    }
    let (shape, second, third, fraction) = match parsed_numbers.as_slice() {
        [first, second] if *second < 60 && *first <= 24 => (TimeShape::Clock, *second, None, None),
        [first] if *first <= 24 => (TimeShape::Clock, 0, None, None),
        [first, second, _fraction]
            if *second < 60
                && separators.len() == 2
                && separators[0].contains(&':')
                && separators[1].contains(&'.') =>
        {
            (
                TimeShape::MinuteDuration,
                *second,
                None,
                Some(numbers[2].clone()),
            )
        }
        [first, second, third]
            if *second < 60
                && *third < 60
                && separators.len() == 2
                && separators.iter().all(|separator| separator.contains(&':')) =>
        {
            (TimeShape::HourDuration, *second, Some(*third), None)
        }
        _ => return None,
    };
    let mut result = TimeValue {
        first,
        second,
        third,
        fraction,
        period,
        timezone,
        shape,
    };
    if result.shape == TimeShape::Clock {
        result.first = normalize_time_clock_hour(result.first, result.period);
    }
    Some(result)
}

fn time_representations_equivalent(canonical: &str, observed: &str) -> bool {
    let Some(left) = time_representation(canonical) else {
        return false;
    };
    let Some(right) = time_representation(observed) else {
        return false;
    };
    if left.shape != right.shape
        || left.first != right.first
        || left.second != right.second
        || left.third != right.third
        || normalized_time_fraction(left.fraction.as_deref())
            != normalized_time_fraction(right.fraction.as_deref())
    {
        return false;
    }
    match (left.timezone.as_deref(), right.timezone.as_deref()) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            normalize_timezone_value(left) == normalize_timezone_value(right)
        }
        _ => false,
    }
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
        "TIME" => Ok(time_representations_equivalent(canonical, observed)),
        _ => Err(PyValueError::new_err(
            "representation equivalence is supported only for CARDINAL, DATE, and TIME",
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
