use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use text_processing_rs::itn::en::{
    cardinal, date, electronic, measure, money, ordinal, punctuation, telephone, time, whitelist,
    word,
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

fn phone_input_is_complete(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|word| word.to_ascii_lowercase())
        .collect();
    if words.is_empty() {
        return false;
    }
    if words.iter().enumerate().any(|(index, word)| {
        let known_digit = word.chars().all(|character| character.is_ascii_digit())
            || (single_sequence_digit(word).is_some()
                && !matches!(word.as_str(), "nought" | "naught" | "nil"));
        let known_number = cardinal::words_to_number(word).is_some();
        let control = match word.as_str() {
            "plus" => index == 0,
            "ssn" => true,
            "is" => index > 0 && words[index - 1] == "ssn",
            "double" | "triple" => words
                .get(index + 1)
                .is_some_and(|next| single_sequence_digit(next).is_some()),
            "dot" => index > 0 && index + 1 < words.len(),
            _ => false,
        };
        !(known_digit
            || known_number
            || control
            || (word.len() == 1
                && word
                    .chars()
                    .all(|character| character.is_ascii_alphabetic())))
    }) {
        return false;
    }
    true
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

const TIMEZONE_NAMES: &[&str] = &[
    "ut", "utc", "gmt", "z", "et", "ct", "mt", "pt", "est", "edt", "cst", "cdt", "mst", "mdt",
    "pst", "pdt", "ast", "adt", "nst", "ndt", "akst", "akdt", "hst", "hdt", "lint", "cet", "cest",
    "eet", "eest", "wet", "west", "bst", "jst", "kst", "wst", "chst", "aest", "aedt", "acst",
    "acdt", "awst", "nzst", "nzdt", "hast", "hadt", "sst", "gst", "sgt", "hkt", "msk", "myt",
    "ist", "irst", "irdt", "pkt", "wat", "pht", "met", "uzt",
];

fn is_timezone_suffix(words: &[String]) -> bool {
    if words.is_empty() {
        return false;
    }
    let offset_start = words
        .iter()
        .position(|word| matches!(word.as_str(), "minus" | "plus"));
    let name_end = offset_start.unwrap_or(words.len());
    if name_end == 0
        || words[..name_end].iter().any(|word| {
            !word
                .chars()
                .all(|character| character.is_ascii_alphabetic())
        })
    {
        return false;
    }
    let name = words[..name_end].join("");
    if !TIMEZONE_NAMES.contains(&name.as_str()) {
        return false;
    }
    let Some(offset_start) = offset_start else {
        return true;
    };
    let offset = &words[offset_start + 1..];
    if offset.is_empty() {
        return false;
    }
    if let Some(colon) = offset.iter().position(|word| word == "colon") {
        if offset[colon + 1..].iter().any(|word| word == "colon") {
            return false;
        }
        return parse_time_number(&offset[..colon])
            .zip(parse_time_number(&offset[colon + 1..]))
            .is_some_and(|(hours, minutes)| hours <= 23 && minutes < 60);
    }
    parse_time_number(offset).is_some_and(|hours| hours <= 23)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MoneyCurrency {
    code: &'static str,
    display: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MoneyAlias {
    phrase: &'static str,
    currency: MoneyCurrency,
    minor: bool,
}

const MONEY_ALIASES: &[MoneyAlias] = &[
    // Common currencies and their minor units.
    MoneyAlias {
        phrase: "united states dollars",
        currency: MoneyCurrency {
            code: "USD",
            display: "$",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "united states dollar",
        currency: MoneyCurrency {
            code: "USD",
            display: "$",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "dollars",
        currency: MoneyCurrency {
            code: "USD",
            display: "$",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "dollar",
        currency: MoneyCurrency {
            code: "USD",
            display: "$",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "united states cents",
        currency: MoneyCurrency {
            code: "USD",
            display: "$",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "cents",
        currency: MoneyCurrency {
            code: "USD",
            display: "$",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "cent",
        currency: MoneyCurrency {
            code: "USD",
            display: "$",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "pounds",
        currency: MoneyCurrency {
            code: "GBP",
            display: "£",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "pound",
        currency: MoneyCurrency {
            code: "GBP",
            display: "£",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "euros",
        currency: MoneyCurrency {
            code: "EUR",
            display: "€",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "euro",
        currency: MoneyCurrency {
            code: "EUR",
            display: "€",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "pence",
        currency: MoneyCurrency {
            code: "GBP",
            display: "£",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "penny",
        currency: MoneyCurrency {
            code: "GBP",
            display: "£",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "rupees",
        currency: MoneyCurrency {
            code: "INR",
            display: "Rs",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "rupee",
        currency: MoneyCurrency {
            code: "INR",
            display: "Rs",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "indian rupees",
        currency: MoneyCurrency {
            code: "INR",
            display: "Rs",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "indian rupee",
        currency: MoneyCurrency {
            code: "INR",
            display: "Rs",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "paise",
        currency: MoneyCurrency {
            code: "INR",
            display: "Rs",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "yen",
        currency: MoneyCurrency {
            code: "JPY",
            display: "¥",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "japanese yen",
        currency: MoneyCurrency {
            code: "JPY",
            display: "¥",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "sen",
        currency: MoneyCurrency {
            code: "JPY",
            display: "¥",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "won",
        currency: MoneyCurrency {
            code: "KRW",
            display: "KRW",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "korean won",
        currency: MoneyCurrency {
            code: "KRW",
            display: "KRW",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "yuan",
        currency: MoneyCurrency {
            code: "CNY",
            display: "CNY",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "chinese yuan",
        currency: MoneyCurrency {
            code: "CNY",
            display: "CNY",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "reals",
        currency: MoneyCurrency {
            code: "BRL",
            display: "R$",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "real",
        currency: MoneyCurrency {
            code: "BRL",
            display: "R$",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "brazilian reals",
        currency: MoneyCurrency {
            code: "BRL",
            display: "R$",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "centavos",
        currency: MoneyCurrency {
            code: "BRL",
            display: "R$",
        },
        minor: true,
    },
    // ISO-style names occurring in the Google corpus.
    MoneyAlias {
        phrase: "norwegian kroner",
        currency: MoneyCurrency {
            code: "NOK",
            display: "NOK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "norwegian krone",
        currency: MoneyCurrency {
            code: "NOK",
            display: "NOK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "danish kroner",
        currency: MoneyCurrency {
            code: "DKK",
            display: "DKK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "danish krone",
        currency: MoneyCurrency {
            code: "DKK",
            display: "DKK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "swedish kronor",
        currency: MoneyCurrency {
            code: "SEK",
            display: "SEK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "swedish krona",
        currency: MoneyCurrency {
            code: "SEK",
            display: "SEK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "swiss francs",
        currency: MoneyCurrency {
            code: "CHF",
            display: "CHF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "swiss franc",
        currency: MoneyCurrency {
            code: "CHF",
            display: "CHF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "german marks",
        currency: MoneyCurrency {
            code: "DEM",
            display: "DM",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "german mark",
        currency: MoneyCurrency {
            code: "DEM",
            display: "DM",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "marks",
        currency: MoneyCurrency {
            code: "DEM",
            display: "DM",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "philippine pesos",
        currency: MoneyCurrency {
            code: "PHP",
            display: "PHP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "argentine pesos",
        currency: MoneyCurrency {
            code: "ARS",
            display: "ARS",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "chilean pesos",
        currency: MoneyCurrency {
            code: "CLP",
            display: "CLP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "mexican pesos",
        currency: MoneyCurrency {
            code: "MXN",
            display: "MXN",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "pesos",
        currency: MoneyCurrency {
            code: "DOP",
            display: "DOP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "polish zlotys",
        currency: MoneyCurrency {
            code: "PLN",
            display: "PLN",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "zlotys",
        currency: MoneyCurrency {
            code: "PLN",
            display: "ZL",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "zloty",
        currency: MoneyCurrency {
            code: "PLN",
            display: "ZL",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "czech korunas",
        currency: MoneyCurrency {
            code: "CZK",
            display: "CZK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "czech koruna",
        currency: MoneyCurrency {
            code: "CZK",
            display: "CZK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "slovak korunas",
        currency: MoneyCurrency {
            code: "SKK",
            display: "SKK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "hungarian forints",
        currency: MoneyCurrency {
            code: "HUF",
            display: "HUF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "azerbaijani manats",
        currency: MoneyCurrency {
            code: "AZN",
            display: "AZN",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "turkmenistani manats",
        currency: MoneyCurrency {
            code: "TMT",
            display: "TMT",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "malaysian ringgit",
        currency: MoneyCurrency {
            code: "MYR",
            display: "MYR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "thai bahts",
        currency: MoneyCurrency {
            code: "THB",
            display: "THB",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "vietnamese dongs",
        currency: MoneyCurrency {
            code: "VND",
            display: "VND",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "indonesian rupiahs",
        currency: MoneyCurrency {
            code: "IDR",
            display: "IDR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "sri lanka rupees",
        currency: MoneyCurrency {
            code: "LKR",
            display: "LKR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "pakistani rupees",
        currency: MoneyCurrency {
            code: "PKR",
            display: "PKR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "pakistani rupee",
        currency: MoneyCurrency {
            code: "PKR",
            display: "PKR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "nepalese rupees",
        currency: MoneyCurrency {
            code: "NPR",
            display: "NPR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "nepalese rupee",
        currency: MoneyCurrency {
            code: "NPR",
            display: "NPR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "bangladeshi takas",
        currency: MoneyCurrency {
            code: "BDT",
            display: "BDT",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "takas",
        currency: MoneyCurrency {
            code: "BDT",
            display: "BDT",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "united arab emirates dirhams",
        currency: MoneyCurrency {
            code: "AED",
            display: "AED",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "saudi riyals",
        currency: MoneyCurrency {
            code: "SAR",
            display: "SAR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "qatari rials",
        currency: MoneyCurrency {
            code: "QAR",
            display: "QAR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "iranian rials",
        currency: MoneyCurrency {
            code: "IRR",
            display: "IRR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "omani rials",
        currency: MoneyCurrency {
            code: "OMR",
            display: "OMR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "israeli new sheqels",
        currency: MoneyCurrency {
            code: "ILS",
            display: "ILS",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "lebanese pounds",
        currency: MoneyCurrency {
            code: "LBP",
            display: "LBP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "lebanese pound",
        currency: MoneyCurrency {
            code: "LBP",
            display: "LBP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "egyptian pounds",
        currency: MoneyCurrency {
            code: "EGP",
            display: "EGP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "sudanese pounds",
        currency: MoneyCurrency {
            code: "SDG",
            display: "SDG",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "south sudanese pounds",
        currency: MoneyCurrency {
            code: "SSP",
            display: "SSP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "australian dollars",
        currency: MoneyCurrency {
            code: "AUD",
            display: "AUD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "australian dollar",
        currency: MoneyCurrency {
            code: "AUD",
            display: "AUD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "canadian dollars",
        currency: MoneyCurrency {
            code: "CAD",
            display: "CAD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "canadian dollar",
        currency: MoneyCurrency {
            code: "CAD",
            display: "CAD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "new zealand dollars",
        currency: MoneyCurrency {
            code: "NZD",
            display: "NZD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "new taiwan dollars",
        currency: MoneyCurrency {
            code: "TWD",
            display: "TWD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "hong kong dollars",
        currency: MoneyCurrency {
            code: "HKD",
            display: "HKD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "hong kong dollar",
        currency: MoneyCurrency {
            code: "HKD",
            display: "HKD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "singapore dollars",
        currency: MoneyCurrency {
            code: "SGD",
            display: "SGD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "bermudian dollars",
        currency: MoneyCurrency {
            code: "BMD",
            display: "BMD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "bahamian dollars",
        currency: MoneyCurrency {
            code: "BSD",
            display: "BSD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "namibian dollars",
        currency: MoneyCurrency {
            code: "NAD",
            display: "NAD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "south african rands",
        currency: MoneyCurrency {
            code: "ZAR",
            display: "ZAR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "maldivian rufiyaas",
        currency: MoneyCurrency {
            code: "MVR",
            display: "MVR",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "botswana pulas",
        currency: MoneyCurrency {
            code: "BWP",
            display: "BWP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "gambian dalasis",
        currency: MoneyCurrency {
            code: "GMD",
            display: "GMD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "kenyan shillings",
        currency: MoneyCurrency {
            code: "KES",
            display: "KES",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "tanzanian shillings",
        currency: MoneyCurrency {
            code: "TZS",
            display: "TZS",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "ugandan shillings",
        currency: MoneyCurrency {
            code: "UGX",
            display: "UGX",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "nigerian nairas",
        currency: MoneyCurrency {
            code: "NGN",
            display: "NGN",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "ethiopian birrs",
        currency: MoneyCurrency {
            code: "ETB",
            display: "ETB",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "rwandan francs",
        currency: MoneyCurrency {
            code: "RWF",
            display: "RWF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "rwandan franc",
        currency: MoneyCurrency {
            code: "RWF",
            display: "RWF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "guinean francs",
        currency: MoneyCurrency {
            code: "GNF",
            display: "GNF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "guinean franc",
        currency: MoneyCurrency {
            code: "GNF",
            display: "GNF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "congolese francs",
        currency: MoneyCurrency {
            code: "CDF",
            display: "CDF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "djiboutian francs",
        currency: MoneyCurrency {
            code: "DJF",
            display: "DJF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "belgian francs",
        currency: MoneyCurrency {
            code: "BEF",
            display: "BEF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "belgian franc",
        currency: MoneyCurrency {
            code: "BEF",
            display: "BEF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "french francs",
        currency: MoneyCurrency {
            code: "FRF",
            display: "FRF",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "portuguese escudos",
        currency: MoneyCurrency {
            code: "PTE",
            display: "PTE",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "cape verde escudos",
        currency: MoneyCurrency {
            code: "CVE",
            display: "CVE",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "spanish pesetas",
        currency: MoneyCurrency {
            code: "ESP",
            display: "ESP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "italian liras",
        currency: MoneyCurrency {
            code: "ITL",
            display: "ITL",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "maltese liras",
        currency: MoneyCurrency {
            code: "MTL",
            display: "MTL",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "vatican liras",
        currency: MoneyCurrency {
            code: "VAL",
            display: "VAL",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "sammarinese liras",
        currency: MoneyCurrency {
            code: "SML",
            display: "SML",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "greek drachmas",
        currency: MoneyCurrency {
            code: "GRD",
            display: "GRD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "austrian schillings",
        currency: MoneyCurrency {
            code: "ATS",
            display: "ATS",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "croatian kunas",
        currency: MoneyCurrency {
            code: "HRK",
            display: "HRK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "bulgarian levs",
        currency: MoneyCurrency {
            code: "BGN",
            display: "BGN",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "serbian dinars",
        currency: MoneyCurrency {
            code: "RSD",
            display: "RSD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "ukrainian hryvnias",
        currency: MoneyCurrency {
            code: "UAH",
            display: "UAH",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "lithuanian litass",
        currency: MoneyCurrency {
            code: "LTL",
            display: "LTL",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "latvian latss",
        currency: MoneyCurrency {
            code: "LVL",
            display: "LVL",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "icelandic kronur",
        currency: MoneyCurrency {
            code: "ISK",
            display: "ISK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "finnish markkas",
        currency: MoneyCurrency {
            code: "FIM",
            display: "FIM",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "irish pounds",
        currency: MoneyCurrency {
            code: "IEP",
            display: "IEP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "british pounds",
        currency: MoneyCurrency {
            code: "GBP",
            display: "£",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "british pound",
        currency: MoneyCurrency {
            code: "GBP",
            display: "£",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "british pence",
        currency: MoneyCurrency {
            code: "GBP",
            display: "£",
        },
        minor: true,
    },
    MoneyAlias {
        phrase: "cypriot pounds",
        currency: MoneyCurrency {
            code: "CYP",
            display: "CYP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "cypriot pound",
        currency: MoneyCurrency {
            code: "CYP",
            display: "CYP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "saint helena pounds",
        currency: MoneyCurrency {
            code: "SHP",
            display: "SHP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "saint helena pound",
        currency: MoneyCurrency {
            code: "SHP",
            display: "SHP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "solomon islands dollars",
        currency: MoneyCurrency {
            code: "SBD",
            display: "SBD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "costa rican colons",
        currency: MoneyCurrency {
            code: "CRC",
            display: "CRC",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "costa rican colon",
        currency: MoneyCurrency {
            code: "CRC",
            display: "CRC",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "aruban florins",
        currency: MoneyCurrency {
            code: "AWG",
            display: "AWG",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "aruban florin",
        currency: MoneyCurrency {
            code: "AWG",
            display: "AWG",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "macanese patacas",
        currency: MoneyCurrency {
            code: "MOP",
            display: "MOP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "macanese pataca",
        currency: MoneyCurrency {
            code: "MOP",
            display: "MOP",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "papua new guinean kina",
        currency: MoneyCurrency {
            code: "PGK",
            display: "PGK",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "fiji dollars",
        currency: MoneyCurrency {
            code: "FJD",
            display: "FJD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "brunei dollars",
        currency: MoneyCurrency {
            code: "BND",
            display: "BND",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "jamaican dollars",
        currency: MoneyCurrency {
            code: "JMD",
            display: "JMD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "barbados dollars",
        currency: MoneyCurrency {
            code: "BBD",
            display: "BBD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "east caribbean dollars",
        currency: MoneyCurrency {
            code: "XCD",
            display: "XCD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "tobago dollars",
        currency: MoneyCurrency {
            code: "TTD",
            display: "TTD",
        },
        minor: false,
    },
    MoneyAlias {
        phrase: "surinamese dollars",
        currency: MoneyCurrency {
            code: "SRD",
            display: "SRD",
        },
        minor: false,
    },
];

macro_rules! money_alias {
    ($phrase:literal, $code:literal, $display:literal, $minor:literal) => {
        MoneyAlias {
            phrase: $phrase,
            currency: MoneyCurrency {
                code: $code,
                display: $display,
            },
            minor: $minor,
        }
    };
}

// Less common names found in the Google corpus. Keeping these as aliases
// lets the parser remain generic while sharing the same currency identity
// with the evaluation normalizer below.
const MONEY_EXTRA_ALIASES: &[MoneyAlias] = &[
    money_alias!(
        "bosnia and herzegovina convertible marks",
        "BAM",
        "BAM",
        false
    ),
    money_alias!("convertible marks", "BAM", "BAM", false),
    money_alias!("moldovan leus", "MDL", "MDL", false),
    money_alias!("moldovan leu", "MDL", "MDL", false),
    money_alias!("seychelles rupees", "SCR", "SCR", false),
    money_alias!("seychelles rupee", "SCR", "SCR", false),
    money_alias!("lao kips", "LAK", "LAK", false),
    money_alias!("lao kip", "LAK", "LAK", false),
    money_alias!("mongolian tugriks", "MNT", "MNT", false),
    money_alias!("mongolian tugrik", "MNT", "MNT", false),
    money_alias!("sierra leonean leones", "SLL", "SLL", false),
    money_alias!("sierra leonean leone", "SLL", "SLL", false),
    money_alias!("cambodian riels", "KHR", "KHR", false),
    money_alias!("cambodian riel", "KHR", "KHR", false),
    money_alias!("afghan afghanis", "AFN", "AFN", false),
    money_alias!("afghan afghani", "AFN", "AFN", false),
    money_alias!("burundian francs", "BIF", "BIF", false),
    money_alias!("burundian franc", "BIF", "BIF", false),
    money_alias!("eritrean nakfa", "ERN", "ERN", false),
    money_alias!("macedonian denars", "MKD", "MKD", false),
    money_alias!("macedonian denar", "MKD", "MKD", false),
    money_alias!("angolan kwanzas", "AOA", "AOA", false),
    money_alias!("angolan kwanza", "AOA", "AOA", false),
    money_alias!("tunisian dinars", "TND", "TND", false),
    money_alias!("tunisian dinar", "TND", "TND", false),
    money_alias!("libyan dinars", "LYD", "LYD", false),
    money_alias!("libyan dinar", "LYD", "LYD", false),
    money_alias!("kuwaiti dinars", "KWD", "KWD", false),
    money_alias!("kuwaiti dinar", "KWD", "KWD", false),
    money_alias!("venezuelan bolivar fuertes", "VEF", "VEF", false),
    money_alias!("estonian kroon", "EEK", "EEK", false),
    money_alias!("samoan talas", "WST", "WST", false),
    money_alias!("samoan tala", "WST", "WST", false),
    money_alias!("honduran lempiras", "HNL", "HNL", false),
    money_alias!("honduran lempira", "HNL", "HNL", false),
    money_alias!("myanma kyats", "MMK", "MMK", false),
    money_alias!("myanma kyat", "MMK", "MMK", false),
    money_alias!("panamanian balboas", "PAB", "PAB", false),
    money_alias!("panamanian balboa", "PAB", "PAB", false),
    money_alias!("mauritanian ouguiyas", "MRO", "MRO", false),
    money_alias!("mauritanian ouguiya", "MRO", "MRO", false),
    money_alias!("belarusian rubles", "BYR", "BYR", false),
    money_alias!("belarusian ruble", "BYR", "BYR", false),
    money_alias!("slovenian tolar", "SIT", "SIT", false),
    money_alias!("ghanaian cedis", "GHS", "GHS", false),
    money_alias!("ghanaian cedi", "GHS", "GHS", false),
    money_alias!("uzbekistan soms", "UZS", "UZS", false),
    money_alias!("uzbekistan som", "UZS", "UZS", false),
    money_alias!("kazakhstani tenges", "KZT", "KZT", false),
    money_alias!("kazakhstani tenge", "KZT", "KZT", false),
    money_alias!("vanuatu vatus", "VUV", "VUV", false),
    money_alias!("vanuatu vatu", "VUV", "VUV", false),
    money_alias!("lesotho lotis", "LSL", "LSL", false),
    money_alias!("lesotho loti", "LSL", "LSL", false),
    money_alias!("syrian pounds", "SYP", "SYP", false),
    money_alias!("syrian pound", "SYP", "SYP", false),
    money_alias!("netherlands antillean guilders", "ANG", "ANG", false),
    money_alias!("netherlands antillean guilder", "ANG", "ANG", false),
    money_alias!("antillean guilders", "ANG", "ANG", false),
    money_alias!("netherlands guilders", "NLG", "NLG", false),
    money_alias!("netherlands guilder", "NLG", "NLG", false),
    money_alias!("papua new guinean kinas", "PGK", "PGK", false),
    money_alias!("trinidad and tobago dollars", "TTD", "TTD", false),
    money_alias!("trinidad and tobago dollar", "TTD", "TTD", false),
    money_alias!("monegasque francs", "MCF", "MCF", false),
    money_alias!("monegasque franc", "MCF", "MCF", false),
    money_alias!("generic francs", "CHF", "CHF", false),
    money_alias!("german pfennigs", "DEM", "DM", true),
    money_alias!("german pfennig", "DEM", "DM", true),
    money_alias!("pfennigs", "DEM", "DM", true),
    money_alias!("pfennig", "DEM", "DM", true),
    money_alias!("irish pennies", "IEP", "IEP", true),
    money_alias!("irish penny", "IEP", "IEP", true),
    money_alias!("euro cents", "EUR", "€", true),
    money_alias!("monegasque centimes", "MCF", "MCF", true),
    money_alias!("canadian cents", "CAD", "CAD", true),
    money_alias!("polish grosz", "PLN", "PLN", true),
    money_alias!("polish groszy", "PLN", "PLN", true),
    money_alias!("czech halers", "CZK", "CZK", true),
    money_alias!("czech haler", "CZK", "CZK", true),
    money_alias!("swiss rappen", "CHF", "CHF", true),
    money_alias!("austrian schilling", "ATS", "ATS", false),
    money_alias!("austrian groschen", "ATS", "ATS", true),
    money_alias!("groschen", "ATS", "ATS", true),
    money_alias!("malaysian sen", "MYR", "MYR", true),
    money_alias!("kazakhstani tiin", "KZT", "KZT", true),
    money_alias!("tiin", "KZT", "KZT", true),
    money_alias!("ghanaian pesewas", "GHS", "GHS", true),
    money_alias!("ghanaian pesewa", "GHS", "GHS", true),
    money_alias!("pesewas", "GHS", "GHS", true),
    money_alias!("pesewa", "GHS", "GHS", true),
    money_alias!("centimos", "ESP", "ESP", true),
    money_alias!("cordoba centavos", "NIO", "NIO", true),
    money_alias!("avos", "MOP", "MOP", true),
    money_alias!("piastre", "LBP", "LBP", true),
    money_alias!("kopek", "RUB", "RUB", true),
    money_alias!("kopec", "RUB", "RUB", true),
    money_alias!("paisa", "INR", "Rs", true),
    money_alias!("jeon", "KRW", "KRW", true),
    money_alias!("haliers", "SKK", "SKK", true),
    money_alias!("halier", "SKK", "SKK", true),
    money_alias!("ore", "NOK", "NOK", true),
    money_alias!("norwegian ore", "NOK", "NOK", true),
    money_alias!("danish ore", "DKK", "DKK", true),
    money_alias!("swedish ore", "SEK", "SEK", true),
    money_alias!("paise", "INR", "Rs", true),
    money_alias!("pakistani paise", "PKR", "PKR", true),
    money_alias!("euro cents", "EUR", "€", true),
    money_alias!("grosz", "PLN", "PLN", true),
    money_alias!("halers", "CZK", "CZK", true),
    money_alias!("rappen", "CHF", "CHF", true),
    money_alias!("kroner", "NOK", "NOK", false),
    money_alias!("kronor", "SEK", "SEK", false),
    money_alias!("dirhams", "AED", "AED", false),
    money_alias!("emirates dirhams", "AED", "AED", false),
    money_alias!("arab emirates dirhams", "AED", "AED", false),
    money_alias!("riyals", "SAR", "SAR", false),
    money_alias!("rials", "QAR", "QAR", false),
    money_alias!("rands", "ZAR", "ZAR", false),
    money_alias!("bahts", "THB", "THB", false),
    money_alias!("dongs", "VND", "VND", false),
    money_alias!("rupiahs", "IDR", "IDR", false),
    money_alias!("hryvnias", "UAH", "UAH", false),
    money_alias!("korunas", "CZK", "CZK", false),
    money_alias!("schillings", "ATS", "ATS", false),
    money_alias!("levs", "BGN", "BGN", false),
    money_alias!("liras", "ITL", "ITL", false),
    money_alias!("pesetas", "ESP", "ESP", false),
    money_alias!("forints", "HUF", "HUF", false),
    money_alias!("manats", "AZN", "AZN", false),
    money_alias!("ringgit", "MYR", "MYR", false),
    money_alias!("patacas", "MOP", "MOP", false),
    money_alias!("pataca", "MOP", "MOP", false),
    money_alias!("florins", "AWG", "AWG", false),
    money_alias!("pulas", "BWP", "BWP", false),
    money_alias!("dalasis", "GMD", "GMD", false),
    money_alias!("nairas", "NGN", "NGN", false),
    money_alias!("birrs", "ETB", "ETB", false),
    money_alias!("leones", "SLL", "SLL", false),
    money_alias!("kinas", "PGK", "PGK", false),
    money_alias!("colons", "CRC", "CRC", false),
    money_alias!("escudos", "CVE", "CVE", false),
    money_alias!("dinars", "RSD", "RSD", false),
    money_alias!("dominican pesos", "DOP", "DOP", false),
    money_alias!("dominican peso", "DOP", "DOP", false),
    money_alias!("algerian dinars", "DZD", "DZD", false),
    money_alias!("algerian dinar", "DZD", "DZD", false),
    money_alias!("kips", "LAK", "LAK", false),
    money_alias!("rubles", "RUB", "RUB", false),
    money_alias!("roubles", "RUB", "RUB", false),
    money_alias!("afghanis", "AFN", "AFN", false),
    money_alias!("ethiopian birr", "ETB", "ETB", false),
    money_alias!("argentine peso", "ARS", "ARS", false),
    money_alias!("spanish peseta", "ESP", "ESP", false),
    money_alias!("israeli new sheqel", "ILS", "ILS", false),
    money_alias!("lithuanian litas", "LTL", "LTL", false),
    money_alias!("costa rican centimos", "CRC", "CRC", true),
    money_alias!("centsas", "LTL", "LTL", true),
    money_alias!("platinum ounces", "XPT", "XPT", false),
    money_alias!("platinum ounce", "XPT", "XPT", false),
    money_alias!("silver ounces", "XAG", "XAG", false),
    money_alias!("silver ounce", "XAG", "XAG", false),
    // Singular and alternate minor-unit names accepted in spoken input.
    money_alias!("ruble", "RUB", "RUB", false),
    money_alias!("rouble", "RUB", "RUB", false),
    money_alias!("peso", "DOP", "DOP", false),
    money_alias!("franc", "CHF", "CHF", false),
    money_alias!("dinar", "RSD", "RSD", false),
    money_alias!("dirham", "AED", "AED", false),
    money_alias!("rial", "QAR", "QAR", false),
    money_alias!("riyal", "SAR", "SAR", false),
    money_alias!("baht", "THB", "THB", false),
    money_alias!("dong", "VND", "VND", false),
    money_alias!("lira", "ITL", "ITL", false),
    money_alias!("peseta", "ESP", "ESP", false),
    money_alias!("forint", "HUF", "HUF", false),
    money_alias!("manat", "AZN", "AZN", false),
    money_alias!("shilling", "KES", "KES", false),
    money_alias!("euro cent", "EUR", "€", true),
    money_alias!("centime", "EUR", "€", true),
    money_alias!("centimes", "EUR", "€", true),
    money_alias!("centavo", "BRL", "R$", true),
    money_alias!("pennies", "GBP", "£", true),
    money_alias!("fen", "CNY", "CNY", true),
    money_alias!("fens", "CNY", "CNY", true),
    money_alias!("jiao", "CNY", "CNY", true),
    money_alias!("jiaos", "CNY", "CNY", true),
    money_alias!("jeons", "KRW", "KRW", true),
    money_alias!("kopeks", "RUB", "RUB", true),
    money_alias!("kopecs", "RUB", "RUB", true),
    money_alias!("rappens", "CHF", "CHF", true),
    money_alias!("piastres", "LBP", "LBP", true),
    money_alias!("fils", "AED", "AED", true),
];

fn money_aliases() -> impl Iterator<Item = MoneyAlias> {
    MONEY_ALIASES
        .iter()
        .chain(MONEY_EXTRA_ALIASES.iter())
        .copied()
}

fn is_money_currency_word(word: &str) -> bool {
    money_aliases().any(|alias| {
        alias
            .phrase
            .split_whitespace()
            .any(|alias_word| alias_word == word)
    })
}

fn money_words(text: &str) -> Vec<String> {
    let normalized: String = text
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if matches!(
                character,
                '-' | '\u{2010}' | '\u{2011}' | '\u{2013}' | '\u{2014}'
            ) {
                ' '
            } else {
                character
            }
        })
        .collect();
    normalized
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| {
                matches!(
                    character,
                    ',' | '.'
                        | ';'
                        | ':'
                        | '!'
                        | '?'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '"'
                        | '\''
                        | '\u{2019}'
                )
            })
            .to_owned()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn money_alias_matches_at(words: &[String], start: usize, phrase: &str) -> bool {
    let alias_words: Vec<&str> = phrase.split_whitespace().collect();
    words
        .get(start..start + alias_words.len())
        .is_some_and(|candidate| {
            candidate
                .iter()
                .map(String::as_str)
                .eq(alias_words.into_iter())
        })
}

fn find_money_alias(words: &[String], minor: bool) -> Option<(usize, usize, MoneyAlias)> {
    let mut found = None;
    for start in 0..words.len() {
        for alias in money_aliases().filter(|alias| alias.minor == minor) {
            let length = alias.phrase.split_whitespace().count();
            if !money_alias_matches_at(words, start, alias.phrase) {
                continue;
            }
            let replace = found.is_none_or(|(old_start, old_end, _)| {
                length > old_end - old_start || (length == old_end - old_start && start < old_start)
            });
            if replace {
                found = Some((start, start + length, alias));
            }
        }
    }
    found
}

fn money_scale_power(word: &str) -> Option<usize> {
    match word {
        "thousand" => Some(3),
        "lakh" | "lakhs" => Some(5),
        "million" => Some(6),
        "crore" | "crores" => Some(7),
        "billion" => Some(9),
        "trillion" => Some(12),
        _ => None,
    }
}

fn parse_money_integer_words(words: &[String]) -> Option<i128> {
    let mut cleaned: Vec<String> = words
        .iter()
        .filter(|word| word.as_str() != "and")
        .cloned()
        .collect();
    if cleaned.is_empty() {
        return None;
    }

    let negative = matches!(
        cleaned.first().map(String::as_str),
        Some("minus" | "negative")
    );
    if negative {
        cleaned.remove(0);
    }
    if cleaned.is_empty() {
        return None;
    }

    // Treat the largest scale as a multiplier for the complete coefficient
    // before it. This covers both conventional forms ("one million five
    // hundred thousand") and corpus forms such as "two thousand five
    // hundred million" (2,500 million).
    if let Some(last_scale) = cleaned
        .iter()
        .rposition(|word| money_scale_power(word).is_some())
    {
        let power = money_scale_power(&cleaned[last_scale])?;
        if cleaned[..last_scale]
            .iter()
            .any(|word| money_scale_power(word) == Some(power))
        {
            let coefficient = parse_money_integer_standard(&cleaned[..last_scale])?;
            let value = coefficient.checked_mul(10_i128.checked_pow(power as u32)?)?;
            return Some(if negative { -value } else { value });
        }
    }
    if let Some(power) = cleaned
        .iter()
        .filter_map(|word| money_scale_power(word))
        .max()
    {
        let index = cleaned
            .iter()
            .rposition(|word| money_scale_power(word) == Some(power))?;
        let coefficient = parse_money_integer_standard(&cleaned[..index])?;
        let remainder = if index + 1 < cleaned.len() {
            parse_money_integer_words(&cleaned[index + 1..])?
        } else {
            0
        };
        let value = coefficient
            .checked_mul(10_i128.checked_pow(power as u32)?)?
            .checked_add(remainder);
        return value.map(|value| if negative { -value } else { value });
    }

    if cleaned.len() >= 2 {
        let single_digit = matches!(
            cleaned[0].as_str(),
            "one" | "two" | "three" | "four" | "five" | "six" | "seven" | "eight" | "nine"
        );
        if single_digit {
            let first = parse_cardinal_number(&cleaned[0])?;
            let rest = parse_cardinal_number(&cleaned[1..].join(" "))?;
            if (10..=99).contains(&rest) {
                return first.checked_mul(100)?.checked_add(rest);
            }
        }
    }
    let value = parse_money_integer_standard(&cleaned)?;
    Some(if negative { -value } else { value })
}

fn parse_money_integer_standard(words: &[String]) -> Option<i128> {
    let mut total = 0_i128;
    let mut current = 0_i128;
    let mut saw_value = false;
    for word in words {
        match word.as_str() {
            "zero" | "oh" | "o" | "nought" | "naught" | "nil" => {
                saw_value = true;
            }
            "one" => {
                current = current.checked_add(1)?;
                saw_value = true;
            }
            "two" => {
                current = current.checked_add(2)?;
                saw_value = true;
            }
            "three" => {
                current = current.checked_add(3)?;
                saw_value = true;
            }
            "four" => {
                current = current.checked_add(4)?;
                saw_value = true;
            }
            "five" => {
                current = current.checked_add(5)?;
                saw_value = true;
            }
            "six" => {
                current = current.checked_add(6)?;
                saw_value = true;
            }
            "seven" => {
                current = current.checked_add(7)?;
                saw_value = true;
            }
            "eight" => {
                current = current.checked_add(8)?;
                saw_value = true;
            }
            "nine" => {
                current = current.checked_add(9)?;
                saw_value = true;
            }
            "ten" => {
                current = current.checked_add(10)?;
                saw_value = true;
            }
            "eleven" => {
                current = current.checked_add(11)?;
                saw_value = true;
            }
            "twelve" => {
                current = current.checked_add(12)?;
                saw_value = true;
            }
            "thirteen" => {
                current = current.checked_add(13)?;
                saw_value = true;
            }
            "fourteen" => {
                current = current.checked_add(14)?;
                saw_value = true;
            }
            "fifteen" => {
                current = current.checked_add(15)?;
                saw_value = true;
            }
            "sixteen" => {
                current = current.checked_add(16)?;
                saw_value = true;
            }
            "seventeen" => {
                current = current.checked_add(17)?;
                saw_value = true;
            }
            "eighteen" => {
                current = current.checked_add(18)?;
                saw_value = true;
            }
            "nineteen" => {
                current = current.checked_add(19)?;
                saw_value = true;
            }
            "twenty" => {
                current = current.checked_add(20)?;
                saw_value = true;
            }
            "thirty" => {
                current = current.checked_add(30)?;
                saw_value = true;
            }
            "forty" => {
                current = current.checked_add(40)?;
                saw_value = true;
            }
            "fifty" => {
                current = current.checked_add(50)?;
                saw_value = true;
            }
            "sixty" => {
                current = current.checked_add(60)?;
                saw_value = true;
            }
            "seventy" => {
                current = current.checked_add(70)?;
                saw_value = true;
            }
            "eighty" => {
                current = current.checked_add(80)?;
                saw_value = true;
            }
            "ninety" => {
                current = current.checked_add(90)?;
                saw_value = true;
            }
            "hundred" => {
                current = current.max(1).checked_mul(100)?;
                saw_value = true;
            }
            word if money_scale_power(word).is_some() => {
                let power = money_scale_power(word)?;
                let component = current
                    .max(1)
                    .checked_mul(10_i128.checked_pow(power as u32)?)?;
                total = total.checked_add(component)?;
                current = 0;
                saw_value = true;
            }
            word if money_scale_power(word).is_none() => {
                let value = word.parse::<i128>().ok()?;
                current = current.checked_add(value)?;
                saw_value = true;
            }
            _ => return None,
        }
    }
    saw_value.then(|| total.checked_add(current)).flatten()
}

fn parse_money_fraction_words(words: &[String]) -> Option<String> {
    let mut fraction = String::new();
    for word in words {
        fraction.push(single_sequence_digit(word)?);
    }
    (!fraction.is_empty()).then_some(fraction)
}

fn normalize_money_decimal(value: &str) -> String {
    let (negative, value) = value
        .strip_prefix('-')
        .map_or((false, value), |rest| (true, rest));
    let (integer, fraction) = value.split_once('.').unwrap_or((value, ""));
    let integer = integer.trim_start_matches('0');
    let integer = if integer.is_empty() { "0" } else { integer };
    let fraction = fraction.trim_end_matches('0');
    let mut output = String::new();
    if negative && (integer != "0" || !fraction.is_empty()) {
        output.push('-');
    }
    output.push_str(integer);
    if !fraction.is_empty() {
        output.push('.');
        output.push_str(fraction);
    }
    output
}

fn multiply_money_decimal(value: &str, power: usize) -> String {
    let negative = value.starts_with('-');
    let value = value.strip_prefix('-').unwrap_or(value);
    let (integer, fraction) = value.split_once('.').unwrap_or((value, ""));
    let digits = format!("{integer}{fraction}");
    let decimal_index = integer.len() + power;
    let result = if decimal_index >= digits.len() {
        format!("{digits}{}", "0".repeat(decimal_index - digits.len()))
    } else {
        format!("{}.{}", &digits[..decimal_index], &digits[decimal_index..])
    };
    let signed = if negative {
        format!("-{result}")
    } else {
        result
    };
    normalize_money_decimal(&signed)
}

fn parse_money_amount(words: &[String]) -> Option<String> {
    if words.is_empty() {
        return None;
    }
    let point = words.iter().position(|word| word == "point");
    if let Some(point) = point {
        let scale = words[point + 1..]
            .iter()
            .position(|word| money_scale_power(word).is_some())
            .map(|index| point + 1 + index);
        if scale.is_some_and(|index| index + 1 != words.len()) {
            return None;
        }
        let fraction_end = scale.unwrap_or(words.len());
        let fraction = parse_money_fraction_words(&words[point + 1..fraction_end])?;
        let integer = if point == 0 {
            0
        } else {
            parse_money_integer_words(&words[..point])?
        };
        let mut amount = normalize_money_decimal(&format!("{integer}.{fraction}"));
        if let Some(scale) = scale {
            amount = multiply_money_decimal(&amount, money_scale_power(&words[scale])?);
        }
        return Some(amount);
    }
    parse_money_integer_words(words).map(|value| value.to_string())
}

fn money_minor_places(alias: MoneyAlias, minor: i128) -> usize {
    match alias.phrase.split_whitespace().last() {
        Some("jeon" | "jiao" | "sen") if minor < 10 => 1,
        _ => 2,
    }
}

fn add_money_minor(major: &str, minor: i128, places: usize) -> Option<String> {
    if minor < 0 {
        return None;
    }
    let base = 10_i128.checked_pow(places as u32)?;
    let negative = major.starts_with('-');
    let major = major.strip_prefix('-').unwrap_or(major);
    let (integer, fraction) = major.split_once('.').unwrap_or((major, ""));
    let current = if fraction.is_empty() {
        0
    } else {
        let mut padded = fraction.to_owned();
        while padded.len() < places {
            padded.push('0');
        }
        padded[..places].parse::<i128>().ok()?
    };
    let combined = current.checked_add(minor)?;
    let carry = combined / base;
    let cents = combined % base;
    let integer = integer.parse::<i128>().ok()?.checked_add(carry)?;
    let fraction = format!("{cents:0places$}");
    Some(normalize_money_decimal(&format!(
        "{}{integer}.{fraction}",
        if negative { "-" } else { "" }
    )))
}

fn format_money_value(currency: MoneyCurrency, amount: &str) -> String {
    if currency
        .display
        .chars()
        .all(|character| character.is_ascii_uppercase())
    {
        format!("{} {}", currency.display, amount)
    } else {
        format!("{}{}", currency.display, amount)
    }
}

fn parse_local_money(text: &str) -> Option<String> {
    let words = money_words(text);
    if words.is_empty() {
        return None;
    }
    let major = find_money_alias(&words, false);
    let minor = find_money_alias(&words, true);
    let major = major.filter(|(major_start, _, _)| {
        minor.is_none_or(|(minor_start, _, _)| minor_start > *major_start)
    });
    let (currency, amount) = if let Some((major_start, major_end, major_alias)) = major {
        let mut amount = parse_money_amount(&words[..major_start])?;
        let minor_amount = minor
            .filter(|(minor_start, _, _)| *minor_start > major_end)
            .and_then(|(minor_start, _, minor_alias)| {
                let start = words[..minor_start]
                    .iter()
                    .rposition(|word| word == "and")
                    .map_or(major_end, |index| index + 1);
                parse_money_integer_words(&words[start..minor_start])
                    .map(|amount| (amount, minor_alias))
            });
        if let Some((minor_amount, minor_alias)) = minor_amount {
            if minor.is_some_and(|(_, minor_end, _)| minor_end < words.len()) {
                return None;
            }
            amount = add_money_minor(
                &amount,
                minor_amount,
                money_minor_places(minor_alias, minor_amount),
            )?;
        } else if major_end < words.len() {
            let remainder = words[major_end..]
                .iter()
                .filter(|word| word.as_str() != "and")
                .cloned()
                .collect::<Vec<_>>();
            if !remainder.is_empty() {
                let implied = parse_money_integer_words(&remainder)?;
                amount = add_money_minor(&amount, implied, 2)?;
            }
        }
        (major_alias.currency, amount)
    } else if let Some((minor_start, _, minor_alias)) = minor {
        if minor.is_some_and(|(_, minor_end, _)| minor_end < words.len()) {
            return None;
        }
        let amount_words = words[..minor_start]
            .iter()
            .filter(|word| !is_money_currency_word(word))
            .cloned()
            .collect::<Vec<_>>();
        let amount = parse_money_integer_words(&amount_words)?;
        let places = money_minor_places(minor_alias, amount);
        let base = 10_i128.checked_pow(places as u32)?;
        let negative = amount < 0;
        let magnitude = amount.checked_abs()?;
        let major = magnitude / base;
        let fraction = magnitude % base;
        let value = if fraction == 0 {
            major.to_string()
        } else {
            normalize_money_decimal(&format!("{major}.{fraction:0places$}"))
        };
        let value = if negative { format!("-{value}") } else { value };
        (minor_alias.currency, value)
    } else {
        return None;
    };
    Some(format_money_value(
        currency,
        &normalize_money_decimal(&amount),
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DecimalValue {
    negative: bool,
    digits: String,
    scale: i64,
}

const MAX_DECIMAL_OUTPUT_CHARS: usize = 1_000_000;

impl DecimalValue {
    fn from_parts(negative: bool, integer: &str, fraction: &str, exponent: i64) -> Option<Self> {
        if integer.is_empty() && fraction.is_empty() {
            return None;
        }
        if !integer.chars().all(|character| character.is_ascii_digit())
            || !fraction.chars().all(|character| character.is_ascii_digit())
        {
            return None;
        }
        let mut digits = format!("{integer}{fraction}");
        let mut scale = i64::try_from(fraction.len()).ok()?.checked_sub(exponent)?;
        let first_nonzero = digits
            .chars()
            .position(|character| character != '0')
            .unwrap_or(digits.len());
        if first_nonzero == digits.len() {
            return Some(Self {
                negative,
                digits: "0".to_owned(),
                scale: 0,
            });
        }
        digits.drain(..first_nonzero);
        while digits.ends_with('0') {
            digits.pop();
            scale = scale.checked_sub(1)?;
        }
        Some(Self {
            negative,
            digits,
            scale,
        })
    }

    fn with_exponent(mut self, exponent: i64) -> Option<Self> {
        if self.digits == "0" {
            return Some(self);
        }
        self.scale = self.scale.checked_sub(exponent)?;
        Some(self)
    }
}

fn decimal_value_to_string(value: &DecimalValue) -> Option<String> {
    let sign_len = usize::from(value.negative);
    let output_len = if value.scale <= 0 {
        let zeros = usize::try_from(value.scale.checked_neg()?).ok()?;
        sign_len
            .checked_add(value.digits.len())?
            .checked_add(zeros)?
    } else {
        let scale = usize::try_from(value.scale).ok()?;
        let magnitude_len = if scale >= value.digits.len() {
            2usize
                .checked_add(scale.checked_sub(value.digits.len())?)?
                .checked_add(value.digits.len())?
        } else {
            value.digits.len().checked_add(1)?
        };
        sign_len.checked_add(magnitude_len)?
    };
    if output_len > MAX_DECIMAL_OUTPUT_CHARS {
        return None;
    }

    let mut output = String::new();
    if value.negative {
        output.push('-');
    }
    if value.scale <= 0 {
        output.push_str(&value.digits);
        let zeros = usize::try_from(value.scale.checked_neg()?).ok()?;
        output.push_str(&"0".repeat(zeros));
        return Some(output);
    }

    let scale = usize::try_from(value.scale).ok()?;
    if scale >= value.digits.len() {
        output.push_str("0.");
        output.push_str(&"0".repeat(scale - value.digits.len()));
        output.push_str(&value.digits);
    } else {
        let split = value.digits.len() - scale;
        output.push_str(&value.digits[..split]);
        output.push('.');
        output.push_str(&value.digits[split..]);
    }
    Some(output)
}

fn decimal_scale_power(word: &str) -> Option<i64> {
    match word {
        "thousand" | "k" => Some(3),
        "million" | "m" | "mn" => Some(6),
        "billion" | "b" | "bn" => Some(9),
        "trillion" | "t" | "tn" => Some(12),
        "quadrillion" | "q" => Some(15),
        "quintillion" => Some(18),
        "sextillion" => Some(21),
        "septillion" => Some(24),
        "octillion" => Some(27),
        "nonillion" => Some(30),
        "decillion" => Some(33),
        "undecillion" => Some(36),
        _ => None,
    }
}

fn decimal_words(text: &str) -> Vec<String> {
    money_words(text)
}

fn parse_decimal_integer_words(words: &[String]) -> Option<i128> {
    if words.is_empty() {
        return None;
    }
    for (index, word) in words.iter().enumerate() {
        let Some(power) = decimal_scale_power(word) else {
            continue;
        };
        if words[..index]
            .iter()
            .any(|candidate| decimal_scale_power(candidate) == Some(power))
        {
            return None;
        }
    }
    let high_scale = words
        .iter()
        .enumerate()
        .filter_map(|(index, word)| {
            decimal_scale_power(word)
                .filter(|power| *power >= 15)
                .map(|power| (index, power))
        })
        .max_by_key(|(_, power)| *power);
    if let Some((index, power)) = high_scale {
        if index == 0 {
            return None;
        }
        let coefficient = parse_decimal_integer_words(&words[..index])?;
        let remainder = if index + 1 < words.len() {
            parse_decimal_integer_words(&words[index + 1..])?
        } else {
            0
        };
        return coefficient
            .checked_mul(10_i128.checked_pow(u32::try_from(power).ok()?)?)?
            .checked_add(remainder);
    }
    if words.iter().any(|word| decimal_scale_power(word).is_some()) {
        return parse_money_integer_words(words);
    }
    parse_cardinal_number(&words.join(" "))
}

fn parse_decimal_fraction_words(words: &[String]) -> Option<String> {
    if words.is_empty() {
        return None;
    }
    words
        .iter()
        .map(|word| single_sequence_digit(word))
        .collect()
}

fn parse_decimal_sign(words: &[String]) -> Option<(bool, &[String])> {
    match words.first().map(String::as_str) {
        Some("minus" | "negative") => Some((true, &words[1..])),
        Some("plus" | "positive") => Some((false, &words[1..])),
        _ => Some((false, words)),
    }
}

fn parse_decimal_word_value(words: &[String]) -> Option<DecimalValue> {
    let (negative, words) = parse_decimal_sign(words)?;
    if words.is_empty() {
        return None;
    }
    if words
        .iter()
        .any(|word| matches!(word.as_str(), "minus" | "negative" | "plus" | "positive"))
    {
        return None;
    }

    let point = words
        .iter()
        .position(|word| matches!(word.as_str(), "point" | "dot"));
    if let Some(point) = point {
        let scale_positions: Vec<usize> = words[point + 1..]
            .iter()
            .enumerate()
            .filter_map(|(index, word)| decimal_scale_power(word).map(|_| index))
            .collect();
        if scale_positions.len() > 1 {
            return None;
        }
        let scale_index = scale_positions.first().map(|index| point + 1 + index);
        let fraction_end = scale_index.unwrap_or(words.len());
        let fraction = parse_decimal_fraction_words(&words[point + 1..fraction_end])?;
        let integer = if point == 0 {
            0
        } else {
            parse_decimal_integer_words(&words[..point])?
        };
        let mut value = DecimalValue::from_parts(negative, &integer.to_string(), &fraction, 0)?;
        if let Some(scale_index) = scale_index {
            value = value.with_exponent(decimal_scale_power(&words[scale_index])?)?;
        }
        return Some(value);
    }

    let integer = parse_decimal_integer_words(words)?;
    DecimalValue::from_parts(negative, &integer.unsigned_abs().to_string(), "", 0).map(
        |mut value| {
            if integer < 0 {
                value.negative = !value.negative;
            }
            value
        },
    )
}

fn parse_decimal_scientific_words(words: &[String]) -> Option<DecimalValue> {
    let times = words.iter().position(|word| word == "times")?;
    if words.get(times + 1..times + 4)? != ["ten", "to", "the"] {
        return None;
    }
    if words[..times].is_empty() || words[times + 4..].is_empty() {
        return None;
    }
    let mantissa = parse_decimal_word_value(&words[..times])?;
    let exponent = i64::try_from(parse_decimal_integer_words(&words[times + 4..])?).ok()?;
    mantissa.with_exponent(exponent)
}

fn decimal_surface_scale_power(word: &str) -> Option<i64> {
    decimal_scale_power(&word.to_ascii_lowercase())
}

fn is_decimal_separator(character: char) -> bool {
    matches!(character, ',' | '.' | '_' | '\'' | ' ')
}

fn valid_decimal_grouped_integer(text: &str) -> Option<String> {
    if text.is_empty() {
        return Some(String::new());
    }
    if text.chars().all(|character| character.is_ascii_digit()) {
        return Some(text.to_owned());
    }

    let groups: Vec<&str> = text.split(is_decimal_separator).collect();
    if groups.len() < 2
        || groups.iter().any(|group| {
            group.is_empty() || !group.chars().all(|character| character.is_ascii_digit())
        })
        || !(1..=3).contains(&groups[0].len())
        || groups[1..].iter().any(|group| group.len() != 3)
    {
        return None;
    }
    Some(groups.concat())
}

fn parse_decimal_surface_mantissa(text: &str) -> Option<(String, String)> {
    if text.is_empty()
        || text
            .chars()
            .any(|character| !character.is_ascii_digit() && !is_decimal_separator(character))
    {
        return None;
    }

    let dot_count = text.chars().filter(|character| *character == '.').count();
    let comma_count = text.chars().filter(|character| *character == ',').count();
    let decimal_separator = match (dot_count, comma_count) {
        (0, 0) => None,
        (1, 0) => Some('.'),
        (0, 1) => {
            let (_, fraction) = text.split_once(',')?;
            (fraction.len() != 3).then_some(',')
        }
        (dot_count, 0) if dot_count > 1 => None,
        (0, comma_count) if comma_count > 1 => None,
        (dot_count, comma_count) => {
            let (dot, comma) = (text.rfind('.')?, text.rfind(',')?);
            let separator = if dot > comma { '.' } else { ',' };
            let count = if separator == '.' {
                dot_count
            } else {
                comma_count
            };
            (count == 1).then_some(separator)
        }
    };

    if let Some(separator) = decimal_separator {
        let (integer, fraction) = text.split_once(separator)?;
        if fraction.is_empty() && integer.is_empty()
            || fraction
                .chars()
                .any(|character| !character.is_ascii_digit())
        {
            return None;
        }
        let integer = valid_decimal_grouped_integer(integer)?;
        return Some((integer, fraction.to_owned()));
    }

    Some((valid_decimal_grouped_integer(text)?, String::new()))
}

fn decimal_surface_attached_scale(text: &str) -> Option<(String, i64)> {
    for suffix in [
        "quadrillion",
        "quintillion",
        "sextillion",
        "septillion",
        "octillion",
        "nonillion",
        "decillion",
        "undecillion",
        "trillion",
        "billion",
        "million",
        "thousand",
        "bn",
        "mn",
        "tn",
        "q",
        "k",
        "b",
        "m",
        "t",
    ] {
        if let Some(prefix) = text.strip_suffix(suffix) {
            if prefix.is_empty() {
                return None;
            }
            return Some((prefix.to_owned(), decimal_surface_scale_power(suffix)?));
        }
    }
    None
}

fn decimal_surface_number(text: &str) -> Option<DecimalValue> {
    let mut text = text
        .trim()
        .to_ascii_lowercase()
        .replace('\u{2212}', "-")
        .replace('\u{00a0}', " ")
        .replace('\u{202f}', " ")
        .replace('\u{2019}', "'")
        .replace('\u{2018}', "'");
    if text.is_empty() {
        return None;
    }
    let has_parentheses = text.contains(['(', ')']);
    let parenthesized = text.starts_with('(') && text.ends_with(')');
    if has_parentheses && !parenthesized {
        return None;
    }
    if parenthesized {
        text = text[1..text.len() - 1].trim().to_owned();
    }

    let mut scale = None;
    let mut pieces: Vec<&str> = text.split_whitespace().collect();
    if pieces.len() > 1 {
        if let Some(power) = pieces
            .last()
            .and_then(|word| decimal_surface_scale_power(word))
        {
            scale = Some(power);
            pieces.pop();
        }
    }
    if pieces.len() > 1 {
        if let Some(power) = pieces
            .first()
            .and_then(|word| decimal_surface_scale_power(word))
        {
            if scale.is_some() {
                return None;
            }
            scale = Some(power);
            pieces.remove(0);
        }
    }
    if pieces.is_empty() {
        return None;
    }
    text = pieces.join(" ");
    if let Some((prefix, power)) = decimal_surface_attached_scale(&text) {
        if scale.is_some() || decimal_surface_attached_scale(&prefix).is_some() {
            return None;
        }
        scale = Some(power);
        text = prefix;
    }

    let has_sign = text.starts_with('-') || text.starts_with('+');
    let negative = parenthesized || text.starts_with('-');
    if has_sign {
        text = text[1..].to_owned();
    }
    let exponent = if text.matches('e').count() > 1 {
        return None;
    } else if let Some(index) = text.find('e') {
        let mantissa = &text[..index];
        let exponent = &text[index + 1..];
        if mantissa.is_empty() || exponent.is_empty() {
            return None;
        }
        let exponent = exponent.parse::<i64>().ok()?;
        text = mantissa.to_owned();
        exponent
    } else {
        0
    };
    let (integer, fraction) = parse_decimal_surface_mantissa(&text)?;
    let value = DecimalValue::from_parts(negative, &integer, &fraction, exponent)?;
    value.with_exponent(scale.unwrap_or(0))
}

fn parse_local_decimal(text: &str) -> Option<String> {
    let words = decimal_words(text);
    if words.is_empty() {
        return None;
    }
    let value = if words.iter().any(|word| word == "times") {
        parse_decimal_scientific_words(&words)?
    } else {
        parse_decimal_word_value(&words).or_else(|| decimal_surface_number(text))?
    };
    decimal_value_to_string(&value)
}

fn decimal_representations_equivalent(canonical: &str, observed: &str) -> bool {
    decimal_surface_number(canonical) == decimal_surface_number(observed)
}

fn normalize_measurement_input(text: &str) -> Option<String> {
    let mut normalized = text.trim().to_ascii_lowercase();
    if normalized.starts_with("negative ") {
        normalized.replace_range(..9, "minus ");
    } else if normalized.starts_with("positive ") {
        normalized.replace_range(..9, "");
    } else if normalized.starts_with("plus ") {
        normalized.replace_range(..5, "");
    }
    normalized = normalized.replace(" dot ", " point ");
    (!normalized.is_empty()).then_some(normalized)
}

fn measurement_output_unit(output: &str) -> Option<String> {
    let tokens: Vec<&str> = output.split_whitespace().collect();
    for (index, token) in tokens.iter().enumerate() {
        if index == 0 {
            if decimal_surface_number(token).is_some() {
                continue;
            }
            return None;
        }
        if index == 1 && index + 1 < tokens.len() && decimal_surface_scale_power(token).is_some() {
            continue;
        }
        return Some(tokens[index..].join(" "));
    }
    None
}

fn is_measurement_number_word(word: &str) -> bool {
    matches!(
        word,
        "minus"
            | "negative"
            | "plus"
            | "positive"
            | "point"
            | "dot"
            | "hundred"
            | "thousand"
            | "million"
            | "billion"
            | "trillion"
            | "quadrillion"
            | "quintillion"
            | "sextillion"
            | "septillion"
            | "octillion"
            | "nonillion"
            | "decillion"
            | "undecillion"
    ) || parse_cardinal_number(word).is_some()
}

fn parse_local_measurement(text: &str) -> Option<String> {
    let normalized = normalize_measurement_input(text)?;
    let upstream = measure::parse(&normalized)?;
    let unit = measurement_output_unit(&upstream)?;
    let words: Vec<&str> = normalized.split_whitespace().collect();
    for split in (1..words.len()).rev() {
        if words[split..]
            .iter()
            .any(|word| is_measurement_number_word(word))
        {
            continue;
        }
        if let Some(number) = parse_local_decimal(&words[..split].join(" ")) {
            return Some(format!("{number} {unit}"));
        }
    }
    None
}

fn canonical_measurement_unit(text: &str) -> Option<String> {
    let compact = text
        .trim()
        .to_ascii_lowercase()
        .replace('²', "2")
        .replace('³', "3")
        .replace('μ', "u")
        .split_whitespace()
        .collect::<String>();
    let canonical = match compact.as_str() {
        "%" | "percent" => "%",
        "m" | "meter" | "meters" => "m",
        "km" | "kilometer" | "kilometers" => "km",
        "cm" | "centimeter" | "centimeters" => "cm",
        "dm" | "decimeter" | "decimeters" => "dm",
        "mm" | "millimeter" | "millimeters" => "mm",
        "um" | "micrometer" | "micrometers" => "um",
        "nm" | "nanometer" | "nanometers" => "nm",
        "ft" | "foot" | "feet" => "ft",
        "mi" | "mile" | "miles" => "mi",
        "sqft" | "squarefoot" | "squarefeet" => "sq ft",
        "sqmi" | "squaremile" | "squaremiles" => "sq mi",
        "m2" | "squaremeter" | "squaremeters" => "m2",
        "km2" | "squarekilometer" | "squarekilometers" => "km2",
        "dm3" | "cubicdecimeter" | "cubicdecimeters" => "dm3",
        "m3" | "cubicmeter" | "cubicmeters" => "m3",
        "km3" | "cubickilometer" | "cubickilometers" => "km3",
        "h" | "hour" | "hours" => "h",
        "min" | "minute" | "minutes" => "min",
        "s" | "second" | "seconds" => "s",
        "mph" => "mph",
        "km/h" | "kilometers/hour" => "km/h",
        "m/s" | "meters/second" => "m/s",
        "gbps" | "gigabits/second" => "gbps",
        "mbps" | "megabits/second" => "mbps",
        "pb" | "petabyte" | "petabytes" => "pb",
        "gb" | "gigabyte" | "gigabytes" => "gb",
        "mb" | "megabyte" | "megabytes" => "mb",
        "kb" | "kilobyte" | "kilobytes" | "kilobit" | "kilobits" => "kb",
        "b" | "byte" | "bytes" => "b",
        "kw" => "kw",
        "mw" => "mw",
        "gw" => "gw",
        "kwh" => "kwh",
        "gwh" => "gwh",
        "mwh" => "mwh",
        "w" | "watt" | "watts" => "w",
        "hp" | "horsepower" => "hp",
        "°c" | "celsius" | "degreecelsius" | "degreescelsius" => "°c",
        "°f" | "fahrenheit" | "degreefahrenheit" | "degreesfahrenheit" => "°f",
        "k" | "kelvin" => "k",
        "mhz" => "mhz",
        "khz" => "khz",
        "hz" => "hz",
        "mv" => "mv",
        "v" | "volt" | "volts" => "v",
        "ms" => "ms",
        "au" => "au",
        "oz" | "ounce" | "ounces" => "oz",
        "kg" | "kilogram" | "kilograms" => "kg",
        "g" | "gram" | "grams" => "g",
        "kl" => "kl",
        "l" | "liter" | "liters" | "litre" | "litres" => "l",
        "ml" | "milliliter" | "milliliters" => "ml",
        "cc" => "cc",
        "ha" | "hectare" | "hectares" => "ha",
        "lm" | "lumen" | "lumens" => "lm",
        value if value.starts_with('/') => {
            let denominator = canonical_measurement_unit(&value[1..])?;
            return Some(format!("/{denominator}"));
        }
        _ => return None,
    };
    Some(canonical.to_owned())
}

fn measurement_surface_representation(text: &str) -> Option<(DecimalValue, String)> {
    let normalized = text
        .trim()
        .replace('\u{2212}', "-")
        .replace('\u{00a0}', " ")
        .replace('\u{202f}', " ");
    for index in 1..=normalized.len() {
        if !normalized.is_char_boundary(index) {
            continue;
        }
        let (number, unit) = normalized.split_at(index);
        let Some(number) = decimal_surface_number(number.trim()) else {
            continue;
        };
        let Some(unit) = canonical_measurement_unit(unit.trim()) else {
            continue;
        };
        return Some((number, unit));
    }
    None
}

fn measurement_representations_equivalent(canonical: &str, observed: &str) -> bool {
    measurement_surface_representation(canonical) == measurement_surface_representation(observed)
}

fn electronic_input_is_complete(text: &str) -> bool {
    let lowered = text.trim().to_ascii_lowercase();
    if lowered.is_empty() || !lowered.is_ascii() {
        return false;
    }
    let words: Vec<&str> = lowered.split_whitespace().collect();
    let at_positions: Vec<usize> = words
        .iter()
        .enumerate()
        .filter_map(|(index, word)| (*word == "at").then_some(index))
        .collect();
    if at_positions.len() > 1 {
        return false;
    }
    if let Some(at) = at_positions.first().copied() {
        if at == 0 || at + 1 == words.len() || words[at + 1] == "dot" {
            return false;
        }
    }

    let is_protocol = [
        "h t t p colon slash slash ",
        "h t t p s colon slash slash ",
        "http colon slash slash ",
        "https colon slash slash ",
    ]
    .iter()
    .any(|prefix| lowered.starts_with(prefix));
    if is_protocol && words.len() <= 4 {
        return false;
    }

    let Some(last_dot) = words.iter().rposition(|word| *word == "dot") else {
        return is_protocol;
    };
    if last_dot == 0 || last_dot + 1 >= words.len() {
        return false;
    }
    if words[..last_dot]
        .iter()
        .zip(words[1..].iter())
        .any(|(left, right)| {
            (*left == "dot" && *right == "dot")
                || (*left == "at" && *right == "dot")
                || (*left == "dot" && *right == "at")
        })
    {
        return false;
    }
    if words[last_dot + 1..]
        .iter()
        .any(|word| matches!(*word, "slash" | "colon"))
    {
        return true;
    }
    // Domain and email parsers otherwise append arbitrary trailing words to
    // the final label. A spoken label may be split into single letters, but a
    // second multi-character word is outside the electronic span.
    words[last_dot + 2..].iter().all(|word| {
        word.len() == 1
            && word
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
    })
}

fn electronic_output_is_valid(value: &str) -> bool {
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        return false;
    }

    let domain = if let Some((local, domain)) = value.split_once('@') {
        if local.is_empty() || domain.is_empty() || value.matches('@').count() != 1 {
            return false;
        }
        domain
    } else {
        value
    };
    let host = domain
        .strip_prefix("http://")
        .or_else(|| domain.strip_prefix("https://"))
        .unwrap_or(domain)
        .split(['/', ':'])
        .next()
        .unwrap_or_default();
    if !host.contains('.') || host.starts_with('.') || host.ends_with('.') {
        return false;
    }
    host.split('.').all(|label| {
        !label.is_empty()
            && label
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '-')
    })
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
    } else if let Some(rest) = text.strip_prefix("plus ") {
        (false, rest)
    } else if let Some(rest) = text.strip_prefix("positive ") {
        (false, rest)
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

const ORDINAL_WORD_VALUES: &[(&str, i128)] = &[
    ("zeroth", 0),
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
    ("sixth", 6),
    ("seventh", 7),
    ("eighth", 8),
    ("ninth", 9),
    ("tenth", 10),
    ("eleventh", 11),
    ("twelfth", 12),
    ("thirteenth", 13),
    ("fourteenth", 14),
    ("fifteenth", 15),
    ("sixteenth", 16),
    ("seventeenth", 17),
    ("eighteenth", 18),
    ("nineteenth", 19),
    ("twentieth", 20),
    ("thirtieth", 30),
    ("fortieth", 40),
    ("fiftieth", 50),
    ("sixtieth", 60),
    ("seventieth", 70),
    ("eightieth", 80),
    ("ninetieth", 90),
];

const ORDINAL_SCALE_VALUES: &[(&str, i128)] = &[
    ("hundredth", 100),
    ("thousandth", 1_000),
    ("millionth", 1_000_000),
    ("billionth", 1_000_000_000),
    ("trillionth", 1_000_000_000_000),
    ("quadrillionth", 1_000_000_000_000_000),
    ("quintillionth", 1_000_000_000_000_000_000),
    ("sextillionth", 1_000_000_000_000_000_000_000),
    ("septillionth", 1_000_000_000_000_000_000_000_000),
    ("octillionth", 1_000_000_000_000_000_000_000_000_000),
    ("nonillionth", 1_000_000_000_000_000_000_000_000_000_000),
    ("decillionth", 1_000_000_000_000_000_000_000_000_000_000_000),
    (
        "undecillionth",
        1_000_000_000_000_000_000_000_000_000_000_000_000,
    ),
];

fn ordinal_word_value(word: &str) -> Option<(i128, bool)> {
    if let Some((_, value)) = ORDINAL_WORD_VALUES.iter().find(|(name, _)| *name == word) {
        return Some((*value, false));
    }
    ORDINAL_SCALE_VALUES
        .iter()
        .find(|(name, _)| *name == word)
        .map(|(_, value)| (*value, true))
}

fn format_ordinal_number(value: i128) -> String {
    let suffix = match value % 100 {
        11..=13 => "th",
        _ => match value % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    };
    format!("{value}{suffix}")
}

fn parse_roman_ordinal(text: &str) -> Option<i128> {
    let mut text = text.trim().to_ascii_uppercase();
    while text.ends_with(['.', ',', ';', ':']) {
        text.pop();
    }
    if text.is_empty() || text.len() > 15 || !text.bytes().all(|byte| b"IVXLCDM".contains(&byte)) {
        return None;
    }
    let value = text
        .bytes()
        .enumerate()
        .map(|(index, byte)| {
            let current = match byte {
                b'I' => 1,
                b'V' => 5,
                b'X' => 10,
                b'L' => 50,
                b'C' => 100,
                b'D' => 500,
                b'M' => 1_000,
                _ => 0,
            };
            let next = text.as_bytes().get(index + 1).map_or(0, |next| match next {
                b'I' => 1,
                b'V' => 5,
                b'X' => 10,
                b'L' => 50,
                b'C' => 100,
                b'D' => 500,
                b'M' => 1_000,
                _ => 0,
            });
            if current < next {
                -current
            } else {
                current
            }
        })
        .sum::<i32>();
    if !(1..=3_999).contains(&value) {
        return None;
    }
    let mut remainder = value;
    let mut canonical = String::new();
    for (unit, symbol) in [
        (1_000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ] {
        while remainder >= unit {
            canonical.push_str(symbol);
            remainder -= unit;
        }
    }
    (canonical == text).then_some(value as i128)
}

fn parse_numeric_ordinal(text: &str) -> Option<i128> {
    let mut text = text.trim().to_ascii_lowercase();
    while text.ends_with(['.', ',', ';', ':']) {
        text.pop();
    }
    let suffix = ["st", "nd", "rd", "th"]
        .iter()
        .find(|suffix| text.ends_with(**suffix))?;
    let digits = text.strip_suffix(suffix)?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let value = digits.parse::<i128>().ok()?;
    let expected = match value % 100 {
        11..=13 => "th",
        _ => match value % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    };
    (expected == *suffix).then_some(value)
}

fn parse_ordinal_words(text: &str) -> Option<i128> {
    let mut normalized = text.trim().to_ascii_lowercase();
    for separator in [
        '-', '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2212}',
    ] {
        normalized = normalized.replace(separator, " ");
    }
    if !normalized.is_ascii() {
        return None;
    }
    let mut words: Vec<&str> = normalized.split_whitespace().collect();
    if words.first() == Some(&"the") {
        words.remove(0);
    }
    if words.is_empty()
        || words
            .iter()
            .any(|word| matches!(*word, "minus" | "negative" | "plus" | "positive"))
    {
        return None;
    }
    if words.len() == 1 {
        if let Some(value) = parse_roman_ordinal(words[0]) {
            return Some(value);
        }
    }
    words.retain(|word| *word != "and");
    let (last, prefix) = words.split_last()?;
    let (value, is_scale) = ordinal_word_value(last)?;
    if is_scale {
        let coefficient = if prefix.is_empty() {
            1
        } else {
            parse_cardinal_number(&prefix.join(" "))?
        };
        return coefficient.checked_mul(value);
    }
    if prefix.is_empty() {
        Some(value)
    } else {
        parse_cardinal_number(&prefix.join(" "))?.checked_add(value)
    }
}

fn parse_local_ordinal(text: &str) -> Option<String> {
    if text.trim().is_empty() || !text.trim().is_ascii() {
        return None;
    }
    let value = parse_numeric_ordinal(text)
        .or_else(|| parse_roman_ordinal(text))
        .or_else(|| parse_ordinal_words(text))?;
    (value >= 0).then(|| format_ordinal_number(value))
}

fn ordinal_representation_value(text: &str) -> Option<i128> {
    parse_numeric_ordinal(text)
        .or_else(|| parse_roman_ordinal(text))
        .or_else(|| parse_ordinal_words(text))
}

fn ordinal_representations_equivalent(canonical: &str, observed: &str) -> bool {
    ordinal_representation_value(canonical) == ordinal_representation_value(observed)
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
    if let Some(aviation) = cardinal::parse_aviation(text)
        .filter(|value| *value != canonical)
        .filter(|value| cardinal_representation_value(value).is_some())
    {
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

fn has_impossible_month_first_ordinal(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    let start = if words
        .first()
        .is_some_and(|word| DATE_WEEKDAYS.contains(word))
    {
        1
    } else {
        0
    };
    if words.len() - start != 3 || !DATE_MONTHS.contains(&words[start]) {
        return false;
    }
    let Some(ordinal) = ordinal::parse(&words[start + 1..].join(" ")) else {
        return false;
    };
    let day = ordinal
        .chars()
        .filter(char::is_ascii_digit)
        .collect::<String>()
        .parse::<u32>();
    day.is_ok_and(|day| day > 31)
}

fn has_impossible_month_first_cardinal(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    let start = if words
        .first()
        .is_some_and(|word| DATE_WEEKDAYS.contains(word))
    {
        1
    } else {
        0
    };
    if words.len() <= start + 1 || !DATE_MONTHS.contains(&words[start]) {
        return false;
    }
    let tail = &words[start + 1..];
    if tail.len() >= 3 && parse_date_year(&tail.join(" "), false).is_some() {
        return false;
    }
    tail.len() >= 3
        && cardinal::words_to_number(&tail[..2].join(" "))
            .is_some_and(|value| (32..1000).contains(&value))
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
        // A month followed by one large number is the month-year form. Any
        // larger number in a date that also has a day position is invalid.
        // Era markers are part of the month-year form as well.
        let month_year = month_index == 0
            && (words.len() == 2
                || (words.len() == 3 && matches!(words[2], "BC" | "BCE" | "CE" | "AD")));
        return month_year && day >= 100;
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
    if has_impossible_month_first_ordinal(&text) || has_impossible_month_first_cardinal(&text) {
        return None;
    }
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
struct MoneyRepresentation {
    currency: String,
    amount: String,
}

const MONEY_SURFACE_MARKERS: &[(&str, &str)] = &[
    ("US$", "USD"),
    ("$US", "USD"),
    ("A$", "AUD"),
    ("AU$", "AUD"),
    ("$A", "AUD"),
    ("$Au", "AUD"),
    ("C$", "CAD"),
    ("CA$", "CAD"),
    ("HK$", "HKD"),
    ("NZ$", "NZD"),
    ("$NZ", "NZD"),
    ("S$", "SGD"),
    ("SI$", "SGD"),
    ("MOP$", "MOP"),
    ("T$", "TWD"),
    ("NT$", "TWD"),
    ("RD$", "DOP"),
    ("EC$", "XCD"),
    ("FJ$", "FJD"),
    ("J$", "JMD"),
    ("JA$", "JMD"),
    ("BZ$", "BZD"),
    ("B$", "BZD"),
    ("BDS$", "BBD"),
    ("G$", "GYD"),
    ("$HK", "HKD"),
    ("$R", "BRL"),
    ("$r", "BRL"),
    ("u$s", "USD"),
    ("ℳ", "DEM"),
    ("S/.", "ESP"),
    ("E£", "GBP"),
    ("£E", "GBP"),
    ("KZL", "PLN"),
    ("MUSD", "USD"),
    ("BUSD", "USD"),
    ("BARS", "ARS"),
    ("TWST", "WST"),
    ("TMNT", "MNT"),
    ("Tk", "BDT"),
    ("Mrs", "INR"),
    ("руб", "RUB"),
    ("MDM", "DEM"),
    ("MATS", "ATS"),
    ("MARS", "ARS"),
    ("MCRC", "CRC"),
    ("MILS", "ILS"),
    ("MNOK", "NOK"),
    ("MBGN", "BGN"),
    ("MSSP", "SSP"),
    ("MMCF", "MCF"),
    ("TMRO", "MRO"),
    ("TYEN", "JPY"),
    ("TRS", "INR"),
    ("BRS", "INR"),
    ("Kwon", "KRW"),
    ("₹", "INR"),
    ("₩", "KRW"),
    ("₽", "RUB"),
    ("₫", "VND"),
    ("₱", "PHP"),
    ("₴", "UAH"),
    ("₺", "TRY"),
    ("₦", "NGN"),
    ("₲", "PYG"),
    ("฿", "THB"),
    ("₡", "CRC"),
    ("₸", "KZT"),
    ("₵", "GHS"),
    ("₭", "LAK"),
    ("₮", "MNT"),
    ("₾", "GEL"),
    ("₼", "AZN"),
];

fn money_surface_marker_boundary(text: &str, start: usize, end: usize, marker: &str) -> bool {
    if !marker
        .chars()
        .all(|character| character.is_ascii_alphabetic())
    {
        return true;
    }
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    !before.is_some_and(|character| character.is_ascii_alphabetic())
        && !after.is_some_and(|character| character.is_ascii_alphabetic())
}

fn money_surface_marker_scale(marker: &str) -> Option<&'static str> {
    match marker {
        "mdm" | "mats" | "mars" | "mcrc" | "mils" | "mnok" | "mbgn" | "mssp" | "mmcf" | "mrs"
        | "musd" => Some("million"),
        "brs" | "bars" | "busd" => Some("billion"),
        "trs" | "tyen" | "tmro" | "twst" | "tmnt" => Some("trillion"),
        "kwon" | "kzl" => Some("thousand"),
        _ => None,
    }
}

fn money_surface_marker(text: &str) -> Option<(&'static str, String)> {
    let lower = text.to_ascii_lowercase();
    let mut best: Option<(usize, usize, &'static str, usize)> = None;
    let mut consider = |marker: &'static str, currency: &'static str| {
        let marker_lower = marker.to_ascii_lowercase();
        let mut offset = 0;
        while let Some(relative) = lower[offset..].find(&marker_lower) {
            let start = offset + relative;
            let end = start + marker_lower.len();
            if money_surface_marker_boundary(&lower, start, end, marker)
                && best.is_none_or(|(_, _, _, length)| marker_lower.len() > length)
            {
                best = Some((start, end, currency, marker_lower.len()));
            }
            offset = end;
        }
    };

    for &(marker, currency) in MONEY_SURFACE_MARKERS {
        consider(marker, currency);
    }
    for alias in money_aliases() {
        consider(alias.currency.display, alias.currency.code);
        consider(alias.currency.code, alias.currency.code);
        consider(alias.phrase, alias.currency.code);
    }

    let (start, end, currency, _) = best?;
    let marker_text = &lower[start..end];
    let mut remainder = String::with_capacity(lower.len() - (end - start));
    remainder.push_str(&lower[..start]);
    remainder.push_str(&lower[end..]);
    if (marker_text.eq_ignore_ascii_case("rs") || marker_text.eq_ignore_ascii_case("tk"))
        && remainder.starts_with('.')
    {
        remainder.remove(0);
    }
    if marker_text
        .chars()
        .all(|character| character.is_ascii_alphabetic())
        && remainder.ends_with('.')
    {
        remainder.pop();
    }
    if let Some(scale) = money_surface_marker_scale(marker_text) {
        remainder.push(' ');
        remainder.push_str(scale);
    }
    Some((currency, remainder))
}

fn money_surface_scale_power(word: &str) -> Option<usize> {
    match word {
        "thousand" | "k" => Some(3),
        "lakh" | "lakhs" | "lac" | "lacs" => Some(5),
        "million" | "m" | "mn" => Some(6),
        "crore" | "crores" | "cr" => Some(7),
        "billion" | "b" | "bn" => Some(9),
        "trillion" | "t" | "tn" => Some(12),
        _ => None,
    }
}

fn money_surface_number(text: &str) -> Option<String> {
    let mut output = String::new();
    let mut decimal = false;
    let mut sign = false;
    for (index, character) in text.chars().enumerate() {
        match character {
            '+' | '-' if index == 0 && !sign => {
                sign = true;
                output.push(character);
            }
            '0'..='9' => output.push(character),
            '.' if !decimal => {
                decimal = true;
                output.push(character);
            }
            ',' | '_' | '\'' | '\u{00a0}' | '\u{202f}' | ' ' => {}
            _ => return None,
        }
    }
    let digits = output
        .strip_prefix('-')
        .or_else(|| output.strip_prefix('+'))
        .unwrap_or(&output)
        .chars()
        .filter(|character| character.is_ascii_digit())
        .count();
    (digits > 0).then(|| normalize_money_decimal(&output))
}

fn money_surface_representation(text: &str) -> Option<MoneyRepresentation> {
    let (currency, mut remainder) = money_surface_marker(text.trim())?;
    let parenthesized = remainder.starts_with('(') && remainder.ends_with(')');
    if parenthesized {
        remainder = remainder[1..remainder.len() - 1].to_owned();
    }

    let mut number_parts: Vec<&str> = Vec::new();
    let mut scale = None;
    for raw in remainder.split_whitespace() {
        let token = raw.trim_matches(|character: char| {
            matches!(character, ',' | ';' | ':' | '/' | '[' | ']' | '{' | '}')
        });
        let token = token.strip_suffix('-').unwrap_or(token);
        if token.is_empty() {
            continue;
        }
        if token == "m" && number_parts.iter().any(|part| part.contains(',')) {
            continue;
        }
        if let Some(power) = money_surface_scale_power(token) {
            if scale.is_some() {
                return None;
            }
            scale = Some(power);
            continue;
        }
        let mut suffix_scale = None;
        for suffix in [
            "trillion", "billion", "million", "thousand", "crore", "lakh", "bn", "mn", "tn", "cr",
            "lac", "k", "t", "b", "m",
        ] {
            if let Some(prefix) = token.strip_suffix(suffix) {
                if !prefix.is_empty() && money_surface_number(prefix).is_some() {
                    suffix_scale = Some((prefix, money_surface_scale_power(suffix)?));
                    break;
                }
            }
        }
        if let Some((prefix, power)) = suffix_scale {
            if scale.is_some() {
                return None;
            }
            number_parts.push(prefix);
            scale = Some(power);
        } else if token
            .chars()
            .all(|character| character.is_ascii_alphabetic())
            && money_aliases().any(|alias| alias.phrase.eq_ignore_ascii_case(token))
        {
            // A repeated currency word, such as "$1 million dollars", is
            // display noise after the marker has already established identity.
        } else if !number_parts.is_empty()
            && token.len() == 1
            && token
                .chars()
                .all(|character| character.is_ascii_alphabetic())
        {
            // Some corpus renderers append a one-letter annotation after the
            // amount. It does not change the monetary value.
        } else {
            number_parts.push(token);
        }
    }

    let number = money_surface_number(&number_parts.join(""))?;
    let amount = scale.map_or(number.clone(), |power| {
        multiply_money_decimal(&number, power)
    });
    let amount = if parenthesized {
        normalize_money_decimal(&format!("-{amount}"))
    } else {
        normalize_money_decimal(&amount)
    };
    Some(MoneyRepresentation {
        currency: currency.to_owned(),
        amount,
    })
}

fn money_representations_equivalent(canonical: &str, observed: &str) -> bool {
    let Some(left) = money_surface_representation(canonical) else {
        return false;
    };
    let Some(right) = money_surface_representation(observed) else {
        return false;
    };
    money_amounts_equivalent(&left.amount, &right.amount) && left.currency == right.currency
}

fn money_amounts_equivalent(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let Some((left_integer, left_fraction)) = left.split_once('.') else {
        return false;
    };
    let Some((right_integer, right_fraction)) = right.split_once('.') else {
        return false;
    };
    left_integer == right_integer
        && left_fraction.trim_start_matches('0') == right_fraction.trim_start_matches('0')
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

fn split_compound_date_token(token: String) -> Vec<String> {
    for weekday in DATE_WEEKDAYS {
        let prefix = &weekday[..3];
        if let Some(month) = token.strip_prefix(prefix) {
            if representation_month(month).is_some() {
                return vec![prefix.to_owned(), month.to_owned()];
            }
        }
    }
    vec![token]
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
    let tokens: Vec<String> = representation_tokens(trimmed)
        .into_iter()
        .flat_map(split_compound_date_token)
        .collect();
    let mut options = Vec::new();
    if tokens.is_empty() {
        return options;
    }

    if tokens[0] == "q" && tokens.len() == 3 {
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
                .any(|weekday| token.len() >= 2 && weekday.starts_with(token))
        })
        .filter(|token| *token != "the")
        .filter(|token| *token != "th" && *token != "st" && *token != "nd" && *token != "rd")
        .collect();

    if numbers.len() == 1 && !letters.is_empty() {
        let era = letters.join("").to_ascii_uppercase();
        if matches!(era.as_str(), "BC" | "BCE" | "CE" | "AD" | "AF") {
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

    if numbers.is_empty() {
        if let Some(value) = roman_value(trimmed) {
            options.push(DateRepresentation::Year(value.to_string()));
        }
        return options;
    }

    if letters.iter().any(|token| {
        representation_month(token).is_none()
            && !DATE_WEEKDAYS
                .iter()
                .any(|weekday| weekday.starts_with(token))
            && !matches!(
                *token,
                "the" | "of" | "th" | "st" | "nd" | "rd" | "a" | "b" | "c" | "d" | "e" | "f" | "s"
            )
            && !(token.len() <= 2
                && token
                    .chars()
                    .all(|character| character.is_ascii_alphabetic()))
    }) {
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

    if numbers.len() == 1 {
        if !letters.is_empty() {
            return options;
        }
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
    if value.is_empty() || !valid_timezone_surface_value(&value) {
        return None;
    }
    Some(value)
}

fn valid_timezone_surface_value(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    let sign = value.find(['-', '+']);
    let name_end = sign.unwrap_or(value.len());
    let name = &value[..name_end];
    if !TIMEZONE_NAMES.contains(&name) {
        return false;
    }
    let Some(sign) = sign else {
        return true;
    };
    let offset = &value[sign + 1..];
    if offset.is_empty() || value[sign + 1..].contains(['-', '+']) {
        return false;
    }
    if let Some((hours, minutes)) = offset.split_once(':') {
        hours.parse::<u32>().is_ok_and(|hours| hours <= 23)
            && minutes.parse::<u32>().is_ok_and(|minutes| minutes < 60)
    } else {
        offset.parse::<u32>().is_ok_and(|hours| hours <= 23)
    }
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
    let mut period_count = 0;
    let mut index = 0;
    while index < tokens.len() {
        match &tokens[index] {
            TimeRepresentationToken::Alphabetic(value) if matches!(value.as_str(), "am" | "pm") => {
                period_count += 1;
            }
            TimeRepresentationToken::Alphabetic(value) if matches!(value.as_str(), "a" | "p") => {
                let mut next = index + 1;
                while matches!(tokens.get(next), Some(TimeRepresentationToken::Symbol('.'))) {
                    next += 1;
                }
                if matches!(
                    tokens.get(next),
                    Some(TimeRepresentationToken::Alphabetic(marker)) if marker == "m"
                ) {
                    period_count += 1;
                    index = next;
                }
            }
            _ => {}
        }
        index += 1;
    }
    if period_count > 1 {
        return None;
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
    if tokens[..first_digit]
        .iter()
        .enumerate()
        .any(|(index, token)| {
            !period_indices.contains(&index)
                && !matches!(token, TimeRepresentationToken::Symbol('.'))
        })
    {
        return None;
    }
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
    let last_non_comma = time_tokens
        .iter()
        .rposition(|token| !matches!(token, TimeRepresentationToken::Symbol(',')));
    if time_tokens.iter().enumerate().any(|(index, token)| {
        matches!(token, TimeRepresentationToken::Symbol(symbol) if !matches!(symbol, ':' | '.' | ',')
            || (*symbol == ',' && last_non_comma.is_some_and(|last| index < last)))
    }) {
        return None;
    }
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
    let timezone = match timezone_start {
        Some(start) => Some(time_timezone_value(&tokens[start..])?),
        None => None,
    };
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
        "DECIMAL" => parse_local_decimal(text),
        "DIGIT_SEQUENCE" => parse_digit_sequence(text),
        "ELECTRONIC" => {
            if electronic_input_is_complete(text) {
                electronic::parse(text).filter(|value| electronic_output_is_valid(value))
            } else {
                None
            }
        }
        "MONEY" => parse_local_money(text).or_else(|| money::parse(text)),
        "MEASUREMENT" => parse_local_measurement(text),
        "ORDINAL" => parse_local_ordinal(text),
        "PUNCTUATION" => punctuation::parse(text),
        "PHONE" => phone_input_is_complete(text)
            .then(|| telephone::parse(text))
            .flatten(),
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
        "DECIMAL" => Ok(decimal_representations_equivalent(canonical, observed)),
        "MEASUREMENT" => Ok(measurement_representations_equivalent(canonical, observed)),
        "MONEY" => Ok(money_representations_equivalent(canonical, observed)),
        "ORDINAL" => Ok(ordinal_representations_equivalent(canonical, observed)),
        "TIME" => Ok(time_representations_equivalent(canonical, observed)),
        _ => Err(PyValueError::new_err(
            "representation equivalence is supported only for CARDINAL, DATE, DECIMAL, MEASUREMENT, TIME, and MONEY",
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
