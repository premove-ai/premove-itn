use super::{parse_digit_sequence, realize, realize_known_kind, single_sequence_digit};

#[test]
fn sequence_digit_reuses_upstream_cardinal_words() {
    assert_eq!(single_sequence_digit("zero"), Some('0'));
    assert_eq!(single_sequence_digit("seven"), Some('7'));
    assert_eq!(single_sequence_digit("9"), Some('9'));
    assert_eq!(single_sequence_digit("ten"), None);
    assert_eq!(single_sequence_digit("thirty"), None);
}

#[test]
fn digit_sequence_accepts_spoken_zero_aliases_in_digit_positions() {
    assert_eq!(
        realize_known_kind("DIGIT_SEQUENCE", "o nine"),
        Some("09".to_owned())
    );
    assert_eq!(
        realize_known_kind("DIGIT_SEQUENCE", "O OH zero"),
        Some("000".to_owned())
    );
    assert_eq!(
        realize_known_kind("DIGIT_SEQUENCE", "o nine four o"),
        Some("0940".to_owned())
    );
}

#[test]
fn digit_sequence_preserves_leading_zeroes() {
    assert_eq!(
        parse_digit_sequence("zero zero seven two"),
        Some("0072".to_owned())
    );
}

#[test]
fn digit_sequence_accepts_mixed_spoken_and_numeric_tokens() {
    assert_eq!(
        parse_digit_sequence("3 seven one 8"),
        Some("3718".to_owned())
    );
}

#[test]
fn digit_sequence_expands_repetition_modifiers() {
    assert_eq!(parse_digit_sequence("double seven"), Some("77".to_owned()));
    assert_eq!(parse_digit_sequence("triple zero"), Some("000".to_owned()));
    assert_eq!(
        parse_digit_sequence("double oh seven"),
        Some("007".to_owned())
    );
    assert_eq!(
        parse_digit_sequence("four double five triple zero"),
        Some("455000".to_owned())
    );
}

#[test]
fn digit_sequence_rejects_unknown_and_incomplete_input() {
    assert_eq!(parse_digit_sequence("seven eighty eight"), None);
    assert_eq!(parse_digit_sequence("one banana two"), None);
    assert_eq!(parse_digit_sequence("double"), None);
    assert_eq!(parse_digit_sequence("double triple seven"), None);
    assert_eq!(parse_digit_sequence(""), None);
}

#[test]
fn forced_realizer_uses_selected_upstream_parser() {
    assert_eq!(
        realize_known_kind(
            "CARDINAL",
            "thirty three thousand three hundred and seventy two"
        ),
        Some("33372".to_owned())
    );
    assert_eq!(realize_known_kind("DIGIT_SEQUENCE", "four thirty"), None);
}

#[test]
fn time_realizer_accepts_bare_spoken_clock_forms() {
    assert_eq!(
        realize_known_kind("TIME", "twelve thirty seven"),
        Some("12:37".to_owned())
    );
    assert_eq!(
        realize_known_kind("TIME", "twenty one fifty five"),
        Some("21:55".to_owned())
    );
    assert_eq!(
        realize_known_kind("TIME", "eighteen forty three"),
        Some("18:43".to_owned())
    );
}

#[test]
fn time_realizer_accepts_spoken_meridiem_variants() {
    assert_eq!(
        realize_known_kind("TIME", "ten fifty p m"),
        Some("10:50 p.m.".to_owned())
    );
    assert_eq!(
        realize_known_kind("TIME", "ten fifty PM"),
        Some("10:50 p.m.".to_owned())
    );
    assert_eq!(
        realize_known_kind("TIME", "ten o five a.m."),
        Some("10:05 a.m.".to_owned())
    );
}

#[test]
fn time_realizer_rejects_invalid_clock_values_and_timezones() {
    assert_eq!(realize_known_kind("TIME", "twenty four ten"), None);
    assert_eq!(realize_known_kind("TIME", "ten sixty"), None);
    assert_eq!(realize_known_kind("TIME", "ten fifty p m i s t"), None);
}

#[test]
fn forced_realizer_rejects_empty_input() {
    assert_eq!(realize_known_kind("TIME", "  "), None);
}

#[test]
fn electronic_realizer_formats_email_addresses() {
    assert_eq!(
        realize_known_kind("ELECTRONIC", "a at gmail dot com"),
        Some("a@gmail.com".to_owned())
    );
    assert_eq!(
        realize_known_kind("ELECTRONIC", "nvidia dot com"),
        Some("nvidia.com".to_owned())
    );
    assert_eq!(
        realize_known_kind("ELECTRONIC", "john dot smith at gmail dot com"),
        Some("john.smith@gmail.com".to_owned())
    );
    assert_eq!(
        realize_known_kind("ELECTRONIC", "example dot com"),
        Some("example.com".to_owned())
    );
}

#[test]
fn phone_realizer_formats_normal_spoken_numbers() {
    assert_eq!(
        realize_known_kind("PHONE", "nine eight two zero five five one two three four"),
        Some("982-055-1234".to_owned())
    );
    assert_eq!(
        realize_known_kind(
            "PHONE",
            "plus one four one five two one two five five five one two three four"
        ),
        Some("+14 152 1255 5 1234".to_owned())
    );
}

#[test]
fn measurement_realizer_formats_values_and_units() {
    assert_eq!(
        realize_known_kind("MEASUREMENT", "two hundred meters"),
        Some("200 m".to_owned())
    );
}

#[test]
fn ordinal_realizer_formats_ordinal_numbers() {
    assert_eq!(
        realize_known_kind("ORDINAL", "twenty first"),
        Some("21st".to_owned())
    );
}

#[test]
fn punctuation_realizer_formats_spoken_symbols() {
    assert_eq!(
        realize_known_kind("PUNCTUATION", "question mark"),
        Some("?".to_owned())
    );
}

#[test]
fn whitelist_realizer_formats_known_abbreviations() {
    assert_eq!(
        realize_known_kind("WHITELIST", "doctor dao"),
        Some("dr. dao".to_owned())
    );
}

#[test]
fn word_realizer_formats_words_with_attached_punctuation() {
    assert_eq!(
        realize_known_kind("WORD", "twenty!"),
        Some("20 !".to_owned())
    );
}

#[test]
fn forced_realizer_accepts_all_configured_upstream_kinds() {
    let cases = [
        ("ELECTRONIC", "a at gmail dot com"),
        ("MEASUREMENT", "two hundred meters"),
        ("ORDINAL", "twenty first"),
        ("PUNCTUATION", "question mark"),
        ("WHITELIST", "doctor dao"),
        ("WORD", "twenty!"),
    ];

    for (kind, source) in cases {
        assert!(
            realize(kind, source)
                .expect("every upstream kind must be supported")
                .is_some(),
            "kind: {kind}"
        );
    }
}

#[test]
fn malformed_digit_sequence_fails_closed() {
    assert_eq!(
        realize_known_kind("DIGIT_SEQUENCE", "seven eighty eight"),
        None
    );
}
