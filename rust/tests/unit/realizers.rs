use super::{
    parse_digit_sequence, realize, realize_known_kind, realize_known_kind_options,
    single_sequence_digit,
};

#[test]
fn sequence_digit_reuses_upstream_cardinal_words() {
    let supported = [
        ("zero", '0'),
        ("oh", '0'),
        ("o", '0'),
        ("nought", '0'),
        ("naught", '0'),
        ("nil", '0'),
        ("one", '1'),
        ("two", '2'),
        ("three", '3'),
        ("four", '4'),
        ("five", '5'),
        ("six", '6'),
        ("seven", '7'),
        ("eight", '8'),
        ("nine", '9'),
        ("9", '9'),
    ];
    for (source, expected) in supported {
        assert_eq!(single_sequence_digit(source), Some(expected), "{source}");
    }
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
    assert_eq!(
        parse_digit_sequence("12 three 045"),
        Some("123045".to_owned())
    );
}

#[test]
fn digit_sequence_expands_repetition_modifiers() {
    assert_eq!(parse_digit_sequence("single seven"), Some("7".to_owned()));
    assert_eq!(parse_digit_sequence("double seven"), Some("77".to_owned()));
    assert_eq!(parse_digit_sequence("triple zero"), Some("000".to_owned()));
    assert_eq!(
        parse_digit_sequence("quadruple nought"),
        Some("0000".to_owned())
    );
    assert_eq!(
        parse_digit_sequence("double oh seven"),
        Some("007".to_owned())
    );
    assert_eq!(
        parse_digit_sequence("double o seven"),
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
    assert_eq!(parse_digit_sequence("single double seven"), None);
    assert_eq!(parse_digit_sequence("quadruple"), None);
    assert_eq!(parse_digit_sequence("double 12"), None);
    assert_eq!(parse_digit_sequence("one and two"), None);
    assert_eq!(parse_digit_sequence("one-two"), None);
    assert_eq!(parse_digit_sequence("1.2"), None);
    assert_eq!(parse_digit_sequence("-1"), None);
    assert_eq!(parse_digit_sequence(""), None);
}

#[test]
fn digit_sequence_is_case_and_ascii_whitespace_insensitive() {
    assert_eq!(
        parse_digit_sequence("  DOUBLE\tOh\nSeven  "),
        Some("007".to_owned())
    );
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
fn cardinal_realizer_canonicalizes_zero_and_case() {
    assert_eq!(realize_known_kind("CARDINAL", "zero"), Some("0".to_owned()));
    assert_eq!(
        realize_known_kind("CARDINAL", "Twelve"),
        Some("12".to_owned())
    );
    assert_eq!(
        realize_known_kind("CARDINAL", "NEGATIVE Forty Two"),
        Some("-42".to_owned())
    );
}

#[test]
fn cardinal_realizer_supports_large_i128_scales() {
    assert_eq!(
        realize_known_kind(
            "CARDINAL",
            "eighteen septillion two hundred thirty four million five"
        ),
        Some("18000000000000000234000005".to_owned())
    );
    assert_eq!(
        realize_known_kind("CARDINAL", "seven octillion"),
        Some("7000000000000000000000000000".to_owned())
    );
    assert_eq!(
        realize_known_kind("CARDINAL", "one undecillion"),
        Some("1000000000000000000000000000000000000".to_owned())
    );
    assert_eq!(realize_known_kind("CARDINAL", "one duodecillion"), None);
    assert_eq!(
        realize_known_kind("CARDINAL", "one septillion two octillion"),
        None
    );
    assert_eq!(
        realize_known_kind("CARDINAL", "one octillion two octillion"),
        None
    );
}

#[test]
fn cardinal_options_exclude_surface_representations() {
    assert_eq!(realize_known_kind_options("CARDINAL", "two"), vec!["2"]);
    assert_eq!(
        realize_known_kind_options("CARDINAL", "twelve thousand three hundred forty five"),
        vec!["12345"]
    );
    assert_eq!(
        realize_known_kind_options("CARDINAL", "minus twelve"),
        vec!["-12"]
    );
}

#[test]
fn cardinal_options_preserve_signed_zero() {
    assert_eq!(
        realize_known_kind("CARDINAL", "minus zero"),
        Some("-0".to_owned())
    );
    assert_eq!(
        realize_known_kind_options("CARDINAL", "minus zero"),
        vec!["-0"]
    );
    assert_eq!(
        realize_known_kind_options("CARDINAL", "fourteen"),
        vec!["14"]
    );
}

#[test]
fn cardinal_options_reject_context_dependent_possessives() {
    assert!(realize_known_kind_options("CARDINAL", "four hundred's").is_empty());
    assert!(realize_known_kind_options("CARDINAL", "two’s").is_empty());
}

#[test]
fn cardinal_options_include_aviation_reading_without_changing_canonical_result() {
    assert_eq!(
        realize_known_kind("CARDINAL", "seven eighty eight"),
        Some("95".to_owned())
    );
    assert_eq!(
        realize_known_kind_options("CARDINAL", "seven eighty eight"),
        vec!["95", "788"]
    );
}

#[test]
fn cardinal_reuses_digit_sequence_for_values_larger_than_i128() {
    let source = "one two three four five six seven eight nine zero \
                  one two three four five six seven eight nine zero \
                  one two three four five six seven eight nine zero \
                  one two three four five six seven eight nine zero";
    let digits = "1234567890123456789012345678901234567890";
    assert_eq!(
        realize_known_kind("CARDINAL", source),
        Some(digits.to_owned())
    );
    assert_eq!(realize_known_kind_options("CARDINAL", source), vec![digits]);
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
