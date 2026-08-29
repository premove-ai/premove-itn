use super::{
    cardinal_representations_equivalent, date_representations_equivalent,
    decimal_representations_equivalent, digit_sequence_representations_equivalent,
    measurement_representations_equivalent, money_representations_equivalent,
    ordinal_representations_equivalent, parse_digit_sequence, phone_representations_equivalent,
    realize, realize_known_kind, realize_known_kind_options, single_sequence_digit,
    time_representations_equivalent,
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
fn digit_sequence_equivalence_ignores_display_separators() {
    assert!(digit_sequence_representations_equivalent("77152-", "77152"));
    assert!(digit_sequence_representations_equivalent(
        "77\u{202f}152",
        "77152"
    ));
    assert!(digit_sequence_representations_equivalent("77–152", "77152"));
    assert!(digit_sequence_representations_equivalent(
        "(771) 52", "77152"
    ));
    assert!(!digit_sequence_representations_equivalent("77152", "77153"));
    assert!(!digit_sequence_representations_equivalent(
        "77152", "77152A"
    ));
    assert!(!digit_sequence_representations_equivalent(
        "(77152", "77152"
    ));
    assert!(!digit_sequence_representations_equivalent(
        "77152)", "77152"
    ));
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
    assert_eq!(
        realize_known_kind("CARDINAL", "plus forty two"),
        Some("42".to_owned())
    );
    assert_eq!(
        realize_known_kind("CARDINAL", "positive forty two"),
        Some("42".to_owned())
    );
    assert_eq!(
        realize_known_kind("CARDINAL", "one hundred and twenty-three"),
        Some("123".to_owned())
    );
    assert_eq!(
        realize_known_kind("CARDINAL", "one hundred twenty\u{2011}three"),
        Some("123".to_owned())
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
    assert_eq!(realize_known_kind_options("CARDINAL", "zero"), vec!["0"]);
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
fn money_realizer_covers_scales_and_minor_units() {
    for (source, expected) in [
        ("one dollar", "$1"),
        ("two thousand five hundred dollars", "$2500"),
        ("one point five million dollars", "$1500000"),
        ("two dollars and fifty cents", "$2.5"),
        ("fifty cents", "$0.5"),
        ("one pound and sixteen pence", "£1.16"),
        ("one lakh rupees", "Rs100000"),
        ("five philippine pesos and thirty centavos", "PHP 5.3"),
        ("fifty euro cents", "€0.5"),
        ("three won and six jeon", "KRW 3.6"),
        ("twelve yen and five sen", "¥12.5"),
        ("two malaysian ringgit and fifty eight sen", "MYR 2.58"),
        ("two thousand ten million rupees", "Rs2010000000"),
        (
            "two thousand five hundred million norwegian kroner",
            "NOK 2500000000",
        ),
        ("twenty-five dollars", "$25"),
        ("one hundred, twenty-three dollars", "$123"),
        ("one euro cent", "€0.01"),
        ("five chinese fen", "CNY 0.05"),
        ("two chinese jiao", "CNY 0.2"),
        ("one ruble", "RUB 1"),
        ("five pennies", "£0.05"),
    ] {
        assert_eq!(
            realize_known_kind("MONEY", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
}

#[test]
fn money_equivalence_ignores_currency_placement_grouping_and_scales() {
    for (canonical, observed) in [
        ("$10000", "$10,000"),
        ("$1000000", "$1 million"),
        ("$1000000", "$1M"),
        ("£1.5", "GBP 1.50"),
        ("Rs100000", "INR 1 lakh"),
        ("PHP 5.3", "₱5.30"),
        ("DM2", "2 DEM"),
        ("NOK 100", "100NOK"),
        ("R$435", "BRL 435.00"),
        ("£50000", "£50,000 M"),
        ("$2000000", "$2 M"),
        ("$5", "USD 5"),
        ("$5 A", "$5 Z"),
    ] {
        assert!(
            money_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
    for (canonical, observed) in [
        ("$5", "CAD 5"),
        ("$5", "A$5"),
        ("$10", "£10"),
        ("$100", "$101"),
        ("NOK 1", "SEK 1"),
        ("$5", "$5 tomorrow"),
        ("$5", "$5 million billion"),
    ] {
        assert!(
            !money_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
}

#[test]
fn money_realizer_rejects_unrelated_suffixes() {
    for source in [
        "twenty dollars U.S.",
        "five euros tomorrow",
        "two pounds weight",
        "one dollar and fifty cents tomorrow",
        "two dollars and five cents only",
        "one point five million billion dollars",
    ] {
        assert_eq!(realize_known_kind("MONEY", source), None, "{source}");
    }
}

#[test]
fn decimal_realizer_covers_integers_fractions_scales_and_exponents() {
    for (source, expected) in [
        ("seventy three", "73"),
        ("one point oh five", "1.05"),
        ("point five", "0.5"),
        ("minus point five", "-0.5"),
        ("plus one point five", "1.5"),
        ("five billion", "5000000000"),
        ("one point five million", "1500000"),
        ("two thousand five hundred million", "2500000000"),
        ("two hundred fifty six quadrillion", "256000000000000000"),
        ("one thousand quadrillion", "1000000000000000000"),
        ("zero point o million", "0"),
        (
            "six point three seven one five times ten to the fourteen",
            "637150000000000",
        ),
        (
            "one point one six four one five three two one eight two six nine three five times ten to the minus ten",
            "0.000000000116415321826935",
        ),
    ] {
        assert_eq!(
            realize_known_kind("DECIMAL", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
}

#[test]
fn decimal_realizer_rejects_partial_or_malformed_spans() {
    for source in [
        "one point five percent",
        "one point five tomorrow",
        "minus minus one",
        "one point five times ten",
        "one point five million billion",
    ] {
        assert_eq!(realize_known_kind("DECIMAL", source), None, "{source}");
    }
}

#[test]
fn decimal_surface_parser_rejects_malformed_numbers() {
    for source in ["e3", "+", "+.", "1..2", "1,2,3", "1.2.3", "1e", "1e+"] {
        assert_eq!(realize_known_kind("DECIMAL", source), None, "{source}");
    }
    assert_eq!(
        realize_known_kind("DECIMAL", "1,234.567"),
        Some("1234.567".to_owned())
    );
    assert_eq!(
        realize_known_kind("DECIMAL", "1 234"),
        Some("1234".to_owned())
    );
}

#[test]
fn decimal_realizer_preserves_negative_zero_and_bounds_expansion() {
    assert_eq!(
        realize_known_kind("DECIMAL", "minus zero point zero"),
        Some("-0".to_owned())
    );
    assert_eq!(realize_known_kind("DECIMAL", "-0"), Some("-0".to_owned()));
    assert!(!decimal_representations_equivalent("-0", "0"));
    assert!(decimal_representations_equivalent("-0.00", "-0"));
    assert_eq!(realize_known_kind("DECIMAL", "1e1000000"), None);
}

#[test]
fn decimal_equivalence_ignores_numeric_formatting() {
    for (canonical, observed) in [
        ("1212.3", "1,212.30"),
        ("5.4 million", "5400000"),
        ("1.5M", "1500000"),
        ("1.2e3", "1200"),
        ("1,5", "1.50"),
        ("1.234,56", "1234.56"),
        ("1.234.567", "1234567"),
        ("1’234’567", "1234567"),
        (".5", "0.500"),
        ("-0.50", "-0.5"),
        ("(1.5)", "-1.5"),
    ] {
        assert!(
            decimal_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
    for (canonical, observed) in [("1.2", "1.3"), ("1.2", "1.2 million"), ("-0.5", "0.5")] {
        assert!(
            !decimal_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
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
fn date_realizer_covers_year_readings_missed_upstream() {
    assert_eq!(
        realize_known_kind("DATE", "nineteen hundred"),
        Some("1900".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "ten eighty seven"),
        Some("1087".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "twenty twenty"),
        Some("2020".to_owned())
    );
}

#[test]
fn date_realizer_covers_short_years_and_weekdays() {
    assert_eq!(
        realize_known_kind("DATE", "november seventeenth o nine"),
        Some("november 17 09".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "thursday april nineteenth"),
        Some("thursday april 19".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "saturday the thirtieth of june twenty twelve"),
        Some("saturday 30 june 2012".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "the twelfth of september twenty five"),
        Some("12 september 25".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "october fifth twenty one fifteen"),
        Some("october 5 2115".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "january ten sixty six"),
        Some("january 1066".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "april thirteen o seven"),
        Some("april 1307".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "the first of march o o"),
        Some("1 march 00".to_owned())
    );
}

#[test]
fn date_realizer_covers_centuries_and_era_variants() {
    assert_eq!(
        realize_known_kind("DATE", "nineteen hundreds"),
        Some("1900s".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "two thousands"),
        Some("2000s".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "fourteen hundred b c e"),
        Some("1400BCE".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "six ten c e"),
        Some("610CE".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "thirty one b c"),
        Some("31BC".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "seven forties"),
        Some("740s".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "twenty three seventies"),
        Some("2370s".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "august eleventh seventeen forty a d"),
        Some("august 11 1740 AD".to_owned())
    );
    assert_eq!(
        realize_known_kind("DATE", "twenty three a f"),
        Some("23AF".to_owned())
    );
}

#[test]
fn date_realizer_rejects_impossible_calendar_dates() {
    assert_eq!(
        realize_known_kind("DATE", "july twenty twelve"),
        Some("july 2012".to_owned())
    );
    assert_eq!(realize_known_kind("DATE", "april thirty first"), None);
    assert_eq!(
        realize_known_kind("DATE", "february twenty ninth twenty twenty three"),
        None
    );
    assert_eq!(
        realize_known_kind("DATE", "february twenty ninth twenty twenty four"),
        Some("february 29 2024".to_owned())
    );
    assert_eq!(realize_known_kind("DATE", "january thirty second"), None);
    assert_eq!(
        realize_known_kind("DATE", "january thirty two twenty twenty"),
        None
    );
    assert_eq!(
        realize_known_kind("DATE", "monday january thirty third"),
        None
    );
}

#[test]
fn cardinal_equivalence_ignores_rendering_policy() {
    for observed in [
        "12345", "12,345", "12 345", "12, 345", "012345", "12,345 ", "12345:", "12345-",
    ] {
        assert!(cardinal_representations_equivalent("12345", observed));
    }
    for observed in ["14", "014", "XIV", "XIIII", "XIV's", "XIVs"] {
        assert!(cardinal_representations_equivalent("14", observed));
    }
    assert!(cardinal_representations_equivalent("-0", "-000"));
    assert!(!cardinal_representations_equivalent("14", "15"));
    assert!(!cardinal_representations_equivalent("-14", "XIV"));
}

#[test]
fn date_equivalence_ignores_unambiguous_rendering_policy() {
    for observed in [
        "4 March 2014",
        "March 4th, 2014",
        "2014-03-04",
        "03/04/2014",
        "Tuesday, Mar. 4, 2014",
        "the 4th March 2014",
        "2014-MAR-04",
    ] {
        assert!(date_representations_equivalent("4 march 2014", observed));
    }
    assert!(date_representations_equivalent(
        "november 17 09",
        "11/17/09"
    ));
    assert!(date_representations_equivalent("1900s", "1900's"));
    assert!(date_representations_equivalent("31BC", "31 B.C."));
    assert!(date_representations_equivalent("610CE", "610 C.E."));
    assert!(date_representations_equivalent("23AF", "23 A.F."));
    assert!(date_representations_equivalent(
        "19 july 3228 BCE",
        "July 19, 3228 B.C.E."
    ));
    assert!(date_representations_equivalent(
        "sunday july 17 1988",
        "SunJuly 17, 1988"
    ));
    assert!(!date_representations_equivalent(
        "4 march 2014",
        "5 March 2014"
    ));
    assert!(!date_representations_equivalent("31BC", "31 AD"));
    assert!(!date_representations_equivalent(
        "4 march 2014",
        "4 march 2014 tomorrow"
    ));
    assert!(!date_representations_equivalent(
        "2020 tomorrow",
        "2020 tomorrow"
    ));
    assert!(!date_representations_equivalent("2020 FOO", "2020 FOO"));
    assert!(!date_representations_equivalent(
        "Q1 2020 FOO",
        "Q1 2020 FOO"
    ));
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
fn time_realizer_accepts_military_duration_and_timezone_forms() {
    for (source, expected) in [
        ("zero hundred", "00:00"),
        ("sixteen hundred", "16:00"),
        ("twenty four hundred", "24:00"),
        ("twenty one o eight u t c", "21:08 UTC"),
        ("ten twenty two a m e t", "10:22 a.m. ET"),
        ("two forty five g m t minus six", "02:45 GMT-6"),
        ("three nineteen p m w s t", "03:19 p.m. WST"),
        ("five fifteen c h s t", "05:15 CHST"),
        (
            "nine minutes three seconds and twenty nine milliseconds",
            "09:03.29",
        ),
        (
            "three hours fifty two minutes and eight seconds",
            "03:52:08",
        ),
    ] {
        assert_eq!(
            realize_known_kind("TIME", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
}

#[test]
fn time_realizer_accepts_relative_and_named_clock_forms() {
    for (source, expected) in [
        ("midnight", "00:00"),
        ("noon", "12:00"),
        ("one o clock", "01:00"),
        ("one thirty in the morning", "01:30 a.m."),
        ("five past six", "06:05"),
        ("twenty minutes past six", "06:20"),
        ("quarter to one", "12:45"),
        ("half to three", "02:30"),
    ] {
        assert_eq!(
            realize_known_kind("TIME", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
}

#[test]
fn time_realizer_rejects_invalid_clock_values_and_timezones() {
    assert_eq!(realize_known_kind("TIME", "twenty five ten"), None);
    assert_eq!(realize_known_kind("TIME", "ten sixty"), None);
    assert_eq!(
        realize_known_kind("TIME", "ten fifty p m i s t"),
        Some("10:50 p.m. IST".to_owned())
    );
    for source in [
        "ten fifty p.m. tomorrow",
        "ten fifty p.m. weight",
        "ten fifty p.m. only",
        "ten fifty p.m. UTC minus twenty five",
    ] {
        assert_eq!(realize_known_kind("TIME", source), None, "{source}");
    }
}

#[test]
fn time_equivalence_ignores_rendering_policy() {
    for (canonical, observed) in [
        ("04:30", "4:30"),
        ("04:30", "04.30"),
        ("04:30 p.m.", "4:30PM"),
        ("04:30 p.m. ET", "4.30 PM et"),
        ("04:30", "0430"),
        ("09:03.29", "9:03.290"),
        ("03:52:08", "3:52:08"),
        ("02:45 GMT-6", "2:45 gmt-06:00"),
        ("02:45 GMT+5:30", "2:45 gmt+05:30"),
        ("01:00 p.m.", "PM1"),
    ] {
        assert!(
            time_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
    for (canonical, observed) in [
        ("04:30", "04:31"),
        ("04:30 a.m.", "04:30 p.m."),
        ("04:30 ET", "04:30 UTC"),
        ("09:03.29", "09:03.30"),
        ("03:52:08", "03:52:09"),
        ("04:30 tomorrow", "04:30 tomorrow"),
        ("04:30 UTC-24", "04:30 UTC-24"),
        ("04:30 GMT+5:60", "04:30 GMT+5:60"),
        ("04:30!", "04:30!"),
        ("04-30", "04-30"),
        ("04:30 p.m. p.m.", "04:30 p.m. p.m."),
    ] {
        assert!(
            !time_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
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
    assert_eq!(
        realize_known_kind(
            "ELECTRONIC",
            "h t t p colon slash slash example dot com slash path"
        ),
        Some("http://example.com/path".to_owned())
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
fn phone_realizer_accepts_silence_separators_and_rejects_partial_spans() {
    assert_eq!(
        realize_known_kind("PHONE", "three two nine two sil three two nine seven"),
        Some("329-23297".to_owned())
    );
    assert_eq!(
        realize_known_kind(
            "PHONE",
            "o sil nine o six eight nine nine sil five six sil seven"
        ),
        Some("090-689-9567".to_owned())
    );
    for source in [
        "three two sil sil nine",
        "three two sil",
        "three two tomorrow",
        "one two point three",
        "one two million",
        "one hundred sil twenty",
    ] {
        assert_eq!(realize_known_kind("PHONE", source), None, "{source}");
    }
}

#[test]
fn phone_equivalence_ignores_display_grouping_but_preserves_structure() {
    for (canonical, observed) in [
        ("3292-3297", "329-23297"),
        ("0-906899-56-7", "090-689-9567"),
        ("(212) 555 0123", "212-555-0123"),
        ("123.123.0.40", "123.123.0.40"),
        ("SSN 799-12-3113", "ssn is 799-12-3113"),
    ] {
        assert!(
            phone_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
    for (canonical, observed) in [
        ("123.123.0.40", "123123040"),
        ("123-4567", "123-4568"),
        ("SSN 799-12-3113", "799-12-3113"),
    ] {
        assert!(
            !phone_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
}

#[test]
fn measurement_realizer_formats_values_and_units() {
    for (source, expected) in [
        ("two hundred meters", "200 m"),
        ("plus three meters", "3 m"),
        ("positive three meters", "3 m"),
        ("negative three meters", "-3 m"),
        ("one dot five meters", "1.5 m"),
        ("one point five million meters", "1500000 m"),
        ("two hundred kilometers per hour", "200 km/h"),
        ("two square feet", "2 sq ft"),
        ("seven inches", "7 in"),
        ("twenty nine pounds", "29 lb"),
        ("seventy eight revolutions per minute", "78 rpm"),
        ("per hour", "/h"),
        ("per square kilometer", "/km²"),
        ("two hundred twenty kilo volts", "220 kV"),
        ("two point four gigahertz", "2.4 GHz"),
        ("four giga liters", "4 GL"),
        ("three peta watts", "3 PW"),
        (
            "thirty three and a third revolutions per minute",
            "33.333333333333 rpm",
        ),
        ("half a kilometer", "0.5 km"),
        ("fifty two thirds of a meter", "17.333333333333 m"),
        ("five milligrams per kilogram", "5 mg/kg"),
        ("five kilo amperes", "5 kA"),
        ("fifty calories", "50 cal"),
        ("five kilobytes", "5 KB"),
        ("five kilobits", "5 kbit"),
        ("twenty five and three quarters pounds", "25.75 lb"),
        ("three per cubic meter", "3 /m³"),
        ("one hundred fifty c c", "150 cc"),
        ("fifty eight seconds", "58 s"),
        ("one hundred eleven mega meters", "111 Mm"),
        ("five sieverts", "5 Sv"),
        ("zero pico henrys", "0 pH"),
        ("one ten thousandth of a millimeter", "0.0001 mm"),
        ("three thousand five thousandths of a meter", "0.6 m"),
        ("fifty one hundred twenty fifths of a micrometer", "0.4 μm"),
        ("foot per second", "ft/s"),
        ("kilogram per kilogram", "kg/kg"),
        ("per c c", "/cc"),
    ] {
        assert_eq!(
            realize_known_kind("MEASUREMENT", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
    for source in [
        "two meters tomorrow",
        "two meters squared",
        "five kg",
        "one point five million million meters",
        "one hundred meters per hour tomorrow",
    ] {
        assert_eq!(realize_known_kind("MEASUREMENT", source), None, "{source}");
    }
}

#[test]
fn measurement_equivalence_ignores_formatting_but_preserves_units() {
    for (canonical, observed) in [
        ("90%", "90 percent"),
        ("1,234.50 km", "1234.5 kilometers"),
        ("315.9/km²", "315.90 / square kilometers"),
        ("2 sq ft", "2 square feet"),
        ("1GB", "1 gigabytes"),
        ("-0 m", "-0.00 meters"),
        ("7\"", "7 inches"),
        ("5'", "5 feet"),
        ("0.1 mi²", "0.10 square miles"),
        ("10 mg/kg", "10 milligrams per kilogram"),
        ("2.4 GHz", "2.40 gigahertz"),
        ("33 1/3 rpm", "33.333333333333 revolutions per minute"),
        ("2½ mm", "2.5 millimeters"),
        ("15/25 km", "0.6 kilometers"),
        ("2.6 g/cm3", "2.6 grams per c c"),
        ("1/10,000mm", "0.0001 millimeters"),
        ("43m", "43 minutes"),
        ("100Gb/s", "100 gigabits per second"),
        ("37.7 p.c.", "37.7 percent"),
    ] {
        assert!(
            measurement_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
    for (canonical, observed) in [("2 m", "2 km"), ("90%", "90 km"), ("1 m/s", "1 km/h")] {
        assert!(
            !measurement_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
}

#[test]
fn ordinal_realizer_formats_ordinal_numbers() {
    for (source, expected) in [
        ("first", "1st"),
        ("the eighth", "8th"),
        ("twenty first", "21st"),
        ("twenty-first", "21st"),
        ("one hundred and twenty-second", "122nd"),
        ("one hundredth", "100th"),
        ("one trillionth", "1000000000000th"),
        ("VIII", "8th"),
        ("XXVth", "25th"),
        ("42nd", "42nd"),
        ("5ª", "5th"),
        ("10ths", "10th"),
        ("1,500th", "1500th"),
    ] {
        assert_eq!(
            realize_known_kind("ORDINAL", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
    for source in [
        "the twenty first extra",
        "twenty fourths",
        "minus first",
        "21rd",
        "IC",
        "XXVrd",
        "",
    ] {
        assert_eq!(realize_known_kind("ORDINAL", source), None, "{source}");
    }
}

#[test]
fn ordinal_equivalence_ignores_rendering_forms() {
    for (canonical, observed) in [
        ("42nd", "forty second"),
        ("VIII", "the eighth"),
        ("I.", "1st"),
        ("01st", "first"),
        ("1000th", "one thousandth"),
        ("5ª", "fifth"),
        ("10ths", "tenths"),
        ("33, 140th", "thirty three thousand one hundred fortieth"),
    ] {
        assert!(
            ordinal_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
    for (canonical, observed) in [("42nd", "43rd"), ("IV", "IIII"), ("1st", "1th")] {
        assert!(
            !ordinal_representations_equivalent(canonical, observed),
            "{canonical} vs {observed}"
        );
    }
}

#[test]
fn punctuation_realizer_formats_spoken_symbols() {
    for (source, expected) in [
        ("question mark", "?"),
        ("Question   Mark", "?"),
        ("full stop", "."),
        ("exclamation", "!"),
        ("semi colon", ";"),
        ("en dash", "–"),
        ("em dash", "—"),
        ("dot dot dot", "..."),
        ("open square bracket", "["),
        ("quotation mark", "\""),
        ("apostrophe", "'"),
        ("backslash", "\\"),
    ] {
        assert_eq!(
            realize_known_kind("PUNCTUATION", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
    for source in ["question mark extra", "sil", "", "🙂"] {
        assert_eq!(realize_known_kind("PUNCTUATION", source), None, "{source}");
    }
}

#[test]
fn whitelist_realizer_formats_known_abbreviations() {
    for (source, expected) in [
        ("doctor dao", "dr. dao"),
        ("Doctor dao", "Dr. dao"),
        ("i like for example ice cream", "i like e.g. ice cream"),
        ("seven eleven stores", "7-eleven stores"),
        ("r t x", "RTX"),
    ] {
        assert_eq!(
            realize_known_kind("WHITELIST", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
    for source in [
        "doctorate dao",
        "for examples only",
        "é doctor",
        "doctor",
        "doctor doctor",
    ] {
        let result = realize_known_kind("WHITELIST", source);
        if source == "doctor" {
            assert_eq!(result, Some("dr.".to_owned()), "{source}");
        } else if source == "doctor doctor" {
            assert_eq!(result, Some("dr. doctor".to_owned()), "{source}");
        } else {
            assert_eq!(result, None, "{source}");
        }
    }
}

#[test]
fn word_realizer_formats_words_with_attached_punctuation() {
    for (source, expected) in [
        ("twenty!", "20 !"),
        ("e s three", "es3"),
        ("E S three hundred", "ES300"),
        ("one hundred and five?", "105 ?"),
        ("twenty…", "20 …"),
        ("123!", "123 !"),
    ] {
        assert_eq!(
            realize_known_kind("WORD", source),
            Some(expected.to_owned()),
            "{source}"
        );
    }
    for source in ["twenty", "e s", "x tomorrow", "🙂!", "twenty!!"] {
        assert_eq!(realize_known_kind("WORD", source), None, "{source}");
    }
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
fn delegated_realizers_reject_unrelated_suffixes() {
    assert_eq!(
        realize_known_kind("ELECTRONIC", "a at gmail dot com tomorrow"),
        None
    );
    for source in [
        "a at gmail dot com dot",
        "a at gmail dot",
        "a at dot com",
        "dot com",
        "http colon slash slash",
    ] {
        assert_eq!(realize_known_kind("ELECTRONIC", source), None, "{source}");
    }
    assert_eq!(
        realize_known_kind(
            "PHONE",
            "nine eight two zero five five one two three four tomorrow"
        ),
        None
    );
    for source in [
        "nine eight two zero five five one two three four double",
        "nine eight two zero five five one two three four plus",
        "nine eight two zero five five one two three four naught",
    ] {
        assert_eq!(realize_known_kind("PHONE", source), None, "{source}");
    }
}

#[test]
fn malformed_digit_sequence_fails_closed() {
    assert_eq!(
        realize_known_kind("DIGIT_SEQUENCE", "seven eighty eight"),
        None
    );
}
