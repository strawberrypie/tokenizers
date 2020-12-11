#[cxx::bridge(namespace = "huggingface::tokenizers")]
pub mod ffi {
    pub enum BertStripAccents {
        DeterminedByLowercase,
        False,
        True,
    }

    extern "C++" {
        include!("tokenizers-cpp/normalizers.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type NormalizedString;
        type Normalizer;

        fn normalized_string(str: &str) -> Box<NormalizedString>;

        fn bert_normalizer(
            clean_text: bool,
            handle_chinese_chars: bool,
            strip_accents: BertStripAccents,
            lowercase: bool,
        ) -> Box<Normalizer>;

        fn strip_normalizer(strip_left: bool, strip_right: bool) -> Box<Normalizer>;

        fn strip_accents_normalizer() -> Box<Normalizer>;

        fn nfc_normalizer() -> Box<Normalizer>;

        fn nfd_normalizer() -> Box<Normalizer>;

        fn nfkc_normalizer() -> Box<Normalizer>;

        fn nfkd_normalizer() -> Box<Normalizer>;

        fn lowercase_normalizer() -> Box<Normalizer>;

        fn nmt_normalizer() -> Box<Normalizer>;

        fn precompiled_normalizer(precompiled_charsmap: &[u8]) -> Result<Box<Normalizer>>;

        fn replace_literal_normalizer(pattern: &str, content: &str) -> Result<Box<Normalizer>>;

        fn replace_regex_normalizer(pattern: &str, content: &str) -> Result<Box<Normalizer>>;

        fn sequence_normalizer(normalizers: Vec<Normalizer>) -> Box<Normalizer>;

        fn normalize(normalizer: &Normalizer, normalized: &mut NormalizedString) -> Result<()>;

        fn get_normalized(normalized: &NormalizedString) -> &str;
        fn get_original(normalized: &NormalizedString) -> &str;
    }
}

use derive_more::{Deref, DerefMut};
use tk::{
    normalizers::{
        replace::ReplacePattern, BertNormalizer, Lowercase, Nmt, Precompiled, Replace, Sequence,
        Strip, StripAccents, NFC, NFD, NFKC, NFKD,
    },
    Normalizer as NormalizerTrait, Result,
};

#[derive(Deref, DerefMut)]
struct NormalizedString(tk::NormalizedString);

#[derive(Deref, DerefMut, Clone)]
pub struct Normalizer(pub tk::NormalizerWrapper);

impl NormalizerTrait for Normalizer {
    fn normalize(&self, normalized: &mut tk::NormalizedString) -> Result<()> {
        self.0.normalize(normalized)
    }
}

use ffi::BertStripAccents;

fn normalized_string(str: &str) -> Box<NormalizedString> {
    Box::new(NormalizedString(str.into()))
}

fn make_normalizer<N: Into<tk::NormalizerWrapper>>(normalizer: N) -> Box<Normalizer> {
    Box::new(Normalizer(normalizer.into()))
}

fn bert_normalizer(
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: BertStripAccents,
    lowercase: bool,
) -> Box<Normalizer> {
    let strip_accents = match strip_accents {
        BertStripAccents::False => Some(false),
        BertStripAccents::True => Some(true),
        BertStripAccents::DeterminedByLowercase => None,
        _ => None,
    };
    make_normalizer(BertNormalizer::new(
        clean_text,
        handle_chinese_chars,
        strip_accents,
        lowercase,
    ))
}

fn strip_normalizer(strip_left: bool, strip_right: bool) -> Box<Normalizer> {
    make_normalizer(Strip::new(strip_left, strip_right))
}

fn strip_accents_normalizer() -> Box<Normalizer> {
    make_normalizer(StripAccents)
}

fn nfc_normalizer() -> Box<Normalizer> {
    make_normalizer(NFC)
}

fn nfd_normalizer() -> Box<Normalizer> {
    make_normalizer(NFD)
}

fn nfkc_normalizer() -> Box<Normalizer> {
    make_normalizer(NFKC)
}

fn nfkd_normalizer() -> Box<Normalizer> {
    make_normalizer(NFKD)
}

fn lowercase_normalizer() -> Box<Normalizer> {
    make_normalizer(Lowercase)
}

fn nmt_normalizer() -> Box<Normalizer> {
    make_normalizer(Nmt)
}

fn precompiled_normalizer(precompiled_charsmap: &[u8]) -> Result<Box<Normalizer>> {
    Ok(make_normalizer(Precompiled::from(precompiled_charsmap)?))
}

fn replace_literal_normalizer(pattern: &str, content: &str) -> Result<Box<Normalizer>> {
    Ok(make_normalizer(Replace::new(
        ReplacePattern::String(pattern.to_string()),
        content,
    )?))
}

fn replace_regex_normalizer(pattern: &str, content: &str) -> Result<Box<Normalizer>> {
    Ok(make_normalizer(Replace::new(
        ReplacePattern::Regex(pattern.to_string()),
        content,
    )?))
}

fn sequence_normalizer(normalizers: Vec<Normalizer>) -> Box<Normalizer> {
    make_normalizer(Sequence::new(
        normalizers.into_iter().map(|n| n.0).collect(),
    ))
}

fn normalize(normalizer: &Normalizer, normalized: &mut NormalizedString) -> Result<()> {
    normalizer.normalize(normalized)
}

fn get_normalized(normalized: &NormalizedString) -> &str {
    normalized.get()
}

fn get_original(normalized: &NormalizedString) -> &str {
    normalized.get_original()
}
