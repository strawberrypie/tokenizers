#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/normalizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
/**
 * @brief A NormalizedString takes care of processing an "original" string to
 * modify it and obtain a "normalized" string.
 */
struct NormalizedString {
    HFT_FFI_WRAPPER(NormalizedString);

public:
    /**
     * @brief Construct a new NormalizedString object.
     *
     * @param str The original string
     */
    explicit NormalizedString(nonstd::string_view str) noexcept
        : inner_(ffi::normalized_string(ffi::to_rust_str(str))) {}

    /**
     * @brief Returns the normalized string.
     */
    operator nonstd::string_view() noexcept { return get_normalized(); }

    /**
     * @brief Returns the normalized string.
     */
    nonstd::string_view get_normalized() noexcept {
        return ffi::to_string_view(ffi::get_normalized(*inner_));
    }

    /**
     * @brief Returns the original string.
     */
    nonstd::string_view get_original() noexcept {
        return ffi::to_string_view(ffi::get_original(*inner_));
    }
};

/**
 * @brief Takes care of string pre-processing.
 */
struct Normalizer {
    HFT_FFI_WRAPPER(Normalizer);

public:
    /**
     * @brief BERT Normalizer.
     *
     * @param clean_text Whether to do the bert basic cleaning:
     *   1. Remove any control characters
     *   2. Replace all sorts of whitespace by the classic one ` `
     * @param handle_chinese_chars Whether to split each Chinese character into
     * a separate token
     * @param strip_accents Whether to strip accents
     * @param lowercase Whether to lowercase the input
     */
    static Normalizer bert(bool clean_text, bool handle_chinese_chars,
                           BertStripAccents strip_accents,
                           bool lowercase) noexcept {
        return {ffi::bert_normalizer(clean_text, handle_chinese_chars,
                                     strip_accents, lowercase)};
    }

    /**
     * @brief This Normalizer strips whitespace from string ends.
     *
     * @param strip_left Whether to strip whitespace on the left
     * @param strip_right Whether to strip whitespace on the right
     */
    static Normalizer strip(bool strip_left, bool strip_right) noexcept {
        return {ffi::strip_normalizer(strip_left, strip_right)};
    }

    /**
     * @brief This Normalizer removes combining marks.
     */
    static Normalizer strip_accents() noexcept {
        return {ffi::strip_accents_normalizer()};
    }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFC Unicode normalization
     * form).
     */
    static Normalizer nfc() noexcept { return {ffi::nfc_normalizer()}; }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFD Unicode normalization
     * form).
     */
    static Normalizer nfd() noexcept { return {ffi::nfd_normalizer()}; }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFKC Unicode normalization
     * form).
     */
    static Normalizer nfkc() noexcept { return {ffi::nfkc_normalizer()}; }

    /**
     * @brief This Normalizer applies
     * [https://unicode.org/reports/tr15/#Norm_Forms](NFKD Unicode normalization
     * form).
     */
    static Normalizer nfkd() noexcept { return {ffi::nfkd_normalizer()}; }

    /**
     * @brief This Normalizer lowercases the string
     */
    static Normalizer lowercase() noexcept {
        return {ffi::lowercase_normalizer()};
    }

    /**
     * @brief This Normalizer replaces all occurences of a string
     *
     * @param pattern The string to be replaced
     * @param replacement The replacement
     */
    static Normalizer replace_literal(nonstd::string_view pattern,
                                      nonstd::string_view replacement) {
        return {ffi::replace_literal_normalizer(ffi::to_rust_str(pattern),
                                                ffi::to_rust_str(replacement))};
    }

    /**
     * @brief This Normalizer replaces all matches of a regular expression
     *
     * @param pattern The pattern to be replaced (uses [Rust regex
     * syntax](https://docs.rs/regex/1.4.2/regex/#syntax), not C++!)
     * @param replacement The replacement
     */
    static Normalizer replace_regex(nonstd::string_view pattern,
                                    nonstd::string_view replacement) {
        return {ffi::replace_regex_normalizer(ffi::to_rust_str(pattern),
                                              ffi::to_rust_str(replacement))};
    }

    // FIXME not supported yet
    // static Normalizer sequence(nonstd::span<Normalizer> normalizers) {
    //     rust::Vec<ffi::Normalizer> normalizers_ffi;
    //     fill_vec(normalizers_ffi, normalizers,
    //              [](Normalizer&& n) { return n.consume(); });
    //     return {ffi::sequence_normalizer(normalizers_ffi)};
    // }

    /**
     * @brief Applies this normalizer to the argument.
     *
     * @param normalized The NormalizedString to be normalized
     */
    HFT_RESULT_VOID normalize(NormalizedString& normalized) {
        HFT_TRY_VOID(ffi::normalize(*inner_, *normalized));
    }
};

/**
 * @brief Builder for BERT normalizer (see Normalizer::bert()).
 */
struct BertNormalizerBuilder {
    HFT_BUILDER_ARG(bool, clean_text, true);
    HFT_BUILDER_ARG(bool, handle_chinese_chars, true);
    HFT_BUILDER_ARG(bool, lowercase, true);

    BertStripAccents strip_accents = BertStripAccents::DeterminedByLowercase;
    HFT_DISABLE_WARNING_PUSH
    HFT_DISABLE_WARNING(-Wshadow, 4458)
    /**
     * @brief Sets whether the accents should be stripped. By default they are
     * stripped if `lowercase` is `true`, not stripped otherwise.
     */
    BertNormalizerBuilder& with_strip_accents(bool strip_accents) {
        this->strip_accents =
            strip_accents ? BertStripAccents::True : BertStripAccents::False;
        return *this;
    }
    HFT_DISABLE_WARNING_POP

    /**
     * @brief Builds the BERT Normalizer.
     */
    Normalizer build() {
        return Normalizer::bert(clean_text, handle_chinese_chars, strip_accents,
                                lowercase);
    }
};

}  // namespace tokenizers
}  // namespace huggingface
