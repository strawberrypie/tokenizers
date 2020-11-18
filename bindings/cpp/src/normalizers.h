#pragma once

#include "tokenizers_util.h"
#include "tokenizers-cpp/src/normalizers.rs.h"

#include <string>

namespace huggingface {
namespace tokenizers {
struct NormalizedString {
    HFT_FFI_WRAPPER(NormalizedString);

public:
    static HFT_RESULT(NormalizedString) from(const std::string& str) {
        HFT_TRY(NormalizedString, {ffi::normalized_string(str)});
    }
};

struct BertNormalizer {
    HFT_FFI_WRAPPER(BertNormalizer);

public:
    BertNormalizer(bool clean_text, bool handle_chinese_chars,
                   BertStripAccents strip_accents, bool lowercase)
        : inner_(ffi::bert_normalizer(clean_text, handle_chinese_chars,
                                      strip_accents, lowercase)){};

    HFT_RESULT_VOID normalize(NormalizedString& normalized) {
        HFT_TRY_VOID(ffi::normalize_bert(*inner_, *normalized));
    }
};

struct BertNormalizerOptions {
    HFT_BUILDER_ARG(bool, clean_text, true);
    HFT_BUILDER_ARG(bool, handle_chinese_chars, true);
    HFT_BUILDER_ARG(bool, lowercase, true);

    BertStripAccents strip_accents = BertStripAccents::DeterminedByLowercase;
#pragma warning(suppress : 4458)
    BertNormalizerOptions& with_strip_accents(bool strip_accents) {
        this->strip_accents =
            strip_accents ? BertStripAccents::True : BertStripAccents::False;
        return *this;
    }

    BertNormalizer build() {
        return BertNormalizer(clean_text, handle_chinese_chars, strip_accents,
                              lowercase);
    }
};

}  // namespace tokenizers
}  // namespace huggingface
