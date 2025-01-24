METRICS = [
    'comet22',
    'comet22-qe',
    'chrf',
    'sacrebleu',
]

OPUS_TEST_SETS = [
    'elitr',
    'elrc',
    'euBookshop',
    'php',
    'tanzil',
    'tedx-fr',
    'tedx-fr_ca',
]

BASIC_FEATURES = [
    "finetune_delta",
    "amount_old_data",
    "amount_new_data",
    "amount_finetune_data",
    "total_data",
    "ratio_new_old_data",
    "count_old_data_english_words_total",
    "count_old_data_english_words_trimmed",
    "count_new_data_english_words_total",
    "count_new_data_english_words_trimmed",
    "count_finetune_data_english_words_total",
    "count_finetune_data_english_words_trimmed",
    "count_old_data_french_words_total",
    "count_old_data_french_words_trimmed",
    "count_new_data_french_words_total",
    "count_new_data_french_words_trimmed",
    "count_finetune_data_french_words_total",
    "count_finetune_data_french_words_trimmed",
    "ratio_new_old_data_english_words_total",
    "ratio_new_old_data_english_words_trimmed",
    "ratio_new_old_data_french_words_total",
    "ratio_new_old_data_french_words_trimmed",
]


SYS_PERF_FEATURES = [
    "curr_comet22",
    "curr_comet22-qe",
    "curr_sacrebleu",
    "curr_chrf",
]
for test_set in OPUS_TEST_SETS:
    for metric in METRICS:
        feature = f"curr_{test_set}_{metric}"
        SYS_PERF_FEATURES.append(feature)

NGRAM_FEATURES = {
    "new_data": [] , 
    "finetune_data": [] ,
}
EMBEDDING_FEATURES = {
    "new_data": [] , 
    "finetune_data": [] ,
}
SENT_OVERLAP_FEATURES = {
    "new_data": [] , 
    "finetune_data": [] ,
}
CONTENT_AWARE_FEATURES = []
for data_set in ["new_data", "finetune_data"]:
    for test_set in ["none"] + OPUS_TEST_SETS:
        if "none" in test_set:
            key = f"{data_set}"
        else:
            key = f"{data_set}-{test_set}"

        if 'new_data' in data_set:
            NGRAM_FEATURES[data_set] = NGRAM_FEATURES[data_set] + [
                f"{key}-2gram_freq_dist_diff-jensenShannon",
                f"{key}-3gram_freq_dist_diff-jensenShannon",
                f"{key}-4gram_freq_dist_diff-jensenShannon",
            ]

        EMBEDDING_FEATURES[data_set] = EMBEDDING_FEATURES[data_set] + [
            f"{key}-sent_embedding_cluster_dist-cosine",
            f"{key}-sent_embedding_cluster_dist-euclidean",
        ]

        SENT_OVERLAP_FEATURES[data_set] = SENT_OVERLAP_FEATURES[data_set] + [
            f"{key}-sent_overlap_ratio",
        ]

    CONTENT_AWARE_FEATURES = CONTENT_AWARE_FEATURES + (
        SENT_OVERLAP_FEATURES[data_set] + NGRAM_FEATURES[data_set] + EMBEDDING_FEATURES[data_set]
    )

# content-aware features for the generic FIPs
base_contentAware_features = [
    'sent_overlap_ratio',
    '2gram_freq_dist_diff-jensenShannon',
    '3gram_freq_dist_diff-jensenShannon',
    '4gram_freq_dist_diff-jensenShannon',
    'sent_embedding_cluster_dist-cosine',
    'sent_embedding_cluster_dist-euclidean',
]

GENERIC_CONTENT_AWARE_FEATURES = []
for data_set in ['new_data', 'finetune_data']:
    for f in base_contentAware_features:
        if 'finetune_data' in data_set and 'gram' in f:
            continue
        GENERIC_CONTENT_AWARE_FEATURES.append(f"{data_set}-{f}")
        GENERIC_CONTENT_AWARE_FEATURES.append(f"{data_set}-test_set-{f}")

GENERIC_SYS_PERF_FEATURES = [
    "curr_comet22",
    "curr_comet22-qe",
    "curr_sacrebleu",
    "curr_chrf",
]
for metric in METRICS:
    GENERIC_SYS_PERF_FEATURES.append(f"curr_test_set_{metric}")


TARGETS = []
for test_set in OPUS_TEST_SETS:
    for metric in METRICS:
        TARGETS.append(f"target_{test_set}_{metric}")
        TARGETS.append(f"delta-target_{test_set}_{metric}")
