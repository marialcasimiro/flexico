# Create an FID

To generate the Finetune Impact Dataset (FID), it is necessary to:

1. finetune models at each "time" interval and evaluate the finetuned
    model on all consecutive "time" intervals, saving all metrics,
    including the sets of "old" and "new" data
    (Note: "time" can be defined either as:
    - weeks ==> finetune_periods = 1 implies finetune each week
    - sentences ==> finetune_periods = 1000 implies finetune after 1000
    new sentences have been received)

2. iterate over every possible combination of finetune intervals and
    compute the desired features (e.g. ratio of sentence overlap)


## Generate metrics files

The first task that needs to be performed when creating an FID is to deploy the target adaptation tactic multiple times and collect metrics.

This is done via script `generate_metrics_files.py`.

The following parameters can be set:

```
    time_interval:      Set time_interval period\n\t1 time_interval period = 1 week if time_interval_type=time\n\totherwise specify number of sentences
    finetune_instant:   When set, finetune the model when this moment is reached. Otherwise, gather metrics for all finetune instants
    finetune_type:      Type of fine-tune [incremental; base]
    time_interval_type: Type of time_interval periodicity [time; sentence]
    gpu_off / gpu_on:   Do not use / use GPU to fine-tune the MT model
    huggingface_token:  Token to login to HuggingFace. This is required to access model comet22-kiwi
```

This generates a set of finetuned models (finetuned_models.zip) and a set of metrics files (tmp_metrics.zip).

## Generate FID features

Once all instances of the tactic's execution have been complete, it is time to generate the features that the FID will use.

This is done via script `generate_fid_features.py`.

Currently, the following groups of features can be computed:

```
    all:        compute all features at once
    complex:    all except basic
    basic:      for example: ["finetune_delta", 
                 "amount_old_data", 
                 "amount_new_data", 
                 "amount_new_source_words", 
                 "amount_new_target_words", 
                 "total_data", 
                 "ratio_new_old_data", 
                 "curr_comet22", 
                 "curr_sacrebleu", 
                 "curr_chrf"]

    sent_overlap_ratio

    sent_fuzzy_score:   ["fuzzy_score-ratio", 
                         "fuzzy_score-tokenSetRatio", 
                         "fuzzy_score-partialTokenSetRatio", 
                         "fuzzy_score-tokenSortRatio", 
                         "fuzzy_score-tokenRatio", 
                         "fuzzy_score-partialTokenRatio", 
                         "fuzzy_score-QRatio"]

    ngram_freq_dist_diff:   ["2gram_freq_dist_diff-jensenShannon", 
                             "3gram_freq_dist_diff-jensenShannon", 
                             "4gram_freq_dist_diff-jensenShannon"]

    sent_embedding_cluster_dist:    ["sent_embedding_cluster_dist-cosine", 
                                     "sent_embedding_cluster_dist-euclidean"]
```

This generates a set of feature files (fid_tmp_files.zip).



## Create FID

Finally, to create the FID, all the feature files previously computed need to be merged.

This is done via script `create_fid.py`.
