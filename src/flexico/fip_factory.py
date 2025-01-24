from abc import ABC, abstractmethod

import pandas as pd
import xgboost as xgb
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

METRICS = ["comet22", "comet22-qe", "chrf", "sacrebleu"]

BASE_CONT_AWARE_FEATURES = [
    "sent_overlap_ratio",
    "2gram_freq_dist_diff-jensenShannon",
    "3gram_freq_dist_diff-jensenShannon",
    "4gram_freq_dist_diff-jensenShannon",
    "sent_embedding_cluster_dist-cosine",
    "sent_embedding_cluster_dist-euclidean",
]


class FIP_factory(ABC):
    def __init__(
        self,
        fid: pd.DataFrame,
        dataset_specific_basic_features: list,
        dataset_specific_content_aware_features: list,
        dataset_specific_sys_perf_features: list,
        splits_dict: dict,
        model_type: str = "rf",
        val_split_percent: float = 0.2,  # 80% for train; 20% for val
        seed: int = 1,
        verbose: bool = False,
    ):
        self.fid = fid
        self.splits_dict = splits_dict
        self.model_type = model_type
        self.val_split_percent = val_split_percent
        self.seed = seed
        self.verbose = verbose

        basic_features = [
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
            "ratio_new_old_data_english_words_total",
            "ratio_new_old_data_english_words_trimmed",
        ] + dataset_specific_basic_features

        sys_perf_features = [
            "curr_comet22",
            "curr_comet22-qe",
            "curr_sacrebleu",
            "curr_chrf",
        ] + dataset_specific_sys_perf_features

        content_aware_features = [
            "new_data-sent_overlap_ratio",
            "new_data-2gram_freq_dist_diff-jensenShannon",
            "new_data-3gram_freq_dist_diff-jensenShannon",
            "new_data-4gram_freq_dist_diff-jensenShannon",
            "new_data-sent_embedding_cluster_dist-cosine",
            "new_data-sent_embedding_cluster_dist-euclidean",
            "finetune_data-sent_overlap_ratio",
            "finetune_data-sent_embedding_cluster_dist-cosine",
            "finetune_data-sent_embedding_cluster_dist-euclidean",
        ] + dataset_specific_content_aware_features

        self.features_dict = {
            "basic": basic_features,
            "sys_perf": sys_perf_features,
            "basic_sys_perf": basic_features + sys_perf_features,
            "contentAware": content_aware_features,
            "all": basic_features + sys_perf_features + content_aware_features,
        }

    @abstractmethod
    def prepare_splits(self):
        pass

    def build_predictor(self):
        predictor_type = self.model_type
        if predictor_type is None:
            clf = None
        elif (
            "rf" in predictor_type
            or "randomForest" in predictor_type
            or "random_forest" in predictor_type
        ):  # RANDOM FOREST
            clf = RandomForestRegressor(
                random_state=self.seed
            )  # n_estimators=5, max_depth=12)
        elif "mlp" in predictor_type:  # MultiLayer Perceptron (MLP)
            clf = MLPRegressor(random_state=self.seed)
        elif "lin" in predictor_type or "linear" in predictor_type:  # LINEAR MODEL
            clf = linear_model.LinearRegression()
        elif "xgb" in predictor_type:  # XGBOOST
            clf = xgb.XGBRegressor(
                eval_metric="mae", random_state=self.seed
            )  # base_score=0, n_estimators=2000, max_depth=12, learning_rate=0.02, subsample=0.8)
        else:
            print(f"[W] Predictor {predictor_type} not implemented")
            clf = None

        if self.verbose:
            print(f"[D] Predictor {predictor_type} implemented as {type(clf)}")
        return clf

    def train_predictor(self, predictor, x_train, y_train, x_val, y_val):
        if "xgb" in self.model_type:
            predictor.fit(
                x_train,
                y_train,
                eval_set=[[x_val, y_val]],
                verbose=100,
                early_stopping_rounds=200,
            )  # base_margin=None)
        else:
            predictor.fit(x_train, y_train)

        return predictor

    def build_fip(self, feature_set: str, target: str):
        features = self.features_dict[feature_set]

        aip = self.build_predictor()

        aip = self.train_predictor(
            predictor=aip,
            x_train=self.splits_dict["train"][features],
            y_train=self.splits_dict["train"][target],
            x_val=self.splits_dict["val"][features],
            y_val=self.splits_dict["val"][target],
        )

        return aip, features


class HKNewsFIP_factory(FIP_factory):
    def __init__(
        self,
        fid: pd.DataFrame,
        test_sets: list = None,
        model_type: str = "rf",
        fid_type: str = "generic",
        val_split_percent: float = 0.2,
        seed: int = 1,
        verbose: bool = False,
    ):
        self.fid = fid
        self.val_split_percent = val_split_percent
        self.verbose = verbose

        train_split, val_split, test_split = self.prepare_splits()
        splits_dict = {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        }

        basic_features = [
            # "count_old_data_chinese_words_total",
            # "count_old_data_chinese_words_trimmed",
            "count_new_data_chinese_words_total",
            "count_new_data_chinese_words_trimmed",
            "count_finetune_data_chinese_words_total",
            "count_finetune_data_chinese_words_trimmed",
            "ratio_new_old_data_chinese_words_total",
            "ratio_new_old_data_chinese_words_trimmed",
        ]

        content_aware_features = []
        sys_perf_features = []
        if "generic" in fid_type:
            content_aware_features.append(
                "finetune_data-2gram_freq_dist_diff-jensenShannon"
            )
            content_aware_features.append(
                "finetune_data-3gram_freq_dist_diff-jensenShannon"
            )
            content_aware_features.append(
                "finetune_data-4gram_freq_dist_diff-jensenShannon"
            )
            for feature in BASE_CONT_AWARE_FEATURES:
                content_aware_features.append(f"new_data-test_set-{feature}")
                content_aware_features.append(f"finetune_data-test_set-{feature}")

            for mt_metric in METRICS:
                sys_perf_features.append(f"curr_test_set_{mt_metric}")
        else:
            for test_set in test_sets:
                for feature in BASE_CONT_AWARE_FEATURES:
                    content_aware_features.append(f"new_data-{test_set}-{feature}")
                    content_aware_features.append(f"finetune_data-{test_set}-{feature}")

                for mt_metric in METRICS:
                    sys_perf_features.append(f"curr_{test_set}_{mt_metric}")

        super().__init__(
            fid,
            basic_features,
            content_aware_features,
            sys_perf_features,
            splits_dict,
            model_type,
            val_split_percent,
            seed,
            verbose,
        )

    def prepare_splits(self):
        if self.verbose:
            print("[D] Creating TRAIN / VAL / TEST splits:")

        df = self.fid

        # use years 1997 and 1998 for train and val
        # use year 1999 for test
        # year 2000 was used to create the fixed test-sets

        train_val_split = df.loc[df["period_end_timestamp"] < "1999-01-01"]
        # TRAIN SPLIT
        train_split = train_val_split.head(
            int(len(train_val_split) * (1 - self.val_split_percent))
        ).copy()

        # VAL SPLIT
        val_split = train_val_split.tail(
            int(len(train_val_split) - len(train_split))
        ).copy()

        assert len(train_split) + len(val_split) == len(train_val_split)

        # TEST SPLIT
        test_split = df.loc[
            (df["prev_finetune_period_start_timestamp"] >= "1999-01-01")
            & (df["period_end_timestamp"] < "2000-01-01")
        ].copy()

        if self.verbose:
            print("\tSET SIZES:")
            print(f"\t\ttrain set: {len(train_split)}")
            print(f"\t\tval set: {len(val_split)}")
            print(f"\t\ttest set: {len(test_split)}")

        return train_split, val_split, test_split


class OpusFIP_factory(FIP_factory):
    def __init__(
        self,
        fid: pd.DataFrame,
        test_sets: list = None,
        model_type: str = "rf",
        fid_type: str = "generic",
        val_split_percent: float = 0.2,
        seed: int = 1,
        verbose: bool = False,
    ):
        self.fid = fid
        self.val_split_percent = val_split_percent
        self.verbose = verbose

        train_split, val_split, test_split = self.prepare_splits()
        splits_dict = {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        }

        basic_features = [
            "count_old_data_french_words_total",
            "count_old_data_french_words_trimmed",
            "count_new_data_french_words_total",
            "count_new_data_french_words_trimmed",
            "count_finetune_data_french_words_total",
            "count_finetune_data_french_words_trimmed",
            "ratio_new_old_data_french_words_total",
            "ratio_new_old_data_french_words_trimmed",
        ]

        content_aware_features = []
        sys_perf_features = []
        if "specific" in fid_type:
            for test_set in test_sets:
                content_aware_features.append(
                    f"finetune_data-{test_set}-sent_overlap_ratio"
                )
                content_aware_features.append(
                    f"finetune_data-{test_set}-sent_embedding_cluster_dist-cosine"
                )
                content_aware_features.append(
                    f"finetune_data-{test_set}-sent_embedding_cluster_dist-euclidean"
                )
                for feature in BASE_CONT_AWARE_FEATURES:
                    content_aware_features.append(f"new_data-{test_set}-{feature}")

                for mt_metric in METRICS:
                    sys_perf_features.append(f"curr_{test_set}_{mt_metric}")
        else:
            content_aware_features.append("finetune_data-test_set-sent_overlap_ratio")
            content_aware_features.append(
                "finetune_data-test_set-sent_embedding_cluster_dist-cosine"
            )
            content_aware_features.append(
                "finetune_data-test_set-sent_embedding_cluster_dist-euclidean"
            )
            for feature in BASE_CONT_AWARE_FEATURES:
                content_aware_features.append(f"new_data-test_set-{feature}")

            for mt_metric in METRICS:
                sys_perf_features.append(f"curr_test_set_{mt_metric}")

        super().__init__(
            fid,
            basic_features,
            content_aware_features,
            sys_perf_features,
            splits_dict,
            model_type,
            val_split_percent,
            seed,
            verbose,
        )

    def prepare_splits(self):
        """Create TRAIN / VAL / TEST splits"""

        train_split_percent = 0.7  # 70% for train + val and 30% for test

        time_interval = self.fid["prev_finetune"].min()

        duration = len(list(self.fid["prev_finetune"].unique()))

        train_val_duration = int(duration * train_split_percent)

        test_start_time = (
            self.fid["prev_finetune"].min() + train_val_duration * time_interval
        )

        train_val_split = self.fid.loc[self.fid["curr_finetune"] < test_start_time]

        val_split_time = train_val_split["prev_finetune"].min() + (
            train_val_duration * time_interval * (1 - self.val_split_percent)
        )

        # TRAIN SPLIT
        train_split = train_val_split.loc[
            train_val_split["curr_finetune"] < val_split_time
        ]

        # VAL SPLIT
        val_split = train_val_split.loc[
            (train_val_split["prev_finetune"] >= val_split_time)
        ].copy()

        # TEST SPLIT
        test_split = self.fid.loc[(self.fid["prev_finetune"] >= test_start_time)].copy()
        test_start_week = test_split["curr_finetune"].to_numpy()[0]

        if self.verbose:
            print("[D] Creating TRAIN / VAL / TEST splits:")
            print(f"\tmin finetune id: {self.fid['prev_finetune'].min()}")
            print(f"\tmax finetune id: {self.fid['curr_finetune'].max()}")
            print(f"\tduration: {duration}")
            print(f"\ttrain_val_duration: {train_val_duration}")
            print(f"\ttest_start_time: {test_start_time}")
            print(f"\tval_split_time: {val_split_time}")

            print(f"\ttest_start_week: {test_start_week}")

            print("\tSET SIZES:")
            print(f"\t\ttrain set: {len(train_split)}")
            print(f"\t\tval set: {len(val_split)}")
            print(f"\t\ttest set: {len(test_split)}")

        return train_split, val_split, test_split
