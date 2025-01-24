from abc import ABC, abstractmethod
from datetime import date, timedelta

import pandas as pd
from datasets import Dataset

from data_preprocessing.utils_data_preprocessing import remove_empty_strings_from_col
from constants import BASE_DATA_DIR


class MyDataset(ABC):

    """
    Base class for creating each dataset instance

    All datasets must have the following columns:
        - source
        - target
    """

    def __init__(self, dataset_name: str, df: pd.DataFrame):
        self.dataset_name = dataset_name

        df = df.loc[(df["target"].notnull()) & (df["source"].notnull())]

        df = remove_empty_strings_from_col(
            df=df,
            col="source",
        )

        df = remove_empty_strings_from_col(
            df=df,
            col="target",
        )

        self.raw_dataset = df.reset_index(drop=True).copy()
        print(f"[D] Dataset has {len(self.raw_dataset)} rows")

    @staticmethod
    def _add_week_col(df: pd.DataFrame):
        # offset if dataset does not start on a monday
        first_day_of_week = df.loc[df["timestamp"] == df["timestamp"].min()][
            "timestamp"
        ].dt.dayofweek.to_numpy()[0]
        df["shifted_timestamp"] = df["timestamp"] - timedelta(
            days=int(first_day_of_week)
        )

        df["year"] = df["shifted_timestamp"].dt.isocalendar().year
        df["year_number"] = df["year"] - df["year"].min()
        df["num_weeks_in_year"] = (
            pd.to_datetime(df["year"].apply(lambda x: date(x, 12, 28)))
            .dt.isocalendar()
            .week
        )
        df["year_week_number"] = df["shifted_timestamp"].dt.isocalendar().week

        new_week_numbers = []
        year_weeks = df["year_week_number"].tolist()
        weeks_in_year = df["num_weeks_in_year"].tolist()
        aux = cumulative_week_number = year_weeks[0]
        for curr_week, last_year_week in zip(year_weeks, weeks_in_year):
            # starting new week
            if curr_week == aux + 1:
                cumulative_week_number += 1
                aux += 1

            # last week of the year
            if curr_week == last_year_week:
                last_week_number = cumulative_week_number

            # new year has started
            if curr_week == 1:
                aux = 1
                cumulative_week_number = last_week_number + 1

            new_week_numbers.append(cumulative_week_number)

        df["week_number"] = new_week_numbers

        df = df.astype({"week_number": "int32"})

        return df.drop(
            columns=[
                "shifted_timestamp",
                "year",
                "year_number",
                "num_weeks_in_year",
                "year_week_number",
            ]
        ).copy()

    def get_raw_dataset(self):
        return self.raw_dataset

    def get_columns_to_drop(self):
        return self.columns_to_drop

    def has_time_columns(self):
        return bool("timestamp" in self.raw_dataset.columns)

    def get_source_lang(self):
        return self.source_lang

    def get_target_lang(self):
        return self.target_lang

    @abstractmethod
    def get_test_set(self, test_set_format: str = "huggingFace"):
        pass

    @abstractmethod
    def get_test_set_names(self):
        pass

    @abstractmethod
    def is_test_set_fixed(self):
        pass


class HknewsDataset(MyDataset):
    def __init__(self, dataset_name, fixed_test_set: bool = False):
        # Additional initialization code specific to the concrete class
        dataset = pd.read_csv(
            f"{BASE_DATA_DIR}hksar_news/processed_hksar_news_dataset_utf8.csv",
            sep="\t",
            dtype=str,
            encoding="utf-8",
        )

        dataset["timestamp"] = pd.to_datetime(
            dataset[["day", "month", "year"]].astype(str).apply(" ".join, 1),
            format="%d %m %Y",
        )

        dataset = dataset.sort_values(by=["timestamp", "en_seq_num"])

        dataset = self._add_week_col(df=dataset)

        self.columns_to_drop = [
            "month",
            "day",
            "en_seq_num",
            "ch_seq_num",
            "source_lang",
            "target_lang",
            "timestamp",
            "week_number",
        ]

        self.source_lang = "english"
        self.target_lang = "chinese"

        super().__init__(
            dataset_name, dataset
        )  # Call the constructor of the abstract class

        # load fixed test sets
        self.fixed_test_set = fixed_test_set
        if fixed_test_set:
            hf_test_dataset = {}
            pandas_test_dataset = {}
            self.test_set_names = [
                "Finance",
                "Entertainment",
                "TravelAndtourism",
                "HealthAndwellness",
                "Sports",
                "Environment",
                "Governance",
            ]
            for topic in self.test_set_names:
                test_dataset_df = pd.read_pickle(
                    f"{BASE_DATA_DIR}hksar_news/hksar_news-{topic}_testSet.pkl"
                )
                pandas_test_dataset[topic] = test_dataset_df.drop(
                    columns=[
                        col
                        for col in list(test_dataset_df.columns)
                        if col not in ("source", "target")
                    ]
                )
                hf_test_dataset[topic] = Dataset.from_pandas(
                    df=pandas_test_dataset[topic]
                ).remove_columns(["__index_level_0__"])

            self.test_set = {
                "pandas": pandas_test_dataset,
                "huggingFace": hf_test_dataset,
            }
        else:
            # for the case when we want to evaluate the following week
            self.test_set = "week_after_finetune"
            self.test_set_names = None

    def get_test_set(self, test_set_format: str = "huggingFace"):
        if "pandas" in test_set_format or "huggingFace" in test_set_format:
            return self.test_set[test_set_format]
        else:
            print(
                f"[E] Unknown fixed test set format {test_set_format}. Defaulting to huggingFace format"
            )
            return self.hf_test_set

    def get_test_set_names(self):
        return self.test_set_names

    def is_test_set_fixed(self):
        return self.fixed_test_set


class OpusDataset(MyDataset):
    def __init__(self, dataset_name, fixed_test_set: bool = False):
        # Additional initialization code specific to the concrete class
        tokens = dataset_name.split("-")
        time_interval = tokens[1].split("_")[1]
        seed = tokens[2].split("_")[1]
        source_lang = tokens[3]
        target_lang = tokens[4]

        dataset = pd.read_pickle(
            f"{BASE_DATA_DIR}fra_eng-datasets/finetune_set-en_fr-chunkSize_{time_interval}-seed_{seed}.pkl",
        )

        if "eng" in source_lang and "fra" in target_lang:
            self.source_lang = "english"
            self.target_lang = "french"
            rename_columns = {"english": "source", "french": "target"}
        elif "fra" in source_lang and "eng" in target_lang:
            self.source_lang = "french"
            self.target_lang = "english"
            rename_columns = {"english": "target", "french": "source"}
        else:
            print(f"[E] Unknown language pair {source_lang}-{target_lang}")
        dataset = dataset.rename(columns=rename_columns)
        super().__init__(
            dataset_name, dataset
        )  # Call the constructor of the abstract class

        self.columns_to_drop = ["original_opus_dataset_name"]

        self.fixed_test_set = fixed_test_set
        # load test set df (which is fixed for this dataset / use-case)
        if fixed_test_set:
            test_set_df = pd.read_pickle(
                f"{BASE_DATA_DIR}/fra_eng-datasets/test_set-en_fr-chunkSize_{time_interval}-seed_{seed}.pkl",
            ).rename(columns=rename_columns)

            self.test_set_names = list(test_set_df.original_opus_dataset_name.unique())
            # test set for hugging face is a dictionary of the several test sets
            #   Dict[str, torch.utils.data.Dataset]
            hf_test_dataset = {}
            pandas_test_dataset = {}
            for opus_dataset_name in self.test_set_names:
                pandas_test_dataset[opus_dataset_name] = test_set_df.loc[
                    test_set_df.original_opus_dataset_name == opus_dataset_name
                ].drop(columns=self.columns_to_drop)

                hf_test_dataset[opus_dataset_name] = Dataset.from_pandas(
                    df=pandas_test_dataset[opus_dataset_name]
                ).remove_columns(["__index_level_0__"])

            self.test_set = {
                "pandas": pandas_test_dataset,
                "huggingFace": hf_test_dataset,
            }
        else:
            # for the case when we want to evaluate the following chunk of sentences
            self.test_set = "sentences_after_finetune"
            self.test_set_names = None

    def get_test_set(self, test_set_format: str = "huggingFace"):
        if "pandas" in test_set_format or "huggingFace" in test_set_format:
            return self.test_set[test_set_format]
        else:
            print(
                f"[E] Unknown fixed test set format {test_set_format}. Defaulting to huggingFace format"
            )
            return self.hf_test_set

    def is_test_set_fixed(self):
        return self.fixed_test_set

    def get_test_set_names(self):
        return self.test_set_names
