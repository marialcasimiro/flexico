import argparse
import os

import pandas as pd

from constants import FID_TMP_FILES_DIR, FID_DIR


def add_df_to_dict(df_dict, df, key, params):
    if key in df_dict:
        # "load" existing dataframe
        tmp_df = df_dict[key]

        # check if the new dataframe has columns
        # that are already in the existing dataframe
        # if there are, remove them
        for col in df.columns:
            if col in tmp_df.columns:
                df = df.drop(col, axis=1)

        # merge dataframes
        new_df = tmp_df.join(df)
        df_dict[key] = new_df
    else:
        df["dataset"] = params["dataset"]
        df["timeIntervalType"] = params["time_interval_type"]
        df["timeInterval"] = params["time_interval"]
        df["finetuneType"] = params["finetune_type"]
        df["curr_finetune"] = params["curr_finetune"]
        df["prev_finetune"] = params["prev_finetune"]
        df_dict[key] = df


def parse_filename(filename: str):
    # key contains the following:
    # currFinetune_XX-prevFinetune_XX-timeIntervalType_XX-timeInterval_XX-finetuneType_XX
    if "dataset" not in filename:
        tokens = filename.split("-", maxsplit=2)
        features = tokens[1].split("_", maxsplit=1)[1]
        key = tokens[2][:-4]
        curr_finetune = tokens[2].split("-")[0].split("_")[1]
        prev_finetune = tokens[2].split("-")[1].split("_")[1]
    else:
        if "hk-news" in filename:
            tokens = filename.split("-", maxsplit=4)
            features = tokens[3].split("_", maxsplit=1)[1]
            key = tokens[4][:-4]
            curr_finetune = tokens[4].split("-")[0].split("_")[1]
            prev_finetune = tokens[4].split("-")[1].split("_")[1]
        else:
            tokens = filename.split("-", maxsplit=3)
            features = tokens[2].split("_", maxsplit=1)[1]
            key = tokens[3][:-4]
            curr_finetune = tokens[3].split("-")[0].split("_")[1]
            prev_finetune = tokens[3].split("-")[1].split("_")[1]

    return key, features, curr_finetune, prev_finetune


def merge_fid_tmp_files(
    params,
    tmp_folder: str = FID_TMP_FILES_DIR,
):
    fid_tmp_dict = {}

    # create dict key with dataframe with all features of the same finetune
    for subdir, _, files in os.walk(tmp_folder):
        print(subdir)
        for filename in files:
            print("#" * 100)
            print(filename)
            if (
                filename.endswith(".pkl")
                and f"timeIntervalType_{params['time_interval_type']}-timeInterval_{params['time_interval']}-finetuneType_{params['finetune_type']}"
                in filename
            ):
                print(filename)
                key, features, curr_finetune, prev_finetune = parse_filename(filename)

                values = pd.read_pickle(os.path.join(subdir, filename))
                values = values.rename(
                    columns={
                        "compute_features_total_time": f"{features}-compute_features_total_time"
                    }
                )

                # if currFinetune and/or prevFinetune is "all", these values cannot be read from the file name
                # this means that the dataframe "values" will have more than one row
                if "all" in filename:
                    for _, row in values.iterrows():
                        row_df = pd.DataFrame(row).transpose().reset_index(drop=True)
                        params["curr_finetune"] = row_df.curr_finetune[0]
                        params["prev_finetune"] = row_df.prev_finetune[0]
                        key = f"currFinetune_{params['curr_finetune']}-prevFinetune_{params['prev_finetune']}-timeIntervalType_{params['time_interval_type']}-timeInterval_{params['time_interval']}-finetuneType_{params['finetune_type']}"
                        add_df_to_dict(
                            df_dict=fid_tmp_dict,
                            df=row_df,
                            key=key,
                            params=params,
                        )
                else:
                    params["curr_finetune"] = curr_finetune
                    params["prev_finetune"] = prev_finetune
                    add_df_to_dict(
                        df_dict=fid_tmp_dict,
                        df=values,
                        key=key,
                        params=params,
                    )

    # merge all dataframes: each key has a dataframe
    fid = pd.concat(list(fid_tmp_dict.values()), axis="index").reset_index(drop=True)

    return fid


def main(exp_params):
    fid = merge_fid_tmp_files(
        params=exp_params, tmp_folder=f"{FID_TMP_FILES_DIR}{exp_params['dataset']}/"
    )

    output_file = (
        f"fid-dataset_{exp_params['dataset']}"
        f"-timeInterval_{exp_params['time_interval']}"
        f"-timeIntervalType_{exp_params['time_interval_type']}"
        f"-finetuneType_{exp_params['finetune_type']}.csv"
    )

    fid.to_csv(path_or_buf=FID_DIR + output_file, index=False)
    print(f"[D] FID saved in:\n\t{FID_DIR}{output_file}")


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-ti",
        "--time_interval",
        help="<Required> Set time_interval period.\n\t1 time_interval period = 1 week if time_interval_type=time\n\totherwise specify number of sentences",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="<Optional> Dataset to be used. DEFAULT = hk-news\n\tOptions: ['kh-news']",
        required=False,
        type=str,
        default="hk-news",
    )

    parser.add_argument(
        "--finetune_type",
        help="<Optional> Type of fine-tune. DEFAULT = incremental\n\tOptions: ['base', 'incremental']\n\tbase: always fine-tune base hugging face model with all the data seen until now\n\tincremental: fine-tune the previously fine-tuned model with new data gathered since",
        required=False,
        type=str,
        default="incremental",
    )

    parser.add_argument(
        "--time_interval_type",
        help="<Optional> Type of time_interval periodicity. DEFAULT = time\n\tOptions: ['time', 'sentence']\n\ttime: time_interval based on how much time has passed since the last adaptation\n\tsentence: time_interval when the specified number of new sentences has been received",
        required=False,
        type=str,
        default="time",
    )

    prog_args = parser.parse_args()
    print(prog_args)

    main(vars(prog_args))
