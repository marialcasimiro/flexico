import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn import base, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import hk_news_features
import opus_eng_fra_features



def make_corr(Vs, dataset, Vtitle=""):
    """Correlation analysis function"""
    cols = Vs
    plt.figure()
    ax = sns.heatmap(dataset[cols].corr(), cmap="RdBu_r", annot=True, center=0.0)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if Vtitle != "":
        plt.title(Vtitle, fontsize=12)
    else:
        plt.title(Vs[0] + " - " + Vs[-1], fontsize=12)
    plt.show()


def plot_feature_importance(
    clf: base.ClassifierMixin,
    model_type: str,
    features: list,
    title: str,
    num_features: int = 25,
):
    """Plot Feature Importance"""

    if model_type == "tree":
        coefs = clf.feature_importances_
    elif model_type == "lin":
        coefs = clf.coef_

    feature_imp = pd.DataFrame(
        sorted(zip(coefs, features)), columns=["Value", "Feature"]
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False).iloc[:num_features],
    )
    plt.tight_layout()
    plt.title(title)
    plt.show()
    print(feature_imp.sort_values(by="Value", ascending=False).iloc[:num_features])

    return feature_imp


# """ Scatter and distribution plots """


def scatter_plot(real: np.ndarray, pred: np.ndarray, title: str):
    _, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(real, pred, label="linear-reg")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, ":", alpha=0.75, zorder=0, label="optimum", color="black")
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set(xlabel="Real", ylabel="Predicted")
    ax.legend(loc="best")
    ax.grid()

    ax.set_title(title)
    plt.show()


def plot_dist_sns(
    real: np.ndarray, pred: np.ndarray, title: str, stat="count", bins="auto"
):
    if "auto" in bins:
        bins = np.histogram_bin_edges(real, bins="auto")

    df = pd.DataFrame({"real": real, "pred": pred})
    print(df)
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(data=df, shrink=0.8, stat=stat, bins=bins, kde=True)
    plt.grid()
    ax.set(xlabel="Benefit of retrain", ylabel=stat)
    ax.set_title(title)
    plt.show()


def plot_distribution(real: np.ndarray, pred: np.ndarray, density: bool = False):
    plt.figure()
    arr, bins, _ = plt.hist(
        real,
        bins="auto",
        label="real",
        density=density,
        edgecolor="black",
        linewidth=1.2,
        rwidth=0.5,
    )
    print(arr)
    print(bins)
    for i in enumerate(arr):
        if arr[i] != 0:
            plt.text(bins[i], arr[i], str(arr[i]))
    arr, bins, _ = plt.hist(
        pred,
        bins=bins,
        label="pred",
        density=density,
        edgecolor="black",
        linewidth=1.2,
        rwidth=0.5,
    )
    print(arr)
    print(bins)
    for i in enumerate(arr):
        if arr[i] != 0:
            plt.text(bins[i], arr[i], str(arr[i]))
    plt.legend()


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list,
):
    """Plot Feature Distribution"""
    for feature in features:
        # default bin size based on:
        # - the variance of the data
        # - the number of observations
        # it is possible to set both/either the:
        # - "binwidth" parameter -- size of bins
        # - "bins" parameter -- number of bins
        sns.displot(df, x=feature)

def plot_feature_train_vs_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
    kde: bool = True,
    num_bins = 10,
):
    for feature in features:
        plt.figure(figsize=(10, 6))
        if kde:
            # Plot the distribution for the train set
            sns.kdeplot(train_df[feature], label='Train', shade=True, color='blue')

            # Plot the distribution for the test set
            sns.kdeplot(test_df[feature], label='Test', shade=True, color='orange')

            plt.ylabel('Density')
        else:
            # Combine the train and test data to find the common bin edges
            all_data = pd.concat([train_df, test_df])
            bins = np.histogram_bin_edges(all_data[feature], bins=num_bins)

            # Plot the histogram for the train set
            sns.histplot(train_df[feature], label='Train', bins=bins, color='blue', alpha=0.5, kde=False)

            # Plot the histogram for the test set
            sns.histplot(test_df[feature], label='Test', bins=bins, color='orange', alpha=0.5, kde=False)

            plt.ylabel('Frequency')

        # Add title and labels
        plt.title(f'{feature} Distribution in Train and Test Sets')
        plt.xlabel(f'{feature}')
        plt.legend()
    


def get_hkNews_splits(
    df: pd.DataFrame,
    train_split_percent: float = 0.7,  # 70% for train and val; 30% for test
    val_split_percent: float = 0.2,  # 80% for train; 20% for val
    fixed_test_set: bool = False,
):
    """Create TRAIN / VAL / TEST splits"""
    print("[D] Creating TRAIN / VAL / TEST splits:")
    if fixed_test_set:
        # use years 1997 and 1998 for train and val
        # use year 1999 for test
        # year 2000 was used to create the fixed test-sets

        train_val_split = df.loc[df["period_end_timestamp"] < "1999-01-01"]
        # TRAIN SPLIT
        train_split = train_val_split.head(
            int(len(train_val_split) * (1 - val_split_percent))
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
    else:
        print(f"\tperiod_end_timestamp min: {df['period_end_timestamp'].min()}")
        print(f"\tperiod_end_timestamp max: {df['period_end_timestamp'].max()}")

        duration = df["period_end_timestamp"].max() - df["period_end_timestamp"].min()
        print(f"\tduration: {duration}")

        train_val_duration = int(duration.days * train_split_percent)
        print(f"\ttrain_val_duration: {train_val_duration}")

        test_start_time = df["period_start_timestamp"].min() + pd.Timedelta(
            days=train_val_duration
        )
        print(f"\ttest_start_time: {test_start_time}")

        train_val_split = df.loc[df["period_end_timestamp"] < test_start_time]

        val_split_time = train_val_split["period_start_timestamp"].min() + pd.Timedelta(
            days=int(train_val_duration * (1 - val_split_percent))
        )
        print(f"\tval_split_time: {val_split_time}")

        # TRAIN SPLIT
        train_split = train_val_split.loc[
            train_val_split["period_end_timestamp"] < val_split_time
        ]

        # VAL SPLIT
        val_split = train_val_split.loc[
            (train_val_split["prev_finetune_period_start_timestamp"] >= val_split_time)
        ].copy()
        print(f"{val_split['period_start_timestamp'].min()}")

        # TEST SPLIT
        test_split = df.loc[
            (df["prev_finetune_period_start_timestamp"] >= test_start_time)
        ].copy()
        test_start_week = test_split["curr_finetune_week"].to_numpy()[0]
        print(f"\ttest_start_week: {test_start_week}")
        print(f"{test_split[['period_start_timestamp', 'period_end_timestamp', ]]}")
        print(
            f"{test_split[['prev_finetune_period_start_timestamp', 'prev_finetune_period_end_timestamp', 'curr_finetune_week']]}"
        )
        print(f"{test_split['prev_finetune_period_start_timestamp'].min()}")

    print("\tSET SIZES:")
    print(f"\t\ttrain set: {len(train_split)}")
    print(f"\t\tval set: {len(val_split)}")
    print(f"\t\ttest set: {len(test_split)}")

    return train_split, val_split, test_split


def get_opusEngFra_splits(
    df: pd.DataFrame,
    train_split_percent: float = 0.7,  # 70% for train and val; 30% for test
    val_split_percent: float = 0.2,  # 80% for train; 20% for val
):
    """Create TRAIN / VAL / TEST splits"""
    print("[D] Creating TRAIN / VAL / TEST splits:")
    print(f"\tmin finetune id: {df['prev_finetune'].min()}")
    print(f"\tmax finetune id: {df['curr_finetune'].max()}")

    time_interval = df["prev_finetune"].min()

    duration = len(list(df["prev_finetune"].unique()))
    print(f"\tduration: {duration}")

    train_val_duration = int(duration * train_split_percent)
    print(f"\ttrain_val_duration: {train_val_duration}")

    test_start_time = df["prev_finetune"].min() + train_val_duration * time_interval
    print(f"\ttest_start_time: {test_start_time}")

    train_val_split = df.loc[df["curr_finetune"] < test_start_time]

    val_split_time = train_val_split["prev_finetune"].min() + (
        train_val_duration * time_interval * (1 - val_split_percent)
    )
    print(f"\tval_split_time: {val_split_time}")

    # TRAIN SPLIT
    train_split = train_val_split.loc[train_val_split["curr_finetune"] < val_split_time]

    # VAL SPLIT
    val_split = train_val_split.loc[
        (train_val_split["prev_finetune"] >= val_split_time)
    ].copy()

    # TEST SPLIT
    test_split = df.loc[(df["prev_finetune"] >= test_start_time)].copy()
    test_start_week = test_split["curr_finetune"].to_numpy()[0]
    print(f"\ttest_start_week: {test_start_week}")

    print("\tSET SIZES:")
    print(f"\t\ttrain set: {len(train_split)}")
    print(f"\t\tval set: {len(val_split)}")
    print(f"\t\ttest set: {len(test_split)}")

    return (
        train_split.loc[train_split['new_data_original_opus_dataset_name'].isin(opus_eng_fra_features.OPUS_TEST_SETS)], 
        val_split.loc[val_split['new_data_original_opus_dataset_name'].isin(opus_eng_fra_features.OPUS_TEST_SETS)], 
        test_split.loc[test_split['new_data_original_opus_dataset_name'].isin(opus_eng_fra_features.OPUS_TEST_SETS)]
    )


def compute_metrics(ground_truth, preds, decimal_places: int = 5, verbose: bool = True):
    """
    Compute metrics: MAPE, MAE, r^2, Pearson
    and Spearman correlation coefficients
    """
    # The mean absolute percentage error
    mape = round(mean_absolute_percentage_error(ground_truth, preds), decimal_places)
    
    # The mean absolute error
    mae = round(mean_absolute_error(ground_truth, preds), decimal_places)
    
    # The coefficient of determination: 1 is perfect prediction
    coef_det = round(r2_score(ground_truth, preds), decimal_places)
    
    # Pearson correlation coefficient
    pearson_corr_coef = round(pearsonr(ground_truth, preds)[0], decimal_places)
    
    # Spearman correlation coefficient
    spearman_corr_coef = round(spearmanr(ground_truth, preds)[0], decimal_places)
    
    if verbose:
        print(f"Mean absolute percentage error: {mape}")
        print(f"Mean absolute error: {mae}")
        print(f"Coefficient of determination (r2): {coef_det}")
        print(f"Pearson Correlation coefficient: {pearson_corr_coef}")
        print(f"Spearman Correlation coefficient: {spearman_corr_coef}")
        # The coefficients
        # print("Coefficients: \n", clf.coef_)
        # The intercept
        # print("Intercept: ", clf.intercept_)
        print("\n")

    return mape, mae, coef_det, pearson_corr_coef, spearman_corr_coef


def build_predictor(predictor_name):
    """Build predictor"""
    if "rf" in predictor_name:  # RANDOM FOREST
        # default:
        # n_estimators = 100
        # max_depth = nodes are expanded until all leaves are pure
        #             or until all leaves contain less than 2 samples)
        clf = RandomForestRegressor(
            random_state=1, 
            n_estimators=100, 
            max_depth=10
        )
    elif 'dt' in predictor_name:
        clf = DecisionTreeRegressor(random_state=1)
    elif "mlp" in predictor_name:  # MultiLayer Perceptron (MLP)
        clf = MLPRegressor(random_state=1)
    elif "lin" in predictor_name:  # LINEAR MODEL
        clf = linear_model.LinearRegression()
    elif "xgb" in predictor_name:  # XGBOOST
        # default:
        # n_estimators = might be 100 as well
        # max_depth = 6
        clf = xgb.XGBRegressor(
            eval_metric="mae",
            random_state=1,
            n_estimators=100,
            max_depth=10,
            early_stopping_rounds=200,
        )
    else:
        print(f"Predictor {predictor_name} not implemented")

    return clf


def train_predictor(predictor_name, predictor, x_train, y_train, x_val, y_val):
    """Train predictor"""
    if "xgb" in predictor_name:
        predictor.fit(
            x_train,
            y_train,
            eval_set=[[x_val, y_val]],
            verbose=100,
        )  # base_margin=None)
    else:
        predictor.fit(x_train, y_train)
        # print(type(predictor))
        # print(f"[D] Predictor {predictor_name} trained with features {predictor.feature_names_in_}")

    return predictor


def normalize_features(
    x_train,  # array-like of shape (n_samples, n_features)
    x_val,  # array-like of shape (n_samples, n_features)
    x_test,  # array-like, sparse matrix of shape (n_samples, n_features)
    scaler_type: str = "standard",
):
    """Normalize features"""
    if scaler_type is None:
        return x_train, x_val, x_test
    else:
        if "standard" in scaler_type:
            scaler = StandardScaler()
        else:  # "min-max" in scaler_type:
            scaler = MinMaxScaler()

        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_val_scaled, x_test_scaled


def sequential_feature_selection(
    target, all_features, val_split, predictor_name, n_features_to_select=5
):
    """Do sequential feature selection"""

    def my_pearsonr(ground_truth, preds):
        return pearsonr(ground_truth, preds)[0]

    # use all features
    x_val = val_split[all_features]
    y_val = val_split[target]

    clf = build_predictor(predictor_name)

    # run sequential feature selection
    sfs = SequentialFeatureSelector(
        clf,
        n_features_to_select=n_features_to_select,
        scoring=make_scorer(my_pearsonr, greater_is_better=True),
    )
    sfs.fit(x_val, y_val)

    # get selected features
    return list(sfs.get_feature_names_out())


def set_type_of_time_cols(df: pd.DataFrame):
    time_cols = [
        'period_start_timestamp', 
        'period_end_timestamp', 
        'prev_finetune_period_start_timestamp', 
        'prev_finetune_period_end_timestamp'
    ]

    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
        
    return df.copy()

def create_general_fid(
    df: pd.DataFrame,
    dataset_name: str,
):

    if "hk-news" in dataset_name:
        test_sets =hk_news_features.HK_NEWS_TEST_SETS
        generic_cont_aware_features = hk_news_features.GENERIC_CONTENT_AWARE_FEATURES
    elif "opus" in dataset_name:
        test_sets = opus_eng_fra_features.OPUS_TEST_SETS
        generic_cont_aware_features = opus_eng_fra_features.GENERIC_CONTENT_AWARE_FEATURES
    else:
        print(f"[E] Dataset {dataset_name} unknown. Defaulting to hk-news dataset.")
        test_sets = hk_news_features.HK_NEWS_TEST_SETS

    general_fid_dict = {
        'test-set': [],
    }
    for col in list(df.columns):
        if any(test_set in col for test_set in test_sets):
            continue
        general_fid_dict[col] = []

    for f in generic_cont_aware_features:
        general_fid_dict[f] = []

    for metric in hk_news_features.METRICS:
        general_fid_dict[f"delta-target_test_set_{metric}"] = []
        general_fid_dict[f"curr_test_set_{metric}"] = []
        general_fid_dict[f"target_test_set_{metric}"] = []

    for index, row in df.iterrows(): 
        for test_set in test_sets:        
            for key in general_fid_dict.keys():
                if 'test-set' in key:
                    general_fid_dict[key].append(test_set)
                elif 'test_set' in key:
                    general_fid_dict[key].append(row[key.replace('test_set', test_set)])
                else:
                    general_fid_dict[key].append(row[key])

    general_fid = pd.DataFrame(general_fid_dict)
    
    if "hk-news" in dataset_name:
        general_fid = set_type_of_time_cols(general_fid)

    return general_fid


def eval_FIPs_offline(
    target: str, 
    features_dict: dict, 
    predictors: list, 
    res_dict: dict, 
    feature_imp_dict: dict, 
    dataset: pd.DataFrame,
    dataset_name: str,
    fip_type: str,
    fid_type: str,
    metrics: list = hk_news_features.METRICS, # metrics are the same regardless of the use-case
    l1o_test_set: list = [],
    create_plots: bool = False,
):

    if "opus" in dataset_name:
        test_sets = opus_eng_fra_features.OPUS_TEST_SETS
        fixed_test_set = True
        if 'test-set' in dataset.columns:
            dataset = dataset.loc[dataset['test-set'].isin(test_sets)].copy()
        train_split, val_split, test_split = get_opusEngFra_splits(dataset)
    else:
        if "hk-news" not in dataset_name:
            print(f"[E] Dataset {dataset_name} unknown. Defaulting to hk-news dataset.")
        test_sets = hk_news_features.HK_NEWS_TEST_SETS
        fixed_test_set = True
        # if target.split("_")[1] in test_sets:
        #     fixed_test_set = True
        # else:
        #     fixed_test_set = False
        if 'test-set' in dataset.columns:
            dataset = dataset.loc[dataset['test-set'].isin(test_sets)].copy()

        train_split, val_split, test_split = get_hkNews_splits(
            dataset,
            fixed_test_set = fixed_test_set,
        )


    counter = len(res_dict)
    for predictor in predictors:
        for feature_set_name, features in features_dict.items():
            
            # for each fixed test-set, use only contentAware
            # features that pertain to that fixed test-set
            fip_features = features
            if fixed_test_set:
                fip_features = []
                for feature in features:
                    if '-' in feature:
                        if feature.split("-")[1] not in test_sets or feature.split("-")[1] == target.split("_")[1]:
                            fip_features.append(feature)
                    else:
                        fip_features.append(feature)
            
            if "seq_feat_sel" in feature_set_name:
                features = sequential_feature_selection(
                    target=target, 
                    all_features=features_dict["all"], 
                    val_split=val_split, 
                    predictor_name=predictor
                )
                
            x_train, x_val, x_test = normalize_features(
                x_train=train_split[fip_features],
                x_val=val_split[fip_features],
                x_test=test_split[fip_features],
                scaler_type = None,
            )
            
            if isinstance(l1o_test_set, list) and len(l1o_test_set) > 0:
                train_split = train_split[~train_split['test-set'].isin(l1o_test_set)].copy()
                val_split = val_split[~val_split['test-set'].isin(l1o_test_set)].copy()
                test_split = test_split.loc[test_split['test-set'].isin(l1o_test_set)].copy()
            else:
                l1o_test_set = 'none'
                
                
            x_train = train_split[fip_features]
            y_train = train_split[target]
            x_val = val_split[fip_features]
            y_val = val_split[target]
            x_test = test_split[fip_features]
            y_test = test_split[target]
            
            print("#" * 50)
            print(
                f"target: {target}"
                f"\npredictor: {predictor}"
                f"\nleave-1-out test-set: {l1o_test_set}"
                f"\nfeature-set: {feature_set_name}"
                
            )
            print(f"ANALYZING {feature_set_name.upper()} FEATURES:\n{fip_features}\n")
    
            print("SET SIZES:")
            print(f"  train set: {len(x_train[fip_features[0]])}")
            if 'test-set' in list(train_split.columns):
                print(f"     unique test-sets: {list(train_split['test-set'].unique())}")
            print(f"  val set: {len(x_val[fip_features[0]])}")
            if 'test-set' in list(val_split.columns):
                print(f"     unique test-sets: {list(val_split['test-set'].unique())}")
            print(f"  test set: {len(x_test[fip_features[0]])}")
            if 'test-set' in list(test_split.columns):
                print(f"     unique test-sets: {list(test_split['test-set'].unique())}")
            
            # build predictor
            clf = build_predictor(predictor)
            # train predictor
            clf = train_predictor(predictor, clf, x_train, y_train, x_val, y_val)
            # predict for the test-set
            y_preds = clf.predict(x_test)


            res_dict[counter] = {}
            res_dict[counter]["fip_type"] = fip_type
            res_dict[counter]["fid_type"] = fid_type
            res_dict[counter]["target"] = target
            if "delta" in target:
                target_type = "delta"
            else:
                target_type = "absolute"
            res_dict[counter]["target-type"] = target_type
            res_dict[counter]["features"] = feature_set_name
            res_dict[counter]["l1o-test_set"] = l1o_test_set if len(l1o_test_set) > 1 else l1o_test_set[0]
            for metric in hk_news_features.METRICS:
                if metric in target:
                    res_dict[counter]["metric"] = metric
            if 'test_set' in target:
                res_dict[counter]["test-set"] = "all_fixed_test_sets"
                res_dict[counter]["tactic"] = 'finetune'
            elif target.split("_")[1] in metrics:
                res_dict[counter]["test-set"] = "time"
                res_dict[counter]["tactic"] = target.split("_")[2]
            else:
                if "tedx-fr_ca" in target:
                    res_dict[counter]["test-set"] = "tedx_fr_ca"
                else:
                    res_dict[counter]["test-set"] = target.split("_")[1]
                res_dict[counter]["tactic"] = "finetune"
            res_dict[counter]["predictor"] = predictor
            for feature_set, x, y in zip(['train', 'val', 'test'], [x_train, x_val, x_test], [y_train, y_val, y_test]):
                print(f"Evaluating {feature_set}-set:")
                mape, mae, coef_det, pearson_corr_coef, spearman_corr_coef = compute_metrics(
                    ground_truth=y,
                    preds=clf.predict(x),
                )
                res_dict[counter][f"{feature_set}-mape"] = round(mape, 4)
                res_dict[counter][f"{feature_set}-mae"] = round(mae, 4)
                res_dict[counter][f"{feature_set}-coef_det"] = round(coef_det, 4)
                res_dict[counter][f"{feature_set}-PCC"] = round(pearson_corr_coef, 4)
                res_dict[counter][f"{feature_set}-SCC"] = round(spearman_corr_coef, 4)
            

            if predictor in ["dt", "rf", "xgb"]: model_type = "tree"
            elif predictor in ["lin"]: model_type="lin"                
            feature_imp = plot_feature_importance(clf, model_type=model_type, features=fip_features, title=target)

            feature_imp_dict[counter] = {}
            feature_imp_dict[counter]["fip_type"] = fip_type
            feature_imp_dict[counter]["fid_type"] = fid_type
            feature_imp_dict[counter]["target"] = target
            feature_imp_dict[counter]["target-type"] = target_type
            feature_imp_dict[counter]["l1o-test_set"] = l1o_test_set if len(l1o_test_set) > 1 else l1o_test_set[0]
            if 'test_set' in target:
                feature_imp_dict[counter]["test-set"] = "all_fixed_test_sets"
                feature_imp_dict[counter]["metric"] = target.split("_")[-1]
                feature_imp_dict[counter]["tactic"] = 'finetune'
            elif target.split("_")[1] in metrics:
                feature_imp_dict[counter]["test-set"] = "time"
                feature_imp_dict[counter]["metric"] = target.split("_")[1]
                feature_imp_dict[counter]["tactic"] = target.split("_")[2]
            else:
                feature_imp_dict[counter]["test-set"] = target.split("_")[1]
                feature_imp_dict[counter]["metric"] = target.split("_")[2]
                feature_imp_dict[counter]["tactic"] = "finetune"
            feature_imp_dict[counter]["predictor"] = predictor
            feature_imp_dict[counter]["feature-set"] = feature_set_name
            feature_imp_dict[counter]["features"] = fip_features
            feature_imp_dict[counter]["feature_imp"] = feature_imp

            if create_plots:
                scatter_plot(real=y_test.to_numpy(), pred=y_preds, title=target)
                if not np.isnan(pearson_corr_coef):
                    plot_dist_sns(y_test.to_numpy(), y_preds, target)

            counter += 1


def highlight_opt(col):
    '''
    highlight the optimal (max/min) value in each row of a dataframe
    '''
    if 'PCC' in col.name or 'pcc' in col.name or 'correlation' in col.name:
        is_opt = col == col.max()
    else:
        is_opt = col == col.min()
        
    return ['font-weight: bold' if v else '' for v in is_opt]


def _get_avg_table(df: pd.DataFrame):

    return df.pivot_table(
        index='predictor', 
        columns='features', 
        values=['MAE', 'PCC'], 
        aggfunc=np.mean,
    ).stack(level=1)

def _get_general_table(df: pd.DataFrame):
    return df[[
        'test-set',
        'predictor',
        'metric',
        'features',
        'MAE',
        'PCC',
    ]].pivot(
        index='test-set', columns=['predictor', 'features'], values=['MAE', 'PCC']
    ).stack(level=0).transpose()

def get_results_table(
    results: pd.DataFrame,
    fid_type: str,
    average: bool = False,
    single: bool = False,
    to_latex: bool = False,
):
    
    if single:
        res = results.loc[
            (results.fid_type == fid_type)
        ]

        if average:
            res_table = _get_avg_table(res)
        else:
            res_table = _get_general_table(res)
            
        res_table = res_table.style.apply(
            lambda col: highlight_opt(col),
            axis=0,
        )
        
        display(res_table)
        
        if to_latex:
            print(res_table.to_latex(convert_css=True))

    else:
        for predictor in results.predictor.unique():
            res = results.loc[
                (results.fid_type == fid_type)
                & (results.predictor == predictor)
            ]

            if average:
                res_table = res_table = _get_avg_table(res)
            else:
                res_table = _get_general_table(res)
                
            res_table = res_table.style.apply(
                    lambda col: highlight_opt(col),
                    axis=0,
                )
            
            display(res_table)
            
            if to_latex:
                print(res_table.to_latex(convert_css=True))

def inject_extreme_samples(
    df: pd.DataFrame,
    percent: float,
    metrics: list,
    test_sets: list,
    features: list = [
        'amount_new_data',
        'amount_finetune_data',
        'sent_overlap_ratio',
        '2gram_freq_dist_diff-jensenShannon',
        '3gram_freq_dist_diff-jensenShannon',
        '4gram_freq_dist_diff-jensenShannon',
        'sent_embedding_cluster_dist-cosine',
        'sent_embedding_cluster_dist-euclidean',
    ],
):
    # add targets to list of features
    for metric in metrics:
        for test_set in test_sets:
            key = f"{test_set}_{metric}"
            features.append([f"curr_{key}"])
    
    # select percent samples at random
    synth_samples = df.sample(frac=percent, random_state=42).copy()

    # modify the content-aware features when 'NEW == OLD'
    #   - amount_new_data <-- amount_old_data
    #   - amount_finetune_data <-- amount_old_data
    #   - distance features <-- 0
    #   - overlap features  <-- 1
    #   - delta for fixed test-sets <-- 0
    for feature in features:
        if "sent_overlap" in feature:
            synth_samples[feature] = 1.0
        elif 'amount' in feature and 'data' in feature:
            synth_samples[feature] = synth_samples['amount_old_data']
        else:
            synth_samples[feature] = 0.0
            
    synth_samples['ratio_new_old_data'] = synth_samples['amount_new_data']/synth_samples['amount_old_data']
    synth_samples['ratio_finetune_old_data'] = synth_samples['amount_finetune_data']/synth_samples['amount_old_data']
    
    # add synthethised samples to df
    new_df = pd.concat(
        [df, synth_samples],
        ignore_index=True,
    )

    return new_df