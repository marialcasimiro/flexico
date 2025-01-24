import re
import time
from collections import Counter

import faiss
import jieba
import nltk
import pandas as pd
from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from rapidfuzz import fuzz
from scipy.spatial.distance import cityblock, cosine, euclidean, jensenshannon
from sentence_transformers import SentenceTransformer

from utils import add_to_dict

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def _extract_nested_lists(input_list):
    # Create an empty list to store the flattened elements
    flattened_list = []

    # Loop over the input list and flatten any nested lists
    for element in input_list:
        if isinstance(element, list):
            # If the element is a list, flatten it and add its elements to the flattened list
            flattened_list.extend(_extract_nested_lists(element))
        else:
            # If the element is not a list, convert it to a string and add it to the flattened list
            flattened_list.append(str(element))

    # Return the flattened list with all elements converted to strings
    return [str(element) for element in flattened_list]


def count_words_by_language(text, language="english"):
    """
    Count the number of words of the input text

    Returns:
    - word_counts: Counter (dict) object with words and corresponding count
    - total_count: Total number of words in text
    - total_count_trimmed: Total number of words in text, excluding
        stop words and non-alphabetic characters
    """

    # Convert sets to flattened lists of strings
    text = _extract_nested_lists(text)

    # Combine the list of strings into a single text
    text = " ".join(text)

    if language in ["chinese", "ch", "zh"]:
        # Tokenize Chinese text using Jieba
        words = list(jieba.cut(text))

        # Exclude non-alphabetic characters, including punctuation
        words_trimmed = [word for word in words if word.isalpha()]

    else:
        # Remove punctuation and convert text to lowercase
        text = re.sub(r"[^\w\s]", "", text.lower())

        # Download stopwords for the specified language
        stop_words = set(stopwords.words(language))

        # Tokenize text
        words = nltk.word_tokenize(text)

        # Exclude stop words
        words_trimmed = [
            word for word in words if word.isalpha() and word.lower() not in stop_words
        ]

    # Count the occurrence of each word (including stopwords)
    word_counts = Counter(words)

    # Calculate the total word count
    total_count = len(words)
    total_count_trimmed = len(words_trimmed)

    return word_counts, total_count, total_count_trimmed


def compute_document_sentence_overlap_ratio(doc1, doc2):
    # Convert sets to flattened lists of strings
    list1 = _extract_nested_lists(doc1)
    list2 = _extract_nested_lists(doc2)

    # Calculate the length of each set
    len1 = len(list1)
    len2 = len(list2)

    # check if any of the sets is empty
    # in this case there is no overlap
    if len1 == 0 or len2 == 0:
        return 0

    # Calculate the number of overlapping sentences
    overlap = len(set(list1) & set(list2))

    # Calculate the sentence overlap ratio
    ratio = overlap / max(len1, len2)

    return ratio


def compute_document_fuzzy_matching_score(doc1, doc2, scorers: dict = None):
    if scorers is None:
        scorers = {"WRatio": fuzz.WRatio}
    # Flatten the lists of lists into lists of strings
    doc1 = _extract_nested_lists(doc1)
    doc2 = _extract_nested_lists(doc2)

    # Convert the documents to lowercase strings
    doc1 = [sentence.lower() for sentence in doc1]
    doc2 = [sentence.lower() for sentence in doc2]

    # Compute the maximum fuzzy matching score between all sentence pairs
    scores = {}
    for scorer_name, scorer in scorers.items():
        max_score = 0
        num_comparisons = 0
        for sentence1 in doc1:
            for sentence2 in doc2:
                # Compute the similarity score between the two sentences
                score = scorer(sentence1, sentence2)
                max_score += score
                num_comparisons += 1

        # Weight the maximum score based on the number of comparisons made
        if num_comparisons > 0:
            max_score = max_score / num_comparisons

        scores[scorer_name] = max_score

    return scores


def get_distance(arr1, arr2, distance_metric: str = "jensenShannon"):
    # Compute the distance between the two frequency distributions
    if distance_metric == "cosine":  # [0; 1]: 0 = perpendicular; 1 = parallel
        distance = cosine(arr1, arr2)
    elif distance_metric == "euclidean":  # [0; unknown]: closer to 0 means more similar
        distance = euclidean(arr1, arr2)
    elif distance_metric == "jensenShannon":  # [0; 1] because base = 2
        distance = jensenshannon(arr1, arr2, base=2)
    elif distance_metric == "manhattan":
        distance = cityblock(arr1, arr2)
    else:
        raise ValueError(
            f"Invalid metric: {distance_metric}. Valid options are 'cosine', 'euclidean', 'manhattan', 'jensenShannon'."
        )

    return distance


def compute_ngram_freq_dist(doc, n):
    """
    compute the n-gram frequency distribution of doc
    """
    # Flatten the lists of lists into lists of strings
    doc = _extract_nested_lists(doc)

    # Flatten the lists of sentences into single strings
    doc_str = " ".join(list(doc))

    # Tokenize the documents into n-grams
    tokens = word_tokenize(doc_str)
    ngrams_list = list(ngrams(tokens, n))

    # Compute the frequency distributions of n-grams
    return FreqDist(ngrams_list)


def compare_ngram_freq_dist(doc1, doc2, n, metrics: list = None):
    """
    compute distance or divergence metric to compare
    the n-gram frequency distributions of docs doc1 and doc2
    """
    if metrics is None:
        metrics = ["jensenShannon"]
    # Check if the inputs are already frequency distributions
    if not isinstance(doc1, FreqDist):
        freq_dist1 = compute_ngram_freq_dist(doc1, n)
    if not isinstance(doc2, FreqDist):
        freq_dist2 = compute_ngram_freq_dist(doc2, n)

    # Ensure that both frequency distributions have the same bins
    keys_union = set(freq_dist1).union(set(freq_dist2))

    # Create new frequency distributions with the same bins
    new_freq_dist1 = FreqDist({k: freq_dist1[k] for k in keys_union})
    new_freq_dist2 = FreqDist({k: freq_dist2[k] for k in keys_union})

    distances = {}
    for metric in metrics:
        # Compute the distance between the two frequency distributions
        distance = get_distance(
            arr1=list(new_freq_dist1.values()),
            arr2=list(new_freq_dist2.values()),
            distance_metric=metric,
        )

        distances[metric] = distance

    return distances


def get_sentence_embeddings(doc1, doc2):
    # Flatten the lists of lists into lists of strings
    doc1 = _extract_nested_lists(doc1)
    doc2 = _extract_nested_lists(doc2)

    # Generate document embeddings
    print("[D] Generating embeddings")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 'all-mpnet-base-v2')
    print("Model loaded")
    doc1_embeddings = model.encode(doc1, batch_size=64, device="cuda")
    print(f"doc1 embeddings shape: {doc1_embeddings.shape}")
    doc2_embeddings = model.encode(doc2, batch_size=64, device="cuda")
    print(f"doc2 embeddings shape: {doc2_embeddings.shape}")

    return {"doc1": doc1_embeddings, "doc2": doc2_embeddings}


def compute_doc_similarity(doc1, doc2, metrics: list = None):
    if metrics is None:
        metrics = ["euclidean"]

    # store how long the different computations take:
    # - time to compute embeddings
    # - time to compute centroids
    # - time to compute all distance metrics
    # - time to compute each distance metric
    times = {}
    centroids = {}  # store the centroids for each embedding
    # store the distances between centroids for each metric
    distances = {}

    # if one of the sets of data is empty
    # we can't comput the distances
    if len(doc1) == 0 or len(doc2) == 0:
        print("[W] doc with len == 0 -- setting distance to -1")
        times["sent_embedding_total_time"] = 0
        times["centroids_total_time"] = 0
    else:
        start_time = time.time()
        doc_embeddings = get_sentence_embeddings(doc1, doc2)
        times["sent_embedding_total_time"] = time.time() - start_time

        start_time = time.time()
        for name, doc in doc_embeddings.items():
            d = doc.shape[1]
            kmeans = faiss.Kmeans(d, k=1, niter=20, verbose=True)
            kmeans.train(doc)
            centroids[name] = kmeans.centroids[0]
        times["centroids_total_time"] = time.time() - start_time

    start_time = time.time()
    for metric in metrics:
        distance_start_time = time.time()
        if len(doc1) == 0 or len(doc2) == 0:
            dist = -1
        else:
            dist = get_distance(
                arr1=centroids["doc1"],
                arr2=centroids["doc2"],
                distance_metric=metric,
            )
        times[f"cluster_{metric}_distance_total_time"] = (
            time.time() - distance_start_time
        )
        print(f"{metric}: {dist}")
        distances[metric] = dist

    times["cluster_distances_total_time"] = time.time() - start_time

    return distances, times


def get_metric_value(
    data: pd.DataFrame,
    metric: str,
    start_time,
    end_time,
    dataset_name: str,
    verbose: bool = False,
):
    tokens = metric.split("_")
    metric_instant = tokens[0]
    metric = tokens[1]
    if len(tokens) >= 3:
        metric_instant = metric_instant + "-" + tokens[2]

    try:
        if "hk-news" in dataset_name:
            metric_value = data.loc[
                (data["finetune_eval_period_start_time"] == start_time)
                & (data["finetune_eval_period_end_time"] == end_time)
            ][f"eval_{metric}"].to_numpy()[0]
        elif "opus_eng_fra" in dataset_name:
            metric_value = data.loc[
                (data["curr_start_index"] == start_time)
                & (data["curr_end_index"] == end_time)
            ][f"eval_{metric}"].to_numpy()[0]

        if verbose:
            print(f"[D]\t{metric_instant} {metric} = {metric_value}")
    except IndexError:
        print(
            f"[E] period [start-time={start_time}, end-time={end_time}] does not exist. Returning -1."
        )
        metric_value = -1

    return metric_value


def compute_sent_overlap_ratio_features(
    features_dict, doc1, doc2, new_data_type, test_set_name, verbose: bool = False
):
    # ratio of sentence overlap feature between old and new data
    if verbose:
        print("[D] Computing ratio of sentence overlap feature!")

    key = f"{new_data_type}-sent_overlap_ratio"
    if test_set_name is not None:
        key = f"{new_data_type}-{test_set_name}-sent_overlap_ratio"

    sent_overlap_ratio_start_time = time.time()
    add_to_dict(
        dictionary=features_dict,
        key=key,
        data=compute_document_sentence_overlap_ratio(doc1, doc2),
        verbose=verbose,
    )
    sent_overlap_ratio_total_time = time.time() - sent_overlap_ratio_start_time

    add_to_dict(
        features_dict,
        f"{key}-_total_time",
        sent_overlap_ratio_total_time,
    )
    if verbose:
        print(
            f"[D] ratio of sentence overlap feature computed in {sent_overlap_ratio_total_time} secs!"
        )


def compute_sent_fuzzy_score_features(
    features_dict: dict, doc1, doc2, new_data_type, test_set_name, verbose: bool = False
):
    """fuzzy matching score between old and new data for each scorer"""

    # tokenRatio: Helper method that returns the max of
    #   token_set_ratio & token_sort_ratio
    #   (faster than manually executing the two functions)
    # partial_token_ratio: Helper method that returns the max of
    #   partial_token_set_ratio & partial_token_sort_ratio
    #   (faster than manually executing the two functions)

    scorers_dict = {
        "ratio": fuzz.ratio,
        "partialRatio": fuzz.partial_ratio,
        "tokenSetRatio": fuzz.token_set_ratio,
        "tokenSortRatio": fuzz.token_sort_ratio,
        # "tokenRatio": fuzz.token_ratio,
        "partialTokenSetRatio": fuzz.partial_token_set_ratio,
        "partialTokenSortRatio": fuzz.partial_token_sort_ratio,
        # "partialTokenRatio": fuzz.partial_token_ratio,
        "WRatio": fuzz.WRatio,
        "QRatio": fuzz.QRatio,
    }
    if verbose:
        print(
            f"[D] Computing fuzzy matching score features for {len(scorers_dict)} different scorers"
        )
    sent_fuzzy_score_start_time = time.time()
    scores = compute_document_fuzzy_matching_score(doc1, doc2, scorers=scorers_dict)
    sent_fuzzy_score_total_time = time.time() - sent_fuzzy_score_start_time

    key = f"{new_data_type}-sent_fuzzy_score_total_time"
    if test_set_name is not None:
        key = f"{new_data_type}-{test_set_name}-sent_fuzzy_score_total_time"
    add_to_dict(features_dict, key, sent_fuzzy_score_total_time)

    for scorer_name, score in scores.items():
        key = f"{new_data_type}-fuzzy_score-{scorer_name}"
        if test_set_name is not None:
            key = f"{new_data_type}-{test_set_name}-fuzzy_score-{scorer_name}"
        add_to_dict(
            dictionary=features_dict,
            key=key,
            data=score,
            verbose=verbose,
        )

    if verbose:
        print(
            f"[D] fuzzy matching score features computed in {sent_fuzzy_score_total_time} secs!"
        )


def compute_ngram_freq_dist_features(
    features_dict: dict, doc1, doc2, new_data_type, test_set_name, verbose: bool = False
):
    # difference between ngram frequency distributions between old and new data
    ngrams_values = [2, 3, 4]
    distances = ["jensenShannon"]  # , "cosine", "euclidean", "manhattan"]

    if verbose:
        print(
            f"[D] Computing ngram frequency distributions features for ngrams {ngrams_values} and distances {distances}"
        )
    ngram_freq_dist_diff_start_time = time.time()
    for ngram in ngrams_values:
        distances = compare_ngram_freq_dist(doc1, doc2, ngram, metrics=distances)
        for distance_name, distance_value in distances.items():
            key = f"{new_data_type}-{ngram}gram_freq_dist_diff-{distance_name}"
            if test_set_name is not None:
                key = f"{new_data_type}-{test_set_name}-{ngram}gram_freq_dist_diff-{distance_name}"
            add_to_dict(
                dictionary=features_dict,
                key=key,
                data=distance_value,
                verbose=verbose,
            )
    ngram_freq_dist_diff_total_time = time.time() - ngram_freq_dist_diff_start_time

    key = f"{new_data_type}-ngram_freq_dist_diff_total_time"
    if test_set_name is not None:
        key = f"{new_data_type}-{test_set_name}-ngram_freq_dist_diff_total_time"

    add_to_dict(
        features_dict,
        key,
        ngram_freq_dist_diff_total_time,
    )
    if verbose:
        print(
            f"[D] ngram frequency distributions features computed in {ngram_freq_dist_diff_total_time} secs!"
        )


def compute_sent_embedding_features(
    features_dict: dict, doc1, doc2, new_data_type, test_set_name, verbose: bool = False
):
    # distance between old and new data embeddings clusters
    distances = ["cosine", "euclidean"]
    if verbose:
        print(
            f"[D] Computing sentence embedding clusters distance features for distances {distances}"
        )
    sent_embedding_cluster_dist_start_time = time.time()
    distances, times = compute_doc_similarity(doc1, doc2, metrics=distances)
    sent_embedding_cluster_dist_total_time = (
        time.time() - sent_embedding_cluster_dist_start_time
    )
    for distance_name, distance_value in distances.items():
        key = f"{new_data_type}-sent_embedding_cluster_dist-{distance_name}"
        if test_set_name is not None:
            key = f"{new_data_type}-{test_set_name}-sent_embedding_cluster_dist-{distance_name}"
        add_to_dict(
            dictionary=features_dict,
            key=key,
            data=distance_value,
            verbose=verbose,
        )
    for time_estimate_type, value in times.items():
        key = f"{new_data_type}-{time_estimate_type}"
        if test_set_name is not None:
            key = f"{new_data_type}-{test_set_name}-{time_estimate_type}"
        add_to_dict(
            dictionary=features_dict,
            key=key,
            data=value,
            verbose=verbose,
        )

    key = f"{new_data_type}-sent_embedding_cluster_dist_total_time"
    if test_set_name is not None:
        key = f"{new_data_type}-{test_set_name}-sent_embedding_cluster_dist_total_time"
    add_to_dict(
        features_dict,
        key,
        sent_embedding_cluster_dist_total_time,
    )
    if verbose:
        print(
            f"[D] sentence embedding clusters distance features computed in {sent_embedding_cluster_dist_total_time} secs!"
        )
