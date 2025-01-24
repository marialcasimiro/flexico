import comet  # From: unbabel-comet
import numpy as np
import torch
from datasets import load_metric
from huggingface_hub.utils import HfHubHTTPError

from data_preprocessing.utils_data_preprocessing import is_empty_sentence


def load_comet_model(model: str):
    try:
        comet_model = comet.load_from_checkpoint(comet.download_model(model))
        return comet_model, 0
    except HfHubHTTPError as e:
        print(f"[E] Error loading comet model: {e}")
        print("\t Login in to HuggingFace again and retrying")
        return None, 1


def get_comet_score(
    model, model_name, sources, predictions, references, gpus=None, batch_size: int = 64
):
    if gpus is None:
        # print(f"[D]\tmps available = {torch.backends.mps.is_available()}")
        # print(f"[D]\tcuda available = {torch.cuda.is_available()}")
        # if torch.backends.mps.is_available():
        #     gpus = 1
        #     accelerator='mps'
        if torch.cuda.is_available():
            gpus = 1
            accelerator = "gpu"
        else:
            gpus = 0
            accelerator = "auto"

    # [dict(zip(data, t)) for t in zip(*data.values())]:
    #   - create a list of dicts, where each dict contains
    #   the src, mt, and, for comet22, reference translations
    if "qe" in model_name:
        data = {"src": sources, "mt": predictions}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
    else:
        data = {"src": sources, "mt": predictions, "ref": references}
        data = [dict(zip(data, t)) for t in zip(*data.values())]

    prediction = model.predict(
        data, gpus=gpus, batch_size=batch_size, accelerator=accelerator
    )
    return {"mean_score": prediction["system_score"], "scores": prediction["scores"]}


def load_metrics(parameters: dict):
    metrics = {}
    print("[D] Loading metrics...")
    for metric_path in parameters["metrics"]:
        if "comet" in metric_path:
            # metric_name = metric_path.split("/")[2][:-3]
            model_version = "22"
            if "20" in metric_path:
                model_version = "20"
            model_type = ""
            if "qe" in metric_path or "kiwi" in metric_path:
                model_type = "-qe"
            metric_name = f"comet{model_version}{model_type}"
            model, error = load_comet_model(metric_path)
            if error:
                return None, error
            metrics[metric_name] = model
        else:
            metric_name = metric_path
            metrics[metric_name] = load_metric(metric_path)
        print(f"[D] Loaded metric {metric_name} from {metric_path}")

    parameters["metrics"] = list(metrics.keys())

    return metrics, 0


def decode_sentences(eval_preds, tokenizer):
    preds, ref_sents, inputs = eval_preds

    # In case the model returns more than the prediction logits
    for arr in [preds, ref_sents, inputs]:
        if isinstance(arr, tuple):
            arr = arr[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    # Replace -100s in the ref_sents as we can't decode them
    ref_sents = np.where(ref_sents != -100, ref_sents, tokenizer.pad_token_id)
    decoded_ref_sents = tokenizer.batch_decode(ref_sents, skip_special_tokens=True)

    # Some simple post-processing
    decoded_inputs = [source.strip() for source in decoded_inputs]
    decoded_preds = [pred.strip() for pred in decoded_preds]
    comet_decoded_ref_sents = [ref_sent.strip() for ref_sent in decoded_ref_sents]
    chrf_decoded_ref_sents = [[ref_sent.strip()] for ref_sent in decoded_ref_sents]

    # search for empty reference sentences.
    # This shouldn't happen but something occurs
    # during the encoding/decoding process that
    # turns some reference sentences into empty sentences
    print("[D] Searching for empty reference sentences")
    empty_ref_sentences_counter = 0
    for dec_pred, dec_ref_sent, dec_input in zip(
        decoded_preds, chrf_decoded_ref_sents, decoded_inputs
    ):
        if is_empty_sentence(dec_ref_sent[0]):
            print(
                f"[W] removing sample:\n[W]\tinput={dec_input}\tpred={dec_pred}\tref_sent={dec_ref_sent[0]}"
            )
            decoded_preds.remove(dec_pred)
            chrf_decoded_ref_sents.remove(dec_ref_sent)
            comet_decoded_ref_sents.remove(dec_ref_sent[0])
            decoded_inputs.remove(dec_input)
            empty_ref_sentences_counter += 1

    print(
        f"[W] {empty_ref_sentences_counter} empty reference sentences found and removed!"
    )

    return (
        decoded_inputs,
        decoded_preds,
        chrf_decoded_ref_sents,
        comet_decoded_ref_sents,
    )


# define compute_metrics function
def compute_metrics(eval_preds, metrics: dict, tokenizer, comet_batch_size: int = 64):
    (
        decoded_inputs,
        decoded_preds,
        chrf_decoded_ref_sents,
        comet_decoded_ref_sents,
    ) = decode_sentences(eval_preds, tokenizer)

    results = {}
    for curr_metric_name, metric in metrics.items():
        print(f"[D] Evaluating with {curr_metric_name}")
        if "comet" in curr_metric_name:
            result = get_comet_score(
                model=metric,
                model_name=curr_metric_name,
                sources=decoded_inputs,
                predictions=decoded_preds,
                references=comet_decoded_ref_sents,
                batch_size=comet_batch_size,
            )
            results[curr_metric_name] = result["mean_score"]
        else:
            result = metric.compute(
                predictions=decoded_preds, references=chrf_decoded_ref_sents
            )
            results[curr_metric_name] = result["score"]

        print(f"[D]\t{curr_metric_name} score={results[curr_metric_name]}")
    return results
