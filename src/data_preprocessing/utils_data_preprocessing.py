import pandas as pd

def read_en_file(input_file: str, encoding: str = "unicode_escape"):
    with open(input_file, "r", encoding=encoding) as f:
        e_article = f.read()

    return e_article


def read_ch_file(input_file: str):
    unknown_encoding_counter = 0

    encodings = ["utf8", "big5", "big5hkscs", "hz", "gb18030", "ISO-8859-10"]

    for e in encodings:
        try:
            with open(input_file, "r", encoding=e) as f:
                c_article = f.read()
        except UnicodeDecodeError:
            # print('[D] Got unicode error with %s , trying different encoding' % e)
            unknown_encoding_counter += 1
        else:
            # print('[D] opening the file with encoding:  %s ' % e)
            break

    error_status = 0
    if unknown_encoding_counter == len(encodings):
        c_article = ""
        error_status = 1

    return c_article, error_status, e


def is_empty_sentence(sent: str):
    return bool(sent.strip() == "" or not sent)


def remove_empty_strings_from_col(
    df: pd.DataFrame,
    col: str,
):
    empty_strings_idx = []

    for index, row in df.iterrows():
        sentence = row[col]
        for char in ["-", "－", "＝", "*", "——", "─"]:
            if char in sentence:
                sentence = sentence.replace(char, "")

        if is_empty_sentence(sentence):
            empty_strings_idx.append(index)

    print(f"[D] Found {len(empty_strings_idx)} empty strings in {col} col")
    return df.drop(empty_strings_idx)
