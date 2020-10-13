from transformers import AutoTokenizer

# jar_path = '/media/E/Data Science/AI/nlp/flaskProject/app/static/VnCoreNLP-1.1.1.jar'
# segmenter = VnCoreNLP(jar_path, annotators='wseg', max_heap_size='-Xmx500m')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# boy_train_file = 'vietnamese-namedb/boy.txt'
# girl_train_file = 'vietnamese-namedb/girl.txt'
MAX_LEN = 10
PAD_ID = 1
START_TOKEN_ID = 0
END_TOKEN_ID = 2
training_dir = 'training_data'


def segment_word(list_names):
    """
    :param list_names: an python list of names. each name contains two word.
    :return: an python list of names which is applied RDR Segmentation
    """
    # return [' '.join(segmenter.tokenize(name)[0]) for name in list_names]
    return [name.strip().replace(' ', '_') for name in list_names]


def encode_bpe(list_segmented_names):
    """
    :param list_segmented_names: a list of names which are applied RDR Segmentation.
    :return: a two dimension array of BPE code, one row is corresponding to a name.
    """
    return [tokenizer.encode(segmented_name) for segmented_name in list_segmented_names]


def pad_bpe_code(list_bpe):
    """
    :param list_bpe: two dimension array, one row contains bpe codes for one boy name
    :return: two dimension array, one row contains bpe codes for one boy name, each row has length equal with MAX_LENGTH
    """
    result = []
    for row in list_bpe:
        if len(row) < MAX_LEN:
            pad_len = MAX_LEN - len(row)
            result.append(row + [PAD_ID] * pad_len)
        if len(row) > MAX_LEN:
            truncate = row.copy()
            truncate = truncate[:MAX_LEN]
            truncate[MAX_LEN-1] = END_TOKEN_ID
            result.append(truncate)
        if len(row) == MAX_LEN:
            result.append(row)
    return result


def create_mask(input_arr):
    """
    :param input_arr: are numpy array.
    :return: a mask array which has same shape with input. it has value 0 where it is bpe code padding and value 1 when
    it not bpe padding.
    """
    arr = input_arr.copy()
    arr[input_arr != PAD_ID] = 1
    arr[input_arr == PAD_ID] = 0
    return arr