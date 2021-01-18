import paddle.fluid.dygraph as D
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModel

import numpy as np

import h5py
import json
import sys

# Change if you want.
# Values are : ernie-2.0-en, ernie-2.0-large-en, ernie-tiny.
# See https://github.com/PaddlePaddle/ERNIE/blob/develop/README.en.md, the table with abbreviations for models.
PRETRAINED_ERNIE = 'ernie-2.0-large-en'


def cache_dataset(data_path,  out_file):
    D.guard().__enter__()  # activate paddle `dygrpah` mode

    model = ErnieModel.from_pretrained(PRETRAINED_ERNIE, force_download=False,
                                       return_additional_info=True)  # Try to get pretrained model from server, make sure you have network connection
    model.eval()
    tokenizer = ErnieTokenizer.from_pretrained('ernie-2.0-large-en')

    with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file.readlines()):
            example = json.loads(line)
            sentences = example["sentences"]

            file_key = example["doc_key"].replace("/", ":")
            group = out_file.create_group(file_key)

            for i, sentence in enumerate(sentences):
                text_id = tokenizer.convert_tokens_to_ids(sentence)
                pair_id = []
                ret_id, _ = tokenizer.build_for_ernie(text_id, pair_id)
                ids = D.to_variable(np.expand_dims(ret_id, 0))  # add a batch dimension
                _, _, additional = model(ids)
                # the particular range 1:len(sentence)+1 removes the start and end token used in Ernie to delimitate the sentences...
                input = additional["hiddens"][0].numpy().squeeze()[1:len(sentence)+1]
                before_last = additional["hiddens"][-2].numpy().squeeze()[1:len(sentence) + 1]
                last = additional["hiddens"][-1].numpy().squeeze()[1:len(sentence)+1]
                ernie_embeddings = np.dstack([input,before_last,last])
                group[str(i)] = ernie_embeddings

            if doc_num % 10 == 0:
                print("Cached {} documents in {}".format(doc_num + 1, data_path))


if __name__ == "__main__":


    with h5py.File("ernie_cache.hdf5", "w") as out_file:
        for json_filename in sys.argv[1:]:
            cache_dataset(json_filename, out_file)
