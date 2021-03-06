import json
import sys

def get_error_count(input_filename, min_dist=None, max_dist=None):
    '''
    :param input_filename: predicted jsonlines filepath
    :return: error_counter, num_examples, num of true_pos, true_neg, false_pos, false_neg

    one line of input file:
    dict_keys(['mention_dist', 'men1_end', 'sentences', 'men1_start', 'pred', 'men2_start', 'men2_end', 'gold_label', 'doc_key'])
    '''
    error_counter = 0
    num_examples = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    if min_dist is not None and max_dist is not None:
        assert min_dist <= max_dist

    def print_speaker(excerpt_speakers, excerpt_sentence):
        start_speech = 0
        current_speaker = excerpt_speakers[0]
        for i, s in enumerate(excerpt_speakers):
            end_speech = i
            if s != current_speaker or i == len(excerpt_sentence) - 1:
                sp = " ".join(excerpt_sentence[start_speech:end_speech + 1])
                print(f"{current_speaker}: {sp}")
                start_speech = i
                current_speaker = s

    # with open(errors, 'w'):
    with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
            example = json.loads(line)
            # for each gold label in cluster check it against pred label
            for idx, label in enumerate(example['gold_label']):
                pred_label = example['pred'][idx]

                if min_dist is not None and example['mention_dist'][idx] < min_dist:
                    continue

                if max_dist is not None and example['mention_dist'][idx] > max_dist:
                    continue

                # calculate number of all mention pairs
                num_examples += 1

                # only a subset of example for error analysis
                if num_examples < 750:

                    # calculate true positives, false positives...
                    if label == 1 and pred_label == 1:
                        true_pos += 1
                    elif label == 0 and pred_label == 0:
                        true_neg +=1
                    elif label == 1 and pred_label == 0:
                        false_neg +=1
                    elif label == 0 and pred_label == 1:
                        false_pos += 1

                    # output errors
                    if label != pred_label:
                        error_counter += 1
                        sentences = [item for sublist in example['sentences'] for item in sublist]
                        print(example['mention_dist'][idx], 'pred:', example['pred'][idx], 'gold:', example['gold_label'][idx])
                        before = max(0, example['men1_start'][idx] - 10)
                        after = min(len(sentences)-1, example['men2_end'][idx]+3)
                        print(" ".join(sentences[example['men1_start'][idx]:example['men1_end'][idx]+1]), '   ==   ', " ".join(sentences[example['men2_start'][idx]:example['men2_end'][idx]+1]))
                        speakers = [item for sublist in example['speakers'] for item in sublist]
                        start_speech = 0

                        if example['mention_dist'][idx] < 15:
                            excerpt_speakers = speakers[before:after]
                            excerpt_sentence = sentences[before:after]
                            print('...')
                            print_speaker(excerpt_speakers, excerpt_sentence)
                            print('...', "(", example['mention_dist'][idx], ")",'\n')
                            #print('...', " ".join(sentences[before:after]), "(", example['mention_dist'][idx], ")", '\n' )
                        else:
                            excerpt_speakers = speakers[before:example['men1_end'][idx]+3]
                            excerpt_sentence = sentences[before:example['men1_end'][idx]+3]
                            print('...')
                            print_speaker(excerpt_speakers, excerpt_sentence)

                            excerpt_speakers = speakers[example['men2_start'][idx]-10:after]
                            excerpt_sentence = sentences[example['men2_start'][idx]-10:after]
                            print('...')
                            print_speaker(excerpt_speakers, excerpt_sentence)
                            print('...', "(", example['mention_dist'][idx], ")",'\n')
                            #print('...', " ".join(sentences[before:example['men1_end'][idx]+3]), '...', " ".join(sentences[example['men2_start'][idx]-10:after]), "(", example['mention_dist'][idx], ")", '\n' )



    return error_counter, num_examples, true_pos, true_neg, false_pos, false_neg

def print_bin_results(filename, min_dist=None, max_dist=None):
    print(
        f"##################################START: Min Distance: {min_dist}, Max Distance: {max_dist}#####################################################\n")
    error_count, num_examples, true_pos, true_neg, false_pos, false_neg = get_error_count(filename, min_dist, max_dist)
    print(f'true_pos, true_neg, false_pos, false_neg, {true_pos, true_neg, false_pos, false_neg}')
    f1 = true_pos / (true_pos + (false_pos+false_neg)/2)
    print(f"F1 score of {f1}")
    print(f"##################################END: Min Distance: {min_dist}, Max Distance: {max_dist}#####################################################\n")


# Execute with a list of predictions created with predict_joshi.py
if __name__ == '__main__':
    for input_filename in sys.argv[1:]:
        print(input_filename)
        print_bin_results(input_filename)
        print_bin_results(input_filename, 0, 50)
        print_bin_results(input_filename, 51, 100)
        print_bin_results(input_filename, 101)
