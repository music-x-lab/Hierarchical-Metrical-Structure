from settings import LMD_MATCHED_FOLDER, RWC_DATASET_PATH
import os
import pretty_midi
from data_preprocess import prepare_quantization, get_piano_roll, get_quantized_melody
import numpy as np
import matplotlib.pyplot as plt

def preprocess_midi(file_path, gt_file_path, require_gt=False, subbeat_count=4):
    try:
        midi = pretty_midi.PrettyMIDI(file_path)
    except:
        return None
    if (gt_file_path is not None and os.path.exists(gt_file_path)):
        midi_gt = pretty_midi.PrettyMIDI(gt_file_path)
    else:
        midi_gt = None
        if (require_gt):
            return None
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, subbeat_count)
    onset_rolls = [get_piano_roll(ins, boundaries, False, ignore_drums=False) for ins in midi.instruments]
    piano_rolls = [get_piano_roll(ins, boundaries, True, ignore_drums=True) for ins in midi.instruments]
    result_roll = np.zeros_like(onset_rolls[0], dtype=np.uint64)
    instrument_ids = np.full(32, -2, dtype=np.int8)
    for i, ins in enumerate(midi.instruments):
        if (i >= 32):
            continue
        result_roll = np.bitwise_or(result_roll, onset_rolls[i].astype(np.uint64) << (i * 2))
        result_roll = np.bitwise_or(result_roll, piano_rolls[i].astype(np.uint64) << (i * 2 + 1))
        instrument_ids[i] = -1 if ins.is_drum else ins.program
    result_downbeat = np.zeros(result_roll.shape[0], dtype=np.uint64)
    result_downbeat[downbeat_bins[downbeat_bins < result_roll.shape[0]]] = 1
    result_downbeat[::subbeat_count] += 1
    result = np.zeros_like(result_downbeat)
    if (midi_gt is not None):
        n_subbeat_gt, downbeat_bins_gt, boundaries_gt, subbeat_time_gt = prepare_quantization(midi_gt, subbeat_count)
        notes = [[np.searchsorted(boundaries_gt, note.start), note.pitch] for note in midi_gt.instruments[0].notes]
        note_dict = np.zeros(max(n_subbeat, n_subbeat_gt), dtype=np.int16)
        for note in notes:
            if (note[1] >= 40):
                note_dict[note[0]] = max(note_dict[note[0]], note[1] - 39)
        for i in range(len(downbeat_bins)):
            if (downbeat_bins[i] < len(result)):
                result[downbeat_bins[i]] = note_dict[downbeat_bins[i]]
        result[result_downbeat < 2] = -1
        print(result_downbeat, result)
    else:
        result -= 1
    result_roll = np.concatenate((result_downbeat[:, None], result[:, None], result_roll), axis=1)
    return result_roll, instrument_ids


def train_test_split(count, filename, ratio=10, names=None):
    shuffled = np.arange(count)
    np.random.seed(6172)
    np.random.shuffle(shuffled)
    result_test = np.arange(ratio - 1, count, ratio)
    result_val = np.arange(ratio - 2, count, ratio)
    result_train = np.setdiff1d(np.setdiff1d(np.arange(count), result_test), result_val)
    f = open(filename, 'w')
    for array in [result_train, result_val, result_test]:
        f.write('%s\n' % (','.join([str(shuffled[i]) for i in array])))
    f.close()
    if (names != None):
        f = open(filename + '.names', 'w')
        for array in [result_train, result_val, result_test]:
            f.write('%s\n' % (','.join([names[shuffled[i]] for i in array])))
        f.close()


def prepare_rwc():
    all_results = []
    lengths = []
    names = []
    for i in range(100):
        file_name = os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', 'RM-P%03d.SMF_SYNC.MID' % (i + 1))
        result = preprocess_midi(file_name,
                                 'annotation\RM-P%03d.SMF_SYNC.MID_gt.mid' % (i + 1), require_gt=True)
        if (result is not None):
            all_results.append(result)
            lengths.append(len(result))
            names.append(os.path.basename(file_name))
    lengths = np.array(lengths)
    all_results = np.concatenate(all_results)
    file_name = 'rwc_multitrack_hierarchy_v6_supervised'
    np.save('data/' + file_name, all_results)
    np.save('data/' + file_name + '.length', lengths)
    train_test_split(len(lengths), 'data/%s.split.txt' % file_name, 7, names)



def prepare_lmd():
    f = open('data/lmd_matched_usable_midi.txt', 'r')
    lines = [line.strip() for line in f.readlines() if line.strip() != '']
    f.close()
    np.random.seed(61)
    np.random.shuffle(lines)
    all_results = []
    lengths = []
    instruments = []
    for i, line in enumerate(lines):
        print('Processing %d/%d: %s' % (i, len(all_results), line))
        result, instrument = preprocess_midi(os.path.join(LMD_MATCHED_FOLDER, line), None)
        i += 1
        if (len(result) < 1024):
            continue
        if (result is not None):
            all_results.append(result)
            lengths.append(len(result))
            instruments.append(instrument)
    lengths = np.array(lengths)
    instruments = np.stack(instruments, axis=0)
    all_results = np.concatenate(all_results)
    file_name = 'lmd_multitrack_hierarchy_v6_unsupervised'
    np.save('data/' + file_name, all_results)
    np.save('data/' + file_name + '.length', lengths)
    np.save('data/' + file_name + '.ins', instruments)
    train_test_split(len(lengths), 'data/%s.split.txt' % file_name)

if __name__ == '__main__':
    prepare_rwc()