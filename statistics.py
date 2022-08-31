from simple_tcn import TCNClassifier, NetworkInterface, N_MIDI_PITCH, CONTEXT_LENGTH
import numpy as np
from midi_structure import get_piano_roll, prepare_quantization, evaluate_result
import pretty_midi
import os
from settings import LMD_MATCHED_FOLDER
import matplotlib.pyplot as plt
import torch
from crf import CRFDecoder

ins_collection = {'others': []}
drum_counter = np.zeros((128, 5), dtype=int)

def load_labels(ref_midi_path, downbeat_bins, subbeat_count=4):
    midi_gt = pretty_midi.PrettyMIDI(ref_midi_path)
    n_subbeat_gt, downbeat_bins_gt, boundaries_gt, subbeat_time_gt = prepare_quantization(midi_gt, subbeat_count)
    if (len(midi_gt.instruments) == 1):
        ins_id = 0
    else:
        assert(midi_gt.instruments[-1].name == 'Layers')
        ins_id = len(midi_gt.instruments) - 1
    notes = [[np.searchsorted(boundaries_gt, note.start), note.pitch] for note in midi_gt.instruments[ins_id].notes]
    note_dict = np.zeros(n_subbeat_gt, dtype=np.int16)
    gt_result = np.zeros(len(downbeat_bins_gt), dtype=np.int16)
    for note in notes:
        if (note[1] >= 40):
            note_dict[note[0]] = max(note_dict[note[0]], note[1] - 39)
    for i in range(min(len(gt_result), len(downbeat_bins))):
        if (downbeat_bins[i] < n_subbeat_gt):
            gt_result[i] = note_dict[downbeat_bins[i]]
    return gt_result

def get_instrument_confidence(midi_path, model_save_name, subbeat_count=4):
    print('Evaluating:', midi_path)
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        print('Midi load failed: %s' % midi_path)
        return None
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, subbeat_count)
    pred_midi_file = os.path.join('output/%s/%s_crf.mid' % (model_save_name, os.path.basename(midi_path)))
    labels = load_labels(pred_midi_file, downbeat_bins)
    conf_file = os.path.join('output/%s/%s_conf.txt' % (model_save_name, os.path.basename(midi_path)))
    conf_ins_file = os.path.join('output/%s/%s_conf_ins.txt' % (model_save_name, os.path.basename(midi_path)))
    if not (os.path.exists(conf_ins_file)):
        print('Conf file not found!')
        return
    conf = np.loadtxt(conf_file)
    f = open(conf_ins_file, 'r')
    ins_list = [token.strip().split(':') for token in f.readline().split(',')]
    f.close()
    n_instruments, n_frames = conf.shape
    assert(n_instruments == len(ins_list))
    piano_rolls = [get_piano_roll(ins, boundaries, False, ignore_drums=True) for ins in midi.instruments]
    onset_rolls = [get_piano_roll(ins, boundaries, True, ignore_drums=True) for ins in midi.instruments]
    drum_rolls = [get_piano_roll(ins, boundaries, True, ignore_drums=False, ignore_non_drums=True) for ins in
                  midi.instruments]
    rolls = []
    ins_names = []
    for j, ins in enumerate(midi.instruments):
        if (ins.is_drum):
            roll = np.concatenate((onset_rolls[j], piano_rolls[j], drum_rolls[j]), axis=-1)
            rolls.append(roll)
            ins_names.append('drums')
    if (len(rolls) > 1):
        rolls = [np.max(rolls, axis=0)]
        ins_names = ['drums']
    for j, ins in enumerate(midi.instruments):
        if (ins.is_drum):
            continue
        if ('mel' in ins.name.lower() or 'vocal' in ins.name.lower()):
            ins_name = 'melody'
        else:
            ins_name = pretty_midi.program_to_instrument_name(ins.program) + '(%d)' % ins.program
        roll = np.concatenate((onset_rolls[j], piano_rolls[j], drum_rolls[j]), axis=-1)
        rolls.append(roll)
        ins_names.append(ins_name)
    assert(len(rolls) == n_instruments)
    for i in range(n_instruments):
        # test if roll is mostly active
        test = np.convolve(rolls[i].sum(axis=-1), np.ones(32), mode='same')
        if ((test == 0).sum() > len(test) * 0.33):
            # print('bad instrument ' + ins_names[i])
            pass
        else:
            # print('good instrument ' + ins_names[i])
            data = conf[i][downbeat_bins[downbeat_bins < len(conf[i])]]
            ins_name = ins_names[i]
            if (ins_name not in ins_collection):
                ins_collection[ins_name] = []
            if (ins_name != 'melody' and ins_name != 'drums'):
                ins_collection['others'].append(data.mean())
            ins_collection[ins_name].append(data.mean())
    if (ins_names[0] == 'drums'):
        drum_roll = rolls[0]
        for i in range(min(len(labels), len(downbeat_bins))):
            if (downbeat_bins[i] >= len(drum_roll)):
                continue
            drum_counter[0, labels[i]] += 1
            drum_counter[rolls[0][downbeat_bins[i], -128:] > 0, labels[i]] += 1
if __name__ == '__main__':
    f = open('data/lmd_matched_usable_midi.txt', 'r')
    lines = [line.strip() for line in f.readlines() if line.strip() != '']
    f.close()
    # np.random.seed(6172)
    # np.random.shuffle(lines)
    for line in lines:
        get_instrument_confidence(os.path.join(LMD_MATCHED_FOLDER, line), 'simple_tcn_v2.0_filtered')
    result = {}
    for ins in ins_collection:
        stats = (np.mean(ins_collection[ins]), np.std(ins_collection[ins]), len(ins_collection[ins]))
        result[ins] = stats
        print(ins, stats[0], stats[1], stats[2], sep='\t')
    print()
    for i in range(128):
        print(pretty_midi.note_number_to_drum_name(i), *drum_counter[i], sep='\t')
