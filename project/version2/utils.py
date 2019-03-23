import os, argparse, time
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import sys
sys.path.append('../')
import os, argparse, time
import pretty_midi
import numpy as np
from copy import deepcopy
import random
from multiprocessing import Pool as ThreadPool

def _windows_from_monophonic_instruments(midi, window_size):
    X, y = [], []
    for m in midi:
        if m is not None:
            melody_instruments = filter_monophonic(m.instruments, 1.0)
            for instrument in melody_instruments:
                if len(instrument.notes) > window_size:
                    windows = _encode_sliding_windows(instrument, window_size)
                    for w in windows:
                        X.append(w[0])
                        y.append(w[1])
    return (np.asarray(X), np.asarray(y))


def get_percent_monophonic(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notes = np.sum(mask, axis=1)
    n = np.count_nonzero(notes)
    single = np.count_nonzero(notes == 1)
    
    return float(single) / float(n)



def filter_monophonic(pm_instruments, percent_monophonic=0.99):
    return [i for i in pm_instruments if \
            get_percent_monophonic(i.get_piano_roll()) >= percent_monophonic]


# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).

def _encode_sliding_windows(pm_instrument, window_size):
    roll = np.copy(pm_instrument.get_piano_roll(fs=5).T)

    # trim beginning silence
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]

    # transform note velocities into 1s
    roll = (roll > 0).astype(float)

    # append a feature: 1 to rests and 0 to notes
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)

    windows = []
    for i in range(0, roll.shape[0] - window_size - 1):
        windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
    return windows


def parse_midi(path):
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except Exception as e:
        raise Exception(("%s\nerror readying midi file %s" % (e, path)))
    return midi


def log(message, verbose):
    if verbose:
        print('[*] {}'.format(message))


def create_experiment_dir(experiment_dir, verbose=False):
    # if the experiment directory was specified and already exists
    if experiment_dir != 'experiments/default' and \
            os.path.exists(experiment_dir):
        # raise an error
        raise Exception('Error: Invalid --experiment_dir, {} already exists' \
                        .format(experiment_dir))

    # if the experiment directory was not specified, create a new numeric folder
    if experiment_dir == 'experiments/default':

        experiments = os.listdir('experiments')
        experiments = [dir_ for dir_ in experiments \
                       if os.path.isdir(os.path.join('experiments', dir_))]

        most_recent_exp = 0
        for dir_ in experiments:
            try:
                most_recent_exp = max(int(dir_), most_recent_exp)
            except ValueError as e:
                # ignrore non-numeric folders in experiments/
                pass

        experiment_dir = os.path.join('experiments',
                                      str(most_recent_exp + 1).rjust(2, '0'))

    os.mkdir(experiment_dir)
    log('Created experiment directory {}'.format(experiment_dir), verbose)
    os.mkdir(os.path.join(experiment_dir, 'checkpoints'))
    log('Created checkpoint directory {}'.format(os.path.join(experiment_dir, 'checkpoints')),
        verbose)
    os.mkdir(os.path.join(experiment_dir, 'tensorboard-logs'))
    log('Created log directory {}'.format(os.path.join(experiment_dir, 'tensorboard-logs')),
        verbose)

    return experiment_dir


def get_callbacks(experiment_dir, checkpoint_monitor='val_acc'):
    callbacks = []

    # save model checkpoints
    filepath = os.path.join(experiment_dir,
                            'checkpoints',
                            'checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

    callbacks.append(ModelCheckpoint(filepath,
                                     monitor=checkpoint_monitor,
                                     verbose=1,
                                     save_best_only=False,
                                     mode='max'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       patience=3,
                                       verbose=1,
                                       mode='auto',
                                       epsilon=0.0001,
                                       cooldown=0,
                                       min_lr=0))

    callbacks.append(TensorBoard(log_dir=os.path.join(experiment_dir, 'tensorboard-logs'),
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=False))

    return callbacks


def get_data_generator(midi_paths,
                       window_size=50,
                       batch_size=32,
                       num_threads=8,
                       max_files_in_ram=170):

    if num_threads > 1:
        # load midi data
        pool = ThreadPool(num_threads)

    load_index = 0

    while True:
        load_files = midi_paths[load_index:load_index + max_files_in_ram]

        load_index = (load_index + max_files_in_ram) % len(midi_paths)

        if num_threads > 1:
            parsed = pool.map(parse_midi, load_files)
        else:
            parsed = map(parse_midi, load_files)


        data = _windows_from_monophonic_instruments(parsed, window_size)

        batch_index = 0
        while batch_index + batch_size < len(data[0]):


            res = (data[0][batch_index: batch_index + batch_size],
                   data[1][batch_index: batch_index + batch_size])

            yield res
            batch_index = batch_index + batch_size


# create a pretty midi file with a single instrument using the one-hot encoding
def _network_output_to_midi(windows,
                            instrument_name='Acoustic Grand Piano',
                            allow_represses=True):

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=instrument_program)

    cur_note = None # an invalid note to start with
    cur_note_start = None
    clock = 0

    for step in windows:

        note_num = np.argmax(step) - 1

        # a note has changed
        if allow_represses or note_num != cur_note:

            # if a note has been played before and it wasn't a rest
            if cur_note is not None and cur_note >= 0:
                # add the last note, now that we have its end time
                note = pretty_midi.Note(velocity=127,
                                        pitch=int(cur_note),
                                        start=cur_note_start,
                                        end=clock)
                instrument.notes.append(note)

            # update the current note
            cur_note = note_num
            cur_note_start = clock

        # update the clock
        clock = clock + 1.0 / 5

    # Add the cello instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    return midi


def generate(model, seeds, window_size, length, num_to_gen, instrument_name):

    # generate a pretty midi file from a model using a seed
    def _gen(model, seed, window_size, length):

        generated = []

        buf = np.copy(seed).tolist()
        while len(generated) < length:
            arr = np.expand_dims(np.asarray(buf), 0)
            pred = model.predict(arr)

            # prob distrobuition sampling
            index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
            pred = np.zeros(seed.shape[1])

            pred[index] = 1
            generated.append(pred)
            buf.pop(0)
            buf.append(pred)

        return generated

    midi = []
    for i in range(0, num_to_gen):
        seed = seeds[random.randint(0, len(seeds) - 1)]
        gen = _gen(model, seed, window_size, length)
        midi.append(_network_output_to_midi(gen, instrument_name))
    return midi
