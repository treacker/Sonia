from keras.models import load_model
import os, argparse, time
import pretty_midi
import numpy as np
from copy import deepcopy
import random
import keras
import utils
from midi2audio import FluidSynth



def generate(model, seeds, window_size, length, num_to_gen, instrument_name):

    # generate a pretty midi file from a model using a seed
    def _gen(model, seed, window_size, length):

        generated = []
        # ring buffer
        buf = np.copy(seed).tolist()
        while len(generated) < length:
            arr = np.expand_dims(np.asarray(buf), 0)
            pred = model.predict(arr)

            # argmax sampling (NOT RECOMMENDED), or...
            # index = np.argmax(pred)

            # prob distrobuition sampling
            index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
            pred = np.zeros(seed.shape[1])

            pred[index] = 1
            generated.append(pred)
            buf.pop(0)
            buf.append(pred)

        return generated

    midis = []
    for i in range(0, num_to_gen):
        seed = seeds[random.randint(0, len(seeds) - 1)]
        gen = _gen(model, seed, window_size, length)
        midis.append(_network_output_to_midi(gen, instrument_name))
    return midis

# create a pretty midi file with a single instrument using the one-hot encoding
# output of keras model.predict
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

    # Iterate over note names, which will be converted to note number later
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


midi_files = [os.path.join("vivaldi", path)
                  for path in os.listdir("vivaldi") \
                  if '.mid' in path or '.midi' in path]

# generate 10 tracks using random seeds

seed_generator = utils.get_data_generator(midi_files,
                                              window_size=50,
                                              batch_size=50,
                                              num_threads=1,
                                              max_files_in_ram=10)

window = 50
length = 100
number = 10
instrument = 'Acoustic Grand Piano'

model = load_model('v3.hdf5')


X, y = next(seed_generator)
generated = generate(model, X, window,
                      length, number, instrument)

if not os.path.isdir('output'):
    os.makedirs('output')

for i, midi in enumerate(generated):
    file = os.path.join('output', '{}.mid'.format(i + 1))
    midi.write(file.format(i + 1))
    fs = FluidSynth('filename') # here should be full path to .sf2 file
    fs.midi_to_audio(file.format(i + 1), os.path.join('output', '{}.wav'.format(i + 1)))

