
  
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

model = load_model('v3.hdf5') # here should be path to model


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

