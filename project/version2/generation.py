from keras.models import load_model
import os, argparse, time
import pretty_midi
import numpy as np
from copy import deepcopy
import random
import keras
import utils
from midi2audio import FluidSynth


midi_files = [os.path.join("vivaldi", path)
                  for path in os.listdir("vivaldi") \
                  if '.mid' in path or '.midi' in path]

# generate 10 tracks using random seeds

print('enter seed (1-50)')
seed = int(input())

seed_generator = utils.get_data_generator(midi_files,
                                              window_size=50,
                                              batch_size=1,
                                              num_threads=1,
                                              max_files_in_ram=10)

window = 50 # length of window
length = 100 # number of events
number = 10 # number of samples
instrument = 'Acoustic Grand Piano'  # full list is here https://www.midi.org/specifications/item/gm-level-1-sound-set


print('enter window size')
window = int(input())
print('enter lenght of sample')
length = int(input())
print('enter number of samples')
number = int(input())
print('enter instrument (for example Acoustic Grand Audio)')
instrument = input()


model = load_model('v5.hdf5') # here should be path to model


X, y = next(seed_generator)

generated = utils.generate(model, X, window,
                      length, number, instrument)

if not os.path.isdir('output'):
    os.makedirs('output')

for i, midi in enumerate(generated):
    file = os.path.join('output', '{}.mid'.format(i + 1))
    midi.write(file.format(i + 1))
    fs = FluidSynth('FluidR3Mono_GM.sf3') # here should be full path to Sound Font file
    fs.midi_to_audio(file.format(i + 1), os.path.join('output', '{}.wav'.format(i + 1)))

