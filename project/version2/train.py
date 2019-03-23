
import os, argparse, time
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM

from keras.optimizers import SGD, Adam



import utils

# model building

OUTPUT_SIZE = 129 # 0-127 notes + 1 for rests


epoch = 0
num_layers = 6
model = Sequential()
for layer_index in range(num_layers):
    kwargs = dict()
    kwargs['units'] = 64
    # if this is the first layer
    if layer_index == 0:
        kwargs['input_shape'] = (50, OUTPUT_SIZE)
        if num_layers == 1:
            kwargs['return_sequences'] = False
        else:
            kwargs['return_sequences'] = True
        model.add(LSTM(**kwargs))
    else:
        # if this is a middle layer
        if layer_index != num_layers - 1:
            kwargs['return_sequences'] = True
            model.add(LSTM(**kwargs))
        else: # this is the last layer
            kwargs['return_sequences'] = False
            model.add(LSTM(**kwargs))
    model.add(Dropout(0.2))

model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))


optimizer = Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])



# actual training



midi_files = [os.path.join("data1", path)
                  for path in os.listdir("data1") \
                  if '.mid' in path or '.midi' in path]

print(len(midi_files))

experiment_dir = utils.create_experiment_dir('experiment_dir4', 1)

val_split = 0.2 

val_split_index = int(float(len(midi_files)) * val_split)


train_generator = utils.get_data_generator(midi_files[0:val_split_index])

val_generator = utils.get_data_generator(midi_files[val_split_index:])

callbacks = utils.get_callbacks(experiment_dir)



batch_size = 60
start_time = time.time()
num_epochs = 10

model.fit_generator(train_generator,
                    steps_per_epoch=len(midi_files) * 600 / batch_size,
                    epochs=num_epochs,
                    validation_data=val_generator,
                    validation_steps=0.2 * len(midi_files) * 600 / batch_size,
                    verbose=1,
                    callbacks=callbacks,
                    initial_epoch=0)

log('Finished in {:.2f} seconds'.format(time.time() - start_time), 1)

