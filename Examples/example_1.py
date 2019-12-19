import os
import numpy
import pandas
import sklearn
import keras
import keras.backend as K


def create_rnn_model_1():
    # Input Layers
    input_layer_1 = keras.layers.Input(shape=(1,), name="input_layer_1")
    input_layer_2 = keras.layers.Input(shape=(4,), name="input_layer_2")
    input_layer_3 = keras.layers.Input(shape=(4,), name="input_layer_3")

    # Embedding Layers
    emb_layer = keras.layers.Embedding(50, 50)
    emb_layer_2 = emb_layer(input_layer_2)
    emb_layer_3 = emb_layer(input_layer_3)

    # RNN Layers
    rnn_layer_1 = keras.layers.GRU(8, activation="relu")(emb_layer_2)
    rnn_layer_2 = keras.layers.GRU(16, activation="relu")(emb_layer_3)

    # Merge all Layers
    x = keras.layers.concatenate([input_layer_1,
                                  rnn_layer_1,
                                  rnn_layer_2])

    output_layer = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.models.Model(
        inputs=[input_layer_1, input_layer_2, input_layer_3],
        outputs=output_layer
    )


def create_rnn_model_2(num_words, training_length):
    model = keras.models.Sequential()

    model.add(
        keras.layers.Embedding(input_dim=num_words, input_length=training_length, output_dim=100, trainable=True, mask_zero=True)
    )

    # model.add(keras.layers.Masking(mask_value=0.0))

    model.add(keras.layers.LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

    model.add(keras.layers.Dense(64, activation='relu'))

    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(num_words, activation='softmax'))

    return model


csv = pandas.read_csv('67f0a094.csv', sep=',')
print(csv.shape, type(csv))
csv_list = csv.values.tolist()
print(len(csv_list), len(csv_list[0]))
# abstracts = csv.values.tolist()[:][0]
abstracts = [x[0] for x in csv_list]
# titles = csv.values.tolist()[:][1]
titles = [x[1] for x in csv_list]

print(abstracts[0], len(abstracts))
print(titles[0], len(titles))

# Create Tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters='!"#$%^&()*+,-./:;<=>?@[\\]_~{|}\t\n',
    lower=True,
    split=' '
)

# Train the Tokenizer on the abstract texts
res_1 = tokenizer.fit_on_texts(abstracts)
print('res_1', res_1)

# Convert list of strings into list of integers
sequences = tokenizer.texts_to_sequences(abstracts)
print('sequence', sequences[0])
print('index_word', len(tokenizer.index_word))
str_0 = ' '.join(tokenizer.index_word[i] for i in sequences[0])
print('str_0', str_0)

# Supervised Learning
features = []
labels = []
training_length = 50

for seq in sequences:
    for i in range(training_length, len(seq)-1):
        features.append(seq[i - training_length : i])
        labels.append(seq[i+1])

print('features', len(features), len(features[0]))
print('labels', len(labels))

features_arr = numpy.array(features)
print('features_arr', features_arr.shape)
labels_arr = numpy.array(labels)
unique_labels_arr = numpy.unique(labels_arr)
print('unique_labels_arr', unique_labels_arr.shape, 'labels_arr', labels_arr.shape)
# One-hot encoding
all_labels = tokenizer.index_word
print(type(all_labels), len(all_labels), list(all_labels.keys())[:10], list(all_labels.values())[:10])

# targets = keras.utils.to_categorical(labels_arr)
targets = numpy.zeros((len(features_arr), len(all_labels)+1), dtype=numpy.int8)
for inx, i in enumerate(labels):
    targets[inx, i] = 1

print('targets', targets.shape)

'''
rnn_model = create_rnn_model_2(targets.shape[1], training_length=training_length)
rnn_model.summary()

rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
call_backs = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
              keras.callbacks.ModelCheckpoint('rnn_model.h5', monitor='val_acc', mode='max', save_best_only=True)]

history = rnn_model.fit(
    x=features_arr,
    y=targets,
    batch_size=1024,
    epochs=2,
    validation_split=0.15,
    callbacks=call_backs,
)
'''
rnn_model = keras.models.load_model('rnn_model.h5')
seq = numpy.random.choice(sequences)
seed_inx = numpy.random.randint(0, len(seq) - training_length)
seed_end = seed_inx + training_length
seed = seq[seed_inx:seed_end]
temp_0 = [tokenizer.index_word[i] for i in seed]
original = ' '.join(temp_0)
print('original', original)

for i in range(training_length):
    print('seed', len(seed))
    preds = rnn_model.predict(numpy.array(seed).reshape(1, -1))[0].astype(numpy.float64)
    exp_preds = numpy.exp(numpy.log(preds))
    softmax_preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, softmax_preds, 1)[0]
    next_inx = numpy.argmax(probas)
    print('next_inx', next_inx, tokenizer.index_word[next_inx])
    # seed += [next_inx]
    seed = seed[1:] + [next_inx]






