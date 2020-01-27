from __future__ import print_function

import numpy as np
from database import *
import math

import keras
from keras import backend as K
from keras.models import Sequential
from keras import layers
from keras.models import model_from_json

from matplotlib import pyplot as plt


def generate_character_set_dictionary(list_of_vocabularies):
    dict = {}
    for language in list_of_vocabularies:
        for word in list_of_vocabularies[language]:
            for ch in word:
                dict[ch] = 0
    return dict


def binarize_language(language):
    language_ids = {"dutch": 0, "english": 1, "french": 2, "ido": 3, "japanese": 4, "korean": 5, "mandarin": 6,
                    "spanish": 7, "thai": 8, "vietnamese": 9}
    langbin = np.zeros(10)
    langbin[language_ids[language]] = 1
    return langbin


def binarize_character(ch, dict, binary_dim):
    x = [0 for _ in range(binary_dim)]
    index = list(dict.keys()).index(ch)
    for digit in range(len(x)):
        value = 2 ** (binary_dim - digit - 1)
        if index >= value:
            x[digit] = 1
            index -= value
    return x


def binarize_word(word, dict, binary_dim):
    x = []
    for ch in word:
        x.append(binarize_character(ch, dict, binary_dim))
    return x


def trainNetworks(train_data, character_dict, binary_dim, output_dim, word_length):
    language_ids = {"dutch": 0, "english": 1, "french": 2, "ido": 3, "japanese": 4, "korean": 5, "mandarin": 6,
                    "spanish": 7, "thai": 8, "vietnamese": 9}

    x = []
    y = []
    languages_each_word_appears_in = {}
    training_size = 0

    # Remove duplicates
    for language in language_ids:
        # print(sorted(train_data[language]))
        train_data[language] = list(set(train_data[language]))
        # print(sorted(train_data[language]))

    for language in language_ids:
        for word in train_data[language]:
            languages_each_word_appears_in[word] = []

    for language in language_ids:
        for word in train_data[language]:
            languages_each_word_appears_in[word].append(language_ids[language])
            # Only add the word to the training set if this is the first time we've seen it
            if len(languages_each_word_appears_in[word]) == 1:
                w = binarize_word(word, character_dict, binary_dim)
                x.append(w)
                training_size += 1

    for word in languages_each_word_appears_in:
        language_representation = [0 for _ in range(output_dim)]
        num_of_languages = len(languages_each_word_appears_in[word])
        # if num_of_languages > 1:
        #     print(languages_each_word_appears_in[word])
        #     print(word)
        v = 1 / num_of_languages
        for lang in languages_each_word_appears_in[word]:
            language_representation[lang] = v
        y.append(language_representation)

    x = np.array(x).reshape(training_size, word_length, binary_dim)
    y = np.array(y).reshape(training_size, output_dim)

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    split_at = len(x) - len(x) // 5
    (x_trn, x_val) = x[:split_at], x[split_at:]
    (y_trn, y_val) = y[:split_at], y[split_at:]

    split_at = len(x_val) - len(x_val) // 2
    (x_val, x_test) = x_val[:split_at], x_val[split_at:]
    (y_val, y_test) = y_val[:split_at], y_val[split_at:]

    trained = False
    best_RNN_val_score = 0
    best_LSTM_val_score = 0
    best_RNN_train_score = 0
    best_LSTM_train_score = 0
    best_RNN_model = None
    best_LSTM_model = None
    H = 0
    log_file = open("log.txt", "w")

    # Comment this block out if you are loading models instead of training
    trained = True
    RNN_hidden_units = []
    LSTM_hidden_units = []
    RNN_training_scores = []
    LSTM_training_scores = []
    RNN_val_scores = []
    LSTM_val_scores = []
    overfit_limit = 2
    LSTM_overfit_count = 0
    RNN_overfit_count = 0
    while RNN_overfit_count < overfit_limit or LSTM_overfit_count < overfit_limit:
        H += 3
        log_file.write("Hidden Units: {}\n".format(H))

        if RNN_overfit_count < overfit_limit:
            RNN_hidden_units.append(H)
            RNN_model = trainSimpleRNN(H, word_length, binary_dim, x_trn, y_trn, x_val, y_val)

            RNN_trn_evaluate = RNN_model.evaluate(x_trn, y_trn)
            # added this so evaluate isn't called twice
            RNN_trn_loss = RNN_trn_evaluate[0]
            RNN_trn_score = RNN_trn_evaluate[1]

            RNN_val_evaluate = RNN_model.evaluate(x_val, y_val)
            RNN_val_loss = RNN_val_evaluate[0]
            RNN_val_score = RNN_val_evaluate[1]

            RNN_training_scores.append(RNN_trn_score)
            RNN_val_scores.append(RNN_val_score)

            if RNN_val_score > best_RNN_val_score:
                best_RNN_val_score = RNN_val_score
                best_RNN_model = RNN_model
                best_RNN_train_score = RNN_trn_score

            if RNN_trn_score > best_RNN_train_score and RNN_val_score < best_RNN_val_score:
                RNN_overfit_count += 1

            log_file.write("RNN training loss: {}\n".format(RNN_trn_loss))
            log_file.write("RNN training accuracy: {}\n".format(RNN_trn_score))
            log_file.write("RNN validation loss:  {}\n".format(RNN_val_loss))
            log_file.write("RNN validation accuracy: {}\n".format(RNN_val_score))
            log_file.write("\n")

        if LSTM_overfit_count < overfit_limit:
            LSTM_hidden_units.append(H)
            LSTM_model = trainLSTM(H, word_length, binary_dim, x_trn, y_trn, x_val, y_val)

            LTSM_trn_evaluate = LSTM_model.evaluate(x_trn, y_trn)
            LSTM_trn_loss = LTSM_trn_evaluate[0]
            LSTM_trn_score = LTSM_trn_evaluate[1]

            LTSM_val_evaluate = LSTM_model.evaluate(x_val, y_val)
            LSTM_val_loss = LTSM_val_evaluate[0]
            LSTM_val_score = LTSM_val_evaluate[1]

            LSTM_training_scores.append(LSTM_trn_score)
            LSTM_val_scores.append(LSTM_val_score)

            if LSTM_val_score > best_LSTM_val_score:
                best_LSTM_val_score = LSTM_val_score
                best_LSTM_model = LSTM_model
                best_LSTM_train_score = LSTM_trn_score

            if LSTM_trn_score > best_LSTM_train_score and LSTM_val_score < best_LSTM_val_score:
                LSTM_overfit_count += 1

            log_file.write("LSTM training loss: {}\n".format(LSTM_trn_loss))
            log_file.write("LSTM training accuracy: {}\n".format(LSTM_trn_score))
            log_file.write("LSTM validation loss: {}\n".format(LSTM_val_loss))
            log_file.write("LSTM validation accuracy: {}\n".format(LSTM_val_score))
            log_file.write("\n")
        log_file.flush()

    if trained:
        fig, axs = plt.subplots(1, 2)

        axs[0].scatter(RNN_hidden_units, RNN_training_scores, c='r', label='Training Scores')
        axs[0].scatter(RNN_hidden_units, RNN_val_scores, c='b', label='Validation Scores')
        axs[0].set_xlabel('Hidden Units')
        axs[0].set_ylabel('Accuracy')
        axs[0].title.set_text('SimpleRNN')
        axs[0].legend(loc='lower right')

        axs[1].scatter(LSTM_hidden_units, LSTM_training_scores, c='r', label='Training Scores')
        axs[1].scatter(LSTM_hidden_units, LSTM_val_scores, c='b', label='Validation Scores')
        axs[1].set_xlabel('Hidden Units')
        axs[1].set_ylabel('Accuracy')
        axs[1].title.set_text('LSTM')
        axs[1].legend(loc='lower right')

        fig.suptitle('Scores during training for SimpleRNN and LSTM')
        plt.show()

    # Save best models to disk
    RNN_json = best_RNN_model.to_json()
    with open("RNN_model.json", "w") as json_file:
        json_file.write(RNN_json)
    best_RNN_model.save_weights("RNN_model.h5")

    LSTM_json = best_LSTM_model.to_json()
    with open("LSTM_model.json", "w") as json_file:
        json_file.write(LSTM_json)
    best_LSTM_model.save_weights("LSTM_model.h5")

    # # For loading saved models
    # json_file = open('RNN_model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # best_RNN_model = model_from_json(loaded_model_json)
    # best_RNN_model.load_weights("RNN_model.h5")
    # best_RNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # json_file = open('LSTM_model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # best_LSTM_model = model_from_json(loaded_model_json)
    # best_LSTM_model.load_weights("LSTM_model.h5")
    # best_LSTM_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Test the models
    RNN_test_score = best_RNN_model.evaluate(x_test, y_test)[1]
    LSTM_test_score = best_LSTM_model.evaluate(x_test, y_test)[1]

    log_file.write("SimpleRNN with {} hidden units: test score was {}\n".format(best_RNN_model.layers[0].units, RNN_test_score))
    log_file.write("LSTM with {} hidden units: test score was {}".format(best_LSTM_model.layers[0].units, LSTM_test_score))
    log_file.close()

    # As a curosity, check how many classifications the models get right for each language
    RNN_correct_classifications_by_class = np.array([0. for _ in range(10)])
    RNN_classifications_by_class = np.array([0. for _ in range(10)])
    LSTM_correct_classifications_by_class = np.array([0. for _ in range(10)])
    LSTM_classifications_by_class = np.array([0. for _ in range(10)])

    for i in range(len(x_test)):
        rowx, rowy = x_test[np.array([i])], y_test[np.array([i])]
        pred = best_RNN_model.predict(rowx, verbose=0)
        if pred.argmax(axis=1) == rowy.argmax(axis=1):
            RNN_correct_classifications_by_class[rowy.argmax(axis=1)[0]] += 1
        RNN_classifications_by_class[rowy.argmax(axis=1)[0]] += 1
        pred = best_LSTM_model.predict(rowx, verbose=0)
        if pred.argmax(axis=1) == rowy.argmax(axis=1):
            LSTM_correct_classifications_by_class[rowy.argmax(axis=1)[0]] += 1
        LSTM_classifications_by_class[rowy.argmax(axis=1)[0]] += 1

    RNN_correct_classifications_by_class = np.divide(RNN_correct_classifications_by_class, RNN_classifications_by_class)
    LSTM_correct_classifications_by_class = np.divide(LSTM_correct_classifications_by_class, LSTM_classifications_by_class)
    print("SimpleRNN running on 10 random test samples:")
    for i in range(10):
        ind = np.random.randint(0, len(x_test))
        rowx, rowy = x_test[np.array([ind])], y_test[np.array([ind])]
        preds = best_RNN_model.predict(rowx, verbose=0)
        print("Prediction: {}".format(preds))
        print("Predicted class: {}".format(preds.argmax(axis=1)))
        print("Correct answer: {}".format(rowy))
        print("Correct class: {}".format(rowy.argmax(axis=1)))
        print("\n")

    # LSTM_hidden_units = best_LSTM_model.
    print("LSTM running on 10 random test samples:")
    for i in range(10):
        ind = np.random.randint(0, len(x_test))
        rowx, rowy = x_test[np.array([ind])], y_test[np.array([ind])]
        preds = best_LSTM_model.predict(rowx, verbose=0)
        print("Prediction: {}".format(preds))
        print("Predicted class: {}".format(preds.argmax(axis=1)))
        print("Correct answer: {}".format(rowy))
        print("Correct class: {}".format(rowy.argmax(axis=1)))
        print("\n")

    print("SimpleRNN with {} hidden units: test score was {}".format(best_RNN_model.layers[0].units, RNN_test_score))
    print("LSTM with {} hidden units: test score was {}".format(best_LSTM_model.layers[0].units, LSTM_test_score))

    plt.bar(list(language_ids.keys()), RNN_correct_classifications_by_class)
    plt.title('SimpleRNN')
    plt.ylabel("Test Score")
    plt.show()

    plt.bar(list(language_ids.keys()), LSTM_correct_classifications_by_class)
    plt.ylabel("Test Score")
    plt.title("LSTM")
    plt.show()


def trainLSTM(H, word_length, binary_dim, x_trn, y_trn, x_val, y_val):
    model = Sequential()
    model.add(layers.LSTM(H, input_shape=(word_length, binary_dim)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_trn, y_trn, epochs=6 * H, batch_size=6 * H, validation_data=(x_val, y_val))

    return model


def trainSimpleRNN(H, word_length, binary_dim, x_trn, y_trn, x_val, y_val):
    model = Sequential()
    model.add(layers.SimpleRNN(H, input_shape=(word_length, binary_dim)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_trn, y_trn, epochs=3 * H, batch_size=3 * H, validation_data=(x_val, y_val))

    return model


def get_longest_word_length(vocabulary_list):
    max_length = 0
    longest_word = ""
    for language in vocabulary_list:
        for word in vocabulary_list[language]:
            if len(word) > max_length:
                max_length = len(word)
                longest_word = word
    return max_length, longest_word


def main():
    vocab = get_vocabularies()

    # Used for current representation of words
    character_dict = generate_character_set_dictionary(vocab)

    if " " not in character_dict:
        character_dict[" "] = 0
    longest_word_length, longest_word = get_longest_word_length(vocab)

    binary_dim = math.ceil(math.log(len(character_dict), 2))
    #print(binary_dim)

    for language in vocab:
        for w in range(len(vocab[language])):
            word = vocab[language][w]
            vocab[language][w] = word + " " * (longest_word_length - len(word))

    output_dimensions = len(vocab)
    #print(output_dimensions)

    trainNetworks(vocab, character_dict, binary_dim, output_dimensions, longest_word_length)


if __name__ == "__main__":
    main()
