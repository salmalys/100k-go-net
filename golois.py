import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import gc

import golois

planes = 31
moves = 361
N = 10000
epochs = 20
batch = 128
filters = 32

# Fonction pour créer un bloc résiduel
def residual_block(input_tensor, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(input_tensor)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None,
                      kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.Add()([x, input_tensor])  # Connexion résiduelle
    x = layers.Activation('relu')(x)
    return x

# Données d'entrée synthétiques
input_data = np.random.randint(2, size=(N, 19, 19, planes)).astype('float32')
policy = keras.utils.to_categorical(np.random.randint(moves, size=(N,)))
value = np.random.randint(2, size=(N,)).astype('float32')
end = np.random.randint(2, size=(N, 19, 19, 2)).astype('float32')
groups = np.zeros((N, 19, 19, 1)).astype('float32')

print("getValidation", flush=True)
golois.getValidation(input_data, policy, value, end)

# Modèle ResNet
input = keras.Input(shape=(19, 19, planes), name='board')
x = layers.Conv2D(filters, 3, activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.0001))(input)

# Ajouter plusieurs blocs résiduels
for i in range(10):  # 10 blocs résiduels
    x = residual_block(x, filters)

# Tête de politique
policy_head = layers.Conv2D(2, 1, activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Dense(moves, activation='softmax',
                           name='policy', kernel_regularizer=regularizers.l2(0.0001))(policy_head)

# Tête de valeur
value_head = layers.Conv2D(1, 1, activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

# Définir le modèle
model = keras.Model(inputs=input, outputs=[policy_head, value_head])
model.summary()

# Compilation du modèle
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy': 1.0, 'value': 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

# Boucle d'entraînement
for i in range(1, epochs + 1):
    print('epoch ' + str(i))
    golois.getBatch(input_data, policy, value, end, groups, i * N)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value},
                        epochs=1, batch_size=batch)
    if i % 5 == 0:
        gc.collect()
    if i % 20 == 0:
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(input_data, [policy, value], verbose=0, batch_size=batch)
        print("val =", val)
        model.save('resnet_model.h5')