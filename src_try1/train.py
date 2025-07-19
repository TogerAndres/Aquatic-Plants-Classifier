# ENTRENADOR MODELO PLANTAS - SOLO BRILLO 🔥

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout,
                                     BatchNormalization, Activation, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 🛠️ Configuración GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 🗂️ Paths
dataset_path = "D:/Informacion/Desktop/Code/ProyectoImpacto/dataSet/Augmented Images"

# 📊 Data Generators SOLO BRILLO
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    brightness_range=[0.8, 1.2]  # 🔆 Solo aumentos de brillo
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# 🧠 Modelo EfficientNetB0 Transfer Learning
def create_efficient_model(num_classes, input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # ❗ Inicialmente congelado

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    return Model(inputs, outputs)

model = create_efficient_model(train_generator.num_classes)

# ⚙️ Compilación inicial
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
)

# 📋 Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_modelo_plantas.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# 🚀 Entrenamiento inicial (cabezas)
history1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# 🔓 Fine-tuning: descongela el modelo base completo
model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
)

history2 = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# 💾 Guardar modelo final
model.save("modelo_plantas_final.keras")
print("✅ Modelo guardado como 'modelo_plantas_final.keras'")

# 📈 Gráficas entrenamiento
def plot_training_history(history1, history2):
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train')
    plt.plot(epochs, val_acc, 'r-', label='Val')
    plt.axvline(x=len(history1.history['accuracy']), color='g', linestyle='--', alpha=0.7, label='Fine-tuning')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train')
    plt.plot(epochs, val_loss, 'r-', label='Val')
    plt.axvline(x=len(history1.history['loss']), color='g', linestyle='--', alpha=0.7, label='Fine-tuning')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history1, history2)

# 🎯 Evaluación final
val_loss, val_acc, val_top3 = model.evaluate(val_generator, verbose=1)
print(f"Precisión validación: {val_acc:.4f}, Top-3: {val_top3:.4f}, Loss: {val_loss:.4f}")
