# -*- coding: utf-8 -*-
"""
Medicinal Plant Identification - Ensemble Model Training
Uses InceptionV3 + ConvNeXtTiny stacked ensemble
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, ConvNeXtTiny
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = 'dataset/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ─────────────────────────────────────────────
# BASE MODEL (InceptionV3 / ConvNeXtTiny)
# ─────────────────────────────────────────────
def create_base_model(model_class):
    base_model = model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    # Unfreeze top 20 layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train both base models
models = [create_base_model(InceptionV3), create_base_model(ConvNeXtTiny)]

for i, model in enumerate(models):
    print(f"\n Training Base Model {i+1}...")
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        ],
        verbose=1
    )

# ─────────────────────────────────────────────
# STACKED ENSEMBLE MODEL
# ─────────────────────────────────────────────
def create_stacked_model(models):
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    features1 = models[0](inputs, training=False)  # InceptionV3
    features2 = models[1](inputs, training=False)  # ConvNeXtTiny

    merged = Concatenate()([features1, features2])

    x = Dense(384, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(train_generator.num_classes, activation='softmax')(x)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.0002,
        first_decay_steps=8 * len(train_generator),
        t_mul=2.0,
        m_mul=0.8,
        alpha=0.0005
    )

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        metrics=['accuracy']
    )
    return model

stacked_model = create_stacked_model(models)

print("\n Training Stacked Ensemble Model...")
stacked_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,
    batch_size=64,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    ],
    verbose=1
)

# ─────────────────────────────────────────────
# EVALUATE & SAVE
# ─────────────────────────────────────────────
accuracy = stacked_model.evaluate(validation_generator)[1]
print(f'\n Ensemble Model Accuracy: {accuracy * 100:.2f}%')

os.makedirs('models', exist_ok=True)
stacked_model.save('models/stacked_model.keras')
print(" Model saved to models/stacked_model.keras")