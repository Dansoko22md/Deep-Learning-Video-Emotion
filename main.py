"""
Video Emotion Recognition - VERSION FINALE ROBUSTE
Avec data augmentation et architecture améliorée
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# GPU optimization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

class Config:
    DATA_PATH = './VideoFlash'
    MODEL_PATH = './models'
    
    # Optimized settings
    FRAMES_PER_VIDEO = 8
    IMG_HEIGHT = 112
    IMG_WIDTH = 112
    CHANNELS = 3
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.0005  # Plus conservateur
    
    EMOTIONS = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    NUM_CLASSES = len(EMOTIONS)
    MAX_VIDEOS = None  # None = utiliser tout le dataset

config = Config()

# ===========================
# DATA AUGMENTATION
# ===========================

def augment_frame(frame):
    """Data augmentation pour améliorer la généralisation"""
    # Random brightness
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        frame = np.clip(frame * factor, 0, 1)
    
    # Random horizontal flip
    if np.random.rand() < 0.5:
        frame = cv2.flip(frame, 1)
    
    return frame

def extract_emotion_from_filename(filename):
    parts = filename.split('_')
    if len(parts) >= 3 and parts[2] in config.EMOTIONS:
        return config.EMOTIONS.index(parts[2])
    return None

def extract_frames_from_video(video_path, num_frames=8, augment=False):
    """Extract frames avec option d'augmentation"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (config.IMG_WIDTH, config.IMG_HEIGHT))
            frame = frame.astype(np.float32) / 255.0
            
            if augment:
                frame = augment_frame(frame)
            
            frames.append(frame)
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.float32))
    
    return np.array(frames[:num_frames], dtype=np.float32)

def get_video_file_list(data_path, max_videos=None):
    video_data = []
    video_files = [f for f in os.listdir(data_path) if f.endswith('.flv')]
    
    for video_file in video_files:
        label = extract_emotion_from_filename(video_file)
        if label is not None:
            video_data.append((os.path.join(data_path, video_file), label))
    
    if max_videos:
        from collections import defaultdict
        by_class = defaultdict(list)
        for path, label in video_data:
            by_class[label].append((path, label))
        
        sampled = []
        per_class = max_videos // len(config.EMOTIONS)
        for items in by_class.values():
            sampled.extend(items[:per_class])
        video_data = sampled
    
    return video_data

# ===========================
# GENERATOR
# ===========================

class VideoDataGenerator(keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size, shuffle=True, augment=False):
        super().__init__()
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(video_paths))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X_batch, y_batch = [], []
        for idx in batch_indices:
            try:
                frames = extract_frames_from_video(
                    self.video_paths[idx], 
                    config.FRAMES_PER_VIDEO,
                    augment=self.augment
                )
                if frames is not None:
                    X_batch.append(frames)
                    y_batch.append(self.labels[idx])
            except:
                continue
        
        if not X_batch:
            X_batch = [np.zeros((config.FRAMES_PER_VIDEO, config.IMG_HEIGHT, 
                                config.IMG_WIDTH, 3), dtype=np.float32)]
            y_batch = [0]
        
        return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.int32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ===========================
# IMPROVED MODEL WITH ATTENTION
# ===========================

def build_improved_cnn_lstm(input_shape, num_classes):
    """
    Architecture améliorée:
    - CNN pour features spatiales
    - LSTM bidirectionnel pour temporel
    - Attention mechanism
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN feature extraction par frame
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    
    # Dense pour réduire dimensionalité
    x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(0.4))(x)
    
    # LSTM bidirectionnel
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    
    # Attention mechanism (simple)
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='cnn_lstm_attention')

# ===========================
# TRAINING
# ===========================

def train_model(train_gen, val_gen):
    input_shape = (config.FRAMES_PER_VIDEO, config.IMG_HEIGHT, 
                   config.IMG_WIDTH, config.CHANNELS)
    model = build_improved_cnn_lstm(input_shape, config.NUM_CLASSES)
    
    # Optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    print(f"\n🔢 Total parameters: {model.count_params():,}")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODEL_PATH, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Custom callback pour afficher les prédictions par classe
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\n📊 Epoch {epoch+1} - Train Acc: {logs['accuracy']:.4f} | "
                f"Val Acc: {logs['val_accuracy']:.4f} | "
                f"Val Loss: {logs['val_loss']:.4f}"
            )
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ===========================
# VISUALIZATION
# ===========================

def plot_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy
    axes[0].plot(epochs, history.history['accuracy'], 'bo-', label='Train', linewidth=2)
    axes[0].plot(epochs, history.history['val_accuracy'], 'rs-', label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(epochs, history.history['loss'], 'bo-', label='Train', linewidth=2)
    axes[1].plot(epochs, history.history['val_loss'], 'rs-', label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_curves.png")
    plt.show()

def evaluate_model(model, test_gen):
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    y_pred_list, y_true_list = [], []
    
    print("Predicting on test set...")
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        y_pred_batch = model.predict(X_batch, verbose=0)
        y_pred_list.extend(np.argmax(y_pred_batch, axis=1))
        y_true_list.extend(y_batch)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(test_gen)} batches")
    
    y_pred = np.array(y_pred_list)
    y_test = np.array(y_true_list)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.EMOTIONS,
                yticklabels=config.EMOTIONS,
                cbar_kws={'label': 'Count'},
                square=True)
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Emotion', fontsize=14)
    plt.xlabel('Predicted Emotion', fontsize=14)
    
    # Ajouter les pourcentages
    for i in range(len(config.EMOTIONS)):
        for j in range(len(config.EMOTIONS)):
            if cm[i].sum() > 0:
                pct = cm[i,j] / cm[i].sum() * 100
                plt.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                        ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: confusion_matrix.png")
    plt.show()
    
    # Report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred, 
                               target_names=config.EMOTIONS,
                               digits=4,
                               zero_division=0))
    
    accuracy = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*70}")
    print(f"✓ FINAL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✓ F1-SCORE:       {f1:.4f}")
    print(f"{'='*70}\n")
    
    return accuracy, f1, cm

# ===========================
# MAIN
# ===========================

def main():
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎬 VIDEO EMOTION RECOGNITION - FINAL VERSION")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"  • Frames per video: {config.FRAMES_PER_VIDEO}")
    print(f"  • Image size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    print(f"  • Learning rate: {config.LEARNING_RATE}")
    print(f"  • Max epochs: {config.EPOCHS}")
    print(f"  • Videos: {config.MAX_VIDEOS or 'ALL'}")
    
    # Load
    print("\n[1] Loading videos...")
    video_data = get_video_file_list(config.DATA_PATH, config.MAX_VIDEOS)
    print(f"✓ Found {len(video_data)} videos")
    
    if not video_data:
        print("❌ No videos found!")
        return
    
    video_paths = [v[0] for v in video_data]
    labels = [v[1] for v in video_data]
    
    # Distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\n📊 Class distribution:")
    for emotion_idx, count in zip(unique, counts):
        print(f"  {config.EMOTIONS[emotion_idx]:>3}: {count:>4} videos ({count/len(labels)*100:.1f}%)")
    
    # Split
    print("\n[2] Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        video_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"  Train: {len(X_train):>4} | Val: {len(X_val):>4} | Test: {len(X_test):>4}")
    
    # Generators
    print("\n[3] Creating data generators...")
    train_gen = VideoDataGenerator(X_train, y_train, config.BATCH_SIZE, 
                                   shuffle=True, augment=True)
    val_gen = VideoDataGenerator(X_val, y_val, config.BATCH_SIZE, shuffle=False)
    test_gen = VideoDataGenerator(X_test, y_test, config.BATCH_SIZE, shuffle=False)
    print(f"  Train: {len(train_gen)} batches | Val: {len(val_gen)} | Test: {len(test_gen)}")
    
    # Train
    print("\n[4] 🏋️  Training model...")
    model, history = train_model(train_gen, val_gen)
    
    # Plot
    print("\n[5] 📊 Generating plots...")
    plot_results(history)
    
    # Evaluate
    print("\n[6] 🎯 Evaluating...")
    accuracy, f1, cm = evaluate_model(model, test_gen)
    
    # Save
    model.save(os.path.join(config.MODEL_PATH, 'final_model.keras'))
    print("✓ Model saved!")
    
    print("\n" + "="*70)
    print(f"🎉 COMPLETE! Final Test Accuracy: {accuracy*100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()