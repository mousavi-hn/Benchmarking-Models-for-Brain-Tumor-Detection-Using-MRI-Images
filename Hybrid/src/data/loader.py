"""Load and preprocess image batches for hybrid quantum-classical training and evaluation."""
import keras
import numpy as np

from dataset import read_image
import src.configs as cfg


class MRISequence(keras.utils.Sequence):
    """Keras sequence that loads and preprocesses image batches from a DataFrame.
    
    The sequence supports deterministic evaluation and optional epoch-wise shuffling
    for training.
    """
    def __init__(self, df, preprocess_func, batch_size=8, target_size=(224, 224), shuffle=False):
        """Initialize the image sequence.
        
        Args:
            df: DataFrame containing image paths and labels.
            preprocess_func: Image preprocessing callable.
            batch_size: Number of samples per batch.
            target_size: Image dimensions expected by the model.
            shuffle: Whether to shuffle sample indices between epochs.
        """
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.preprocess_func = preprocess_func
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        """Return the number of batches in one epoch."""
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        """Load and preprocess one batch.
        
        Args:
            idx: Zero-based batch index.
        
        Returns:
            tuple: Batch image tensor and corresponding binary labels.
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        images = []
        labels = []

        for _, row in batch_df.iterrows():
            img = read_image(row["filepath"], self.target_size)
            img = self.preprocess_func(img)
            images.append(img)
            labels.append(float(row["label"]))

        x = np.asarray(images, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        return x, y

    def on_epoch_end(self):
        """Shuffle sample indices at the end of an epoch when enabled."""
        if self.shuffle:
            rng = np.random.default_rng(cfg.SEED)
            rng.shuffle(self.indices)


def make_generators(preprocess_func, train_df, val_df, test_df):
    """Create training, validation, and test data loaders.
    
    Args:
        preprocess_func: Backbone-specific image preprocessing callable.
        train_df: Training examples.
        val_df: Validation examples.
        test_df: Test examples.
        target_size: Requested image dimensions.
        batch_size: Number of samples per batch.
    
    Returns:
        tuple: Training, validation, and test generators/sequences.
    """
    train_seq = MRISequence(
        train_df, preprocess_func, batch_size=cfg.BATCH_SIZE,
        target_size=cfg.IMG_SIZE, shuffle=True
    )
    val_seq = MRISequence(
        val_df, preprocess_func, batch_size=cfg.BATCH_SIZE,
        target_size=cfg.IMG_SIZE, shuffle=False
    )
    test_seq = MRISequence(
        test_df, preprocess_func, batch_size=cfg.BATCH_SIZE,
        target_size=cfg.IMG_SIZE, shuffle=False
    )
    return train_seq, val_seq, test_seq

