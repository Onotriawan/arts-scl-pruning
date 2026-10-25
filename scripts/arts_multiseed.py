import argparse
import gc
import gzip
import os
import random
import shutil
import tempfile
import time
import traceback

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF info/warning spam

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Default: data/ArabicTrafficSigns next to the repository root (see README).
# Override with --data-root <folder> or by editing these paths.
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ArabicTrafficSigns")
TRAIN_DIR = os.path.join(_ROOT, "train")
TRAIN_CSV = os.path.join(_ROOT, "Labels_Train_Arabic_Traffic_Signs_24_2200.csv")
TEST_DIR = os.path.join(_ROOT, "test")
TEST_CSV = os.path.join(_ROOT, "Labels_Test_Arabic_Traffic_Signs_24_518.csv")

# --------------------------------------------------------------------------
# Hyper-parameters (unchanged from the paper)
# --------------------------------------------------------------------------
HIDDEN_UNITS = 512
PROJECTION_UNITS = 128
DROPOUT_RATE = 0.5
TEMPERATURE = 0.1
INPUT_SHAPE = (224, 224, 3)
IM_SIZE = (224, 224)
NUM_EPOCHS = 50          # max epochs for every supervised stage (early stopping)
PRETRAIN_EPOCHS = 50     # fixed stage-1 epochs (SCL and ce_two_stage)
PATIENCE = 8
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
SPLIT_SEED = 42          # train/val split kept FIXED across seeds
VAL_FRACTION = 0.25      # 0.2 / (0.6 + 0.2), as in the notebook

PRUNING_EPOCHS = 30
PRUNE_RAMP_EPOCHS = 20   # sparsity reaches FINAL_SPARSITY at the end of this epoch
INITIAL_SPARSITY = 0.30
FINAL_SPARSITY = 0.80
PRUNING_FREQUENCY = 100

# Table 6 re-implementations, trained with OUR split, augmentation, head,
# optimiser and early stopping, so only the architecture differs:
#   resnet_v1   : ImageNet ResNet50 (V1)  ~ ResNet V1 of ref. [19]/b13
#   densenet121 : ImageNet DenseNet121   ~ ref. [20]/b14
#   le_cnn      : small CNN on 32x32     ~ ref. [33]/b17
# LE-CNN re-implementation (ref. b17). The paper describes 3 conv blocks +
# ReLU + max-pool, 3 dropout layers, flatten, softmax, 32x32 input, Adam 1e-3.
# Filter counts / dropout rates are NOT given in our text: CHECK ref. b17 and
# edit these values so the re-implementation matches it.
LECNN_FILTERS = (32, 64, 128)
LECNN_DROPOUT = (0.25, 0.25, 0.5)
LECNN_LR = 1e-3

REP_DATA = None   # ~100 training images for int8 calibration (set in main)
INT8_MAX_S_PER_IMAGE = 0.5   # fewer latency runs if int8 inference is slower than this
# int8 TFLite runs slowly on desktop CPUs without optimised int8 kernels (~2.5 s per
# image on the RTX-3060 laptop). Its accuracy is therefore checked only on the first
# seed, for these configs, on a class-stratified subset (INT8_PER_CLASS per class).
INT8_EVAL_CONFIGS = {"baseline", "pruned_baseline", "scl", "pruned_scl"}
INT8_PER_CLASS = 5

HEAD_LAYERS = ["dense_1", "batch_norm_1", "dense_2", "batch_norm_2", "output_dense"]

# group -> configs it produces (a pruned config reuses the group's trained model)
GROUPS = {
    "baseline": ["baseline", "pruned_baseline"],
    "scl": ["scl", "pruned_scl"],
    "ce_two_stage": ["ce_two_stage", "pruned_ce_two_stage"],
    "frozen_imagenet": ["frozen_imagenet", "pruned_frozen_imagenet"],
    # Stage 1 with class-balanced batches (BALANCED_P classes x BALANCED_K images).
    # With random batches of 16 from 24 classes ~53 % of anchors have no positive.
    "scl_balanced": ["scl_balanced", "pruned_scl_balanced"],
    "ce_two_stage_balanced": ["ce_two_stage_balanced", "pruned_ce_two_stage_balanced"],
    "resnet_v1": ["resnet_v1"],
    "densenet121": ["densenet121"],
    "le_cnn": ["le_cnn"],
}
DEFAULT_GROUPS = ["baseline", "scl", "ce_two_stage", "frozen_imagenet"]
BALANCED_P, BALANCED_K = 8, 2   # 8 classes x 2 images = batch of 16 (same batch size):
                                # every anchor has >= 1 positive and negatives from 7 classes
ALL_CONFIGS = [c for g in GROUPS.values() for c in g]


# ==========================================================================
# Data
# ==========================================================================
def read_csv_data(csv_file, img_dir, im_size=IM_SIZE):
    df = pd.read_csv(csv_file, header=None)          # FIX 1: no header row
    img_col, label_col = df.columns[0], df.columns[1]
    X, y = [], []
    for _, row in df.iterrows():
        base = str(row[img_col])
        img = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            p = os.path.join(img_dir, base + ext)
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                break
        if img is None:
            print(f"  could not read {base}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(cv2.resize(img, im_size, interpolation=cv2.INTER_AREA))
        y.append(int(row[label_col]))
    return np.array(X), np.array(y)


def set_seed(seed, deterministic):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as e:  # noqa: BLE001
            print("  op determinism not available:", e)


# ==========================================================================
# Models
# ==========================================================================
def build_fns():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_model_optimization as tfmot
    sparsity = tfmot.sparsity.keras
    tf.get_logger().setLevel("ERROR")
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    BACKBONES = {
        "resnet50v2": (keras.applications.ResNet50V2,
                       keras.applications.resnet_v2.preprocess_input),
        "densenet121": (keras.applications.DenseNet121,
                        keras.applications.densenet.preprocess_input),
        "resnet50": (keras.applications.ResNet50,          # ResNet V1
                     keras.applications.resnet50.preprocess_input),
    }

    def make_augmentation():
        return keras.Sequential([
            layers.Normalization(),
            layers.RandomWidth(0.1),
            layers.RandomHeight(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ], name="augmentation")

    def create_encoder(backbone="resnet50v2"):
        ctor, prep = BACKBONES[backbone]
        net = ctor(include_top=False, input_shape=INPUT_SHAPE, weights="imagenet",
                   pooling="avg")
        inputs = layers.Input(shape=INPUT_SHAPE)
        x = make_augmentation()(inputs)
        x = layers.Lambda(prep, name="preprocess")(x)
        outputs = net(x)
        return keras.Model(inputs, outputs, name=f"{backbone}_encoder")

    def compile_ce(model, lr=LEARNING_RATE):
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=keras.optimizers.Adam(learning_rate=lr),
                      metrics=["accuracy"])
        return model

    def create_classifier(encoder, num_classes, trainable=True):
        for layer in encoder.layers:
            layer.trainable = trainable
        inputs = layers.Input(shape=INPUT_SHAPE)
        x = encoder(inputs)
        x = layers.Dense(HIDDEN_UNITS, activation="relu", name="dense_1")(x)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)
        x = layers.Dense(HIDDEN_UNITS // 2, activation="relu", name="dense_2")(x)
        x = layers.BatchNormalization(name="batch_norm_2")(x)
        x = layers.Dropout(DROPOUT_RATE, name="dropout_2")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="output_dense")(x)
        return compile_ce(keras.Model(inputs, outputs, name="traffic_sign_classifier"))

    def create_le_cnn(num_classes):
        inputs = layers.Input(shape=INPUT_SHAPE)
        x = layers.Resizing(32, 32, name="resize_32")(inputs)
        x = layers.Rescaling(1.0 / 255)(x)
        for f, d in zip(LECNN_FILTERS, LECNN_DROPOUT):
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(d)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return compile_ce(keras.Model(inputs, outputs, name="le_cnn"), lr=LECNN_LR)

    def add_projection_head(encoder):
        inputs = keras.Input(shape=INPUT_SHAPE)
        x = encoder(inputs)
        x = layers.Dense(PROJECTION_UNITS, activation="relu", name="projection_dense")(x)
        x = layers.BatchNormalization(name="projection_batch_norm")(x)
        return keras.Model(inputs, x, name="encoder_with_projection")

    def add_linear_ce_head(encoder, num_classes):
        inputs = keras.Input(shape=INPUT_SHAPE)
        x = encoder(inputs)
        x = layers.Dense(num_classes, activation="softmax", name="stage1_ce_head")(x)
        return compile_ce(keras.Model(inputs, x, name="encoder_with_ce_head"))

    class SupervisedContrastiveLoss(keras.losses.Loss):
        def __init__(self, temperature=0.1, name=None):
            super().__init__(name=name)
            self.temperature = temperature

        def call(self, labels, feature_vectors):
            labels = tf.reshape(labels, [-1, 1])
            fv = tf.math.l2_normalize(feature_vectors, axis=1)
            logits = tf.matmul(fv, tf.transpose(fv)) / self.temperature
            mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
            logits_mask = 1.0 - tf.eye(tf.shape(labels)[0])
            mask = mask * logits_mask
            logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
            exp_logits = tf.exp(logits) * logits_mask
            log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)
            mlpp = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-12)
            return tf.reduce_mean(-mlpp)

    def create_pruned_model(base_model, train_size):
        steps_per_epoch = int(np.ceil(train_size / BATCH_SIZE))
        schedule = sparsity.PolynomialDecay(
            initial_sparsity=INITIAL_SPARSITY, final_sparsity=FINAL_SPARSITY,
            begin_step=0, end_step=steps_per_epoch * PRUNE_RAMP_EPOCHS,
            frequency=PRUNING_FREQUENCY)

        def apply(layer):
            if isinstance(layer, layers.Dense):
                return sparsity.prune_low_magnitude(layer, pruning_schedule=schedule)
            return layer
        return compile_ce(tf.keras.models.clone_model(base_model, clone_function=apply))

    class BestAfterEpoch(keras.callbacks.Callback):
        """Keep the best val_loss weights, only from epochs >= start_epoch."""
        def __init__(self, start_epoch):
            super().__init__()
            self.start_epoch = start_epoch
            self.best = np.inf
            self.best_weights = None
            self.best_epoch = None

        def on_epoch_end(self, epoch, logs=None):
            if epoch + 1 >= self.start_epoch and logs["val_loss"] < self.best:
                self.best = logs["val_loss"]
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch + 1

    def early_stopping():
        return [keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                              patience=PATIENCE,
                                              restore_best_weights=True, verbose=0)]

    return dict(tf=tf, keras=keras, layers=layers, sparsity=sparsity,
                create_encoder=create_encoder, create_classifier=create_classifier,
                create_le_cnn=create_le_cnn, add_projection_head=add_projection_head,
                add_linear_ce_head=add_linear_ce_head,
                SupervisedContrastiveLoss=SupervisedContrastiveLoss,
                create_pruned_model=create_pruned_model, BestAfterEpoch=BestAfterEpoch,
                early_stopping=early_stopping)


# ==========================================================================
# Export model (inference graph without the random-augmentation layers)
# ==========================================================================
def make_export_model(model, F, X_check):
    """Rebuild the classifier without augmentation layers (they are identity at
    inference). Used for TFLite, FLOPs and latency. Falls back to the original
    model if the structure is different (e.g. LE-CNN)."""
    keras, layers = F["keras"], F["layers"]
    try:
        enc = [l for l in model.layers if l.name.endswith("_encoder")][0]
        aug = enc.get_layer("augmentation")
        pre = enc.get_layer("preprocess")
        backbone = enc.layers[-1]
        inp = keras.Input(shape=INPUT_SHAPE)
        x = aug.layers[0](inp)            # Normalization (identity unless adapted)
        x = pre(x)
        x = backbone(x, training=False)
        for name in HEAD_LAYERS:
            x = model.get_layer(name)(x, training=False)
        exp = keras.Model(inp, x, name="export")
        diff = float(np.max(np.abs(model.predict(X_check, verbose=0) -
                                   exp.predict(X_check, verbose=0))))
        if diff > 1e-3:
            print(f"  export model differs from model (max diff {diff:.2e}); using model")
            return model, diff
        return exp, diff
    except Exception as e:  # noqa: BLE001
        print("  export model not built, using model itself:", e)
        return model, np.nan


# ==========================================================================
# Measurements
# ==========================================================================
def _gz_size(path):
    with open(path, "rb") as fi, gzip.open(path + ".gz", "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    return os.path.getsize(path + ".gz")


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def measure_sizes(model, export, F):
    """All sizes in MB = 10^6 bytes. Returns (dict, tflite_f32, tflite_int8)."""
    tf = F["tf"]
    out = {}
    tfl32 = tfl8 = None
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "w.h5")
        model.save(p, include_optimizer=False)
        out["h5_weights_MB"] = os.path.getsize(p) / 1e6
        out["h5_weights_gz_MB"] = _gz_size(p) / 1e6

        p = os.path.join(d, "opt.h5")
        model.save(p)  # default: WITH optimizer state if the model was trained
        out["h5_with_opt_MB"] = os.path.getsize(p) / 1e6

        for tag in ["f32", "int8"]:
            try:
                conv = tf.lite.TFLiteConverter.from_keras_model(export)
                if tag == "int8":   # full-integer post-training quantization
                    conv.optimizations = [tf.lite.Optimize.DEFAULT]
                    if REP_DATA is not None:
                        def rep_gen():
                            for img in REP_DATA:
                                yield [img[None].astype(np.float32)]
                        conv.representative_dataset = rep_gen
                data = conv.convert()
                p = os.path.join(d, f"{tag}.tflite")
                _write(p, data)
                out[f"tflite_{tag}_MB"] = os.path.getsize(p) / 1e6
                out[f"tflite_{tag}_gz_MB"] = _gz_size(p) / 1e6
                if tag == "f32":
                    tfl32 = data
                else:
                    tfl8 = data
            except Exception as e:  # noqa: BLE001
                print(f"  TFLite {tag} conversion failed:", e)
                out[f"tflite_{tag}_MB"] = out[f"tflite_{tag}_gz_MB"] = np.nan
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out, tfl32, tfl8


def measure_sparsity(model, F):
    Dense = F["layers"].Dense
    dz = dt = 0
    dense_params = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            k = layer.kernel.numpy()
            dz += int((k == 0).sum())
            dt += k.size
            dense_params += layer.count_params()
    ws = model.get_weights()
    allz = sum(int((w == 0).sum()) for w in ws)
    allt = sum(w.size for w in ws)
    return {"dense_kernel_sparsity": dz / dt if dt else np.nan,
            "model_sparsity": allz / allt, "nonzero_params": allt - allz,
            "dense_param_share": dense_params / allt}


def dense_head_mflops(model, F):
    Dense = F["layers"].Dense
    return sum(2 * l.kernel.shape[0] * l.kernel.shape[1]
               for l in model.layers if isinstance(l, Dense)) / 1e6


def count_gflops(model, F):
    tf = F["tf"]
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2)
        fn = tf.function(lambda x: model(x, training=False))
        cf = fn.get_concrete_function(tf.TensorSpec([1, *INPUT_SHAPE], tf.float32))
        frozen = convert_variables_to_constants_v2(cf)
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        info = tf.compat.v1.profiler.profile(graph=frozen.graph,
                                             run_meta=tf.compat.v1.RunMetadata(),
                                             cmd="op", options=opts)
        return info.total_float_ops / 1e9
    except Exception as e:  # noqa: BLE001
        print("  FLOPs count failed:", e)
        return np.nan


def measure_latency(model, F, device, n_warmup=20, n_runs=200):
    """Median batch-1 latency in ms."""
    tf = F["tf"]
    try:
        with tf.device(device):
            x = tf.random.uniform([1, *INPUT_SHAPE], 0, 255)
            fn = tf.function(lambda inp: model(inp, training=False))
            for _ in range(n_warmup):
                fn(x).numpy()
            ts = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                fn(x).numpy()
                ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))
    except Exception as e:  # noqa: BLE001
        print(f"  latency on {device} failed:", e)
        return np.nan


def _interpreter(tfl, F):
    it = F["tf"].lite.Interpreter(model_content=tfl, num_threads=os.cpu_count())
    it.allocate_tensors()
    return it, it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]


def tflite_latency(tfl, F, n_warmup=10, n_runs=100):
    if tfl is None:
        return np.nan
    it, i_in, i_out = _interpreter(tfl, F)
    x = np.random.uniform(0, 255, [1, *INPUT_SHAPE]).astype(np.float32)
    t0 = time.perf_counter()
    it.set_tensor(i_in, x)
    it.invoke()
    if time.perf_counter() - t0 > INT8_MAX_S_PER_IMAGE:   # very slow kernels: few runs
        n_warmup, n_runs = 1, 3
    for _ in range(n_warmup):
        it.set_tensor(i_in, x)
        it.invoke()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        it.set_tensor(i_in, x)
        it.invoke()
        it.get_tensor(i_out)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def int8_check(tfl, F, X, y, y_pred_float, per_class):
    """int8 vs float32 on a class-stratified subset of the test set.
    Returns (int8 accuracy, float accuracy, int8/float agreement, n)."""
    if tfl is None:
        return np.nan, np.nan, np.nan, 0
    idx = np.concatenate([np.where(y == c)[0][:per_class] for c in np.unique(y)])
    it, i_in, i_out = _interpreter(tfl, F)
    p8 = []
    t0 = time.time()
    for xi in X[idx]:
        it.set_tensor(i_in, xi[None].astype(np.float32))
        it.invoke()
        p8.append(int(it.get_tensor(i_out).argmax()))
    p8 = np.array(p8)
    print(f"  int8 check on {len(idx)} images took {time.time() - t0:.0f} s")
    return ((p8 == y[idx]).mean(), (y_pred_float[idx] == y[idx]).mean(),
            (p8 == y_pred_float[idx]).mean(), len(idx))


def peak_gpu_memory_mb(model, F):
    tf = F["tf"]
    if not tf.config.list_physical_devices("GPU"):
        return np.nan
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        model(tf.random.uniform([1, *INPUT_SHAPE], 0, 255), training=False).numpy()
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
    except Exception as e:  # noqa: BLE001
        print("  GPU memory stats failed:", e)
        return np.nan


def evaluate(model, X_test, y_test, F, measure_cost, smoke, cfg):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted",
                                                  zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro",
                                                     zero_division=0)
    row = {"accuracy": accuracy_score(y_test, y_pred), "precision": p, "recall": r,
           "f1": f1, "precision_macro": pm, "recall_macro": rm, "f1_macro": f1m,
           "n_errors": int((y_pred != y_test).sum()), "n_test": int(len(y_test)),
           "total_params": model.count_params()}
    row.update(measure_sparsity(model, F))
    export, row["export_max_diff"] = make_export_model(model, F, X_test[:16])
    sizes, tfl32, tfl8 = measure_sizes(model, export, F)
    row.update(sizes)
    if measure_cost and cfg in INT8_EVAL_CONFIGS:
        (row["int8_acc_subset"], row["float_acc_subset"], row["int8_float_agreement"],
         row["int8_subset_n"]) = int8_check(tfl8, F, X_test, y_test, y_pred,
                                            1 if smoke else INT8_PER_CLASS)
    if measure_cost:
        n = 10 if smoke else 200
        row["gflops"] = count_gflops(export, F)
        row["dense_head_mflops"] = dense_head_mflops(model, F)
        row["latency_gpu_ms"] = (measure_latency(export, F, "/GPU:0", n_runs=n)
                                 if F["tf"].config.list_physical_devices("GPU") else np.nan)
        row["latency_cpu_ms"] = measure_latency(export, F, "/CPU:0", n_runs=max(n // 4, 5))
        row["latency_tflite_f32_ms"] = tflite_latency(tfl32, F, n_runs=max(n // 2, 5))
        row["latency_tflite_int8_ms"] = tflite_latency(tfl8, F, n_runs=max(n // 2, 5))
        row["peak_gpu_mem_MB"] = peak_gpu_memory_mb(export, F)
    return row, y_pred


# ==========================================================================
# Training of one group for one seed
# ==========================================================================
def run_group(group, seed, data, out_dir, done, args):
    configs = [c for c in GROUPS[group] if (c, seed) not in done]
    if not configs:
        return
    F = build_fns()
    tf, keras = F["tf"], F["keras"]
    X_tr, y_tr, X_val, y_val, X_te, y_te, n_cls = data
    steps = int(np.ceil(len(X_tr) / BATCH_SIZE))
    measure_cost = ((seed == args.seeds[0]) or args.cost_every_seed) and not args.no_cost

    def make_ds():
        tr = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
              .shuffle(1000, seed=seed, reshuffle_each_iteration=True)
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
        va = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
        return tr, va

    def make_balanced_ds():
        """Infinite stream of class-balanced batches (P classes x K images)."""
        rng = np.random.default_rng(seed + 1000)
        by_cls = [np.where(y_tr == c)[0] for c in np.unique(y_tr)]

        def gen():
            while True:
                cls = rng.choice(len(by_cls), BALANCED_P, replace=False)
                idx = np.concatenate([rng.choice(by_cls[c], BALANCED_K,
                                                 replace=len(by_cls[c]) < BALANCED_K)
                                      for c in cls])
                rng.shuffle(idx)
                yield X_tr[idx], y_tr[idx]
        sig = (tf.TensorSpec((None, *INPUT_SHAPE), tf.as_dtype(X_tr.dtype)),
               tf.TensorSpec((None,), tf.as_dtype(y_tr.dtype)))
        return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)

    def record(cfg, model, budget):
        row, y_pred = evaluate(model, X_te, y_te, F, measure_cost, args.smoke, cfg)
        row.update({"seed": seed, "config": cfg, **budget})
        np.save(os.path.join(out_dir, f"pred_{cfg}_seed{seed}.npy"), y_pred)
        if args.save_models:
            mdir = os.path.join(out_dir, "models")
            os.makedirs(mdir, exist_ok=True)
            model.save(os.path.join(mdir, f"{cfg}_seed{seed}.h5"), include_optimizer=False)
        path = os.path.join(out_dir, "runs.csv")
        if os.path.exists(path):   # keep columns aligned if new columns appear
            old = pd.read_csv(path)
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        else:
            pd.DataFrame([row]).to_csv(path, index=False)
        print(f"  [{cfg} | seed {seed}] acc={row['accuracy']:.4f} errors={row['n_errors']} "
              f"dense_sparsity={row['dense_kernel_sparsity']:.3f} "
              f"h5={row['h5_weights_MB']:.2f}MB gz={row['h5_weights_gz_MB']:.2f}MB "
              f"int8={row.get('tflite_int8_MB', np.nan):.2f}MB "
              f"int8_agree={row.get('int8_float_agreement', np.nan):.3f}")

    def fit_es(model, tr, va):
        h = model.fit(tr, validation_data=va, epochs=NUM_EPOCHS, verbose=args.verbose,
                      callbacks=F["early_stopping"]())
        return len(h.history["loss"])

    def stage2(enc, tr, va):
        clf = F["create_classifier"](enc, n_cls, trainable=False)
        return clf, fit_es(clf, tr, va)

    set_seed(seed, args.deterministic)
    tr, va = make_ds()
    t0 = time.time()
    pre_ep = 0

    # ---------- train the group's main model ----------
    if group in ("baseline", "resnet_v1", "densenet121"):
        backbone = {"baseline": "resnet50v2", "resnet_v1": "resnet50",
                    "densenet121": "densenet121"}[group]
        clf = F["create_classifier"](F["create_encoder"](backbone), n_cls, trainable=True)
        ep = fit_es(clf, tr, va)
    elif group == "le_cnn":
        clf = F["create_le_cnn"](n_cls)
        ep = fit_es(clf, tr, va)
    elif group == "frozen_imagenet":
        clf, ep = stage2(F["create_encoder"](), tr, va)
    elif group in ("scl", "ce_two_stage", "scl_balanced", "ce_two_stage_balanced"):
        enc = F["create_encoder"]()
        if group.startswith("scl"):
            pre = F["add_projection_head"](enc)
            pre.compile(loss=F["SupervisedContrastiveLoss"](TEMPERATURE),
                        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        else:  # identical stage-1 budget, cross-entropy instead of SupCon
            pre = F["add_linear_ce_head"](enc, n_cls)
        if group.endswith("_balanced"):   # same number of steps per epoch as before
            pre.fit(make_balanced_ds(), epochs=PRETRAIN_EPOCHS, steps_per_epoch=steps,
                    verbose=args.verbose)
        else:
            pre.fit(tr, epochs=PRETRAIN_EPOCHS, verbose=args.verbose)
        pre_ep = PRETRAIN_EPOCHS
        clf, ep = stage2(enc, tr, va)
    else:
        raise ValueError(group)

    main_cfg = GROUPS[group][0]
    if main_cfg in configs:
        record(main_cfg, clf, {"train_time_s": time.time() - t0, "stage1_epochs": pre_ep,
                               "stage2_epochs": ep, "grad_steps": (pre_ep + ep) * steps})

    # ---------- pruned variant (fine-tuned from the trained model) ----------
    if len(GROUPS[group]) > 1 and GROUPS[group][1] in configs:
        tr, va = make_ds()
        pruned = F["create_pruned_model"](clf, len(X_tr))
        best = F["BestAfterEpoch"](PRUNE_RAMP_EPOCHS)
        t0 = time.time()
        pruned.fit(tr, validation_data=va, epochs=PRUNING_EPOCHS, verbose=args.verbose,
                   callbacks=[F["sparsity"].UpdatePruningStep(), best])
        if best.best_weights is not None:
            pruned.set_weights(best.best_weights)
        final = F["sparsity"].strip_pruning(pruned)
        final.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        record(GROUPS[group][1], final,
               {"train_time_s": time.time() - t0, "stage1_epochs": 0,
                "stage2_epochs": PRUNING_EPOCHS, "grad_steps": PRUNING_EPOCHS * steps,
                "best_epoch": best.best_epoch})

    keras.backend.clear_session()
    gc.collect()


# ==========================================================================
# Summary + statistics
# ==========================================================================
def summarize(out_dir, extra_dirs=()):
    from scipy import stats
    frames = [pd.read_csv(os.path.join(d, "runs.csv")) for d in [out_dir, *extra_dirs]]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["config", "seed"], keep="first")
    order = [c for c in ALL_CONFIGS if c in df.config.unique()]
    seeds = sorted(int(s) for s in df.seed.unique())
    L =["# Results over seeds\n",
         f"Seeds: {seeds} (n = {len(seeds)});  test images: {df.n_test.iloc[0]}\n"]

    def fmt(v, s=100):
        v = v * s
        n = len(v)
        ci = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        return f"{v.mean():.2f} ± {v.std(ddof=1):.2f} [CI ±{ci:.2f}] ({v.min():.2f}–{v.max():.2f})"

    # ---- classification ----
    metrics = ["accuracy", "precision", "recall", "f1", "f1_macro"]
    metrics = [m for m in metrics if m in df.columns]
    L.append("\n## Classification, % (mean ± SD [95 % CI half-width] (min–max))\n")
    L.append("| Config | n | " + " | ".join(metrics) + " | errors |")
    L.append("|---" * (len(metrics) + 3) + "|")
    for c in order:
        d = df[df.config == c]
        L.append(f"| {c} | {len(d)} | " + " | ".join(fmt(d[m]) for m in metrics)
                 + f" | {fmt(d.n_errors, 1)} |")

    # ---- training budget (fairness) ----
    L.append("\n## Training budget (mean over seeds)\n")
    bcols = [c for c in ["stage1_epochs", "stage2_epochs", "grad_steps", "train_time_s",
                         "best_epoch"] if c in df.columns]
    L.append(_table(df.groupby("config")[bcols].mean().reindex(order).round(1)))

    # ---- size and cost ----
    L.append("\n## Size (MB = 10^6 bytes) and inference cost (cost from first seed)\n")
    scols = ["total_params", "dense_param_share", "dense_kernel_sparsity", "model_sparsity",
             "nonzero_params", "h5_with_opt_MB", "h5_weights_MB", "h5_weights_gz_MB",
             "tflite_f32_MB", "tflite_f32_gz_MB", "tflite_int8_MB", "tflite_int8_gz_MB",
             "int8_subset_n", "float_acc_subset", "int8_acc_subset", "int8_float_agreement",
             "gflops", "dense_head_mflops", "latency_gpu_ms", "latency_cpu_ms",
             "latency_tflite_f32_ms", "latency_tflite_int8_ms", "peak_gpu_mem_MB",
             "export_max_diff"]
    scols = [c for c in scols if c in df.columns]
    L.append(_table(df.groupby("config")[scols].mean().reindex(order).T.round(4)))

    # ---- LaTeX rows ----
    L.append("\n## LaTeX rows: accuracy / precision / recall / F1 (mean $\\pm$ SD)\n```latex")
    for c in order:
        d = df[df.config == c]
        L.append(f"{c} & " + " & ".join(
            f"{d[m].mean()*100:.2f} $\\pm$ {d[m].std(ddof=1)*100:.2f}"
            for m in ["accuracy", "precision", "recall", "f1"]) + r" \\")
    L.append("```")

    # ---- paired tests ----
    acc = df.pivot_table(index="seed", columns="config", values="accuracy")
    L.append("\n## Paired comparisons across seeds (accuracy, percentage points)\n")
    pairs = [("baseline", "scl", "SCL vs end-to-end baseline (paper's comparison)"),
             ("ce_two_stage", "scl", "SCL vs SAME procedure with CE  <- effect of SCL itself"),
             ("frozen_imagenet", "scl", "SCL vs no stage 1 (frozen ImageNet encoder)"),
             ("baseline", "ce_two_stage", "two-stage procedure vs end-to-end (procedure effect)"),
             ("pruned_baseline", "pruned_scl", "after pruning: SCL vs baseline"),
             ("pruned_ce_two_stage", "pruned_scl", "after pruning: SCL vs CE two-stage"),
             ("scl", "scl_balanced", "balanced vs random batches for SCL"),
             ("ce_two_stage_balanced", "scl_balanced", "balanced SCL vs balanced CE  <- effect of SCL"),
             ("baseline", "scl_balanced", "balanced SCL vs end-to-end baseline"),
             ("ce_two_stage", "ce_two_stage_balanced", "balanced vs random batches for CE"),
             ("pruned_baseline", "pruned_scl_balanced", "after pruning: balanced SCL vs baseline")]
    for a, b, label in pairs:
        if a in acc and b in acc:
            L.append(_paired(acc[a], acc[b], f"{b} − {a}", label, stats))

    # pruning drop: is SCL more robust than an equally frozen CE encoder?
    L.append("\n## Accuracy change caused by pruning (pruned − unpruned, pp)\n")
    drops = {}
    for base in ["baseline", "scl", "ce_two_stage", "frozen_imagenet", "scl_balanced",
                 "ce_two_stage_balanced"]:
        pc = f"pruned_{base}"
        if base in acc and pc in acc:
            drops[base] = (acc[pc] - acc[base]).dropna()
            dd = drops[base] * 100
            L.append(f"- {base}: {dd.mean():+.2f} ± {dd.std(ddof=1):.2f} pp "
                     f"(per seed: {', '.join(f'{v:+.2f}' for v in dd)})")
    for a, b in [("baseline", "scl"), ("ce_two_stage", "scl"), ("frozen_imagenet", "scl"),
                 ("baseline", "scl_balanced"), ("ce_two_stage_balanced", "scl_balanced")]:
        if a in drops and b in drops:
            L.append(_paired(drops[a], drops[b], f"drop({b}) − drop({a})",
                             "pruning robustness difference", stats))

    # ---- McNemar per seed ----
    from scipy.stats import binomtest
    yt_path = os.path.join(out_dir, "y_test.npy")
    if os.path.exists(yt_path):
        y_true = np.load(yt_path)
        for a, b in [("baseline", "scl"), ("ce_two_stage", "scl"),
                     ("baseline", "scl_balanced"), ("ce_two_stage_balanced", "scl_balanced")]:
            L.append(f"\n## Exact McNemar test per seed: {a} vs {b}\n")
            for s in seeds:
                def _find(cfg):
                    for d in [out_dir, *extra_dirs]:
                        f = os.path.join(d, f"pred_{cfg}_seed{s}.npy")
                        if os.path.exists(f):
                            return f
                    return ""
                pa, pb = _find(a), _find(b)
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    continue
                a_ok, b_ok = np.load(pa) == y_true, np.load(pb) == y_true
                n01, n10 = int((~a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())
                p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                L.append(f"- seed {s}: only {b} correct = {n01}, only {a} correct = {n10}, "
                         f"p = {p:.3f}")

    text = "\n".join(L)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def _table(df):
    try:
        return df.to_markdown()
    except ImportError:
        return df.to_string()


def _paired(a, b, name, label, stats):
    m = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    diff = (m.b - m.a) * 100
    s = f"- {name} ({label}): mean {diff.mean():+.2f} pp"
    if len(m) > 1:
        s += f", SD {diff.std(ddof=1):.2f}"
        if (diff != 0).any():
            s += f", paired t p = {stats.ttest_rel(m.b, m.a).pvalue:.3f}"
            try:
                s += f", Wilcoxon p = {stats.wilcoxon(m.b, m.a).pvalue:.3f}"
            except ValueError:
                pass
    s += f"; better in {(diff > 0).sum()}/{len(diff)} seeds, ties {(diff == 0).sum()}"
    return s


# ==========================================================================
def main():
    global NUM_EPOCHS, PRETRAIN_EPOCHS, PRUNING_EPOCHS, PRUNE_RAMP_EPOCHS, PATIENCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, choices=list(GROUPS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--deterministic", action="store_true",
                    help="tf op determinism (slower, may fail for some GPU ops)")
    ap.add_argument("--cost-every-seed", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--merge", nargs="*", default=[],
                    help="extra result folders to include in the summary (e.g. results_cloud)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick end-to-end check: 1 seed, 1-2 epochs, all groups")
    ap.add_argument("--verbose", type=int, default=2)
    ap.add_argument("--data-root", default=None,
                    help="folder containing train/, test/ and the two label CSVs "
                         "(overrides the Windows paths at the top, e.g. on RunPod)")
    ap.add_argument("--save-models", action="store_true",
                    help="keep each trained model as <out>/models/<config>_seed<N>.h5 "
                         "(weights only, ~100 MB each)")
    ap.add_argument("--no-cost", action="store_true",
                    help="skip latency/FLOPs/memory/int8 checks (use on cloud GPUs so "
                         "all reported speed numbers come from one machine)")
    args = ap.parse_args()

    if args.smoke:
        NUM_EPOCHS = PRETRAIN_EPOCHS = 1
        PRUNING_EPOCHS, PRUNE_RAMP_EPOCHS, PATIENCE = 2, 1, 1
        args.seeds = [0]
        args.groups = list(GROUPS)
        if args.out == "results":
            args.out = "results_smoke"
    os.makedirs(args.out, exist_ok=True)

    if args.summarize:
        summarize(args.out, args.merge)
        return

    from sklearn.model_selection import train_test_split
    print("Loading data ...")
    tr_dir, tr_csv, te_dir, te_csv = TRAIN_DIR, TRAIN_CSV, TEST_DIR, TEST_CSV
    if args.data_root:
        r = args.data_root
        tr_dir, te_dir = os.path.join(r, "train"), os.path.join(r, "test")
        tr_csv = os.path.join(r, os.path.basename(TRAIN_CSV.replace("\\", "/")))
        te_csv = os.path.join(r, os.path.basename(TEST_CSV.replace("\\", "/")))
    print(f"  data: {tr_csv}")
    X_full, y_full = read_csv_data(tr_csv, tr_dir)
    X_te, y_te = read_csv_data(te_csv, te_dir)
    print(f"  train file: {len(X_full)} images, test file: {len(X_te)} images "
          f"(expected 2200 / 518)")
    if (len(X_full), len(X_te)) != (2200, 518):
        print("  WARNING: image counts differ from the dataset description!")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=VAL_FRACTION, stratify=y_full, random_state=SPLIT_SEED)
    print(f"  split -> train {len(X_tr)}, val {len(X_val)}, test {len(X_te)}")
    np.save(os.path.join(args.out, "y_test.npy"), y_te)
    global REP_DATA
    REP_DATA = X_tr[np.random.RandomState(0).choice(len(X_tr), 100, replace=False)]
    data = (X_tr, y_tr, X_val, y_val, X_te, y_te, len(np.unique(y_full)))

    runs = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(runs):
        prev = pd.read_csv(runs)
        done = set(zip(prev.config, prev.seed))

    for s in args.seeds:
        for g in args.groups:
            print(f"\n===== seed {s} | {g} =====")
            try:
                run_group(g, s, data, args.out, done, args)
            except Exception:  # keep going; the failed group is retried next run
                traceback.print_exc()
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()

    summarize(args.out)


if __name__ == "__main__":
    main()
