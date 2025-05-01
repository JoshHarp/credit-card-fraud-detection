import sklearn
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from keras import ops
from keras import layers
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

#Load Dataset
df = pd.read_csv("data/kaggle_credit_card_transactions_preprocessed(in).csv", encoding='latin-1')

#Data preprocessing
#Convert latitude and longitude to numeric for scaling
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df.dropna(subset=['latitude', 'longitude'], inplace=True)



df = df.drop(columns=["Merchant State"],axis=1)

#Convert features to numeric so model can interpret
df['Zip'] = df['Zip'].fillna(0)
df['Hour'] = df['Time'].apply(lambda value: int(value.split(":")[0]))
df['Minutes'] = df['Time'].apply(lambda value: int(value.split(":")[1]))

df.drop(['Time'], axis=1, inplace=True)

df['Merchant Name'] = df['Merchant Name'].astype("object")
df['Card'] = df['Card'].astype("object")
df['Use Chip'] = df['Use Chip'].astype("object")
df['Zip'] = df['Zip'].astype("object")
df["Merchant City"]=LabelEncoder().fit_transform(df["Merchant City"])
df["Use Chip"]=LabelEncoder().fit_transform(df["Use Chip"])


df= df.drop(columns=['Unnamed: 0'], errors='ignore')

for col in df.columns:
    col_type = df[col].dtype
    if col_type == 'object':
        df[col] = df[col].fillna("")

#Feature selection
not_fraud = df[df['Fraud'] == 0].drop(columns=['Fraud'])
fraud = df[df['Fraud'] == 1].drop(columns=['Fraud'])
full_data = df

print(df.info())


y_benign_all = np.full(len(not_fraud),0)
y_malicious_all = np.full(len(fraud),1)



#Train data only on normal data
X_train, X_test, y_train, y_test = train_test_split(not_fraud, y_benign_all, test_size=0.33, random_state=42)

X_test_both = np.vstack((X_test, fraud))
y_test_both = np.concatenate((y_test, y_malicious_all))


#Scale the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_both_scaled = scaler.transform(X_test_both)



# -------------------------------
# Model 1: Local Outlier Factor
# -------------------------------

n_neig = [3, 5, 10, 25, 50]

print("True Labels:")
print(y_test_both)
print()

for k in n_neig:
    print(f"--- Local Outlier Factor with n_neighbors = {k} ---")
    clf = LocalOutlierFactor(n_neighbors=k, novelty=True)
    clf.fit(X_train_scaled)
    preds = clf.predict(X_test_both_scaled)

    preds_new = np.where(preds, preds == 1, 0)
    preds_new = np.where(preds_new, preds_new == -1, 1)

    pred_scores = clf.decision_function(X_test_both_scaled)
    pred_scores = [x * -1 for x in pred_scores]

    print(f"Precision: {precision_score(y_test_both, preds_new)}")
    print(f"Recall: {recall_score(y_test_both, preds_new)}")
    print(f"F1 Score: {f1_score(y_test_both, preds_new)}")
    print(f"ROC AUC (labels): {roc_auc_score(y_test_both, preds_new)}")
    print(f"ROC AUC (scores): {roc_auc_score(y_test_both, pred_scores)}")
    print()



# -------------------------------
# Model 2: Isolation Forest
# -------------------------------


n_est = [10, 25, 50, 100, 250, 500, 1000, 2000]

for ne in n_est:
    print(f"--- Isolation Forest with n_estimators = {ne} ---")
    clf = IsolationForest(random_state=0, n_estimators=ne).fit(X_train_scaled)
    preds = clf.predict(X_test_both_scaled)

    preds_new = np.where(preds, preds == 1, 0)
    preds_new = np.where(preds_new, preds_new == -1, 1)

    pred_scores = clf.decision_function(X_test_both_scaled)
    pred_scores = [x * -1 for x in pred_scores]

    print(f"Precision: {precision_score(y_test_both, preds_new)}")
    print(f"Recall: {recall_score(y_test_both, preds_new)}")
    print(f"F1 Score: {f1_score(y_test_both, preds_new)}")
    print(f"ROC AUC (labels): {roc_auc_score(y_test_both, preds_new)}")
    print(f"ROC AUC (scores): {roc_auc_score(y_test_both, pred_scores)}")
    print()





# -------------------------------
# Model 3: One-Class SVM
# -------------------------------


kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for ker in kernels:
    print(f"--- One-Class SVM with kernel = {ker} ---")
    clf = OneClassSVM(gamma='auto', kernel=ker).fit(X_train_scaled)
    preds = clf.predict(X_test_both_scaled)

    preds_new = np.where(preds, preds == 1, 0)
    preds_new = np.where(preds_new, preds_new == -1, 1)

    pred_scores = clf.decision_function(X_test_both_scaled)
    pred_scores = [x * -1 for x in pred_scores]

    print(f"Precision: {precision_score(y_test_both, preds_new)}")
    print(f"Recall: {recall_score(y_test_both, preds_new)}")
    print(f"F1 Score: {f1_score(y_test_both, preds_new)}")
    print(f"ROC AUC (labels): {roc_auc_score(y_test_both, preds_new)}")
    print(f"ROC AUC (scores): {roc_auc_score(y_test_both, pred_scores)}")
    print()



# -------------------------------
# Model 4: Stacked Autoencoder
# -------------------------------
# Define encoder


stacked_encoder = keras.models.Sequential([
    keras.layers.Input(shape=(23,)),

    keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(32, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(3, activation="relu"),
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Input(shape=(3,)),

    keras.layers.Dense(32, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(23, activation="linear")
])

stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

stacked_ae.compile(loss="mean_squared_error",
                   optimizer=keras.optimizers.Nadam(learning_rate=1e-4), metrics=["mse"])



history = stacked_ae.fit(X_train_scaled, X_train_scaled, epochs=100, batch_size=32,
                         validation_data=(X_train_scaled, X_train_scaled))

decoded = stacked_ae.predict(X_train_scaled)

mse = np.power(X_train_scaled - decoded, 2)

mse_r = []

print(np.shape(mse))
print(type(mse))

for i in range(0, np.shape(mse)[0]):
    mse_i = np.mean(abs(mse[i]))
    mse_r.append(mse_i)                         # one element per instance

mse_avg = np.mean(mse_r)

#Percentile-based Thresholding
anomaly_t1 = np.percentile(mse_r, 90)  # 90th percentile
anomaly_t2 = np.percentile(mse_r, 95)  # 95th percentile
anomaly_t3 = np.percentile(mse_r,99 )  # 99th percentile

print(anomaly_t1)
print(anomaly_t2)
print(anomaly_t3)

decoded_test = stacked_ae.predict(X_test_both_scaled)
decoded_test

mse = np.power(X_test_both_scaled - decoded_test, 2)

mse_r = []

for i in range(0, np.shape(mse)[0]):
    mse_i = np.mean(abs(mse[i]))
    mse_r.append(mse_i)                         # one element per instance

pred_classes = []

for i in range(0, np.shape(mse_r)[0]):
    if mse_r[i] >= anomaly_t1:
        pred_classes.append(1)        # anomaly
    else:
        pred_classes.append(0)        # normal

#90th Percentile
print("--- Results at Threshold t1 (90th percentile) ---")
print(f"Precision: {precision_score(y_test_both, pred_classes)}")
print(f"Recall: {recall_score(y_test_both, pred_classes)}")
print(f"F1 Score: {f1_score(y_test_both, pred_classes)}")
print()

pred_classes = []

for i in range(0, np.shape(mse_r)[0]):
    if mse_r[i] >= anomaly_t2:
        pred_classes.append(1)        # anomaly
    else:
        pred_classes.append(0)        # normal

#95th Percentile
print("--- Results at Threshold t1 (95th percentile) ---")
print(f"Precision: {precision_score(y_test_both, pred_classes)}")
print(f"Recall: {recall_score(y_test_both, pred_classes)}")
print(f"F1 Score: {f1_score(y_test_both, pred_classes)}")
print()


pred_classes = []

for i in range(0, np.shape(mse_r)[0]):
    if mse_r[i] >= anomaly_t3:
        pred_classes.append(1)        # anomaly
    else:
        pred_classes.append(0)        # normal

#99 Percentile
print("--- Results at Threshold t1 (99th percentile) ---")
print(f"Precision: {precision_score(y_test_both, pred_classes)}")
print(f"Recall: {recall_score(y_test_both, pred_classes)}")
print(f"F1 Score: {f1_score(y_test_both, pred_classes)}")
print()


print(f"ROC AUC Score: {roc_auc_score(y_test_both, mse_r)}")
print()




# -------------------------------
# Model 4: #Variational Auto-Encoder
# -------------------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class VEncoder(keras.Model):
    def __init__(self, input_dim: int, latent_dim: int, **kwargs):
        super().__init__(**kwargs)

        encoder_inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(512, activation="relu",kernel_initializer="he_normal")(encoder_inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)




        x = layers.Dense(256, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)



        x = layers.Dense(128, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)



        x = layers.Dense(64, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)


        x = layers.Dense(32, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def call(self, x):
        return self.encoder(x)

class VDecoder(keras.Model):
        def __init__(self, latent_dim: int, output_dim: int, **kwargs):
            super().__init__(**kwargs)

            latent_inputs = keras.Input(shape=(latent_dim,))


            x = layers.Dense(32, activation="relu")(latent_inputs)
            x = layers.BatchNormalization()(x)

            x = layers.Dense(64,activation="relu")(x)
            x = layers.BatchNormalization()(x)



            x = layers.Dense(128,activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)



            x = layers.Dense(256,activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)


            decoder_outputs = layers.Dense(output_dim, activation="linear")(x)

            self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        def call(self, x):
            return self.decoder(x)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    # also used during validation
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def calculate_reconstruction_loss(self, loss, data, reconstruction):
        """
        In case of computer vision tasks use the following:
            keras.losses.binary_crossentropy(data, reconstruction),
            axis=(1, 2),
        """

        return ops.mean(ops.sum(loss(data, reconstruction)))


    def calculate_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        return kl_loss

    def calculate_total_loss(self, reconstruction_loss, kl_loss):
        return reconstruction_loss + kl_loss * 3  # * (reconstruction_loss // kl_loss)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = self.calculate_reconstruction_loss(loss, data, reconstruction)
            kl_loss = self.calculate_kl_loss(z_mean, z_log_var)
            total_loss = self.calculate_total_loss(reconstruction_loss, kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z)

        reconstruction_loss = self.calculate_reconstruction_loss(loss, data, reconstruction)
        kl_loss = self.calculate_kl_loss(z_mean, z_log_var)
        total_loss = self.calculate_total_loss(reconstruction_loss, kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


loss = keras.losses.MeanSquaredError()


encoder = VEncoder(23, 32)
decoder = VDecoder(32, 23)

vae = VAE(encoder, decoder)



optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

vae.compile(optimizer=optimizer)



# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6
)


history = vae.fit(
    X_train_scaled,
    epochs=300,
    batch_size=128,
    validation_data=(X_train_scaled),
    validation_split=0.2,
    callbacks=[early_stopping,reduce_lr]
)


_, _, decoded = vae.predict(X_train_scaled)
print(np.shape(decoded))
print(decoded[0:3])

mse = np.power(X_train_scaled - decoded, 2)

mse_r = []

for i in range(np.shape(mse)[0]):
    mse_i = np.mean(abs(mse[i]))
    mse_r.append(mse_i)  # one element per instance

mse_avg = np.mean(mse_r)

# Use percentiles for thresholds
anomaly_t1 = np.percentile(mse_r, 25)
anomaly_t2 = np.percentile(mse_r, 50)
anomaly_t3 = np.percentile(mse_r, 95)

print(f"Adjusted Thresholds - t1: {anomaly_t1}, t2: {anomaly_t2}, t3: {anomaly_t3}")
print()

# Predict on test data
_, _, decoded_test = vae.predict(X_test_both_scaled)

mse = np.power(X_test_both_scaled - decoded_test, 2)

mse_r = []

for i in range(np.shape(mse)[0]):
    mse_i = np.mean(abs(mse[i]))
    mse_r.append(mse_i)

# Threshold 1 (25th percentile)
pred_classes = []

for i in range(np.shape(mse_r)[0]):
    if mse_r[i] >= anomaly_t1:
        pred_classes.append(1)  # anomaly
    else:
        pred_classes.append(0)  # normal

print("--- Results at Threshold t1 (25th percentile) ---")
print(f"Precision: {precision_score(y_test_both, pred_classes)}")
print(f"Recall: {recall_score(y_test_both, pred_classes)}")
print(f"F1 Score: {f1_score(y_test_both, pred_classes)}")
print()

# Threshold 2 (50th percentile)
pred_classes = []

for i in range(np.shape(mse_r)[0]):
    if mse_r[i] >= anomaly_t2:
        pred_classes.append(1)  # anomaly
    else:
        pred_classes.append(0)  # normal

print("--- Results at Threshold t2 (50th percentile) ---")
print(f"Precision: {precision_score(y_test_both, pred_classes)}")
print(f"Recall: {recall_score(y_test_both, pred_classes)}")
print(f"F1 Score: {f1_score(y_test_both, pred_classes)}")
print()

# Threshold 3 (95th percentile)
pred_classes = []

for i in range(np.shape(mse_r)[0]):
    if mse_r[i] >= anomaly_t3:
        pred_classes.append(1)  # anomaly
    else:
        pred_classes.append(0)  # normal

print("--- Results at Threshold t3 (95th percentile) ---")
print(f"Precision: {precision_score(y_test_both, pred_classes)}")
print(f"Recall: {recall_score(y_test_both, pred_classes)}")
print(f"F1 Score: {f1_score(y_test_both, pred_classes)}")
print()

# ROC AUC Score for reconstruction error scores
print("--- ROC AUC for Reconstruction Errors ---")
print(f"ROC AUC Score: {roc_auc_score(y_test_both, mse_r)}")
print()