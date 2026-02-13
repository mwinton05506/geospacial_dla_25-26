import os
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from .EpochLogger import EpochLogger

class DenseCL(tf.keras.Model):

    def __init__(
        self, 
        backbone, 
        momentum=0.999, 
        temperature=0.2, 
        lambda_weight=0.5,
        queue_size=4096,
        warmup_epochs=10,
    ):
        super(DenseCL, self).__init__()
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        
        # The Backbone
        self.backbone = backbone
        self.m = momentum
        self.tau = temperature
        self.l_weight = lambda_weight
        
        # Queue parameters
        self.proj_dim = 128
        self.queue_size = queue_size
        
        # Warm-up period for the global loss (L_q) before introducing the dense loss (L_r)
        self.warmup_epochs = warmup_epochs
        
        # Heads
        self.global_head = self._build_head("global_head", is_dense=False)
        self.dense_head = self._build_head("dense_head", is_dense=True)

        # Momentum Teacher (cloned)
        self.momentum_backbone = models.clone_model(backbone)
        self.momentum_global_head = self._build_head("m_global_head", is_dense=False)
        self.momentum_dense_head = self._build_head("m_dense_head", is_dense=True)
        
        # Note being trained via backprop; updated via momentum update rule
        self.momentum_backbone.trainable = False
        self.momentum_global_head.trainable = False
        self.momentum_dense_head.trainable = False

        # Build heads so weight copying below is effective.
        feat_shape = self.backbone.output_shape[1:]
        dummy_feat = tf.zeros((1, *feat_shape))
        _ = self.global_head(dummy_feat)
        _ = self.dense_head(dummy_feat)
        _ = self.momentum_global_head(dummy_feat)
        _ = self.momentum_dense_head(dummy_feat)

        # Init teacher weights
        self.momentum_backbone.set_weights(backbone.get_weights())
        self.momentum_global_head.set_weights(self.global_head.get_weights())
        self.momentum_dense_head.set_weights(self.dense_head.get_weights())
        
        # Queue for negative samples (for global contrastive loss)
        init_queue = tf.math.l2_normalize(
            tf.random.normal(
                shape=(self.queue_size, self.proj_dim), dtype=tf.float32
            ),
            axis=-1,
        )
        self.global_queue = tf.Variable(
            init_queue,
            trainable=False,
            name="global_queue",
            dtype=tf.float32,
        )
        self.global_queue_ptr = tf.Variable(
            0, trainable=False, dtype=tf.int32, name="global_queue_ptr"
        )

    def _build_head(self, name, is_dense=False):
        # Makes the global head (for L_q) or dense head (for L_r) based on the is_dense flag.
        if not is_dense:
            return Sequential(
                [
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(2048, activation="relu"),
                    # Drop out is used here to reduce overfitting on the small dataset
                    layers.Dropout(0.2),
                    layers.Dense(128),
                ],
                name=name,
            )
        else:
            return Sequential(
                [
                    layers.Conv2D(2048, (1, 1), activation="relu"),
                    # Drop out is used here to reduce overfitting on the small dataset
                    layers.Dropout(0.2),
                    layers.Conv2D(128, (1, 1)),
                ],
                name=name,
            )

    def get_dense_correspondence(self, q, k):
        b, h, w, c = tf.shape(q)[0], tf.shape(q)[1], tf.shape(q)[2], tf.shape(q)[3]
        f1 = tf.reshape(q, (b, h * w, c))
        f2 = tf.reshape(k, (b, h * w, c))
        # Same as cosine sim since q and k are already normalized
        sim = tf.linalg.matmul(f1, f2, transpose_b=True)
        idx = tf.math.argmax(sim, axis=-1)
        return tf.gather(f2, idx, batch_dims=1)

    def info_nce(self, q, k):
        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        # Same as cosine sim since q and k are already normalized
        logits = tf.matmul(q, k, transpose_b=True) / tf.cast(self.tau, tf.float32)
        labels = tf.range(tf.shape(q)[0])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.cast(tf.reduce_mean(loss), tf.float32)

    def global_info_nce_with_queue(self, q, k):
        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        
        queue = tf.cast(self.global_queue, tf.float32)

        l_pos = tf.reduce_sum(q * k, axis=-1, keepdims=True)
        l_neg = tf.matmul(q, queue, transpose_b=True)

        logits = tf.concat([l_pos, l_neg], axis=1) / self.tau

        labels = tf.zeros(tf.shape(q)[0], dtype=tf.int32)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        return loss

    def _dequeue_and_enqueue(self, keys):
        keys = tf.cast(keys, tf.float32)
        
        batch_size = tf.shape(keys)[0]
        queue_size = self.queue_size

        ptr = self.global_queue_ptr

        idx = tf.math.mod(ptr + tf.range(batch_size), queue_size)

        updated_queue = tf.tensor_scatter_nd_update(
            self.global_queue,
            tf.expand_dims(idx, axis=1),
            keys
        )
        self.global_queue.assign(updated_queue)

        new_ptr = tf.math.mod(ptr + batch_size, queue_size)
        self.global_queue_ptr.assign(new_ptr)

    def call(self, x, training=False):
        feat = self.backbone(x, training=training)
        g = tf.math.l2_normalize(self.global_head(feat), axis=-1)
        d = tf.math.l2_normalize(self.dense_head(feat), axis=-1)
        return g, d

    @tf.function
    def train_step(self, data):
        # Data from our pipeline is (view_q, view_k)
        x_q, x_k = data
        
        epoch = tf.cast(self.current_epoch, tf.int32)

        with tf.GradientTape() as tape:
            # Student forward pass (on query view)
            feat_q = self.backbone(x_q, training=True)
            q_g = tf.math.l2_normalize(self.global_head(feat_q), axis=-1)
            q_d = tf.math.l2_normalize(self.dense_head(feat_q), axis=-1)

            # Teacher forward pass (on key view) - no gradients
            feat_k = self.momentum_backbone(x_k, training=False)
            k_g = tf.math.l2_normalize(self.momentum_global_head(feat_k), axis=-1)
            k_d = tf.math.l2_normalize(self.momentum_dense_head(feat_k), axis=-1)
            k_g = tf.stop_gradient(k_g)
            k_d = tf.stop_gradient(k_d)

            # Global Contrastive Loss (L_q)
            if epoch < self.warmup_epochs:
                # During warm-up, only use the current batch's momentum features as negatives
                l_g = self.info_nce(q_g, k_g)
            else:
                # After warm-up, use the MoCo-style queue for negatives
                l_g = self.global_info_nce_with_queue(q_g, k_g)

            # Dense Contrastive Loss (L_r)
            matched_k = self.get_dense_correspondence(q_d, k_d)

            # Treat every pixel as a sample for the dense loss
            q_d_flat = tf.reshape(q_d, (-1, self.proj_dim))
            k_d_flat = tf.reshape(matched_k, (-1, self.proj_dim))
            l_d = self.info_nce(q_d_flat, k_d_flat)

            # Combined Total Loss
            if epoch < self.warmup_epochs:
                # Warm-up: focus on global loss for first 10 epochs
                total_loss = l_g
            else:
                total_loss = (1 - self.l_weight) * l_g + self.l_weight * l_d

        train_vars = (
            self.backbone.trainable_variables
            + self.global_head.trainable_variables
            + self.dense_head.trainable_variables
        )
        grads = tape.gradient(total_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        
        self._update_teacher()
        self._dequeue_and_enqueue(k_g)

        return {"loss": total_loss, "global": l_g, "dense": l_d}

    @tf.function
    def _update_teacher(self):
        s_vars = (
            self.backbone.variables
            + self.global_head.variables
            + self.dense_head.variables
        )
        t_vars = (
            self.momentum_backbone.variables
            + self.momentum_global_head.variables
            + self.momentum_dense_head.variables
        )

        for v_s, v_t in zip(s_vars, t_vars):
            v_t.assign(v_t * self.m + v_s * (1 - self.m))