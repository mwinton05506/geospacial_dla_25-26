import tensorflow as tf
from tensorflow.keras import layers, models, Sequential

class DenseCL(tf.keras.Model):

    def __init__(self, backbone, momentum=0.999, temperature=0.2, lambda_weight=0.5):
        super(DenseCL, self).__init__()

        # The Backbone (e.g., ResNet-50)
        # We need the output of the last conv layer for DenseCL
        self.backbone = backbone
        self.m = momentum
        self.tau = temperature
        self.l_weight = lambda_weight

        # Heads
        self.global_head = self._build_head("global_head", is_dense=False)
        self.dense_head = self._build_head("dense_head", is_dense=True)

        # Momentum Teacher (cloned)
        self.momentum_backbone = models.clone_model(backbone)
        self.momentum_global_head = self._build_head("m_global_head", is_dense=False)
        self.momentum_dense_head = self._build_head("m_dense_head", is_dense=True)

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

    def _build_head(self, name, is_dense=False):
        if not is_dense:
            return Sequential(
                [
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(2048, activation="relu"),
                    layers.Dense(128),
                ],
                name=name,
            )
        else:
            return Sequential(
                [
                    layers.Conv2D(2048, (1, 1), activation="relu"),
                    layers.Conv2D(128, (1, 1)),
                ],
                name=name,
            )

    def l2_norm(self, x):
        return tf.math.l2_normalize(x, axis=-1)

    def get_dense_correspondence(self, q, k):
        b, h, w, c = tf.shape(q)[0], tf.shape(q)[1], tf.shape(q)[2], tf.shape(q)[3]
        f1 = tf.reshape(q, (b, h * w, c))
        f2 = tf.reshape(k, (b, h * w, c))
        sim = tf.linalg.matmul(f1, f2, transpose_b=True)
        idx = tf.math.argmax(sim, axis=-1)
        return tf.gather(f2, idx, batch_dims=1)

    def info_nce(self, q, k):
        # Distributed-friendly labels
        logits = tf.matmul(q, k, transpose_b=True) / self.tau
        labels = tf.range(tf.shape(q)[0])
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        )

    def call(self, x, training=False):
        feat = self.backbone(x, training=training)
        g = self.l2_norm(self.global_head(feat))
        d = self.l2_norm(self.dense_head(feat))
        return g, d

    # @tf.function
    def train_step(self, data):
        # Data from our pipeline is (view_q, view_k)
        x_q, x_k = data

        with tf.GradientTape() as tape:
            # --- STUDENT FORWARD ---
            feat_q = self.backbone(x_q, training=True)
            q_g = self.l2_norm(self.global_head(feat_q))
            q_d = self.l2_norm(self.dense_head(feat_q))

            # --- TEACHER FORWARD ---
            feat_k = self.momentum_backbone(x_k, training=False)
            k_g = tf.stop_gradient(self.l2_norm(self.momentum_global_head(feat_k)))
            k_d = tf.stop_gradient(self.l2_norm(self.momentum_dense_head(feat_k)))

            # --- LOSS CALCULATION ---
            # 1. Global Contrastive Loss (L_q)
            l_g = self.info_nce(q_g, k_g)

            # 2. Dense Contrastive Loss (L_r)
            matched_k = self.get_dense_correspondence(q_d, k_d)

            # Treat every pixel as a sample for the dense loss
            q_d_flat = tf.reshape(q_d, (-1, 128))
            k_d_flat = tf.reshape(matched_k, (-1, 128))
            l_d = self.info_nce(q_d_flat, k_d_flat)

            # 3. Combined Total Loss (Eq 3)
            total_loss = (1 - self.l_weight) * l_g + self.l_weight * l_d

        # --- OPTIMIZATION ---
        train_vars = (
            self.backbone.trainable_variables
            + self.global_head.trainable_variables
            + self.dense_head.trainable_variables
        )
        grads = tape.gradient(total_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        # --- MOMENTUM UPDATE (Optimized for performance) ---
        self._update_teacher()

        return {"loss": total_loss, "global": l_g, "dense": l_d}

    # @tf.function
    def _update_teacher(self):
        """Relates to Section 3.1: Momentum-updated encoder."""
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
