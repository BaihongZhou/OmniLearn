import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import utils
from tensorflow.keras.losses import mse, mae
from tensorflow.keras.models import Model
from PET import PET, FourierProjection, get_encoding
from layers import StochasticDepth, LayerScale
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import PolynomialDecay


class ProcessDiscriminator(keras.Model):
    def __init__(self, input_dim, num_processes=3, hidden_units=64):
        super().__init__()
        self.net = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(num_processes, activation="softmax")  # Multiclass classification
        ])

    def call(self, x):
        return self.net(x)


class PET_jetnet(keras.Model):
    """Score based generative model"""

    def __init__(
            self,
            num_feat,
            num_jet,
            num_classes=2,
            num_part=150,
            num_diffusion=6,
            feature_drop=0.1,
            projection_dim=128,
            local=True,
            K=5,
            num_local=2,
            num_layers=8,
            num_class_layers=2,
            num_gen_layers=3,
            num_heads=8,
            drop_probability=0.0,
            simple=False,
            layer_scale=True,
            layer_scale_init=1e-5,
            talking_head=False,
            mode='generator',
            fine_tune=False,
            model_name=None,
            use_mean=False,
            dropout=0.0,
            lambda_adv=1.0,
            num_adv_classes=3,
            num_steps=100,

            rho=3.0,
            P_mean=-1.2,
            P_std=1.2,
            sigma_data=0.5,
            S_churn=0.0,
            S_min=0.0,
            S_noise=1.0,
    ):
        super(PET_jetnet, self).__init__()

        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.max_part = num_part
        self.projection_dim = projection_dim
        self.layer_scale_init = layer_scale_init
        self.num_steps = num_steps
        self.num_diffusion = num_diffusion
        self.ema = 0.999
        self.shape = (-1, 1, 1)

        self.sigma_max = 80
        self.sigma_min = 0.002
        self.rho = rho  # better balance between low and high noise
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.S_churn = S_churn,
        self.S_min = S_min,
        self.S_max = float('inf'),
        self.S_noise = S_noise,

        self.model_part = PET(
            num_feat=num_feat,
            num_jet=num_jet,
            num_classes=num_classes,
            num_keep=11,
            local=local,
            K=K,
            num_layers=num_layers,
            drop_probability=drop_probability,
            simple=simple,
            layer_scale=layer_scale,
            layer_scale_init=layer_scale_init,
            talking_head=talking_head,
            mode=mode,
            feature_drop=feature_drop,
            num_local=num_local,
            num_heads=num_heads,
            num_class_layers=num_class_layers,
            num_gen_layers=num_gen_layers,
            num_diffusion=self.num_diffusion,
            dropout=dropout,
            class_activation=None
        )

        if fine_tune:
            assert model_name is not None, "ERROR: Model name is necessary if fine tune is on"
            self.model_part.load_weights(model_name, by_name=True, skip_mismatch=True)
            # self.model_part.ema_body.trainable=False

        self.body = self.model_part.ema_body
        self.head = self.model_part.ema_generator_head

        # Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_classes))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None, 1))
        inputs_features = Input(shape=(None, num_feat))
        inputs_points = Input(shape=(None, 2))

        x = inputs_mask * (inputs_features - 0) / 1

        output_body = self.body([x, inputs_points, inputs_mask, inputs_time])
        outputs = self.head([output_body, inputs_jet, inputs_mask, inputs_time, inputs_cond])

        self.model_part = keras.Model(
            inputs=[
                inputs_features, inputs_points, inputs_mask,
                inputs_jet, inputs_time, inputs_cond
            ],
            outputs=outputs
        )

        self.ema_body = keras.models.clone_model(self.body)
        self.ema_head = keras.models.clone_model(self.head)

        # self.ema_part = keras.models.clone_model(self.model_part)
        self.loss_tracker = keras.metrics.Mean(name="loss")

        # Add this to __init__:
        self.sigma_tracker = tf.keras.metrics.Mean(name="sigma_mean")
        self.collected_sigmas = []

        self.logger = logging.getLogger(__name__)

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        # return [self.loss_tracker, self.adv_loss_tracker]
        return [self.loss_tracker, self.sigma_tracker]

    def compile(self, body_optimizer, head_optimizer):
        super(PET_jetnet, self).compile(experimental_run_tf_function=False,
                                        weighted_metrics=[],
                                        # run_eagerly=True
                                        )
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer

    def prior_sde(self, dimensions):
        return tf.random.normal(dimensions, dtype=tf.float32)

    def _update_sigma_range(self, sigma_min_val, sigma_max_val):
        sigma_min_val = float(sigma_min_val.numpy())
        sigma_max_val = float(sigma_max_val.numpy())
        self.sigma_min = min(self.sigma_min, sigma_min_val)
        self.sigma_max = max(self.sigma_max, sigma_max_val)

    def edm_preconditioned_network(self, model_part, x_noisy, sigma, features, points, mask, cond):

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / tf.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1.0 / tf.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = tf.math.log(sigma + 1e-5) / 4.0

        model_input = c_in * x_noisy
        model_time = c_noise

        v_pred = model_part([
            features,
            points,
            mask,
            model_input, model_time, cond
        ])
        return c_skip * x_noisy + c_out * v_pred

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        weight = x['input_weight']

        P_mean = self.P_mean
        P_std = self.P_std
        sigma_data = self.sigma_data

        with tf.GradientTape(persistent=True) as tape:
            rnd_normal = tf.random.normal((batch_size, 1), dtype=tf.float32)
            sigma = tf.exp(rnd_normal * P_std + P_mean)
            sigma = tf.cast(sigma, tf.float32)
            loss_weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

            eps = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
            x_clean = x['input_jet']
            x_noisy = x_clean + sigma * eps

            pred = self.edm_preconditioned_network(
                self.model_part,
                x_noisy,
                sigma,
                x['input_features'],
                x['input_points'],
                x['input_mask'],
                y
            )
            target = x_clean

            loss = tf.reduce_mean(loss_weight * tf.square(pred - target))
            if weight is not None:
                loss = tf.reduce_sum(weight * loss) / tf.reduce_sum(weight)

            # Safely record scalar stats
            sigma_max_batch = tf.reduce_max(sigma)
            sigma_min_batch = tf.reduce_min(sigma)
            tf.py_function(self._update_sigma_range, [sigma_min_batch, sigma_max_batch], [])

            total_loss = loss

        self.body_optimizer.minimize(total_loss, self.body.trainable_variables, tape=tape)
        self.optimizer.minimize(total_loss, self.head.trainable_variables, tape=tape)

        for weight, ema_weight in zip(self.head.weights, self.ema_head.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        for weight, ema_weight in zip(self.body.weights, self.ema_body.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        self.loss_tracker.update_state(loss)
        self.sigma_tracker.update_state(sigma)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        weight = x['input_weight']

        P_mean = self.P_mean
        P_std = self.P_std
        sigma_data = self.sigma_data

        rnd_normal = tf.random.normal((batch_size, 1), dtype=tf.float32)
        sigma = tf.exp(rnd_normal * P_std + P_mean)
        sigma = tf.cast(sigma, tf.float32)
        loss_weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

        eps = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
        x_clean = x['input_jet']
        x_noisy = x_clean + sigma * eps

        pred = self.edm_preconditioned_network(
            self.model_part,
            x_noisy,
            sigma,
            x['input_features'],
            x['input_points'],
            x['input_mask'],
            y
        )
        target = x_clean

        loss = tf.reduce_mean(loss_weight * tf.square(pred - target))
        if weight is not None:
            loss = tf.reduce_sum(weight * loss) / tf.reduce_sum(weight)

        self.loss_tracker.update_state(loss)
        self.sigma_tracker.update_state(sigma)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        return self.model(x)

    def generate(
            self, nsplit,
            cond,
            particles,
            points,
            mask,
            use_tqdm=False,
            candidate=10
    ):
        jet_info = []
        jet_total = []

        part_split = np.array_split(particles, nsplit)
        mask_split = np.array_split(mask, nsplit)
        point_split = np.array_split(points, nsplit)
        cond_split = np.array_split(cond, nsplit)

        self.logger.info(f"Max sigma: {self.sigma_max}, Min sigma: {self.sigma_min}")

        # iterable = tqdm(splits,desc='Processing Splits',total=len(splits)) if use_tqdm else splits
        for i in tqdm(range(nsplit), desc='Processing Splits') if use_tqdm else range(nsplit):

            part = part_split[i]
            mask = mask_split[i]
            point = point_split[i]
            cond = cond_split[i]

            jet_candidate = []
            for _ in range(candidate):
                jet = self.edm_sampler(
                    part=part,
                    point=point,
                    mask=mask,
                    cond=cond,
                    model_part=self.model_part,
                    data_shape=(part.shape[0], self.num_jet),
                    num_steps=self.num_steps,
                    sigma_min=self.sigma_min,
                    sigma_max=self.sigma_max,
                    rho=self.rho,
                    S_churn=self.S_churn,
                    S_min=self.S_min,
                    S_max=float('inf'),
                    S_noise=self.S_noise,
                ).numpy()

                jet_candidate.append(jet)

            total_jets = np.concatenate(jet_candidate, 1)
            total_jets = np.array(total_jets).reshape(-1, candidate, jet.shape[1])
            jet_total.append(total_jets)
        return np.concatenate(jet_total)

    def logsnr_schedule_cosine(self, t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t, tf.float32) + b))

    def inv_logsnr_schedule_cosine(self, logsnr, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr, tf.float32))) / a - b / a

    def get_logsnr_alpha_sigma(self, sigma, shape=None):
        logsnr = -tf.math.log(tf.square(sigma))
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))

        if shape is not None:
            alpha = tf.reshape(alpha, shape)
            sigma = tf.reshape(sigma, shape)
            logsnr = tf.reshape(logsnr, shape)

        return logsnr, tf.cast(alpha, tf.float32), tf.cast(sigma, tf.float32)

    @tf.function
    def DDPMSampler(self,
                    part, point, mask, cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    w=0.1,
                    num_steps=100):

        """Generate samples from score-based models with DDPM method.
        
        Args:
        cond: Conditional input
        model: Trained score model to use
        data_shape: Format of the data
        const_shape: Format for constants, should match the data_shape in dimensions
        jet: input jet conditional information if used
        mask: particle mask if used

        Returns: 
        Samples.
        """

        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps, shape=const_shape)
            s = self.inv_logsnr_schedule_cosine(0.5 * (logsnr + logsnr_))
            logsnr_s, alpha_s, sigma_s = self.get_logsnr_alpha_sigma(s, shape=const_shape)

            model_body, model_head = model

            v = model_body([part, point, mask, t], training=False)
            v = model_head([v, x, mask, t, cond], training=False)

            eps = v * alpha + x * sigma
            u = alpha_s / alpha * x - sigma_s * tf.math.expm1(0.25 * (logsnr_ - logsnr)) * eps

            v = model_body([part, point, mask, s], training=False)
            v = model_head([v, u, mask, s, cond], training=False)

            eps = v * alpha_s + u * sigma_s
            mean = alpha_s * u - sigma_s * v

            x = alpha_ * mean + sigma_ * eps
        return mean

    @tf.function
    def DDIMSampler(self,
                    part, point, mask, cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    w=0.1,
                    num_steps=100,
                    eta=1.0):

        """
        Generate samples from score-based models with DDIM method.

        Args:
        cond: Conditional input
        model: Trained score model to use
        data_shape: Format of the data
        const_shape: Format for constants, should match the data_shape in dimensions
        part, point, mask: Additional input components
        w: Weight parameter for condition (optional)
        num_steps: Number of sampling steps
        eta: Noise scaling factor (0 for deterministic sampling, >0 for stochastic sampling)

        Returns:
        Samples.
        """

        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,
                shape=const_shape
            )

            # Compute the predicted epsilon using the model
            model_body, model_head = model
            v = model_body([part, point, mask, t], training=False)
            v = model_head([v, x, mask, t, cond], training=False)
            eps = v * alpha + x * sigma

            # Update x using DDIM deterministic update rule
            pred_x0 = (x - sigma * eps) / alpha  # Predicted x_0
            x = alpha_ * pred_x0 + sigma_ * (eta * eps)  # Add noise if eta > 0

        return x  # Return the final sample

    @tf.function
    def edm_sampler(
            self,
            part, point, mask, cond,
            model_part,
            data_shape=None,
            num_steps=18,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            S_churn=0.0,
            S_min=0.0,
            S_max=float('inf'),
            S_noise=1.0
    ):
        def sigma_schedule(n):
            i = tf.cast(tf.range(n), tf.float64)
            ramp = i / tf.cast(n - 1, tf.float64)
            inv_rho = 1.0 / rho
            sigmas = (sigma_max ** inv_rho + ramp * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
            return tf.concat([sigmas, tf.zeros_like(sigmas[:1])], axis=0)

        sigmas = sigma_schedule(num_steps)
        sigmas = tf.cast(sigmas, tf.float32)

        batch_size = tf.shape(cond)[0]
        x_next = tf.random.normal(data_shape, dtype=tf.float32) * sigmas[0]

        for i in tf.range(num_steps):
            t_cur = sigmas[i]
            t_next = sigmas[i + 1]

            gamma = tf.where(
                (t_cur >= S_min) & (t_cur <= S_max),
                tf.minimum(S_churn / num_steps, tf.sqrt(2.0) - 1.0),
                0.0
            )

            t_hat_scalar = t_cur + gamma * t_cur
            t_hat = tf.ones((batch_size, 1), dtype=tf.float32) * tf.reshape(t_hat_scalar, [])

            x_cur = x_next
            x_hat = x_cur + tf.sqrt(t_hat_scalar ** 2 - t_cur ** 2) * S_noise * tf.random.normal(data_shape,
                                                                                                 dtype=tf.float32)

            denoised = self.edm_preconditioned_network(model_part, x_hat, t_hat, part, point, mask, cond)
            d_cur = (x_hat - denoised) / t_hat_scalar
            x_next = x_hat + (t_next - t_hat_scalar) * d_cur

            if i < num_steps - 1:
                t_next_batch = tf.ones((batch_size, 1), dtype=tf.float32) * tf.reshape(t_next, [])
                denoised_next = self.edm_preconditioned_network(model_part, x_next, t_next_batch, part, point, mask,
                                                                cond)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_hat + (t_next - t_hat_scalar) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
