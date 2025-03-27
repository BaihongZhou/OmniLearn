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

        self.sigma_max = 1.0  # noise matches data scale
        self.sigma_min = 0.01  # small but not vanishing
        self.rho = 3.0  # better balance between low and high noise

        # self.adv_model = ProcessDiscriminator(input_dim=self.num_jet, num_processes=num_adv_classes)
        # self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")
        # self.num_adv_classes = num_adv_classes
        # self.adv_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Or Lion if you like
        # self.lambda_adv_schedule = PolynomialDecay(
        #     initial_learning_rate=0.0,
        #     decay_steps=50000,  # total steps or estimated steps
        #     end_learning_rate=lambda_adv,
        #     power=1.0  # linear ramp-up
        # )

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

    def train_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        weight = x['input_weight']

        raw_file = x['input_file']
        # raw_file_onehot = tf.one_hot(tf.cast(raw_file, tf.int32), depth=self.num_adv_classes)

        with tf.GradientTape(persistent=True) as tape:
            # Diffusion training
            # t = tf.random.uniform((batch_size, 1))
            # logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
            rho = self.rho

            u = tf.random.uniform((batch_size, 1))
            sigma = self.sigma_max * (self.sigma_min / self.sigma_max) ** (u ** (1 / self.rho))
            t = tf.math.log(sigma + 1e-5)

            eps = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
            perturbed_x = x['input_jet'] + sigma * eps
            v_jet = -sigma * eps

            v_pred = self.model_part([
                x['input_features'],
                x['input_points'],
                x['input_mask'],
                perturbed_x, t, y
            ])
            # v_jet = alpha * eps - sigma * x['input_jet']

            # Base diffusion loss
            loss = tf.reduce_mean(tf.square(v_pred - v_jet))
            if weight is not None:
                loss = tf.reduce_sum(weight * loss) / tf.reduce_sum(weight)

            # Adversarial training
            # process_logits = self.adv_model(tf.stop_gradient(v_pred))
            # adv_loss = tf.keras.losses.categorical_crossentropy(raw_file_onehot, process_logits)
            # adv_loss = tf.reduce_mean(adv_loss)
            #
            # current_step = tf.cast(self.optimizer.iterations, tf.float32)
            # lambda_adv = self.lambda_adv_schedule(current_step)

            total_loss = loss  # - lambda_adv * adv_loss

        # Update generator (PET)
        self.body_optimizer.minimize(total_loss, self.body.trainable_variables, tape=tape)
        self.optimizer.minimize(total_loss, self.head.trainable_variables, tape=tape)

        # Update adversary
        # with tf.GradientTape() as adv_tape:
        #     process_logits = self.adv_model(v_pred)
        #     adv_loss = tf.keras.losses.categorical_crossentropy(raw_file_onehot, process_logits)
        #     adv_loss = tf.reduce_mean(adv_loss)
        #
        # adv_grads = adv_tape.gradient(adv_loss, self.adv_model.trainable_variables)
        # self.adv_optimizer.apply_gradients(zip(adv_grads, self.adv_model.trainable_variables))

        # Update logs
        self.loss_tracker.update_state(loss)
        # self.adv_loss_tracker.update_state(adv_loss)
        self.sigma_tracker.update_state(tf.reduce_mean(sigma))

        # EMA update
        for weight, ema_weight in zip(self.head.weights, self.ema_head.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        for weight, ema_weight in zip(self.body.weights, self.ema_body.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        x, y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        weight = x['input_weight']

        raw_file = x['input_file']
        # raw_file_onehot = tf.one_hot(tf.cast(raw_file, tf.int32), depth=self.num_adv_classes)

        # t = tf.random.uniform((batch_size, 1))
        # logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        rho = self.rho

        u = tf.random.uniform((batch_size, 1))
        sigma = self.sigma_max * (self.sigma_min / self.sigma_max) ** (u ** (1 / self.rho))
        t = tf.math.log(sigma + 1e-5)

        eps = tf.random.normal((batch_size, self.num_jet), dtype=tf.float32)
        perturbed_x = x['input_jet'] + sigma * eps
        v_jet = -sigma * eps

        v_pred = self.model_part([
            x['input_features'],
            x['input_points'],
            x['input_mask'],
            perturbed_x, t, y
        ])
        # v_jet = alpha * eps - sigma * x['input_jet']

        # Reconstruction loss
        loss = tf.reduce_mean(tf.square(v_pred - v_jet))
        if weight is not None:
            loss = tf.reduce_sum(weight * loss) / tf.reduce_sum(weight)

        self.loss_tracker.update_state(loss)
        self.sigma_tracker.update_state(tf.reduce_mean(sigma))

        # Optional: track adversarial loss during test
        # if raw_file is not None:
        #     adv_pred = self.adv_model(v_pred)
        #     adv_loss = tf.keras.losses.categorical_crossentropy(raw_file_onehot, adv_pred)
        #     adv_loss = tf.reduce_mean(adv_loss)
        #     self.adv_loss_tracker.update_state(adv_loss)

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

        # iterable = tqdm(splits,desc='Processing Splits',total=len(splits)) if use_tqdm else splits
        for i in tqdm(range(nsplit), desc='Processing Splits') if use_tqdm else range(nsplit):

            part = part_split[i]
            mask = mask_split[i]
            point = point_split[i]
            cond = cond_split[i]

            jet_candidate = []
            for _ in range(candidate):
                # jet = self.DDPMSampler(part,point,mask,cond,
                #                        [self.ema_body,self.ema_head],
                #                        data_shape=[part.shape[0],self.num_jet],
                #                        w = 0.0,
                #                        num_steps = self.num_steps,
                #                        const_shape = [-1,1]).numpy()
                # jet = self.DDIMSampler(
                #     part, point, mask, cond,
                #     [self.ema_body, self.ema_head],
                #     data_shape=[part.shape[0], self.num_jet],
                #     w=0.0,
                #     num_steps=self.num_steps,
                #     const_shape=[-1, 1]
                # ).numpy()
                jet = self.EDMSampler(
                    part, point, mask, cond,
                    [self.ema_body, self.ema_head],
                    data_shape=[part.shape[0], self.num_jet],
                    num_steps=self.num_steps,
                    const_shape=[-1, 1],
                    sigma_max=self.sigma_max,
                    sigma_min=self.sigma_min,
                    rho=self.rho
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

    # def get_logsnr_alpha_sigma(self, time, shape=None):
    #     logsnr = self.logsnr_schedule_cosine(time)
    #     alpha = tf.sqrt(tf.math.sigmoid(logsnr))
    #     sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
    #
    #     if shape is not None:
    #         alpha = tf.reshape(alpha, shape)
    #         sigma = tf.reshape(sigma, shape)
    #         logsnr = tf.reshape(logsnr, shape)
    #
    #     return logsnr, tf.cast(alpha, tf.float32), tf.cast(sigma, tf.float32)

    def get_logsnr_alpha_sigma(self, sigma, shape=None):
        logsnr = -tf.math.log(tf.square(sigma))
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))

        if shape is not None:
            alpha = tf.reshape(alpha, shape)
            sigma = tf.reshape(sigma, shape)
            logsnr = tf.reshape(logsnr, shape)

        return logsnr, tf.cast(alpha, tf.float32), tf.cast(sigma, tf.float32)

    def logsnr_from_sigma(self, sigma):
        return -tf.math.log(tf.square(sigma))  # log(SNR) = -log(σ²)

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
    def EDMSampler(
            self,
            part, point, mask, cond,
            model,
            data_shape=None,
            const_shape=None,
            sigma_max=80.0,
            sigma_min=0.002,
            rho=7.0,
            num_steps=18
    ):
        def sigma_schedule(n):
            i = tf.range(n, dtype=tf.float32)
            ramp = i / (n - 1)
            inv_rho = 1.0 / rho
            return tf.convert_to_tensor(
                (sigma_max ** inv_rho + ramp * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho,
                dtype=tf.float32
            )

        batch_size = cond.shape[0]
        x = tf.random.normal(data_shape, dtype=tf.float32) * sigma_max
        sigmas = sigma_schedule(num_steps)

        model_body, model_head = model

        for i in tf.range(num_steps):
            sigma = tf.reshape(sigmas[i], const_shape)
            sigma_next = tf.reshape(sigmas[i + 1] if i + 1 < num_steps else 0.0, const_shape)

            log_sigma = tf.math.log(sigma + 1e-5)
            t = tf.ones([batch_size, 1], dtype=tf.float32) * tf.reshape(log_sigma, [])
            v = model_body([part, point, mask, t], training=False)
            d = model_head([v, x, mask, t, cond], training=False)

            dt = sigma_next - sigma
            # 🧪 Euler-only step (skip Heun)
            x = x + d * dt
            x_pred = x

            if i + 1 < num_steps:
                log_sigma_next = tf.math.log(sigma_next + 1e-5)
                t_next = tf.ones([batch_size, 1], dtype=tf.float32) * tf.reshape(log_sigma_next, [])
                v_next = model_body([part, point, mask, t_next], training=False)
                d_next = model_head([v_next, x_pred, mask, t_next, cond], training=False)
                x = x + 0.5 * (d + d_next) * dt
            else:
                x = x_pred

        return x
