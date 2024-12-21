import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import utils
from tensorflow.keras.losses import mse, mae
from tensorflow.keras.models import Model
from PET import PET, FourierProjection, get_encoding
from layers import StochasticDepth,LayerScale
from tqdm import tqdm

class PET_jetnet(keras.Model):
    """Score based generative model"""
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 num_part = 150,
                 num_diffusion = 3,
                 feature_drop = 0.0,
                 projection_dim = 128,
                 local = True, K = 5,
                 num_local = 2, 
                 num_layers = 8, num_class_layers=2,
                 num_heads = 4,drop_probability = 0.0,
                 simple = False, layer_scale = True,
                 layer_scale_init = 1e-5,        
                 talking_head = False,
                 mode = 'generator',                 
                 fine_tune = False,
                 model_name = None,
                 use_mean=False):
        super(PET_jetnet, self).__init__()


        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.max_part = num_part
        self.projection_dim = projection_dim
        self.layer_scale_init = layer_scale_init
        self.num_steps = 200
        self.num_diffusion = num_diffusion
        self.ema=0.999
        self.shape = (-1,1,1)
        # The paramete to control the weight of the observation loss
        self.lambda_obs = 1.0

        self.model_part  = PET(num_feat=num_feat,
                               num_jet=num_jet,
                               num_classes=num_classes,
                               local = local,
                               num_layers = num_layers, 
                               drop_probability = drop_probability,
                               simple = simple, layer_scale = layer_scale,
                               talking_head = talking_head,
                               mode = mode,                               
                               )


        if fine_tune:
            assert model_name is not None, "ERROR: Model name is necessary if fine tune is on"
            self.model_part.load_weights(model_name,by_name=True,skip_mismatch=True)
            #self.model_part.ema_body.trainable=False            

            
        if use_mean:
            self.mean, self.std = self.get_mean()
        else:
            self.mean = 0.0
            self.std = 1.0            

        self.body = self.model_part.ema_body        
        self.head = self.model_part.ema_generator_head
                
                
        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_classes))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None,1))
        inputs_features = Input(shape=(None, num_feat))
        inputs_points = Input(shape=(None, 2))


        x = inputs_mask*(inputs_features-self.mean)/self.std
        
        output_body = self.body([x,inputs_points,inputs_mask,inputs_time])
        outputs = self.head([output_body,inputs_jet,inputs_mask,inputs_time,inputs_cond])
        
        self.model_part = keras.Model(inputs=[inputs_features,inputs_points,inputs_mask,
                                              inputs_jet,inputs_time,inputs_cond],
                                      outputs=outputs)
        
              
        self.ema_body = keras.models.clone_model(self.body)
        self.ema_head = keras.models.clone_model(self.head)

        #self.ema_part = keras.models.clone_model(self.model_part)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.obs_loss_tracker = keras.metrics.Mean(name="obs_loss")

        self.multistep_coefficients = [
            tf.constant([1], shape=(1, 1, 1, 1), dtype=tf.float32),
            tf.constant([-1, 3], shape=(2, 1, 1, 1), dtype=tf.float32) / 2,
            tf.constant([5, -16, 23], shape=(3, 1, 1,1), dtype=tf.float32) / 12,
            tf.constant([-9, 37, -59, 55], shape=(4, 1,1, 1), dtype=tf.float32)
            / 24,
            tf.constant(
                [251, -1274, 2616, -2774, 1901], shape=(5, 1,1, 1), dtype=tf.float32
            )
            / 720,
        ]
        

    def get_mean(self):
        #Mean and std from JetClass pretrained model to be used during fine-tuning
        mean_pet = tf.constant([0.0, 0.0,-0.0278,
                                0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0],
                               shape=(1, 1, self.num_feat), dtype=tf.float32)
        std_pet = tf.constant([0.215,0.215,0.070,
                               1.0,1.0,1.0,1.0,1.0,
                               1.0,1.0,1.0,1.0,1.0],
                              shape=(1, 1, self.num_feat), dtype=tf.float32)

        if self.max_part == 150:
            mean_sample = tf.constant([0.0, 0.0, -0.0217,
                                       0.0,0.0,0.0,0.0,0.0,
                                       0.0,0.0,0.0,0.0,0.0],
                                      shape=(1, 1, self.num_feat), dtype=tf.float32)
            std_sample =  tf.constant([0.115, 0.115, -0.054,
                                       1.0,1.0,1.0,1.0,1.0,
                                       1.0,1.0,1.0,1.0,1.0],
                                      shape=(1, 1, self.num_feat), dtype=tf.float32)
        elif self.max_part == 30:
            mean_sample = tf.constant([0.0, 0.0, -0.035,
                                       0.0,0.0,0.0,0.0,0.0,
                                       0.0,0.0,0.0,0.0,0.0],
                                      shape=(1, 1, self.num_feat), dtype=tf.float32)
            std_sample = tf.constant([0.09, 0.09,  0.067, 
                                      1.0,1.0,1.0,1.0,1.0,
                                      1.0,1.0,1.0,1.0,1.0],
                                     shape=(1, 1, self.num_feat), dtype=tf.float32)

        return (mean_sample-mean_pet)/std_pet, std_sample/std_pet

        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker, self.obs_loss_tracker]


    def compile(self,body_optimizer,head_optimizer):
        super(PET_jetnet, self).compile(experimental_run_tf_function=False,
                                        weighted_metrics=[],
                                        #run_eagerly=True
        )
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer

    
    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    

    def train_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]

        with tf.GradientTape(persistent=True) as tape:            
            t = tf.random.uniform((batch_size,1))                
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)
            
            eps = tf.random.normal((batch_size,self.num_jet),dtype=tf.float32)
            perturbed_x = alpha*x['input_jet'] + eps * sigma

            v_pred, obs = self.model_part([x['input_features'],
                                      x['input_points'],
                                      x['input_mask'],
                                    perturbed_x,t,y])
            
            
            v_jet = alpha * eps - sigma * x['input_jet']
            
            obs_loss = self.lambda_obs * tf.reduce_mean(tf.square(obs - x['input_obs']))
            
            loss = tf.reduce_mean(tf.square(v_pred-v_jet)) + obs_loss


        self.body_optimizer.minimize(loss,self.body.trainable_variables,tape=tape)                   
        self.optimizer.minimize(loss,self.head.trainable_variables,tape=tape)

        
        self.loss_tracker.update_state(loss)
        self.obs_loss_tracker.update_state(obs_loss)
                        
        for weight, ema_weight in zip(self.head.weights, self.ema_head.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        for weight, ema_weight in zip(self.body.weights, self.ema_body.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        return {m.name: m.result() for m in self.metrics}


    def test_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]


        t = tf.random.uniform((batch_size,1))                
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)
            
        eps = tf.random.normal((batch_size,self.num_jet),dtype=tf.float32)
        perturbed_x = alpha*x['input_jet'] + eps * sigma
        
        v_pred, obs = self.model_part([x['input_features'],
                                  x['input_points'],
                                  x['input_mask'],
                                  perturbed_x,t,y])
            
            
        v_jet = alpha * eps - sigma * x['input_jet']
        obs_loss = self.lambda_obs * tf.reduce_mean(tf.square(obs - x['input_obs']))
        loss = tf.reduce_mean(tf.square(v_pred-v_jet)) + obs_loss
            
        self.loss_tracker.update_state(loss)
        self.obs_loss_tracker.update_state(obs_loss)
        return {m.name: m.result() for m in self.metrics}
            
    def call(self,x):        
        return self.model(x)

    def generate(self,nsplit,
                 cond,
                 particles,
                 points,
                 mask,
                 use_tqdm=False):
        jet_info = []
        jet_total = []


        part_split = np.array_split(particles,nsplit)
        mask_split = np.array_split(mask,nsplit)
        point_split = np.array_split(points,nsplit)
        cond_split = np.array_split(cond, nsplit)
        
        
        #iterable = tqdm(splits,desc='Processing Splits',total=len(splits)) if use_tqdm else splits
        for i in tqdm(range(nsplit), desc='Processing Splits') if use_tqdm else range(nsplit):

            part = part_split[i]
            mask = mask_split[i]
            point = point_split[i]
            cond = cond_split[i]
            
            jet_candidate = []
            for _ in range(10):
                # jet = self.DDPMSampler(part,point,mask,cond,
                #                        [self.ema_body,self.ema_head],
                #                        data_shape=[part.shape[0],self.num_jet],
                #                        w = 0.0,
                #                        num_steps = self.num_steps,
                #                        const_shape = [-1,1]).numpy()
                jet = self.DDIMSampler(part,point,mask,cond,
                                        [self.ema_body,self.ema_head],
                                        data_shape=[part.shape[0],self.num_jet],
                                        w = 0.0,
                                        num_steps = self.num_steps,
                                        const_shape = [-1,1]).numpy()
                jet_candidate.append(jet)
            
            total_jets = np.concatenate(jet_candidate,1)
            total_jets = np.array(total_jets).reshape(-1, 10, jet.shape[1])
            jet_total.append(total_jets)
        return np.concatenate(jet_total)


    def select_nu(self,nu,part):
        mean_lep = [9.53190298e+00, 0.0,  0.0, 1.12493561e+02, 0.0]
        std_lep = [9.40143725,  1.0,  1.0, 233.26712552,  1.0]
        mean_nu = [-1.0836015, 1.6860425, 2.312119, -0.8071776, 1.2596784, 1.8461846]
        std_nu = [11.045118, 11.041079, 269.44342, 13.143577, 13.144216, 356.6083]

        def revert_prep(x,mean,std):
            return std*x + mean

        def get_pxyz(arr):
            pT = arr[:,0]
            eta = arr[:,1]
            phi = arr[:,2]
            E = arr[:,3]
            px = pT*np.cos(phi)
            py = pT*np.sin(phi)
            pz = pT*np.sinh(eta)
            return np.stack([px,py,pz],-1)
        
        def get_ptetaphi(arr):
            px = arr[:,0]
            py = arr[:,1]
            pz = arr[:,2]
            pT = np.sqrt(px**2 + py**2)
            eta = 0.5*np.log((np.sqrt(px**2 + py**2 + pz**2) + pz)/(np.sqrt(px**2 + py**2 + pz**2) - pz))
            phi = np.arctan2(py,px)
            return np.stack([pT,eta,phi],-1)

        # Use mininumum Delta mass tau selectioon;
        lepton_0_0 = get_pxyz(revert_prep(part[:,0],mean_lep,std_lep))[:,None]
        lepton_0_1 = get_pxyz(revert_prep(part[:,1],mean_lep,std_lep))[:,None]
        lepton_1_0 = get_pxyz(revert_prep(part[:,2],mean_lep,std_lep))[:,None]
        lepton_1_1 = get_pxyz(revert_prep(part[:,3],mean_lep,std_lep))[:,None]
        lepton_0 = lepton_0_0 + lepton_0_1
        lepton_1 = lepton_1_0 + lepton_1_1
        new_nu = revert_prep(nu,mean_nu,std_nu)
        new_nu_0 = new_nu[:,:,:3]
        new_nu_1 = new_nu[:,:,3:]
        e_lep_0 = np.sqrt(np.sum(lepton_0**2,-1,keepdims=True))
        e_lep_1 = np.sqrt(np.sum(lepton_1**2,-1,keepdims=True))
        e_nu_0 = np.sqrt(np.sum(new_nu_0**2,-1,keepdims=True))
        e_nu_1 = np.sqrt(np.sum(new_nu_1**2,-1,keepdims=True))
        tau_0 = lepton_0 + new_nu_0
        tau_1 = lepton_1 + new_nu_1
        m_tau_0 = (e_lep_0 + e_nu_0)**2 - np.sum(tau_0**2,-1,keepdims=True)
        m_tau_1 = (e_lep_1 + e_nu_1)**2 - np.sum(tau_1**2,-1,keepdims=True)
        idx = np.argmin(np.abs(m_tau_0 - 1.777) + np.abs(m_tau_1 - 1.777), 1, keepdims=True)
    
        return np.take_along_axis(nu, idx, axis=1)


    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))

    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr,tf.float32)))/a -b/a


    def get_logsnr_alpha_sigma(self,time,shape=None):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))

        if shape is not None:
            alpha = tf.reshape(alpha, shape)
            sigma = tf.reshape(sigma, shape)
            logsnr = tf.reshape(logsnr,shape)
            
        return logsnr, tf.cast(alpha,tf.float32), tf.cast(sigma,tf.float32)


           
    @tf.function
    def DDPMSampler(self,
                    part,point,mask,cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    w = 0.1,
                    num_steps = 100):
        
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
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t,shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,shape=const_shape)
            s = self.inv_logsnr_schedule_cosine(0.5*(logsnr + logsnr_))
            logsnr_s, alpha_s, sigma_s = self.get_logsnr_alpha_sigma(s,shape=const_shape)
             
            model_body, model_head = model

            v = model_body([part,point,mask,t], training=False)
            v,_ = model_head([v,x,mask,t,cond],training=False)
            
            eps = v * alpha + x * sigma
            u = alpha_s/alpha* x - sigma_s*tf.math.expm1(0.25*(logsnr_ - logsnr))*eps

            v = model_body([part,point,mask,s], training=False)
            v,_ = model_head([v,u,mask,s,cond],training=False)

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
            v,_ = model_head([v, x, mask, t, cond], training=False)
            eps = v * alpha + x * sigma

            # Update x using DDIM deterministic update rule
            pred_x0 = (x - sigma * eps) / alpha  # Predicted x_0
            x = alpha_ * pred_x0 + sigma_ * (eta * eps)  # Add noise if eta > 0

        return x  # Return the final sample
