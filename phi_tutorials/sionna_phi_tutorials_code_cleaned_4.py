# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)

# This function removes nulled subcarriers from any tensor having the shape of a resource grid
remove_nulled_scs = RemoveNulledSubcarriers(rg)

# The LS channel estimator will provide channel estimates and error variances
ls_est = LSChannelEstimator(rg, interpolation_type="nn")

# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = LMMSEEqualizer(rg, sm)

# The demapper produces LLR for all coded bits
demapper = Demapper("app", "qam", num_bits_per_symbol)

# The decoder provides hard-decisions on the information bits
decoder = LDPC5GDecoder(encoder, hard_out=True)

# OFDM CHannel
ofdm_channel = OFDMChannel(channel_model, rg, add_awgn=True, normalize_channel=False, return_channel=True)
channel_freq = ApplyOFDMChannel(add_awgn=True)
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)


# ## Uplink Transmissions in the Frequency Domain
# 
# We now simulate a batch of uplink transmissions. We keep references to the estimated and actual channel frequency responses.

# In[8]:


ebno_db = 10
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, num_ut, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)

a, tau = channel_model(num_time_samples=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

y = channel_freq(x_rg, h_freq, no)
h_hat, err_var = ls_est (y, no)
x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
print("BER: {}".format(compute_ber(b, b_hat).numpy()))


# ### Compare Estimated and Actual Frequency Responses
# We can now compare the estimated frequency responses and ground truth:

# In[9]:


# In the example above, we assumed perfect CSI, i.e.,
# h_hat correpsond to the exact ideal channel frequency response.
h_perf = remove_nulled_scs(h_freq)[0,0,0,0,0,0]

# We now compute the LS channel estimate from the pilots.
h_est = h_hat[0,0,0,0,0,0]

plt.figure()
plt.plot(np.real(h_perf))
plt.plot(np.imag(h_perf))
plt.plot(np.real(h_est), "--")
plt.plot(np.imag(h_est), "--")
plt.xlabel("Subcarrier index")
plt.ylabel("Channel frequency response")
plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
plt.title("Comparison of channel frequency responses");


# ### Understand the Difference Between the Channel Models
# Before we proceed with more advanced simulations, it is important to understand the differences
# between the UMi, UMa, and RMa models. In the following code snippet, we compute the empirical cummulative
# distribution function (CDF) of the condition number of the channel frequency response matrix
# between all receiver and transmit antennas.

# In[10]:


def cond_hist(scenario):
    """Generates a histogram of the channel condition numbers"""
    
    # Setup a CIR generator
    if scenario == "umi":
        channel_model = UMi(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "uma":
        channel_model = UMa(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "rma":
        channel_model = RMa(carrier_frequency=carrier_frequency,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    
    topology = gen_topology(1024, num_ut, scenario)

    # Set the topology
    channel_model.set_topology(*topology)
    
    # Generate random CIR realizations
    # As we nned only a single sample in time, the sampling_frequency
    # does not matter.
    cir = channel_model(1, 1)
    
    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    
    h = tf.squeeze(h)
    h = tf.transpose(h, [0,3,1,2])
    
    # Compute condition number
    c = np.reshape(np.linalg.cond(h), [-1])
    
    # Compute normalized histogram
    hist, bins = np.histogram(c, 100, (1, 100))
    hist = hist/np.sum(hist)
    return bins[:-1], hist

plt.figure()
for cdl_model in ["umi", "uma", "rma"]:
    bins, hist = cond_hist(cdl_model)
    plt.plot(bins, np.cumsum(hist))
plt.xlim([0,40])
plt.legend(["UMi", "UMa", "RMa"]);
plt.xlabel("Channel Condition Number")
plt.ylabel("CDF")
plt.title("CDF of the channel condition number");


# From the figure above, you can observe that the UMi and UMa models
# are better conditioned than the RMa model. This makes 
# them more suitable for MIMO transmissions as we will observe in the next
# section.
# 
# It is also interesting to look at the channel frequency responses of these different models, as done in the next cell:

# In[11]:


def freq_response(scenario):
    """Generates an example frequency response"""
    
    # Setup a CIR generator
    if scenario == "umi":
        channel_model = UMi(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "uma":
        channel_model = UMa(carrier_frequency=carrier_frequency,
                                      o2i_model="low",
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    elif scenario == "rma":
        channel_model = RMa(carrier_frequency=carrier_frequency,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
    
    topology = gen_topology(1, num_ut, scenario)

    # Set the topology
    channel_model.set_topology(*topology)
    
    # Generate random CIR realizations
    # As we nned only a single sample in time, the sampling_frequency
    # does not matter.
    cir = channel_model(1, 1)
    
    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h = tf.squeeze(h)

    return h[0,0]

plt.figure()
for cdl_model in ["umi", "uma", "rma"]:
    h = freq_response(cdl_model)
    plt.plot(np.real(h))
plt.legend(["UMi", "UMa", "RMa"]);
plt.xlabel("Subcarrier Index")
plt.ylabel(r"$\Re(h)$")
plt.title("Channel frequency response");


# The RMa model has significantly less frequency selectivity than the other models which makes channel estimation easier.

# ### Setup a Sionna Block for BER simulations

# In[12]:


class Model(Block):
    """Simulate OFDM MIMO transmissions over a 3GPP 38.901 model"""
    def __init__(self, scenario, perfect_csi):
        super().__init__()
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        
        # Internally set parameters
        self._carrier_frequency = 3.5e9
        self._fft_size = 128
        self._subcarrier_spacing = 30e3
        self._num_ofdm_symbols = 14
        self._cyclic_prefix_length = 20
        self._pilot_ofdm_symbol_indices = [2, 11]      
        self._num_bs_ant = 8
        self._num_ut = 4
        self._num_ut_ant = 1
        self._num_bits_per_symbol = 2 
        self._coderate = 0.5
    
        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change. 
        bs_ut_association = np.zeros([1, self._num_ut])
        bs_ut_association[0, :] = 1
        self._rx_tx_association = bs_ut_association
        self._num_tx = self._num_ut
        self._num_streams_per_tx = self._num_ut_ant

        # Setup an OFDM Resource Grid
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        # Setup StreamManagement
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=1,
                                 polarization="single",
                                 polarization_type="V",
                                 antenna_pattern="omni",
                                 carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        # Configure the channel model
        if self._scenario == "umi":
            self._channel_model = UMi(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "uma":
            self._channel_model = UMa(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "rma":
            self._channel_model = RMa(carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)

        # Instantiate other building blocks
        self._binary_source = BinarySource()
        self._qam_source = QAMSource(self._num_bits_per_symbol)

        self._n = int(self._rg.num_data_symbols*self._num_bits_per_symbol) # Number of coded bits
        self._k = int(self._n*self._coderate)                              # Number of information bits
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._decoder = LDPC5GDecoder(self._encoder)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
                                         normalize_channel=True, return_channel=True)

        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(self._rg)
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)

    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_ut,
                                self._scenario,
                                min_ut_velocity=0.0,
                                max_ut_velocity=0.0)

        self._channel_model.set_topology(*topology)

    @tf.function # Run in graph mode. See the following guide: https://www.tensorflow.org/guide/function
    def call(self, batch_size, ebno_db):        
        self.new_topology(batch_size) 
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, self._num_tx, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)   
        y, h = self._ofdm_channel(x_rg, no)
        if self._perfect_csi:
            h_hat = self._remove_nulled_subcarriers(h)
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est (y, no)
        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)
        return b, b_hat 


# If you do not want to run the simulations (which can take quite some time) yourself, you can skip the next cell and simply look at the results in the next cell.

# In[13]:


SIMS = {
    "ebno_db" : list(np.arange(-5, 17, 2.0)),
    "scenario" : ["umi", "uma", "rma"],
    "perfect_csi" : [True, False],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for scenario in SIMS["scenario"]:
    for perfect_csi in SIMS["perfect_csi"]:

        model = Model(scenario=scenario,
                      perfect_csi=perfect_csi)

        ber, bler = sim_ber(model,
                            SIMS["ebno_db"],
                            batch_size=128,
                            max_mc_iter=1000,
                            num_target_block_errors=1000,
                            target_bler=1e-3)

        SIMS["ber"].append(list(ber.numpy()))
        SIMS["bler"].append(list(bler.numpy()))

SIMS["duration"] = time.time() -  start


# In[14]:


plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")

i=0
legend = []
for scenario in SIMS["scenario"]:
    for perfect_csi in SIMS["perfect_csi"]:
        if scenario=="umi":
            r = "r"
            t = "UMi"
        elif scenario=="uma":
            r = "b"
            t = "UMa"
        else:
            r = "g"
            t = "RMa"
        if perfect_csi:
            r += "-"
        else:
            r += "--"

        plt.semilogy(SIMS["ebno_db"], SIMS["bler"][i], r);
        s = "{} - {} CSI".format(t,"perf." if perfect_csi else "imperf.")

        legend.append(s)
        i += 1
plt.legend(legend)
plt.ylim([1e-4, 1])
plt.title("Multiuser 4x8 MIMO Uplink over Different 3GPP 38.901 Models");


-e 
# --- End of Realistic_Multiuser_MIMO_Simulations.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Basic MIMO Simulations
# In this notebook, you will learn how to setup simulations of MIMO transmissions over
# a flat-fading channel.
# 
# Here is a schematic diagram of the system model with all required components:


# You will learn how to:
# 
# * Use the FastFadingChannel class
# * Apply spatial antenna correlation
# * Implement LMMSE detection with perfect channel knowledge
# * Run BER/SER simulations
# 
# We will first walk through the configuration of all components of the system model, before building an end-to-end model which will allow you to run efficiently simulations with different parameter settings.

# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Simple uncoded transmission](#Simple-uncoded-transmission)
#     * [Adding spatial correlation](#Adding-spatial-correlation)
# * [Extension to channel coding](#Extension-to-channel-coding)
#     * [BER simulations using a Sionna Block](#BER-simulations-using-a-Sionna-Block)

# ### GPU Configuration and Imports

# In[1]:


import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducability
sionna.phy.config.seed = 42


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sionna.phy import Block
from sionna.phy.utils import ebnodb2no, compute_ser, compute_ber, PlotBER
from sionna.phy.channel import FlatFadingChannel, KroneckerModel
from sionna.phy.channel.utils import exp_corr_mat
from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.mapping import SymbolDemapper, Mapper, Demapper, BinarySource, QAMSource
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder


# ## Simple uncoded transmission
# 
# We will consider point-to-point transmissions from a transmitter with `num_tx_ant` antennas to a receiver
# with `num_rx_ant` antennas. The transmitter applies no precoding and sends independent data stream from each antenna.
# 
# Let us now generate a batch of random transmit vectors of random 16QAM symbols:

# In[3]:


num_tx_ant = 4
num_rx_ant = 16
num_bits_per_symbol = 4
batch_size = 1024
qam_source = QAMSource(num_bits_per_symbol)
x = qam_source([batch_size, num_tx_ant])
print(x.shape)


# Next, we will create an instance of the `FlatFadingChannel` class to simulate transmissions over
# an i.i.d. Rayleigh fading channel. The channel will also add AWGN with variance `no`.
# As we will need knowledge of the channel realizations for detection, we activate the `return_channel` flag.

# In[4]:


channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=True)
no = 0.2 # Noise variance of the channel

# y and h are the channel output and channel realizations, respectively.
y, h = channel(x, no)
print(y.shape)
print(h.shape)


# Using the perfect channel knowledge, we can now implement an LMMSE equalizer to compute soft-symbols.
# The noise covariance matrix in this example is just a scaled identity matrix which we need to provide to the
# `lmmse_equalizer`.

# In[5]:


s = tf.cast(no*tf.eye(num_rx_ant, num_rx_ant), y.dtype)
x_hat, no_eff = lmmse_equalizer(y, h, s)


# Let us know have a look at the transmitted and received constellations:

# In[6]:


plt.axes().set_aspect(1.0)
plt.scatter(np.real(x_hat), np.imag(x_hat));
plt.scatter(np.real(x), np.imag(x));


# As expected, the soft symbols `x_hat` are scattered around the 16QAM constellation points.
# The equalizer output `no_eff` provides an estimate of the effective noise variance for each soft-symbol.

# In[7]:


print(no_eff.shape)


# One can confirm that this estimate is correct by comparing the MSE between the transmitted and equalized symbols against the average estimated effective noise variance:

# In[8]:


noise_var_eff = np.var(x-x_hat)
noise_var_est = np.mean(no_eff)
print(noise_var_eff)
print(noise_var_est)


# The last step is to make hard decisions on the symbols and compute the SER:

# In[9]:


symbol_demapper = SymbolDemapper("qam", num_bits_per_symbol, hard_out=True)

# Get symbol indices for the transmitted symbols
x_ind = symbol_demapper(x, no)

# Get symbol indices for the received soft-symbols
x_ind_hat = symbol_demapper(x_hat, no)

compute_ser(x_ind, x_ind_hat)


# ### Adding spatial correlation
# 
# It is very easy add spatial correlation to the `FlatFadingChannel` using the `SpatialCorrelation` class.
# We can, e.g., easily setup a Kronecker (`KroneckerModel`) (or two-sided) correlation model using exponetial correlation matrices (`exp_corr_mat`).

# In[10]:


# Create transmit and receive correlation matrices
r_tx = exp_corr_mat(0.4, num_tx_ant)
r_rx = exp_corr_mat(0.9, num_rx_ant)

# Add the spatial correlation model to the channel
channel.spatial_corr = KroneckerModel(r_tx, r_rx)


# Next, we can validate that the channel model applies the desired spatial correlation by creating a large batch of channel realizations from which we compute the empirical transmit and receiver covariance matrices:

# In[11]:


h = channel.generate(1000000)

# Compute empirical covariance matrices
r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)/num_rx_ant
r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)/num_tx_ant

# Test that the empirical results match the theory
assert(np.allclose(r_tx, r_tx_hat, atol=1e-2))
assert(np.allclose(r_rx, r_rx_hat, atol=1e-2))


# Now, we can transmit the same symbols `x` over the channel with spatial correlation and compute the SER:

# In[12]:


y, h = channel(x, no)
x_hat, no_eff = lmmse_equalizer(y, h, s)
x_ind_hat = symbol_demapper(x_hat, no)
compute_ser(x_ind, x_ind_hat)


# The result cleary show the negative effect of spatial correlation in this setting.
# You can play around with the `a` parameter defining the exponential correlation matrices and see its impact on the SER. 

# ## Extension to channel coding
# So far, we have simulated uncoded symbol transmissions. With a few lines of additional code, we can extend what we have done to coded BER simulations. We need the following additional components:

# In[13]:


n = 1024 # codeword length
k = 512  # number of information bits per codeword
coderate = k/n # coderate
batch_size = 32

binary_source = BinarySource()
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder, hard_out=True)
mapper = Mapper("qam", num_bits_per_symbol)
demapper = Demapper("app", "qam", num_bits_per_symbol)


# Next we need to generate random QAM symbols through mapping of coded bits.
# Reshaping is required to bring `x` into the needed shape.

# In[14]:


b = binary_source([batch_size, num_tx_ant, k])
c = encoder(b)
x = mapper(c)
x_ind = symbol_demapper(x, no) # Get symbol indices for SER computation later on
shape = tf.shape(x)
x = tf.reshape(x, [-1, num_tx_ant])
print(x.shape)


# We will now transmit the symbols over the channel:

# In[15]:


y, h = channel(x, no)
x_hat, no_eff = lmmse_equalizer(y, h, s)


# And then demap the symbols to LLRs prior to decoding them. Note that we need to bring `x_hat` and `no_eff` back to the desired shape for decoding.

# In[16]:


x_ind_hat.shape


# In[17]:


x_hat = tf.reshape(x_hat, shape)
no_eff = tf.reshape(no_eff, shape)

llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)

x_ind_hat = symbol_demapper(x_hat, no)
ber = compute_ber(b, b_hat).numpy()
print("Uncoded SER : {}".format(compute_ser(x_ind, x_ind_hat)))
print("Coded BER : {}".format(compute_ber(b, b_hat)))


# Despite the fairly high SER, the BER is very low, thanks to the channel code.

# ### BER simulations using a Sionna Block

# Next, we will wrap everything that we have done so far in a Sionna Block for convenient BER simulations and comparison of system parameters.
# Note that we use the `@tf.function(jit_compile=True)` decorator which will speed-up the simulations tremendously. See [https://www.tensorflow.org/guide/function](https://www.tensorflow.org/guide/function) for further information.

# In[18]:


class Model(Block):
    def __init__(self, spatial_corr=None):
        super().__init__()
        self.n = 1024
        self.k = 512
        self.coderate = self.k/self.n
        self.num_bits_per_symbol = 4
        self.num_tx_ant = 4
        self.num_rx_ant = 16
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self.channel = FlatFadingChannel(self.num_tx_ant,
                                         self.num_rx_ant,
                                         spatial_corr=spatial_corr,
                                         add_awgn=True,
                                         return_channel=True)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        b = self.binary_source([batch_size, self.num_tx_ant, self.k])
        c = self.encoder(b)

        x = self.mapper(c)
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.num_tx_ant])

        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        no *= np.sqrt(self.num_rx_ant)

        y, h = self.channel(x, no)
        s = tf.complex(no*tf.eye(self.num_rx_ant, self.num_rx_ant), 0.0)

        x_hat, no_eff = lmmse_equalizer(y, h, s)

        x_hat = tf.reshape(x_hat, shape)
        no_eff = tf.reshape(no_eff, shape)

        llr = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)

        return b,  b_hat


# We can now instantiate different version of this model and use the `PlotBer` class for easy Monte-Carlo simulations.

# In[19]:


ber_plot = PlotBER()


# In[20]:


model1 = Model()

ber_plot.simulate(model1,
        np.arange(-2.5, 0.25, 0.25),
        batch_size=4096,
        max_mc_iter=1000,
        num_target_block_errors=100,
        legend="Uncorrelated",
        show_fig=False);


# In[21]:


r_tx = exp_corr_mat(0.4, num_tx_ant)
r_rx = exp_corr_mat(0.7, num_rx_ant)
model2 = Model(KroneckerModel(r_tx, r_rx))

ber_plot.simulate(model2,
        np.arange(0,2.6,0.25),
        batch_size=4096,
        max_mc_iter=1000,
        num_target_block_errors=200,
        legend="Kronecker model");

-e 
# --- End of Simple_MIMO_Simulation.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Part 1: Getting Started with Sionna

# This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model.
# You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
# 
# The tutorial is structured in four notebooks:
# 
# - **Part I: Getting started with Sionna**
# 
# - Part II: Differentiable Communication Systems
# 
# - Part III: Advanced Link-level Simulations
# 
# - Part IV: Toward Learned Receivers
# 

# The [official documentation](https://nvlabs.github.io/sionna) provides key material on how to use Sionna and how its components are implemented.

# * [Imports & Basics](#Imports-&-Basics)
# * [A note on random number generation](#A-note-on-random-number-generation)
# * [Sionna Data-flow and Design Paradigms](#Sionna-Data-flow-and-Design-Paradigms)
# * [Hello, Sionna!](#Hello,-Sionna!)
# * [Communication Systems as Models](#Communication-Systems-as-sionna-blocks)
# * [Forward Error Correction](#Forward-Error-Correction-(FEC))
# * [Eager vs. Graph Mode](#Eager-vs-Graph-Mode)
# * [Exercise](#Exercise)

# ## Imports & Basics

# In[1]:


import os # Configure which GPU 
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For plotting
get_ipython().run_line_magic('matplotlib', 'inline')
# also try %matplotlib widget

import matplotlib.pyplot as plt

# for performance measurements
import time


# We can now access Sionna functions within the `sn` namespace.
# 
# **Hint**: In Jupyter notebooks, you can run bash commands with `!`.

# In[2]:


get_ipython().system('nvidia-smi')


# ## A note on random number generation
# When Sionna is loaded, it instantiates random number generators (RNGs) for [Python](https://docs.python.org/3/library/random.html#alternative-generator),
# [NumPy](https://numpy.org/doc/stable/reference/random/generator.html), and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/random/Generator). You can optionally set a seed which will make all of your
# results deterministic, as long as only these RNGs are used. In the cell below,
# you can see how this seed is set and how the different RNGs can be used.

# In[3]:


sionna.phy.config.seed = 40

# Python RNG - use instead of
# import random
# random.randint(0, 10)
print(sionna.phy.config.py_rng.randint(0,10))

# NumPy RNG - use instead of
# import numpy as np
# np.random.randint(0, 10)
print(sionna.phy.config.np_rng.integers(0,10))

# TensorFlow RNG - use instead of
# import tensorflow as tf
# tf.random.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32)
print(sionna.phy.config.tf_rng.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32))


# ## Sionna Data-flow and Design Paradigms
# 
# Sionna inherently parallelizes simulations via *batching*, i.e., each element in the batch dimension is simulated independently.
# 
# This means the first tensor dimension is always used for *inter-frame* parallelization similar to an outer *for-loop* in Matlab/NumPy simulations, but operations can be operated in parallel.
# 
# To keep the dataflow efficient, Sionna follows a few simple design principles:
# 
# * Signal-processing components are implemented as an individual Sionna Blocks..
# * `tf.float32` is used as preferred datatype and `tf.complex64` for complex-valued datatypes, respectively.  
# This allows simpler re-use of components (e.g., the same scrambling layer can be used for binary inputs and LLR-values).
# * `tf.float64`/`tf.complex128` are available when high precision is needed.
# * Models can be developed in *eager mode* allowing simple (and fast) modification of system parameters.
# * Number crunching simulations can be executed in the faster *graph mode* or even *XLA* acceleration (experimental) is available for most components.
# * Whenever possible, components are automatically differentiable via [auto-grad](https://www.tensorflow.org/guide/autodiff) to simplify the deep learning design-flow.
# * Code is structured into sub-packages for different tasks such as channel coding, mapping,... (see [API documentation](http://nvlabs.github.io/sionna/phy/api/phy.html) for details).
# 
# These paradigms simplify the re-useability and reliability of our components for a wide range of communications related applications.

# ## Hello, Sionna!

# Let's start with a very simple simulation: Transmitting QAM symbols over an AWGN channel. We will implement the system shown in the figure below.


# We will use upper case for naming simulation parameters that are used throughout this notebook
# 
# Every layer needs to be initialized once before it can be used.
# 
# **Tip**: Use the [API documentation](http://nvlabs.github.io/sionna/phy/api/phy.html) to find an overview of all existing components.
# You can directly access the signature and the docstring within jupyter via `Shift+TAB`.
# 
# *Remark*: Most layers are defined to be complex-valued.
# 
# We first need to create a QAM constellation.

# In[4]:


NUM_BITS_PER_SYMBOL = 2 # QPSK
constellation = sionna.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

constellation.show();


# **Task:** Try to change the modulation order, e.g., to 16-QAM.

# We then need to setup a mapper to map bits into constellation points. The mapper takes as parameter the constellation.
# 
# We also need to setup a corresponding demapper to compute log-likelihood ratios (LLRs) from received noisy samples.

# In[5]:


mapper = sionna.phy.mapping.Mapper(constellation=constellation)

# The demapper uses the same constellation object as the mapper
demapper = sionna.phy.mapping.Demapper("app", constellation=constellation)


# **Tip**: You can access the signature+docstring via `?` command and print the complete class definition via `??` operator.
# 
# Obviously, you can also access the source code via [https://github.com/nvlabs/sionna/](https://github.com/nvlabs/sionna/).

# In[6]:


# print class definition of the Constellation class
get_ipython().run_line_magic('pinfo2', 'sionna.phy.mapping.Mapper')


# As can be seen, the `Mapper` class inherits from `Block`, i.e., implements a *Sionna Block*. These blocks can be connected by simply feeding the output of one block to the next block. This allows to simply build complex systems. 

# Sionna provides as utility a binary source to sample uniform i.i.d. bits.

# In[7]:


binary_source = sionna.phy.mapping.BinarySource()


# Finally, we need the AWGN channel.

# In[8]:


awgn_channel = sionna.phy.channel.AWGN()


# Sionna provides a utility function to compute the noise power spectral density ratio $N_0$ from the energy per bit to noise power spectral density ratio $E_b/N_0$ in dB and a variety of parameters such as the coderate and the nunber of bits per symbol.

# In[9]:


no = sionna.phy.utils.ebnodb2no(ebno_db=10.0,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here


# We now have all the components we need to transmit QAM symbols over an AWGN channel.
# 
# Sionna natively supports multi-dimensional tensors.
# 
# Most layers operate at the last dimension and can have arbitrary input shapes (preserved at output).

# In[10]:


BATCH_SIZE = 64 # How many examples are processed by Sionna in parallel

bits = binary_source([BATCH_SIZE,
                      1024]) # Blocklength
print("Shape of bits: ", bits.shape)

x = mapper(bits)
print("Shape of x: ", x.shape)

y = awgn_channel(x, no)
print("Shape of y: ", y.shape)

llr = demapper(y, no)
print("Shape of llr: ", llr.shape)


# In *Eager* mode, we can directly access the values of each tensor. This simplifies debugging.

# In[11]:


num_samples = 8 # how many samples shall be printed
num_symbols = int(num_samples/NUM_BITS_PER_SYMBOL)

print(f"First {num_samples} transmitted bits: {bits[0,:num_samples]}")
print(f"First {num_symbols} transmitted symbols: {np.round(x[0,:num_symbols], 2)}")
print(f"First {num_symbols} received symbols: {np.round(y[0,:num_symbols], 2)}")
print(f"First {num_samples} demapped llrs: {np.round(llr[0,:num_samples], 2)}")


# Let's visualize the received noisy samples.

# In[12]:


plt.figure(figsize=(8,8))
plt.axes().set_aspect(1)
plt.grid(True)
plt.title('Channel output')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.scatter(tf.math.real(y), tf.math.imag(y))
plt.tight_layout()


# **Task:** One can play with the SNR to visualize the impact on the received samples.
# 
# **Advanced Task:** Compare the LLR distribution for "app" demapping with "maxlog" demapping.
# The [Bit-Interleaved Coded Modulation](https://nvlabs.github.io/sionna/phy/tutorials/Bit_Interleaved_Coded_Modulation.html) example notebook can be helpful for this task.
# 

# ## Communication Systems as Sionna Blocks

# It is typically more convenient to wrap a Sionna-based communication system into a Sionna Block acting as end-to-end model.
# 
# These models can be simply built by stacking different Sionna components (i.e., Sionna Blocks).
# 
# The following cell implements the previous system as a end-to-end model.
# 
# The key functions that need to be defined are `__init__()`, which instantiates the required components, and `__call()__`, which performs forward pass through the end-to-end system.

# In[13]:


class UncodedSystemAWGN(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol, block_length):
        """
        A Sionna Block for uncoded transmission over the AWGN channel

        Parameters
        ----------
        num_bits_per_symbol: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.

        block_length: int
            The number of bits per transmitted message block (will be the codeword length later).

        Input
        -----
        batch_size: int
            The batch_size of the Monte-Carlo simulation.

        ebno_db: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.

        Output
        ------
        (bits, llr):
            Tuple:

        bits: tf.float32
            A tensor of shape `[batch_size, block_length] of 0s and 1s
            containing the transmitted information bits.

        llr: tf.float32
            A tensor of shape `[batch_size, block_length] containing the
            received log-likelihood-ratio (LLR) values.
        """

        super().__init__() # Must call the block initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

    # @tf.function # Enable graph execution to speed things up
    def call(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sionna.phy.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y,no)
        return bits, llr


# We need first to instantiate the model.

# In[14]:


model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)


# Sionna provides a utility to easily compute and plot the bit error rate (BER).

# In[15]:


EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 2000 # How many examples are processed by Sionna in parallel

ber_plots = sionna.phy.utils.PlotBER("AWGN")
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=True);


# The `sionna.phy.utils.PlotBER` object stores the results and allows to add additional simulations to the previous curves.
# 
# *Remark*: In Sionna, a block error is defined to happen if for two tensors at least one position in the last dimension differs (i.e., at least one bit wrongly received per codeword).
# The bit error rate the total number of erroneous positions divided by the total number of transmitted bits.

# ## Forward Error Correction (FEC)
# 
# We now add channel coding to our transceiver to make it more robust against transmission errors. For this, we will use [5G compliant low-density parity-check (LDPC) codes and Polar codes](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3214).
# You can find more detailed information in the notebooks [Bit-Interleaved Coded Modulation (BICM)](https://nvlabs.github.io/sionna/phy/tutorials/Bit_Interleaved_Coded_Modulation.html) and [5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes](https://nvlabs.github.io/sionna/phy/tutorials/5G_Channel_Coding_Polar_vs_LDPC_Codes.html).

# In[16]:


k = 12
n = 20

encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k, n)
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)


# Let us encode some random input bits.

# In[17]:


BATCH_SIZE = 1 # one codeword in parallel
u = binary_source([BATCH_SIZE, k])
print("Input bits are: \n", u.numpy())

c = encoder(u)
print("Encoded bits are: \n", c.numpy())


# One of the fundamental paradigms of Sionna is batch-processing.
# Thus, the example above could be executed for arbitrary batch-sizes to simulate `batch_size` codewords in parallel.
# 
# However, Sionna can do more - it supports *N*-dimensional input tensors and, thereby, allows the processing of multiple samples of multiple users and several antennas in a single command line.
# Let's say we want to encode `batch_size` codewords of length `n` for each of the `num_users` connected to each of the `num_basestations`. 
# This means in total we transmit `batch_size` * `n` * `num_users` * `num_basestations` bits.

# In[18]:


BATCH_SIZE = 10 # samples per scenario
num_basestations = 4
num_users = 5 # users per basestation
n = 1000 # codeword length per transmitted codeword
coderate = 0.5 # coderate

k = int(coderate * n) # number of info bits per codeword

# instantiate a new encoder for codewords of length n
encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k, n)

# the decoder must be linked to the encoder (to know the exact code parameters used for encoding)
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder,
                                    hard_out=True, # binary output or provide soft-estimates
                                    return_infobits=True, # or also return (decoded) parity bits
                                    num_iter=20, # number of decoding iterations
                                    cn_type="boxplus-phi") # also try "minsum" decoding

# draw random bits to encode
u = binary_source([BATCH_SIZE, num_basestations, num_users, k])
print("Shape of u: ", u.shape)

# We can immediately encode u for all users, basetation and samples
# This all happens with a single line of code
c = encoder(u)
print("Shape of c: ", c.shape)

print("Total number of processed bits: ", np.prod(c.shape))


# This works for arbitrary dimensions and allows a simple extension of the designed system to multi-user or multi-antenna scenarios.
# 
# Let us now replace the LDPC code by a Polar code. The API remains similar.

# In[19]:


k = 64
n = 128

encoder = sionna.phy.fec.polar.Polar5GEncoder(k, n)
decoder = sionna.phy.fec.polar.Polar5GDecoder(encoder,
                                      dec_type="SCL") # you can also use "SCL"


# *Advanced Remark:* The 5G Polar encoder/decoder class directly applies rate-matching and the additional CRC concatenation. 
# This is all done internally and transparent to the user.
# 
# In case you want to access low-level features of the Polar codes, please use `sionna.fec.polar.PolarEncoder` and the desired decoder (`sionna.fec.polar.PolarSCDecoder`, `sionna.fec.polar.PolarSCLDecoder` or `sionna.fec.polar.PolarBPDecoder`).
# 
# Further details can be found in the tutorial notebook on [5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes](https://nvlabs.github.io/sionna/phy/tutorials/5G_Channel_Coding_Polar_vs_LDPC_Codes.html).
# 


# In[20]:


class CodedSystemAWGN(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol, n, coderate):
        super().__init__() # Must call the Sionna block initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n*coderate)
        self.coderate = coderate
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)

        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)

        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    #@tf.function # activate graph execution to speed things up
    def call(self, batch_size, ebno_db):
        no = sionna.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)

        bits = self.binary_source([batch_size, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y,no)
        bits_hat = self.decoder(llr)
        return bits, bits_hat


# In[21]:


CODERATE = 0.5
BATCH_SIZE = 2000

model_coded_awgn = CodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                   n=2048,
                                   coderate=CODERATE)
ber_plots.simulate(model_coded_awgn,
                   ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 15),
                   batch_size=BATCH_SIZE,
                   num_target_block_errors=500,
                   legend="Coded",
                   soft_estimates=False,
                   max_mc_iter=15,
                   show_fig=True,
                   forward_keyboard_interrupt=False);


# As can be seen, the `BerPlot` class uses multiple stopping conditions and stops the simulation after no error occured at a specifc SNR point.

# **Task**: Replace the coding scheme by a Polar encoder/decoder or a convolutional code with Viterbi decoding.

# ## Eager vs Graph Mode
# 
# So far, we have executed the example in *eager* mode. 
# This allows to run TensorFlow ops as if it was written NumPy and simplifies development and debugging.
# 
# However, to unleash Sionna's full performance, we need to activate *graph* mode which can be enabled with the function decorator *@tf.function()*.
# 
# We refer to [TensorFlow Functions](https://www.tensorflow.org/guide/function) for further details.
# 

# In[22]:


@tf.function() # enables graph-mode of the following function
def run_graph(batch_size, ebno_db):
    # all code inside this function will be executed in graph mode, also calls of other functions
    print(f"Tracing run_graph for values batch_size={batch_size} and ebno_db={ebno_db}.") # print whenever this function is traced
    return model_coded_awgn(batch_size, ebno_db)


# In[23]:


batch_size = 10 # try also different batch sizes
ebno_db = 1.5

# run twice - how does the output change?
run_graph(batch_size, ebno_db)


# In graph mode, Python code (i.e., *non-TensorFlow code*) is only executed whenever the function is *traced*.
# This happens whenever the input signature changes.
# 
# As can be seen above, the print statement was executed, i.e., the graph was traced again.
# 
# To avoid this re-tracing for different inputs, we now input tensors.
# You can see that the function is now traced once for input tensors of same dtype.
# 
# See [TensorFlow Rules of Tracing](https://www.tensorflow.org/guide/function#rules_of_tracing) for details.
# 
# **Task:** change the code above such that tensors are used as input and execute the code with different input values. Understand when re-tracing happens.
# 
# *Remark*: if the input to a function is a tensor its signature must change and not *just* its value. For example the input could have a different size or datatype.
# For efficient code execution, we usually want to avoid re-tracing of the code if not required.

# In[24]:


# You can print the cached signatures with
print(run_graph.pretty_printed_concrete_signatures())


# We now compare the throughput of the different modes.

# In[25]:


repetitions = 4 # average over multiple runs
batch_size = BATCH_SIZE # try also different batch sizes
ebno_db = 1.5

# --- eager mode ---
t_start = time.perf_counter()
for _ in range(repetitions):
    bits, bits_hat = model_coded_awgn(tf.constant(batch_size, tf.int32),
                                tf.constant(ebno_db, tf. float32))
t_stop = time.perf_counter()
# throughput in bit/s
throughput_eager = np.size(bits.numpy())*repetitions / (t_stop - t_start) / 1e6

print(f"Throughput in Eager mode: {throughput_eager :.3f} Mbit/s")
# --- graph mode ---
# run once to trace graph (ignored for throughput)
run_graph(tf.constant(batch_size, tf.int32),
          tf.constant(ebno_db, tf. float32))

t_start = time.perf_counter()
for _ in range(repetitions):
    bits, bits_hat = run_graph(tf.constant(batch_size, tf.int32),
                                tf.constant(ebno_db, tf. float32))
t_stop = time.perf_counter()
# throughput in bit/s
throughput_graph = np.size(bits.numpy())*repetitions / (t_stop - t_start) / 1e6

print(f"Throughput in graph mode: {throughput_graph :.3f} Mbit/s")



# Let's run the same simulation as above in graph mode.

# In[26]:


ber_plots.simulate(run_graph,
                   ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 12),
                   batch_size=BATCH_SIZE,
                   num_target_block_errors=500,
                   legend="Coded (Graph mode)",
                   soft_estimates=True,
                   max_mc_iter=100,
                   show_fig=True,
                   forward_keyboard_interrupt=False);


# **Task:** TensorFlow allows to *compile* graphs with [XLA](https://www.tensorflow.org/xla). Try to further accelerate the code with XLA (`@tf.function(jit_compile=True)`).
# 
# *Remark*: XLA is still an experimental feature and not all TensorFlow (and, thus, Sionna) functions support XLA.
# 
# **Task 2:** Check the GPU load with `!nvidia-smi`. Find the best tradeoff between batch-size and throughput for your specific GPU architecture.

# ## Exercise
# 
# Simulate the coded bit error rate (BER) for a Polar coded and 64-QAM modulation.
# Assume a codeword length of n = 200 and coderate = 0.5.
# 
# **Hint**: For Polar codes, successive cancellation list decoding (SCL) gives the best BER performance.
# However, successive cancellation (SC) decoding (without a list) is less complex.
# 
# 

# In[27]:


n = 200
coderate = 0.5

# *You can implement your code here*

-e 
# --- End of Sionna_tutorial_part1.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Part 2: Differentiable Communication Systems

# This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model.
# You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
# 
# The tutorial is structured in four notebooks:
# 
# - Part I: Getting started with Sionna
# 
# - **Part II: Differentiable Communication Systems**
# 
# - Part III: Advanced Link-level Simulations
# 
# - Part IV: Toward Learned Receivers

# The [official documentation](https://nvlabs.github.io/sionna/phy) provides key material on how to use Sionna and how its components are implemented.

# * [Imports](#Imports)
# * [Gradient Computation Through End-to-end Systems](#Gradient-Computation-Through-End-to-end-Systems)
# * [Creating Custom Layers](#Creating-Custom-Layers)
# * [Setting up Training Loops](#Setting-up-Training-Loops)

# ## Imports

# In[1]:


import os # Configure which GPU 
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna as sn
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# For saving complex Python data structures efficiently
import pickle

# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

# Set seed for reproducable results
sn.phy.config.seed = 42


# ## Gradient Computation Through End-to-end Systems<a class="anchor" id="Gradient-Computation-Through-End-to-end-Systems"></a>

# Let's start by setting up a simple communication system that transmit bits modulated as QAM symbols over an AWGN channel.
# 
# However, compared to what we have previously done, we now make the constellation
# *trainable*. With Sionna, achieving this by assigning trainable points to a
# `Constellation` instance.

# In[2]:


# Binary source to generate uniform i.i.d. bits
binary_source = sn.phy.mapping.BinarySource()

# 256-QAM constellation
NUM_BITS_PER_SYMBOL = 6
qam_constellation = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

# Make a trainable constellation initialized with QAM points
trainable_points = tf.Variable(tf.stack([tf.math.real(qam_constellation.points),
                                         tf.math.imag(qam_constellation.points)], axis=0))

constellation = sn.phy.mapping.Constellation("custom",
                                             num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                             points = tf.complex(trainable_points[0], trainable_points[1]),
                                             normalize=True,
                                             center=True)

# Mapper and demapper
mapper = sn.phy.mapping.Mapper(constellation=constellation)
demapper = sn.phy.mapping.Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = sn.phy.channel.AWGN()


# As we have already seen, we can now easily simulate forward passes through the system we have just setup

# In[3]:


BATCH_SIZE = 128 # How many examples are processed by Sionna in parallel
EBN0_DB = 17.0 # Eb/N0 in dB

no = sn.phy.utils.ebnodb2no(ebno_db=EBN0_DB,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here

bits = binary_source([BATCH_SIZE,
                        1200]) # Blocklength
x = mapper(bits)
y = awgn_channel(x, no)
llr = demapper(y,no)


# Just for fun, let's visualize the channel inputs and outputs

# In[4]:


plt.figure(figsize=(8,8))
plt.axes().set_aspect(1.0)
plt.grid(True)
plt.scatter(tf.math.real(y), tf.math.imag(y), label='Output')
plt.scatter(tf.math.real(x), tf.math.imag(x), label='Input')
plt.legend(fontsize=20);


# Let's now *optimize* the constellation through *stochastic gradient descent* (SGD). As we will see, this is made very easy by Sionna.
# 
# We need to define a *loss function* that we will aim to minimize.
# 
# We can see the task of the receiver as jointly solving, for each received symbol, `NUM_BITS_PER_SYMBOL` binary classification problems in order to reconstruct the transmitted bits.
# Therefore, a natural choice for the loss function is the *binary cross-entropy* (BCE) applied to each bit and to each received symbol.
# 
# *Remark:* The LLRs computed by the demapper are *logits* on the transmitted bits, and can therefore be used as-is to compute the BCE without any additional processing.
# *Remark 2:* The BCE is closely related to an achieveable information rate for bit-interleaved coded modulation systems [1,2]
# 
# [1] Georg Böcherer, "Principles of Coded Modulation", [available online](http://www.georg-boecherer.de/bocherer2018principles.pdf)
# 
# [2] F. Ait Aoudia and J. Hoydis, "End-to-End Learning for OFDM: From Neural Receivers to Pilotless Communication," in IEEE Transactions on Wireless Communications, vol. 21, no. 2, pp. 1049-1063, Feb. 2022, doi: 10.1109/TWC.2021.3101364.

# In[5]:


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

print(f"BCE: {bce(bits, llr)}")


# One iteration of SGD consists in three steps:
# 1. Perform a forward pass through the end-to-end system and compute the loss function
# 2. Compute the gradient of the loss function with respect to the trainable weights
# 3. Apply the gradient to the weights
# 
# To enable gradient computation, we need to perform the forward pass (step 1) within a `GradientTape`

# In[6]:


with tf.GradientTape() as tape:
    bits = binary_source([BATCH_SIZE,
                            1200]) # Blocklength
    mapper.constellation.points = tf.complex(trainable_points[0], trainable_points[1])
    x = mapper(bits)
    y = awgn_channel(x, no)
    llr = demapper(y,no)
    loss = bce(bits, llr)


# Using the ``GradientTape``, computing the gradient is done as follows

# In[7]:


gradient = tape.gradient(loss, [trainable_points])


# `gradient` is a list of tensors, each tensor corresponding to a trainable variable of our model.
# 
# For this model, we only have a single trainable tensor: The constellation of shape [`2`, `2^NUM_BITS_PER_SYMBOL`], the first dimension corresponding to the real and imaginary components of the constellation points.
# 
# *Remark:* It is important to notice that the gradient computation was performed *through the demapper and channel*, which are conventional non-trainable algorithms implemented as *differentiable* Sionna blocks. This key feature of Sionna enables the training of end-to-end communication systems that combine both trainable and conventional and/or non-trainable signal processing algorithms.

# In[8]:


for g in gradient:
    print(g.shape)


# Applying the gradient (third step) is performed using an *optimizer*. [Many optimizers are available as part of TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers), and we use in this notebook ``Adam``.

# In[9]:


optimizer = tf.keras.optimizers.Adam(1e-2)


# Using the optimizer, the gradients can be applied to the trainable weights to update them

# In[10]:


optimizer.apply_gradients(zip(gradient, tape.watched_variables()));


# Let's compare the constellation before and after the gradient application

# In[11]:


fig = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL).show()
fig.axes[0].scatter(trainable_points[0], trainable_points[1], label='After SGD')
fig.axes[0].legend();


# The SGD step has led to slight change in the position of the constellation points. Training of a communication system using SGD consists in looping over such SGD steps until a stop criterion is met.

# ## Creating Custom Layers<a class="anchor" id="Creating-Custom-Layers"></a>

# Custom trainable (or not trainable) algorithms can be implemented as [Keras layers](https://keras.io/api/layers/) or Sionna blocks. All Sionna components, such as the mapper, demapper, channel... are implemented as Sionna blocks.
# 
# To illustrate how this can be done, the next cell implements a simple neural network-based demapper which consists of three dense layers.

# In[12]:


class NeuralDemapper(Layer): # Inherits from Keras Layer

    def __init__(self):
        super().__init__()

        # The three dense layers that form the custom trainable neural network-based demapper
        self.dense_1 = Dense(64, 'relu')
        self.dense_2 = Dense(64, 'relu')
        self.dense_3 = Dense(NUM_BITS_PER_SYMBOL, None) # The last layer has no activation and therefore outputs logits, i.e., LLRs

    def call(self, y):

        # y : complex-valued with shape [batch size, block length]
        # y is first mapped to a real-valued tensor with shape
        #  [batch size, block length, 2]
        # where the last dimension consists of the real and imaginary components
        # The dense layers operate on the last dimension, and treat the inner dimensions as batch dimensions, i.e.,
        # all the received symbols are independently processed.
        nn_input = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        z = self.dense_1(nn_input)
        z = self.dense_2(z)
        z = self.dense_3(z) # [batch size, number of symbols per block, number of bits per symbol]
        llr = tf.reshape(z, [tf.shape(y)[0], -1]) # [batch size, number of bits per block]
        return llr


# A custom Keras layer is used as any other Sionna layer, and therefore integration to a Sionna-based communication is straightforward.
# 
# The following model uses the neural demapper instead of the conventional demapper. It takes at initialization a parameter that indicates if the model is intantiated to be trained or evaluated. When instantiated to be trained, the loss function is returned. Otherwise, the transmitted bits and LLRs are returned.

# In[13]:


class End2EndSystem(Model): # Inherits from Keras Model

    def __init__(self, training):

        super().__init__() # Must call the Keras model initializer

        qam_constellation = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL) 

        self.points = tf.Variable(tf.stack([tf.math.real(qam_constellation.points),
                                            tf.math.imag(qam_constellation.points)], axis=0))

        self.constellation = sn.phy.mapping.Constellation("custom",
                                                    num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                                    points = tf.complex(self.points[0], self.points[1]),
                                                    normalize=True,
                                                    center=True)
        self.mapper = sn.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = NeuralDemapper() # Intantiate the NeuralDemapper custom layer as any other
        self.binary_source = sn.phy.mapping.BinarySource()
        self.awgn_channel = sn.phy.channel.AWGN()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function
        self.training = training

    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.phy.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits

        # Assign points to constellation
        self.mapper.constellation.points = tf.complex(self.points[0], self.points[1])

        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y)  # Call the NeuralDemapper custom layer as any other
        if self.training:
            loss = self.bce(bits, llr)
            return loss
        else:
            return bits, llr


# When a model that includes a neural network is created, the neural network weights are randomly initialized typically leading to very poor performance.
# 
# To see this, the following cell benchmarks the previously defined untrained model against a conventional baseline.

# In[14]:


EBN0_DB_MIN = 10.0
EBN0_DB_MAX = 20.0


###############################
# Baseline
###############################

class Baseline(Model): # Inherits from Keras Model

    def __init__(self):

        super().__init__() # Must call the Keras model initializer

        self.constellation = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
        self.mapper = sn.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.phy.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.phy.mapping.BinarySource()
        self.awgn_channel = sn.phy.channel.AWGN()

    @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.phy.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y,no)
        return bits, llr

###############################
# Benchmarking
###############################

baseline = Baseline()
model = End2EndSystem(False)
ber_plots = sn.phy.utils.PlotBER("Neural Demapper")
ber_plots.simulate(baseline,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);
ber_plots.simulate(model,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Untrained model",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=True);


# ## Setting up Training Loops <a class="anchor" id="Setting-up-Training-Loops"></a>

# Training of end-to-end communication systems consists in iterating over SGD steps.
# 
# The next cell implements a training loop of `NUM_TRAINING_ITERATIONS` iterations.
# The training SNR is set to $E_b/N_0 = 15$ dB.
# 
# At each iteration:
# - A forward pass through the end-to-end system is performed within a gradient tape
# - The gradients are computed using the gradient tape, and applied using the Adam optimizer
# - The estimated loss is periodically printed to follow the progress of training

# In[15]:


# Number of iterations used for training
NUM_TRAINING_ITERATIONS = 10000

# Instantiating the end-to-end model for training
model_train = End2EndSystem(training=True)

# Adam optimizer (SGD variant)
optimizer = tf.keras.optimizers.Adam()

# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model_train(BATCH_SIZE, 15.0) # The model is assumed to return the BMD rate
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")


# The weights of the trained model are saved using [pickle](https://docs.python.org/3/library/pickle.html).

# In[16]:


# Save the weightsin a file
weights = model_train.get_weights()
with open('weights-neural-demapper', 'wb') as f:
    pickle.dump(weights, f)


# Finally, we evaluate the trained model and benchmark it against the previously introduced baseline.
# 
# We first instantiate the model for evaluation and load the saved weights.

# In[17]:


# Instantiating the end-to-end model for evaluation
model = End2EndSystem(training=False)
# Run one inference to build the layers and loading the weights
model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
with open('weights-neural-demapper', 'rb') as f:
    weights = pickle.load(f)
    model.set_weights(weights)


# The trained model is then evaluated.

# In[18]:


# Computing and plotting BER
ber_plots.simulate(model,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100,
                  legend="Trained model",
                  soft_estimates=True,
                  max_mc_iter=100,
                  show_fig=True);

-e 
# --- End of Sionna_tutorial_part2.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Part 3: Advanced Link-level Simulations

# This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model.
# You will also learn how to write custom trainable blocks by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
# 
# The tutorial is structured in four notebooks:
# 
# - Part I: Getting started with Sionna
# 
# - Part II: Differentiable Communication Systems
# 
# - **Part III: Advanced Link-level Simulations**
# 
# - Part IV: Toward Learned Receivers
# 

# The [official documentation](https://nvlabs.github.io/sionna/phy) provides key material on how to use Sionna and how its components are implemented.

# * [Imports](#Imports)
# * [OFDM Resource Grid and Stream Management](#OFDM-Resource-Grid-and-Stream-Management)
# * [Antenna Arrays](#Antenna-Arrays)
# * [Channel Model](#Channel-Model)
# * [Uplink Transmission in the Frequency Domain](#Uplink-Transmission-in-the-Frequency-Domain)

# ## Imports

# In[1]:


import os # Configure which GPU 
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna as sn
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# For the implementation of the Keras models
from tensorflow.keras import Model

# Set seed for reproducable results
sn.phy.config.seed = 42


# ## OFDM Resource Grid and Stream Management

# We will setup a realistic SIMO point-to-point link between a mobile user terminal (UT) and a base station (BS). The system we will setup is shown in the figure below.


# ### Stream Management
# 
# For any type of MIMO simulations, it is required to setup a `StreamManagement` object.
# It determines which transmitters and receivers communicate data streams with each other.
# In our scenario, we will configure a single UT equipped with a single antenna and a single BS equipped with multiple antennas.
# Whether the UT or BS is considered as a transmitter depends on the link direction, which can be
# either uplink or downlink. The `StreamManagement` has many properties that are used by other components,
# such as precoding and equalization.
# 
# We will configure the system here such that the number of streams per transmitter
# is equal to the number of UT antennas.

# In[2]:


# Define the number of UT and BS antennas
NUM_UT = 1
NUM_BS = 1
NUM_UT_ANT = 1
NUM_BS_ANT = 4

# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
NUM_STREAMS_PER_TX = NUM_UT_ANT

# Create an RX-TX association matrix.
# RX_TX_ASSOCIATION[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change.
# For example, considering a system with 2 RX and 4 TX, the RX-TX
# association matrix could be
# [ [1 , 1, 0, 0],
#   [0 , 0, 1, 1] ]
# which indicates that the RX 0 receives from TX 0 and 1, and RX 1 receives from
# TX 2 and 3.
#
# In this notebook, as we have only a single transmitter and receiver,
# the RX-TX association matrix is simply:
RX_TX_ASSOCIATION = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
STREAM_MANAGEMENT = sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)


# ### OFDM Resource Grid

# Next, we configure an OFDM `ResourceGrid` spanning multiple OFDM symbols.
# The resource grid contains data symbols and pilots and is equivalent to a
# *slot* in 4G/5G terminology. Although it is not relevant for our simulation, we null the DC subcarrier
# and a few guard carriers to the left and right of the spectrum. Also a cyclic prefix is added.
# 
# During the creation of the `ResourceGrid`, a `PilotPattern` is automatically generated.
# We could have alternatively created a `PilotPattern` first and then provided it as initialization parameter.
# When multiple streams are considered, the corresponding pilot patterns must be orthogonal.
# By default, orthogonal pilots are setup when considering such systems.

# In[3]:


RESOURCE_GRID = sn.phy.ofdm.ResourceGrid(num_ofdm_symbols=14,
                                         fft_size=76,
                                         subcarrier_spacing=30e3,
                                         num_tx=NUM_UT,
                                         num_streams_per_tx=NUM_STREAMS_PER_TX,
                                         cyclic_prefix_length=6,
                                         pilot_pattern="kronecker",
                                         pilot_ofdm_symbol_indices=[2,11])
RESOURCE_GRID.show();


# In[4]:


RESOURCE_GRID.pilot_pattern.show();


# **Task:** You can try different pilot patterns, FFT size, number of OFDM symbols, and visualize how it affects the resource grid.
# 
# See the notebook [MIMO OFDM Transmissions over CDL](https://nvlabs.github.io/sionna/phy/tutorials/MIMO_OFDM_Transmissions_over_CDL.html) for more advanced examples.

# ## Antenna Arrays

# We need to configure the antenna arrays used by the UT and BS.
# This can be ignored for simple channel models, such as `AWGN`, `RayleighBlockFading`, or `TDL` which do not account for antenna array geometries and antenna radiation patterns. However, other models, such as `CDL`, `UMi`, `UMa`, and `RMa` from the 3GPP 38.901 specification, require it.
# 
# 
# An `AntennaArray` is always defined in the y-z plane. Its final orientation will be determined by the orientation of the UT or BS. This parameter can be configured in the `ChannelModel` that we will create later.

# In[5]:


CARRIER_FREQUENCY = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

UT_ARRAY = sn.phy.channel.tr38901.Antenna(polarization="single",
                                          polarization_type="V",
                                          antenna_pattern="38.901",
                                          carrier_frequency=CARRIER_FREQUENCY)
UT_ARRAY.show();

BS_ARRAY = sn.phy.channel.tr38901.AntennaArray(num_rows=1,
                                               num_cols=int(NUM_BS_ANT/2),
                                               polarization="dual",
                                               polarization_type="cross",
                                               antenna_pattern="38.901", # Try 'omni'
                                               carrier_frequency=CARRIER_FREQUENCY)
BS_ARRAY.show();


# In[6]:


BS_ARRAY.show_element_radiation_pattern();


# **Task:** You can try different antenna pattern ("omni"), polarization, and array geometries.

# ## Channel Model

# Sionna implements the CDL, TDL, UMi, UMa, and RMa models from [3GPP TR 38.901](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173), as well as Rayleigh block fading.
# 
# Note that:
# * TDL only supports SISO
# * CDL only supports single-user, possibly with multiple antenna
# * UMi, UMa, and RMa support single- and multi-user
# 
# *Remark:* The TDL and CDL models correspond to fixed power delay profiles and fixed angles.


# We consider the 3GPP CDL model family in this notebook.

# In[7]:


DELAY_SPREAD = 100e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value.

DIRECTION = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.

CDL_MODEL = "C"       # Suitable values are ["A", "B", "C", "D", "E"]

SPEED = 10.0          # UT speed [m/s]. BSs are always assumed to be fixed.
                     # The direction of travel will chosen randomly within the x-y plane.

# Configure a channel impulse reponse (CIR) generator for the CDL model.
CDL = sn.phy.channel.tr38901.CDL(CDL_MODEL,
                                 DELAY_SPREAD,
                                CARRIER_FREQUENCY,
                                UT_ARRAY,
                                BS_ARRAY,
                                DIRECTION,
                                min_speed=SPEED)


# The instance `CDL` of the CDL model can be used to generate batches of random realizations of continuous-time
# channel impulse responses, consisting of complex gains `a` and delays `tau` for each path. 
# To account for time-varying channels, a channel impulse responses is sampled at the `sampling_frequency` for `num_time_samples` samples.
# For more details on this, please have a look at the API documentation of the channel models.
# 
# In order to model the channel in the frequency domain, we need `num_ofdm_symbols` samples that are taken once per `ofdm_symbol_duration`, which corresponds to the length of an OFDM symbol plus the cyclic prefix.

# In[8]:


BATCH_SIZE = 128 # How many examples are processed by Sionna in parallel

a, tau = CDL(batch_size=BATCH_SIZE,
             num_time_steps=RESOURCE_GRID.num_ofdm_symbols,
             sampling_frequency=1/RESOURCE_GRID.ofdm_symbol_duration)


# The path gains `a` have shape\
# `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`\
# and the delays `tau` have shape\
# `[batch_size, num_rx, num_tx, num_paths]`.

# In[9]:


print("Shape of the path gains: ", a.shape)
print("Shape of the delays:", tau.shape)


# The delays are assumed to be static within the time-window of interest. Only the complex path gains change over time.
# The following two figures depict the channel impulse response at a particular time instant and the time-evolution of the gain of one path, respectively.

# In[10]:


plt.figure()
plt.title("Channel impulse response realization")
plt.stem(tau[0,0,0,:]/1e-9, np.abs(a)[0,0,0,0,0,:,0])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")

plt.figure()
plt.title("Time evolution of path gain")
plt.plot(np.arange(RESOURCE_GRID.num_ofdm_symbols)*RESOURCE_GRID.ofdm_symbol_duration/1e-6, np.real(a)[0,0,0,0,0,0,:])
plt.plot(np.arange(RESOURCE_GRID.num_ofdm_symbols)*RESOURCE_GRID.ofdm_symbol_duration/1e-6, np.imag(a)[0,0,0,0,0,0,:])
plt.legend(["Real part", "Imaginary part"])

plt.xlabel(r"$t$ [us]")
plt.ylabel(r"$a$");


# See the notebook [Realistic Multiuser MIMO Simulations](https://nvlabs.github.io/sionna/phy/tutorials/Realistic_Multiuser_MIMO_Simulations.html) for more advanced examples.

# ## Uplink Transmission in the Frequency Domain

# We are now ready to simulate a transmission.
# 
# In the following, the channel is simulated in the frequency domain. Therefore, the channel is assumed to be constant over the duration of an OFDM symbol, which leads to not simulating the intercarrier interference (ICI) that could occur due to channel aging over the duration of OFDM symbols.
# 
# The `OFDMChannel` layer is used to simulate the channel in the frequency domain and takes care of sampling channel impulse responses, computing the frequency responses, and applying the channel transfer function to the channel inputs (including AWGN).
# 
# Note that it is also possible to simulate the channel in time domain using the `TimeChannel` layer, which enables simulation of ICI.
# For more information, please have a look at the API documentation.

# In[11]:


NUM_BITS_PER_SYMBOL = 2 # QPSK
CODERATE = 0.5

# Number of coded bits in a resource grid
n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL)
# Number of information bits in a resource groud
k = int(n*CODERATE)

# The binary source will create batches of information bits
binary_source = sn.phy.mapping.BinarySource()

# The encoder maps information bits to coded bits
encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k, n)

# The mapper maps blocks of information bits to constellation symbols
mapper = sn.phy.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)

# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = sn.phy.ofdm.ResourceGridMapper(RESOURCE_GRID)

# Frequency domain channel
channel = sn.phy.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)

# The LS channel estimator will provide channel estimates and error variances
ls_est = sn.phy.ofdm.LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")

# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = sn.phy.ofdm.LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)

# The demapper produces LLR for all coded bits
demapper = sn.phy.mapping.Demapper("app", "qam", NUM_BITS_PER_SYMBOL)

# The decoder provides hard-decisions on the information bits
decoder = sn.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)


# Let's now simulate the transmission, and look at the shape of the layers outputs at each stage.
# 
# The utility function `ebnodb2no` takes as additional input the resource grid to account for the pilots when computing the noise power spectral density ratio $N_0$ from the energy per bit to noise power spectral density ratio $E_b/N_0$ (in dB).

# In[12]:


no = sn.phy.utils.ebnodb2no(ebno_db=10.0,
                            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                            coderate=CODERATE,
                            resource_grid=RESOURCE_GRID)

# Transmitter
bits = binary_source([BATCH_SIZE, NUM_UT, RESOURCE_GRID.num_streams_per_tx, k])
print("Shape of bits: ", bits.shape)
codewords = encoder(bits)
print("Shape of codewords: ", codewords.shape)
x = mapper(codewords)
print("Shape of x: ", x.shape)
x_rg = rg_mapper(x)
print("Shape of x_rg: ", x_rg.shape)

# Channel
y, h_freq = channel(x_rg, no)
print("Shape of y_rg: ", y.shape)
print("Shape of h_freq: ", h_freq.shape)

# Receiver
h_hat, err_var = ls_est (y, no)
print("Shape of h_hat: ", h_hat.shape)
print("Shape of err_var: ", err_var.shape)
x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
print("Shape of x_hat: ", x_hat.shape)
print("Shape of no_eff: ", no_eff.shape)
llr = demapper(x_hat, no_eff)
print("Shape of llr: ", llr.shape)
bits_hat = decoder(llr)
print("Shape of bits_hat: ", bits_hat.shape)


# The next cell implements the previous system as an end-to-end model.
# 
# Moreover, a boolean given as parameter to the initializer enables using either LS estimation or perfect CSI, as shown in the figure below.


# In[13]:


class OFDMSystem(Model): # Inherits from Keras Model

    def __init__(self, perfect_csi):
        super().__init__() # Must call the Keras model initializer

        self.perfect_csi = perfect_csi

        n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL) # Number of coded bits
        k = int(n*CODERATE) # Number of information bits
        self.k = k

        # The binary source will create batches of information bits
        self.binary_source = sn.phy.mapping.BinarySource()

        # The encoder maps information bits to coded bits
        self.encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k, n)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = sn.phy.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = sn.phy.ofdm.ResourceGridMapper(RESOURCE_GRID)

        # Frequency domain channel
        self.channel = sn.phy.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_est = sn.phy.ofdm.LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = sn.phy.ofdm.LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)

        # The demapper produces LLR for all coded bits
        self.demapper = sn.phy.mapping.Demapper("app", "qam", NUM_BITS_PER_SYMBOL)

        # The decoder provides hard-decisions on the information bits
        self.decoder = sn.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    @tf.function # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=CODERATE, resource_grid=RESOURCE_GRID)

        # Transmitter
        bits = self.binary_source([batch_size, NUM_UT, RESOURCE_GRID.num_streams_per_tx, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)

        # Channel
        y, h_freq = self.channel(x_rg, no)

        # Receiver
        if self.perfect_csi:
            h_hat, err_var = h_freq, 0.
        else:
            h_hat, err_var = self.ls_est (y, no)
        x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        bits_hat = self.decoder(llr)

        return bits, bits_hat


# In[14]:


EBN0_DB_MIN = -8.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 3.0 # Maximum value of Eb/N0 [dB] for simulations

ber_plots = sn.phy.utils.PlotBER("OFDM over 3GPP CDL")

model_ls = OFDMSystem(False)
ber_plots.simulate(model_ls,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="LS Estimation",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

model_pcsi = OFDMSystem(True)
ber_plots.simulate(model_pcsi,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Perfect CSI",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

ber_plots();

-e 
# --- End of Sionna_tutorial_part3.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Part 4: Toward Learned Receivers

# This tutorial will guide you through Sionna, from its basic principles to the implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model.
# You will also learn how to write custom trainable layers by implementing a state of the art neural receiver, and how to train and evaluate end-to-end communication systems.
# 
# The tutorial is structured in four notebooks:
# 
# - Part I: Getting started with Sionna
# 
# - Part II: Differentiable Communication Systems
# 
# - Part III: Advanced Link-level Simulations
# 
# - **Part IV: Toward Learned Receivers**

# The [official documentation](https://nvlabs.github.io/sionna/phy) provides key material on how to use Sionna and how its components are implemented.

# * [Imports](#Imports)
# * [Simulation Parameters](#Simulation-Parameters)
# * [Implemention of an Advanced Neural Receiver](#Implemention-of-an-Advanced-Neural-Receiver)
# * [Training the Neural Receiver](#Training-the-Neural-Receiver)
# * [Benchmarking the Neural Receiver](#Benchmarking-the-Neural-Receiver)
# * [Conclusion](#Conclusion)

# ## Imports<a class="anchor" id="import"></a>

# In[1]:


import os # Configure which GPU 
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna as sn
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e) 

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For saving complex Python data structures efficiently
import pickle

# For plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

# Set seed for reproducable results
sn.phy.config.seed = 42


# ## Simulation Parameters

# In[2]:


# Bit per channel use
NUM_BITS_PER_SYMBOL = 2 # QPSK

# Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MIN = -3.0

# Maximum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0

# How many examples are processed by Sionna in parallel
BATCH_SIZE = 128

# Coding rate
CODERATE = 0.5

# Define the number of UT and BS antennas
NUM_UT = 1
NUM_BS = 1
NUM_UT_ANT = 1
NUM_BS_ANT = 2

# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
NUM_STREAMS_PER_TX = NUM_UT_ANT

# Create an RX-TX association matrix.
# RX_TX_ASSOCIATION[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change.
# For example, considering a system with 2 RX and 4 TX, the RX-TX
# association matrix could be
# [ [1 , 1, 0, 0],
#   [0 , 0, 1, 1] ]
# which indicates that the RX 0 receives from TX 0 and 1, and RX 1 receives from
# TX 2 and 3.
#
# In this notebook, as we have only a single transmitter and receiver,
# the RX-TX association matrix is simply:
RX_TX_ASSOCIATION = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
STREAM_MANAGEMENT = sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

RESOURCE_GRID = sn.phy.ofdm.ResourceGrid(num_ofdm_symbols=14,
                                         fft_size=76,
                                         subcarrier_spacing=30e3,
                                         num_tx=NUM_UT,
                                         num_streams_per_tx=NUM_STREAMS_PER_TX,
                                         cyclic_prefix_length=6,
                                         pilot_pattern="kronecker",
                                         pilot_ofdm_symbol_indices=[2,11])

# Carrier frequency in Hz.
CARRIER_FREQUENCY = 2.6e9

# Antenna setting
UT_ARRAY = sn.phy.channel.tr38901.Antenna(polarization="single",
                                          polarization_type="V",
                                          antenna_pattern="38.901",
                                          carrier_frequency=CARRIER_FREQUENCY)
BS_ARRAY = sn.phy.channel.tr38901.AntennaArray(num_rows=1,
                                               num_cols=int(NUM_BS_ANT/2),
                                               polarization="dual",
                                               polarization_type="cross",
                                               antenna_pattern="38.901", # Try 'omni'
                                               carrier_frequency=CARRIER_FREQUENCY)

# Nominal delay spread in [s]. Please see the CDL documentation
# about how to choose this value.
DELAY_SPREAD = 100e-9

# The `direction` determines if the UT or BS is transmitting.
# In the `uplink`, the UT is transmitting.
DIRECTION = "uplink"

# Suitable values are ["A", "B", "C", "D", "E"]
CDL_MODEL = "C"

# UT speed [m/s]. BSs are always assumed to be fixed.
# The direction of travel will chosen randomly within the x-y plane.
SPEED = 10.0

# Configure a channel impulse reponse (CIR) generator for the CDL model.
CDL = sn.phy.channel.tr38901.CDL(CDL_MODEL,
                                 DELAY_SPREAD,
                                 CARRIER_FREQUENCY,
                                 UT_ARRAY,
                                 BS_ARRAY,
                                 DIRECTION,
                                 min_speed=SPEED)


# ## Implemention of an Advanced Neural Receiver

# We will implement a state-of-the-art neural receiver that operates over the entire resource grid of received symbols.

# The neural receiver computes LLRs on the coded bits from the received resource grid of frequency-domain baseband symbols.


# As shown in the following figure, the neural receiver substitutes to the channel estimator, equalizer, and demapper.


# As in [1] and [2], a neural receiver using residual convolutional layers is implemented.
# 
# Convolutional layers are leveraged to efficienly process the 2D resource grid that is fed as an input to the neural receiver.

# Residual (skip) connections are used to avoid gradient vanishing [3].
# 
# For convenience, a Keras layer that implements a *residual block* is first defined. The Keras layer that implements the neural receiver is built by stacking such blocks. The following figure shows the architecture of the neural receiver.


# In[3]:


class ResidualBlock(Layer):

    def __init__(self):
        super().__init__()

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_1 = Conv2D(filters=128,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=128,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs

        return z

class NeuralReceiver(Layer):

    def __init__(self):
        super().__init__()

        # Input convolution
        self._input_conv = Conv2D(filters=128,
                                  kernel_size=[3,3],
                                  padding='same',
                                  activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = Conv2D(filters=NUM_BITS_PER_SYMBOL,
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=None)

    def call(self, y, no):

        # Assuming a single receiver, remove the num_rx dimension
        y = tf.squeeze(y, axis=1)

        # Feeding the noise power in log10 scale helps with the performance
        no = sn.phy.utils.log10(no)

        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
        no = sn.phy.utils.insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1]
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        # Input conv
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)

        # Reshape the input to fit what the resource grid demapper is expected
        z = sn.phy.utils.insert_dims(z, 2, 1)

        return z


# The task of the receiver is to jointly solve, for each resource element, `NUM_BITS_PER_SYMBOL` binary classification problems in order to reconstruct the transmitted bits.
# Therefore, a natural choice for the loss function is the *binary cross-entropy* (BCE) applied to each bit and to each received symbol.
# 
# *Remark:* The LLRs computed by the demapper are *logits* on the transmitted bits, and can therefore be used as-is to compute the BCE without any additional processing.
# *Remark 2:* The BCE is closely related to an achieveable information rate for bit-interleaved coded modulation systems [4,5]
# 
# The next cell defines an end-to-end communication system using the neural receiver layer.
# 
# At initialization, the paramater `training` indicates if the system is instantiated to be trained (`True`) or evaluated (`False`).
# 
# If the system is instantiated to be trained, the outer encoder and decoder are not used as they are not required for training. Moreover, the estimated BCE is returned.
# This significantly reduces the computational complexity of training.
# 
# If the system is instantiated to be evaluated, the outer encoder and decoder are used, and the transmited information and corresponding LLRs are returned.

# In[4]:


class OFDMSystemNeuralReceiver(Model): # Inherits from Keras Model

    def __init__(self, training):
        super().__init__() # Must call the Keras model initializer

        self.training = training

        n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL) # Number of coded bits
        k = int(n*CODERATE) # Number of information bits
        self.k = k
        self.n = n

        # The binary source will create batches of information bits
        self.binary_source = sn.phy.mapping.BinarySource()

        # The encoder maps information bits to coded bits
        self.encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k, n)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = sn.phy.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = sn.phy.ofdm.ResourceGridMapper(RESOURCE_GRID)

        # Frequency domain channel
        self.channel = sn.phy.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=False)

        # Neural receiver
        self.neural_receiver = NeuralReceiver()

        # Used to extract data-carrying resource elements
        self.rg_demapper = sn.phy.ofdm.ResourceGridDemapper(RESOURCE_GRID, STREAM_MANAGEMENT)

        # The decoder provides hard-decisions on the information bits
        self.decoder = sn.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

        # Loss function
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function

    @tf.function # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=CODERATE, resource_grid=RESOURCE_GRID)

        # The neural receiver is expected no to have shape [batch_size].
        if len(no.shape) == 0:
            no = tf.fill([batch_size], no)

        # Transmitter
        # Outer coding is only performed if not training
        if self.training:
            codewords = self.binary_source([batch_size, NUM_UT, NUM_UT_ANT, self.n])
        else:
            bits = self.binary_source([batch_size, NUM_UT, NUM_UT_ANT, self.k])
            codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)

        # Channel
        y = self.channel(x_rg, no)

        # Receiver
        llr = self.neural_receiver(y, no)
        llr = self.rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
        llr = tf.reshape(llr, [batch_size, NUM_UT, NUM_UT_ANT, self.n]) # Reshape the LLRs to fit what the outer decoder is expected
        if self.training:
            loss = self.bce(codewords, llr)
            return loss
        else:
            bits_hat = self.decoder(llr)
            return bits, bits_hat


# ## Training the Neural Receiver

# The next cell implements a training loop of `NUM_TRAINING_ITERATIONS` iterations.
# 
# At each iteration:
# - A batch of SNRs $E_b/N_0$ is sampled
# - A forward pass through the end-to-end system is performed within a gradient tape
# - The gradients are computed using the gradient tape, and applied using the Adam optimizer
# - A progress bar is periodically updated to follow the progress of training
# 
# After training, the weights of the models are saved in a file using [pickle](https://docs.python.org/3/library/pickle.html).
# 
# Executing the next cell will take quite a while. If you do not want to train your own neural receiver, you can download the weights [here](https://drive.google.com/file/d/15txi7jAgSYeg8ylx5BAygYnywcGFw9WH/view?usp=sharing) and use them later on.

# In[5]:


train = False # Chane to train your own model
if train :
    # Number of iterations used for training
    NUM_TRAINING_ITERATIONS = 100000

    # Instantiating the end-to-end model for training
    model = OFDMSystemNeuralReceiver(training=True)

    # Adam optimizer (SGD variant)
    optimizer = tf.keras.optimizers.Adam()

    # Training loop
    for i in range(NUM_TRAINING_ITERATIONS):
        # Sample a batch of SNRs.
        ebno_db = tf.random.uniform(shape=[BATCH_SIZE], minval=EBN0_DB_MIN, maxval=EBN0_DB_MAX)
        # Forward pass
        with tf.GradientTape() as tape:
            loss = model(BATCH_SIZE, ebno_db)
        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Print progress
        if i % 100 == 0:
            print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")
    # Save the weightsin a file
    weights = model.get_weights()
    with open('weights-ofdm-neuralrx', 'wb') as f:
        pickle.dump(weights, f)


# ## Benchmarking the Neural Receiver

# We evaluate the trained model and benchmark it against the previously introduced baselines.
# 
# We first define and evaluate the baselines.

# In[6]:


class OFDMSystem(Model): # Inherits from Keras Model

    def __init__(self, perfect_csi):
        super().__init__() # Must call the Keras model initializer

        self.perfect_csi = perfect_csi

        n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL) # Number of coded bits
        k = int(n*CODERATE) # Number of information bits
        self.k = k

        # The binary source will create batches of information bits
        self.binary_source = sn.phy.mapping.BinarySource()

        # The encoder maps information bits to coded bits
        self.encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k, n)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = sn.phy.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = sn.phy.ofdm.ResourceGridMapper(RESOURCE_GRID)

        # Frequency domain channel
        self.channel = sn.phy.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_est = sn.phy.ofdm.LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = sn.phy.ofdm.LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)

        # The demapper produces LLR for all coded bits
        self.demapper = sn.phy.mapping.Demapper("app", "qam", NUM_BITS_PER_SYMBOL)

        # The decoder provides hard-decisions on the information bits
        self.decoder = sn.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    @tf.function # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=CODERATE, resource_grid=RESOURCE_GRID)

        # Transmitter
        bits = self.binary_source([batch_size, NUM_UT, RESOURCE_GRID.num_streams_per_tx, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)

        # Channel
        y, h_freq = self.channel(x_rg, no)

        # Receiver
        if self.perfect_csi:
            h_hat, err_var = h_freq, 0.
        else:
            h_hat, err_var = self.ls_est (y, no)
        x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        bits_hat = self.decoder(llr)

        return bits, bits_hat


# In[7]:


ber_plots = sn.phy.utils.PlotBER("Advanced neural receiver")

baseline_ls = OFDMSystem(False)
ber_plots.simulate(baseline_ls,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline: LS Estimation",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

baseline_pcsi = OFDMSystem(True)
ber_plots.simulate(baseline_pcsi,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline: Perfect CSI",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);


# We then instantiate and evaluate the end-to-end system equipped with the neural receiver.

# In[8]:


# Instantiating the end-to-end model for evaluation
model_neuralrx = OFDMSystemNeuralReceiver(training=False)

# Run one inference to build the layers and loading the weights
model_neuralrx(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
with open('weights-ofdm-neuralrx', 'rb') as f:
    weights = pickle.load(f)
    model_neuralrx.set_weights(weights)


# In[9]:


# Computing and plotting BER
ber_plots.simulate(model_neuralrx,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100,
                  legend="Neural Receiver",
                  soft_estimates=True,
                  max_mc_iter=100,
                  show_fig=True);


# ## Conclusion <a class="anchor" id="Conclusion"></a>

# We hope you are excited about Sionna - there is much more to be discovered:
# 
# - TensorBoard debugging available
# - Scaling to multi-GPU simulation is simple
# - See the [available tutorials](https://nvlabs.github.io/sionna/phy/tutorials.html) for more examples
# 
# And if something is still missing - the project is open-source: you can modify, add, and extend any component at any time.

# ## References

# [1] [M. Honkala, D. Korpi and J. M. J. Huttunen, "DeepRx: Fully Convolutional Deep Learning Receiver," in IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 3925-3940, June 2021, doi: 10.1109/TWC.2021.3054520](https://ieeexplore.ieee.org/abstract/document/9345504).
# 
# [2] [F. Ait Aoudia and J. Hoydis, "End-to-end Learning for OFDM: From Neural Receivers to Pilotless Communication," in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2021.3101364](https://ieeexplore.ieee.org/abstract/document/9508784).
# 
# [3] [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
-e 
# --- End of Sionna_tutorial_part4.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Weighted Belief Propagation Decoding
# 
# This notebooks implements the *Weighted Belief Propagation* (BP) algorithm as proposed by Nachmani *et al.* in [1].
# The main idea is to leverage BP decoding by additional trainable weights that scale each outgoing variable node (VN) and check node (CN) message. These weights provide additional degrees of freedom and can be trained by stochastic gradient descent (SGD) to improve the BP performance for the given code. If all weights are initialized with *1*, the algorithm equals the *classical* BP algorithm and, thus, the concept can be seen as a generalized BP decoder.
# 
# Our main focus is to show how Sionna can lower the barrier-to-entry for state-of-the-art research.
# For this, you will investigate:
# 
# * How to implement the multi-loss BP decoding with Sionna
# * How a single scaling factor can lead to similar results
# * What happens for training of the 5G LDPC code
# 
# The setup includes the following components:
# 
# - LDPC BP Decoder
# - Gaussian LLR source
# 
# Please note that we implement a simplified version of the original algorithm consisting of two major simplifications:
# 
# 1. ) Only outgoing variable node (VN) messages are weighted. This is possible as the VN operation is linear and it would only increase the memory complexity without increasing the *expressive* power of the neural network.
# 
# 2. ) We use the same shared weights for all iterations. This can potentially influence the final performance, however, simplifies the implementation and allows to run the decoder with different number of iterations.
# 
# 
# **Note**: If you are not familiar with all-zero codeword-based simulations please have a look into the [Bit-Interleaved Coded Modulation](https://nvlabs.github.io/sionna/phy/tutorials/Bit_Interleaved_Coded_Modulation.html) example notebook first.
# 
# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Weighted BP for BCH Codes](#Weighted-BP-for-BCH-Codes)
#     * [Weights *before* Training and Simulation of BER](#Weights-before-Training-and-Simulation-of-BER)
#     * [Training](#Training)
#     * [Results](#Results)
# * [Further Experiments](#Further-Experiments)
#     * [Damped BP](#Damped-BP)
#     * [Learning the 5G LDPC Code](#Learning-the-5G-LDPC-Code)
# * [References](#References)


# ## GPU Configuration and Imports

# In[1]:


import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

import tensorflow as tf
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation

# Import required Sionna components
from sionna.phy.fec.ldpc import LDPCBPDecoder, LDPC5GEncoder, LDPC5GDecoder, WeightedBPCallback
from sionna.phy.fec.utils import GaussianPriorSource, load_parity_check_examples, llr2mi
from sionna.phy.utils import ebnodb2no, hard_decisions
from sionna.phy.utils.metrics import compute_ber
from sionna.phy.utils.plotting import PlotBER
from tensorflow.keras.losses import BinaryCrossentropy
from sionna.phy import Block

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# ## Weighted BP for BCH Codes

# First, we define the trainable model consisting of:
# 
# - LDPC BP decoder
# - Gaussian LLR source
# 
# The idea of the multi-loss function in [1] is to average the loss overall iterations, i.e., not just the final estimate is evaluated. This requires to call the BP decoder *iteration-wise* by setting `num_iter=1` and `return_state=True` such that the decoder will perform a single iteration and returns its current estimate while also providing the internal messages for the next iteration.
# 
# A few comments:
# 
# - We assume the transmission of the all-zero codeword. This allows to train and analyze the decoder without the need of an encoder. Remark: The final decoder can be used for arbitrary codewords.
# - We directly generate the channel LLRs with `GaussianPriorSource`. The equivalent LLR distribution could be achieved by transmitting the all-zero codeword over an AWGN channel with BPSK modulation.
# - For the proposed *multi-loss* [1] (i.e., the loss is averaged over all iterations), we need to access the decoders intermediate output after each iteration. This is done by calling the decoding function multiple times while setting `return_state` to True, i.e., the decoder continuous the decoding process at the last message state.
# 
# 
# The BP decoder itself does not have any trainable weights. However, the LDPCBPDecoder API allows to register custom callback functions after each VN/CN node update step. In this tutorial, we use the `WeightedBPCallback` to apply trainable weights to each exchanged internal decoder message. Similarly, offset-corrected BP can be made trainable.

# In[2]:


class WeightedBP(Block):
    """System model for BER simulations of weighted BP decoding.

    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.

    Parameters
    ----------
    pcm: ndarray
        The parity-check matrix of the code under investigation.

    num_iter: int
        Number of BP decoding iterations.

    Input
    -----
    batch_size: int or tf.int
        The batch_size used for the simulation.

    ebno_db: float or tf.float
        A float defining the simulation SNR.

    Output
    ------
    (u, u_hat, loss):
        Tuple:

    u: tf.float32
        A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.

    u_hat: tf.float32
        A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.

    loss: tf.float32
        Binary cross-entropy loss between `u` and `u_hat`.
    """

    def __init__(self, pcm, num_iter=5):
        super().__init__()

        # add trainable weights via decoder callbacks
        self.edge_weights = WeightedBPCallback(num_edges=np.sum(pcm))

        # init components
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1, # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     return_state=True, # decoder stores internal messages after call
                                     hard_out=False, # we need to access soft-information
                                     cn_type="boxplus",
                                     v2c_callbacks=[self.edge_weights])# register callback to make the decoder trainable

        # used to generate llrs during training (see example notebook on all-zero codeword trick)
        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter

        self._bce = BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=self.decoder.coderate)

        # all-zero CW to calculate loss / BER
        c = tf.zeros([batch_size, self.decoder.n])

        # Gaussian LLR source
        llr = self.llr_source([batch_size, self.decoder.n], noise_var)

        # --- implement multi-loss as proposed by Nachmani et al. [1]---
        loss = 0
        msg_v2c = None # internal state of decoder
        for i in range(self._num_iter):
            c_hat, msg_v2c = self.decoder(llr, msg_v2c=msg_v2c) # perform one decoding iteration; decoder returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration

        loss /= self._num_iter # scale loss by number of iterations

        return c, c_hat, loss


# Load a parity-check matrix used for the experiment. We use the same BCH(63,45) code as in [1].
# The code can be replaced by any parity-check matrix of your choice. 

# In[3]:


pcm_id = 1 # (63,45) BCH code parity check matrix
pcm, k , n, coderate = load_parity_check_examples(pcm_id=pcm_id, verbose=True)

num_iter = 10 # set number of decoding iterations

# and initialize the model
model = WeightedBP(pcm=pcm, num_iter=num_iter)


# **Note**: weighted BP tends to work better for small number of iterations.
# The effective gains (compared to the baseline with same number of iterations)
# vanish with more iterations.

# ### Weights *before* Training and Simulation of BER
# 
# Let us plot the weights after initialization of the decoder to verify that everything is properly initialized.
# This is equivalent the *classical* BP decoder.

# In[4]:


# count number of weights/edges
print("Total number of weights: ", np.size(model.edge_weights.weights))

# and show the weight distribution
model.edge_weights.show_weights()


# We first simulate (and store) the BER performance *before* training.
# For this, we use the `PlotBER` class, which provides a convenient way to store the results for later comparison.

# In[5]:


# SNR to simulate the results
ebno_dbs = np.array(np.arange(1, 7, 0.5))
mc_iters = 100 # number of Monte Carlo iterations

# we generate a new PlotBER() object to simulate, store and plot the BER results
ber_plot = PlotBER("Weighted BP")

# simulate and plot the BER curve of the untrained decoder
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend="Untrained",
                  soft_estimates=True,
                  max_mc_iter=mc_iters,
                  forward_keyboard_interrupt=False,
                  graph_mode="graph");


# ### Training
# We now train the model for a fixed number of SGD training iterations.
# 
# **Note**: this is a very basic implementation of the training loop.
# You can also try more sophisticated training loops with early stopping, different hyper-parameters or optimizers etc. 

# In[6]:


# training parameters
batch_size = 1000
train_iter = 200
ebno_db = 4.0
clip_value_grad = 10 # gradient clipping for stable training convergence

# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        b, llr, loss = model(batch_size, ebno_db)
    grads = tape.gradient(loss, tape.watched_variables())
    grads = [tf.clip_by_value(g, -clip_value_grad, clip_value_grad, name=None) for g in grads]
    optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    return loss, b, llr

for it in range(0, train_iter):
    loss, b, llr = train_step()

    # calculate and print intermediate metrics
    # only for information
    # this has no impact on the training
    if it%10==0: # evaluate every 10 iterations
        # calculate ber from received LLRs
        b_hat = hard_decisions(llr) # hard decided LLRs first
        ber = compute_ber(b, b_hat)
        # and print results
        mi = llr2mi(llr, -2*b+1).numpy() # calculate bit-wise mutual information
        l = loss.numpy() # copy loss to numpy for printing
        print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())


# ### Results
# 
# After training, the weights of the decoder have changed.
# In average, the weights are smaller after training.

# In[7]:


model.edge_weights.show_weights() # show weights AFTER training


# And let us compare the new BER performance.
# For this, we can simply call the ber_plot.simulate() function again as it internally stores all previous results (if `add_results` is True).

# In[8]:


ebno_dbs = np.array(np.arange(1, 7, 0.5))
batch_size = 10000
mc_ites = 100

ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend="Trained",
                  max_mc_iter=mc_iters,
                  soft_estimates=True,
                  graph_mode="graph");


# ## Further Experiments
# 
# You will now see that the memory footprint can be drastically reduced by using the same weight for all messages.
# In the second part we will apply the concept to the 5G LDPC codes.

# ### Damped BP 
# 
# It is well-known that scaling of LLRs / messages can help to improve the performance of BP decoding in some scenarios [3,4].
# In particular, this works well for very short codes such as the code we are currently analyzing.
# 
# We now follow the basic idea of [2] and scale all weights with the same scalar.

# In[9]:


# get weights of trained model
weights_bp = model.edge_weights.weights

# calc mean value of weights
damping_factor = tf.reduce_mean(weights_bp)

# set all weights to the SAME constant scaling
weights_damped = tf.ones_like(weights_bp) * damping_factor

# and apply the new weights
model.edge_weights.weights.assign(weights_damped)

# let us have look at the new weights again
model.edge_weights.show_weights()

# and simulate the BER again
leg_str = f"Damped BP (scaling factor {damping_factor.numpy():.3f})"
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend=leg_str,
                  max_mc_iter=mc_iters,
                  soft_estimates=True,
                  graph_mode="graph");


# When looking at the results, we observe almost the same performance although we only scale by a single scalar.
# This implies that the number of weights of our model is by far too large and the memory footprint could be reduced significantly.
# However, isn't it fascinating to see that this simple concept of weighted BP leads to the same results as the concept of *damped BP*?
# 
# **Note**: for more iterations it could be beneficial to implement an individual damping per iteration.

# ### Learning the 5G LDPC Code
# 
# In this Section, you will experience what happens if we apply the same concept to the 5G LDPC code (including rate matching).
# 
# For this, we need to define a new model.

# In[10]:


class WeightedBP5G(Block):
    """System model for BER simulations of weighted BP decoding for 5G LDPC codes.

    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.

    Parameters
    ----------
    k: int
        Number of information bits per codeword.

    n: int
        Codeword length.

    num_iter: int
        Number of BP decoding iterations.

    Input
    -----
    batch_size: int or tf.int
        The batch_size used for the simulation.

    ebno_db: float or tf.float
        A float defining the simulation SNR.

    Output
    ------
    (u, u_hat, loss):
        Tuple:

    u: tf.float32
        A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.

    u_hat: tf.float32
        A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.

    loss: tf.float32
        Binary cross-entropy loss between `u` and `u_hat`.
    """
    def __init__(self, k, n, num_iter=20):
        super().__init__()

        # we need to initialize an encoder for the 5G parameters
        self.encoder = LDPC5GEncoder(k, n)

        # add trainable weights via decoder callbacks
        self.edge_weights = WeightedBPCallback(
                            num_edges=int(np.sum(self.encoder.pcm)))

        self.decoder = LDPC5GDecoder(self.encoder,
                                     num_iter=1, # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     return_state=True,
                                     hard_out=False,
                                     prune_pcm=False,
                                     cn_type="boxplus",
                                     v2c_callbacks=[self.edge_weights,]) # register callback

        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter
        self._coderate = k/n

        self._bce = BinaryCrossentropy(from_logits=True)
    def call(self, batch_size, ebno_db):

        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=self._coderate)

        # BPSK modulated all-zero CW
        c = tf.zeros([batch_size, k]) # decoder only returns info bits

        # use fake llrs from GA
        # works as BP is symmetric
        llr = self.llr_source([batch_size, n], noise_var)

        # --- implement multi-loss is proposed by Nachmani et al. ---
        loss = 0
        msg_v2c = None
        for i in range(self._num_iter):
            c_hat, msg_v2c = self.decoder(llr, msg_v2c=msg_v2c) # perform one decoding iteration; decoder returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration

        return c, c_hat, loss


# In[11]:


# generate model
num_iter = 10
k = 400
n = 800

model5G = WeightedBP5G(k, n, num_iter=num_iter)

# generate baseline BER
ebno_dbs = np.array(np.arange(0, 4, 0.25))
mc_iters = 100 # number of monte carlo iterations
ber_plot_5G = PlotBER("Weighted BP for 5G LDPC")

# simulate the untrained performance
ber_plot_5G.simulate(model5G,
                     ebno_dbs=ebno_dbs,
                     batch_size=1000,
                     num_target_bit_errors=2000, # stop sim after 2000 bit errors
                     legend="Untrained",
                     soft_estimates=True,
                     max_mc_iter=mc_iters,
                     graph_mode="graph");


# And let's train this new model.

# In[12]:


# training parameters
batch_size = 1000
train_iter = 200
clip_value_grad = 10 # gradient clipping seems to be important

# smaller training SNR as the new code is longer (=stronger) than before
ebno_db = 1.5 # rule of thumb: train at ber = 1e-2

# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        b, llr, loss = model5G(batch_size, ebno_db)
    grads = tape.gradient(loss, tape.watched_variables())
    grads = [tf.clip_by_value(g, -clip_value_grad, clip_value_grad, name=None) for g in grads]
    optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    return loss, b, llr

# and let's go
for it in range(0, train_iter):
    loss, b, llr = train_step()

    # calculate and print intermediate metrics
    if it%10==0:
        # calculate ber
        b_hat = hard_decisions(llr)
        ber = compute_ber(b, b_hat)
        # and print results
        mi = llr2mi(llr, -2*b+1).numpy() # calculate bit-wise mutual information
        l = loss.numpy()
        print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())


# We now simulate the new results and compare it to the untrained results.

# In[13]:


ebno_dbs = np.array(np.arange(0, 4, 0.25))
batch_size = 1000
mc_iters = 100

ber_plot_5G.simulate(model5G,
                     ebno_dbs=ebno_dbs,
                     batch_size=batch_size,
                     num_target_bit_errors=2000, # stop sim after 2000 bit errors
                     legend="Trained",
                     max_mc_iter=mc_iters,
                     soft_estimates=True,
                     graph_mode="graph");


# Unfortunately, we observe only very minor gains for the 5G LDPC code. We empirically observed that gain vanishes for more iterations and longer codewords, i.e., for most practical use-cases of the 5G LDPC code the gains are only minor.
# 
# However, there may be other `codes on graphs` that benefit from the principle idea of weighted BP - or other channel setups? Feel free to adjust this notebook and train for your favorite code / channel.
# 
# Other ideas for own experiments:
# 
# - Implement weighted BP with unique weights per iteration.
# - Apply the concept to (scaled) min-sum decoding as in [5].
# - Can you replace the complete CN update by a neural network?
# - Verify the results from all-zero simulations for a *real* system simulation with explicit encoder and random data
# - What happens in combination with higher order modulation?

# ## References
# 
# [1] E. Nachmani, Y. Be’ery and D. Burshtein, "Learning to Decode Linear Codes Using Deep Learning,"
# IEEE Annual Allerton Conference on Communication, Control, and Computing (Allerton), pp. 341-346., 2016. https://arxiv.org/pdf/1607.04793.pdf
# 
# [2] M. Lian, C. Häger, and H. Pfister, "What can machine learning teach us about communications?" IEEE Information Theory Workshop (ITW), pp. 1-5. 2018.
# 
# [3] ] M. Pretti, “A message passing algorithm with damping,” J. Statist. Mech.: Theory Practice, p. 11008, Nov. 2005.
# 
# [4] J.S. Yedidia, W.T. Freeman and Y. Weiss, "Constructing free energy approximations and Generalized Belief Propagation algorithms," IEEE Transactions on Information Theory, 2005.
# 
# [5] E. Nachmani, E. Marciano, L. Lugosch, W. Gross, D. Burshtein and Y. Be’ery, "Deep learning methods for improved decoding of linear codes," IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp.119-131, 2018.
-e 
# --- End of Weighted_BP_Algorithm.ipynb ---

