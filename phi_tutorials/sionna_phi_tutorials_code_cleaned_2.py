
# In this notebook, you will learn how to create a channel model from a [generator](https://wiki.python.org/moin/Generators). This can be used, e.g., to import datasets of channel impulse responses.

# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Simulation Parameters](#Simulation-Parameters)
# * [Creating a Simple Dataset](#Creating-a-Simple-Dataset)
# * [Generators](#Generators)
# * [Use the Channel Model for OFDM Transmissions](#Use-the-Channel-Model-for-OFDM-Transmissions)

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
import h5py

config = sionna.phy.config
config.seed = 42 # Set seed for reproducible random number generation


# ## Simulation Parameters

# In[2]:


num_rx = 2
num_rx_ant = 2
num_tx = 1
num_tx_ant = 8
num_time_steps = 100
num_paths = 10


# ## Creating a Simple Dataset

# To illustrate how to load dataset, we will first create one.
# 
# The next cell creates a very small HDF5 file storing Gaussian distributed i.i.d. path coefficients and uniformly distributed i.i.d. path delays.

# In[3]:


# Number of examples in the dataset
dataset_size = 1000

# Random path coefficients
a_shape = [dataset_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
a = (config.np_rng.normal(size=a_shape) + 1j*config.np_rng.normal(size=a_shape))/np.sqrt(2)

# Random path delays
tau = config.np_rng.uniform(size=[dataset_size, num_rx, num_tx, num_paths])


# In[4]:


filename = 'my_dataset.h5'
hf = h5py.File(filename, 'w')
hf.create_dataset('a', data=a)
hf.create_dataset('tau', data=tau)
hf.close()


# ## Generators

# The first step to load a dataset is to create a [generator](https://wiki.python.org/moin/Generators).
# A generator is a callable object, i.e., a function or a class that implements the `__call__()` method, and that behaves like an iterator.
# 
# The next cell shows how to create a generator that parses an HDF5 file storing path coefficients and delays.
# Note that how the HDF5 file is parsed depends on its structure. The following generator is specific to the dataset previously created.
# 
# If you have another dataset, you will need to change the way it is parsed in the generator. The generator can also carry out any type of desired pre-processing of your data, e.g., normalization.

# In[5]:


class HD5CIRGen:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self):
        with h5py.File(self.filename, 'r') as hf:
            for im in zip(hf["a"], hf["tau"]):
                a = im[0]
                tau = im[1]
                # One could do some preprocessing on the dataset here
                # ...
                yield im


# In[6]:


generator = HD5CIRGen(filename)


# We can use the generator to sample the first 5 items of the dataset:

# In[7]:


i = 0
for (a,tau) in generator():
    print(a.shape)
    print(tau.shape)
    i = i + 1
    if i == 5:
        break


# Let us create a channel model from this dataset:

# In[8]:


from sionna.phy.channel import CIRDataset

batch_size = 64 # The batch_size cannot be changed after the creation of the channel model
channel_model = CIRDataset(generator,
                           batch_size,
                           num_rx,
                           num_rx_ant,
                           num_tx,
                           num_tx_ant,
                           num_paths,
                           num_time_steps)


# We can now sample from this dataset in the same way as we would from a stochastic channel model:

# In[9]:


# Note that the arguments batch_size, num_time_steps, and smapling_frequency
# of the __call__ function are ignored as they are already specified by the dataset.
a, tau = channel_model()


# In[10]:


print(a.shape)
print(a.dtype)
print(tau.shape)
print(tau.dtype)


# ## Use the Channel Model for OFDM Transmissions
# 
# The following code demonstrates how you can use the channel model to generate channel frequency responses that can be used for the simulation of communication system based on OFDM.

# In[11]:


# Create an OFDM resource grid
# Each time step is assumed to correspond to one OFDM symbol over which it is constant.
resource_grid = sionna.phy.ofdm.ResourceGrid(
                                num_ofdm_symbols=num_time_steps,
                                fft_size=76,
                                subcarrier_spacing=15e3,
                                num_tx=num_tx,
                                num_streams_per_tx=num_tx_ant)


# In[12]:


ofdm_channel = sionna.phy.channel.GenerateOFDMChannel(channel_model, resource_grid)


# In[13]:


# Generate a batch of frequency responses
# Shape: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
h_freq = ofdm_channel()
print(h_freq.shape)


# In[14]:


# Delete dataset
get_ipython().run_line_magic('rm', 'my_dataset.h5')

-e 
# --- End of CIR_Dataset.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Using the DeepMIMO Dataset with Sionna
# 
# In this example, you will learn how to use the ray-tracing based DeepMIMO dataset.

# [DeepMIMO](https://deepmimo.net/) is a generic dataset that enables a wide range of machine/deep learning applications for MIMO systems. It takes as input a set of parameters (such as antenna array configurations and time-domain/OFDM parameters) and generates MIMO channel realizations, corresponding locations, angles of arrival/departure, etc., based on these parameters and on a ray-tracing scenario selected [from those available in DeepMIMO](https://deepmimo.net/scenarios/).
# 
# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Configuration of DeepMIMO](#Configuration-of-DeepMIMO)
# * [Using DeepMIMO with Sionna](#Using-DeepMIMO-with-Sionna)
# * [Link-level Simulations using Sionna and DeepMIMO](#Link-level-Simulations-using-Sionna-and-DeepMIMO)
# * [DeepMIMO License and Citation](#DeepMIMO-License-and-Citation)

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

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
                            LMMSEEqualizer, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.channel import subcarrier_frequencies, ApplyOFDMChannel, \
                               GenerateOFDMChannel, CIRDataset
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import ebnodb2no, sim_ber


# ## Configuration of DeepMIMO

# DeepMIMO provides multiple [scenarios](https://deepmimo.net/scenarios/) that one can select from. In this example, we use the O1 scenario with the carrier frequency set to 60 GHz (O1_60). To run this example, please download the "O1_60" data files [from this page](https://deepmimo.net/scenarios/o1-scenario/). The downloaded zip file should be extracted into a folder, and the parameter `DeepMIMO_params['dataset_folder']` should be set to point to this folder, as done below.
# 
# To use DeepMIMO with Sionna, the DeepMIMO dataset first needs to be generated. The generated DeepMIMO dataset contains channels  for different locations of the users and basestations. The layout of the O1 scenario is shown in the figure below. 
# 
# 
# In this example, we generate a dataset that consists of channels for the links from the basestation 6 to the users located on the rows 400 to 450. Each of these rows consists of 181 user locations, resulting in $51 \times 181 = 9231$ basestation-user channels.
# 
# The antenna arrays in the DeepMIMO dataset are defined through the x-y-z axes. In the following example, a single-user MISO downlink is considered. The basestation is equipped with a uniform linear array of 16 elements spread along the x-axis. The users are each equipped with a single antenna. These parameters can be configured using the code below (for more information about the DeepMIMO parameters, please check [the DeepMIMO configurations](https://deepmimo.net/versions/v2-python/)).

# In[3]:


# Import DeepMIMO
try:
    import DeepMIMO
except ImportError as e:
    # Install DeepMIMO if package is not already installed
    import os
    os.system("pip install DeepMIMO")
    import DeepMIMO

# Channel generation
DeepMIMO_params = DeepMIMO.default_params() # Load the default parameters
DeepMIMO_params['dataset_folder'] = r'./scenarios' # Path to the downloaded scenarios
DeepMIMO_params['scenario'] = 'O1_60' # DeepMIMO scenario
DeepMIMO_params['num_paths'] = 10 # Maximum number of paths
DeepMIMO_params['active_BS'] = np.array([6]) # Basestation indices to be included in the dataset

# Selected rows of users, whose channels are to be generated.
DeepMIMO_params['user_row_first'] = 400 # First user row to be included in the dataset
DeepMIMO_params['user_row_last'] = 450 # Last user row to be included in the dataset

# Configuration of the antenna arrays
DeepMIMO_params['bs_antenna']['shape'] = np.array([16, 1, 1]) # BS antenna shape through [x, y, z] axes
DeepMIMO_params['ue_antenna']['shape'] = np.array([1, 1, 1]) # UE antenna shape through [x, y, z] axes

# The OFDM_channels parameter allows choosing between the generation of channel impulse
# responses (if set to 0) or frequency domain channels (if set to 1).
# It is set to 0 for this simulation, as the channel responses in frequency domain
# will be generated using Sionna.
DeepMIMO_params['OFDM_channels'] = 0

# Generates a DeepMIMO dataset
DeepMIMO_dataset = DeepMIMO.generate_data(DeepMIMO_params)


# ### Visualization of the dataset
# 
# To provide a better understanding of the user and basestation locations, we next visualize the locations of the users, highlighting the first active row of users (row 400), and basestation 6.

# In[4]:


plt.figure(figsize=(12,8))

## User locations
active_bs_idx = 0 # Select the first active basestation in the dataset
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 1], # y-axis location of the users
         DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 0], # x-axis location of the users
         s=1, marker='x', c='C0', label='The users located on the rows %i to %i (R%i to R%i)'%
           (DeepMIMO_params['user_row_first'], DeepMIMO_params['user_row_last'],
           DeepMIMO_params['user_row_first'], DeepMIMO_params['user_row_last']))
# First 181 users correspond to the first row
plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 1],
         DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 0],
         s=1, marker='x', c='C1', label='First row of users (R%i)'% (DeepMIMO_params['user_row_first']))

## Basestation location
plt.scatter(DeepMIMO_dataset[active_bs_idx]['location'][1],
         DeepMIMO_dataset[active_bs_idx]['location'][0],
         s=50.0, marker='o', c='C2', label='Basestation')

plt.gca().invert_xaxis() # Invert the x-axis to align the figure with the figure above
plt.ylabel('x-axis')
plt.xlabel('y-axis')
plt.grid()
plt.legend();


# ## Using DeepMIMO with Sionna
# 
# The DeepMIMO Python package provides [a Sionna-compliant channel impulse response generator](https://nvlabs.github.io/sionna/phy/tutorials/CIR_Dataset.html#Generators) that adapts the structure of the DeepMIMO dataset to be consistent with Sionna.
# 
# An adapter is instantiated for a given DeepMIMO dataset. In addition to the dataset, the adapter takes the indices of the basestations and users, to generate the channels between these basestations and users:
# 
# `DeepMIMOSionnaAdapter(DeepMIMO_dataset, bs_idx, ue_idx)`
# 
# 
# **Note:** `bs_idx` and `ue_idx` set the links from which the channels are drawn. For instance, if `bs_idx = [0, 1]` and `ue_idx = [2, 3]`, the adapter then outputs the 4 channels formed by the combination of the first and second basestations with the third and fourth users.
# 
# The default behavior for `bs_idx` and `ue_idx` are defined as follows:
# - If value for `bs_idx` is not given, it will be set to `[0]` (i.e., the first basestation in the `DeepMIMO_dataset`).
# - If value for `ue_idx` is not given, then channels are provided for the links between the `bs_idx` and all users (i.e., `ue_idx=range(len(DeepMIMO_dataset[0]['user']['channel']))`.
# - If the both `bs_idx` and `ue_idx` are not given, the channels between the first basestation and all the users are provided by the adapter. For this example, `DeepMIMOSionnaAdapter(DeepMIMO_dataset)` returns the channels from the basestation 6 and the 9231 available user locations.
# 
# **Note:** The adapter assumes basestations are transmitters and users are receivers. Uplink channels can be obtained using (transpose) reciprocity.
# 
# ### Random Sampling of Multi-User Channels
# 
# 
# 
# In order to randomly sample channels from all the available user locations considering `num_rx` users, one may set `ue_idx` as in the following cell. In this example, the channels will be randomly chosen from the links between the basestation 6 and the 9231 available user locations.

# In[5]:


from DeepMIMO import DeepMIMOSionnaAdapter

# Number of receivers for the Sionna model.
# MISO is considered here.
num_rx = 1

# The number of UE locations in the generated DeepMIMO dataset
num_ue_locations = len(DeepMIMO_dataset[0]['user']['channel']) # 9231
# Pick the largest possible number of user locations that is a multiple of ``num_rx``
ue_idx = np.arange(num_rx*(num_ue_locations//num_rx))
# Optionally shuffle the dataset to not select only users that are near each others
np.random.shuffle(ue_idx)
# Reshape to fit the requested number of users
ue_idx = np.reshape(ue_idx, [-1, num_rx]) # In the shape of (floor(9231/num_rx) x num_rx)

DeepMIMO_Sionna_adapter = DeepMIMOSionnaAdapter(DeepMIMO_dataset, ue_idx=ue_idx)


# ## Link-level Simulations using Sionna and DeepMIMO
# 
# In the following cell, we define a Sionna model implementing the end-to-end link.
# 
# **Note:** The Sionna CIRDataset object shuffles the DeepMIMO channels provided by the adapter. Therefore, channel samples are passed through the model in a random order.

# In[6]:


class LinkModel(Block):
    def __init__(self,
                 DeepMIMO_Sionna_adapter,
                 carrier_frequency,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing = 60e3,
                 batch_size = 64
                ):
        super().__init__()

        self._batch_size = batch_size
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # CIRDataset to parse the dataset
        self._CIR = CIRDataset(DeepMIMO_Sionna_adapter,
                               self._batch_size,
                               DeepMIMO_Sionna_adapter.num_rx,
                               DeepMIMO_Sionna_adapter.num_rx_ant,
                               DeepMIMO_Sionna_adapter.num_tx,
                               DeepMIMO_Sionna_adapter.num_tx_ant,
                               DeepMIMO_Sionna_adapter.num_paths,
                               DeepMIMO_Sionna_adapter.num_time_steps)

        # System parameters
        self._carrier_frequency = carrier_frequency
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 76
        self._num_ofdm_symbols = 14
        self._num_streams_per_tx = DeepMIMO_Sionna_adapter.num_rx
        self._dc_null = False
        self._num_guard_carriers = [0, 0]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 4
        self._coderate = 0.5

        # Setup the OFDM resource grid and stream management
        self._sm = StreamManagement(np.ones([DeepMIMO_Sionna_adapter.num_rx, 1], int), self._num_streams_per_tx)
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=DeepMIMO_Sionna_adapter.num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        # Components forming the link

        # Codeword length
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        # Number of information bits per codeword
        self._k = int(self._n * self._coderate)

        # OFDM channel
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._ofdm_channel = GenerateOFDMChannel(self._CIR, self._rg, normalize_channel=True)
        self._channel_freq = ApplyOFDMChannel(add_awgn=True)

        # Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._zf_precoder = RZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        # Receiver
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="lin_time_avg")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

    def call(self, batch_size, ebno_db):

        # Transmitter
        b = self._binary_source([self._batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        # Generate the OFDM channel
        h_freq = self._ofdm_channel()
        # Precoding
        x_rg, g = self._zf_precoder(x_rg, h_freq)

        # Apply OFDM channel
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        y = self._channel_freq(x_rg, h_freq, no)

        # Receiver
        h_hat, err_var = self._ls_est (y, no)
        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)

        return b, b_hat


# We next evaluate the setup with different $E_b/N_0$ values to obtain BLER curves.

# In[7]:


sim_params = {
              "ebno_db": np.linspace(-7, -5.25, 10),
              "cyclic_prefix_length" : 0,
              "pilot_ofdm_symbol_indices" : [2, 11],
              }
batch_size = 64
model = LinkModel(DeepMIMO_Sionna_adapter=DeepMIMO_Sionna_adapter,
                  carrier_frequency=DeepMIMO_params['scenario_params']['carrier_freq'],
                  cyclic_prefix_length=sim_params["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=sim_params["pilot_ofdm_symbol_indices"])
ber, bler = sim_ber(model,
                    sim_params["ebno_db"],
                    batch_size=batch_size,
                    max_mc_iter=100,
                    num_target_block_errors=100,
                    graph_mode="graph")


# In[8]:


plt.figure(figsize=(12,8))
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.semilogy(sim_params["ebno_db"], bler)


# ## DeepMIMO License and Citation
# 
# A. Alkhateeb, “[DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications](https://arxiv.org/pdf/1902.06435.pdf),” in Proc. of Information Theory and Applications Workshop (ITA), San Diego, CA, Feb. 2019.
# 
# To use the DeepMIMO dataset, please check the license information [here](https://deepmimo.net/license/).
-e 
# --- End of DeepMIMO.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# ## Discover Sionna
# 
# This example notebook will guide you through the basic principles and illustrates the key features of [Sionna](https://nvlabs.github.io/sionna).
# With only a few commands, you can simulate the PHY-layer link-level performance for many 5G-compliant components, including easy visualization of the results.
# 

# ### Load Required Packages
# 
# The Sionna python package must be [installed](https://nvlabs.github.io/sionna/installation.html).

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

import numpy as np
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# IPython "magic function" for inline plots
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# **Tip**: you can run bash commands in Jupyter via the `!` operator.

# In[2]:


get_ipython().system('nvidia-smi')


# In case multiple GPUs are available, we restrict this notebook to single-GPU usage. You can ignore this command if only one GPU is available.
# 
# Further, we want to avoid that this notebook instantiates the whole GPU memory when initialized and set `memory_growth` as active.
# 
# *Remark*: Sionna does not require a GPU. Everything can also run on your CPU - but you may need to wait a little longer.

# In[3]:


# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Index of the GPU to be used
    try:
        #tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


# ### Sionna Data-flow and Design Paradigms
# 
# Sionna inherently parallelizes simulations via *batching*, i.e., each element in the batch dimension is simulated independently.
# 
# This means the first tensor dimension is always used for *inter-frame* parallelization similar to an outer *for-loop* in Matlab/NumPy simulations.
# 
# To keep the dataflow efficient, Sionna follows a few simple design principles:
# 
# * Signal-processing components are implemented as individual [Sionna Blocks](https://nvlabs.github.io/sionna/phy/api/developers.html#sionna.phy.Block). 
# * `tf.float32` is used as preferred datatype and `tf.complex64` for complex-valued datatypes, respectively.  
# This allows simpler re-use of components (e.g., the same scrambling layer can be used for binary inputs and LLR-values).
# * Models can be developed in *eager mode* allowing simple (and fast) modification of system parameters.
# * Number crunching simulations can be executed in the faster *graph mode* or even *XLA* acceleration is available for most components.
# * Whenever possible, components are automatically differentiable via [auto-grad](https://www.tensorflow.org/guide/autodiff) to simplify the deep learning design-flow.
# * Code is structured into sub-packages for different tasks such as channel coding, mapping,... (see [API documentation](https://nvlabs.github.io/sionna/phy/api/phy.html) for details).
# 
# The division into individual blocks simplifies deployment and all blocks and functions comes with unittests to ensure their correct behavior.
# 
# These paradigms simplify the re-useability and reliability of our components for
# a wide range of communications related applications.
# 
# ### A note on random number generation
# When Sionna is loaded, it instantiates random number generators (RNGs) for [Python](https://docs.python.org/3/library/random.html#alternative-generator),
# [NumPy](https://numpy.org/doc/stable/reference/random/generator.html), and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/random/Generator). You can optionally set a seed which will make all of your
# results deterministic, as long as only these RNGs are used. In the cell below,
# you can see how this seed is set and how the different RNGs can be used.

# In[4]:


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


# ### Let's Get Started - The First Blocks (*Eager Mode*)
# 
# Every block needs to be initialized once before it can be used.
# 
# **Tip**: use the [API documentation](https://nvlabs.github.io/sionna/phy/api/phy.html) to find an overview of all existing components.
# 
# We now want to transmit some symbols over an AWGN channel.
# First, we need to initialize the corresponding block.

# In[5]:


channel = sionna.phy.channel.AWGN() # init AWGN channel block


# In this first example, we want to add Gaussian noise to some given values of `x`.
# 
# Remember - the first dimension is the *batch-dimension*.
# 
# We simulate 2 message frames each containing 4 symbols.
# 
# *Remark*: the [AWGN channel](https://nvlabs.github.io/sionna/phy/channel.wireless.html#sionna.phy.channel.AWGN) is defined to be complex-valued.

# In[6]:


# define a (complex-valued) tensor to be transmitted
x = tf.constant([[0., 1.5, 1., 0.],[-1., 0., -2, 3 ]], dtype=tf.complex64)

# let's have look at the shape
print("Shape of x: ", x.shape)
print("Values of x: ", x)


# We want to simulate the channel at an SNR of 5 dB.
# For this, we can simply *call* the previously defined block `channel`.
# 
# A Sionna block acts pretty much like a function: it has an input and returns the processed output.
# 
# *Remark*: Each time this cell is executed a new noise realization is drawn.

# In[7]:


ebno_db = 5

# calculate noise variance from given EbNo
no = sionna.phy.utils.ebnodb2no(ebno_db = ebno_db,
                                num_bits_per_symbol=2, # QPSK
                                coderate=1) 
y = channel(x, no)

print("Noisy symbols are: ", y)


# ### Batches and Multi-dimensional Tensors
# 
# Sionna natively supports multi-dimensional tensors.
# 
# Most blocks operate at the last dimension and can have arbitrary input shapes (preserved at output).
# 
# Let us assume we want to add a CRC-24 check to 64 codewords of length 500 (e.g., different CRC per sub-carrier).
# Further, we want to parallelize the simulation over a batch of 100 samples.

# In[8]:


batch_size = 100 # outer level of parallelism
num_codewords = 64 # codewords per batch sample
info_bit_length = 500 # info bits PER codeword

source = sionna.phy.mapping.BinarySource() # yields random bits

u = source([batch_size, num_codewords, info_bit_length]) # call the source layer
print("Shape of u: ", u.shape)

# initialize an CRC encoder with the standard compliant "CRC24A" polynomial
encoder_crc = sionna.phy.fec.crc.CRCEncoder("CRC24A")
decoder_crc = sionna.phy.fec.crc.CRCDecoder(encoder_crc) # connect to encoder

# add the CRC to the information bits u
c = encoder_crc(u) # returns a list [c, crc_valid]
print("Shape of c: ", c.shape)
print("Processed bits: ", np.size(c.numpy()))

# we can also verify the results
# returns list of [info bits without CRC bits, indicator if CRC holds]
u_hat, crc_valid = decoder_crc(c) 
print("Shape of u_hat: ", u_hat.shape)
print("Shape of crc_valid: ", crc_valid.shape)

print("Valid CRC check of first codeword: ", crc_valid.numpy()[0,0,0])


# We want to do another simulation but for 5 independent users.
# 
# Instead of defining 5 different tensors, we can simply add another dimension.

# In[9]:


num_users = 5

u = source([batch_size, num_users, num_codewords, info_bit_length]) 
print("New shape of u: ", u.shape)

# We can re-use the same encoder as before
c = encoder_crc(u)
print("New shape of c: ", c.shape)
print("Processed bits: ", np.size(c.numpy()))


# Often a good visualization of results helps to get new research ideas.
# Thus, Sionna has built-in plotting functions.
# 
# Let's have look at a 16-QAM constellation.

# In[10]:


constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol=4)
constellation.show();


# ### First Link-level Simulation
# 
# We can already build powerful code with a few simple commands.
# 
# As mentioned earlier, Sionna aims at hiding system complexity into blocks.
# However, we still want to provide as much flexibility as possible.
# Thus, most blocks have several choices of init parameters, but often the default choice is a good start.
# 
# **Tip**: the [API documentation](https://nvlabs.github.io/sionna/phy/api/phy.html) provides many helpful references and implementation details.

# In[11]:


# system parameters
n_ldpc = 500 # LDPC codeword length 
k_ldpc = 250 # number of info bits per LDPC codeword
coderate = k_ldpc / n_ldpc
num_bits_per_symbol = 4 # number of bits mapped to one symbol (cf. QAM)


# Often, several different algorithms are implemented, e.g., the demapper supports  *"true app"* demapping, but also *"max-log"* demapping.
# 
# The check-node (CN) update function of the LDPC BP decoder also supports multiple algorithms.

# In[12]:


demapping_method = "app" # try "max-log"
cn_update = "boxplus" # try "boxplus-phy"


# Let us initialize all required components for the given system parameters.

# In[13]:


binary_source = sionna.phy.mapping.BinarySource()
encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k_ldpc, n_ldpc)
constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol)
mapper = sionna.phy.mapping.Mapper(constellation=constellation)
channel = sionna.phy.channel.AWGN()
demapper = sionna.phy.mapping.Demapper(demapping_method,
                                       constellation=constellation)
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder,
                                            hard_out=True, cn_update=cn_update,
                                            num_iter=20)


# We can now run the code in *eager mode*. This allows us to modify the structure at any time - you can try a different `batch_size` or a different SNR `ebno_db`.

# In[14]:


# simulation parameters
batch_size = 1000
ebno_db = 4

# Generate a batch of random bit vectors
b = binary_source([batch_size, k_ldpc])

# Encode the bits using 5G LDPC code
print("Shape before encoding: ", b.shape)
c = encoder(b)
print("Shape after encoding: ", c.shape)

# Map bits to constellation symbols
x = mapper(c)
print("Shape after mapping: ", x.shape)

# Transmit over an AWGN channel at SNR 'ebno_db'
no = sionna.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
y = channel(x, no)
print("Shape after channel: ", y.shape)

# Demap to LLRs
llr = demapper(y, no)
print("Shape after demapping: ", llr.shape)

# LDPC decoding using 20 BP iterations
b_hat = decoder(llr)
print("Shape after decoding: ", b_hat.shape)

# calculate BERs
c_hat = tf.cast(tf.less(0.0, llr), tf.float32) # hard-decided bits before dec.
ber_uncoded = sionna.phy.utils.compute_ber(c, c_hat)

ber_coded = sionna.phy.utils.compute_ber(b, b_hat)

print("BER uncoded = {:.3f} at EbNo = {:.1f} dB".format(ber_uncoded, ebno_db))
print("BER after decoding = {:.3f} at EbNo = {:.1f} dB".format(ber_coded, ebno_db))
print("In total {} bits were simulated".format(np.size(b.numpy())))


# Just to summarize: we have simulated the transmission of 250,000 bits including higher-order modulation and channel coding!
# 
# But we can go even faster with the *TF graph execution*!

# ### Setting up the End-to-end Model
# 
# We now define a block that wraps the entire link-level simultaions into a single
# callable which more convenient for training and Monte-Carlo simulations.
# 
# We simulate the transmission over a time-varying multi-path channel (the *TDL-A* model from 3GPP TR38.901).
# For this, OFDM and a *conventional* bit-interleaved coded modulation (BICM) scheme with higher order modulation is used.
# The information bits are protected by a 5G-compliant LDPC code.
# 
# *Remark*: Due to the large number of parameters, we define them as dictionary.

# In[15]:


class e2e_model(sionna.phy.Block):
    """Example model for end-to-end link-level simulations.
    
    Parameters
    ----------
    params: dict
        A dictionary defining the system parameters.

    Input
    -----
    batch_size: int or tf.int
        The batch_sizeused for the simulation.

    ebno_db: float or tf.float
        A float defining the simulation SNR.

    Output
    ------
    (b, b_hat): 
        Tuple:

    b: tf.float32
        A tensor of shape `[batch_size, k]` containing the transmitted
        information bits.

    b_hat: tf.float32
        A tensor of shape `[batch_size, k]` containing the receiver's
        estimate of the transmitted information bits.
    """
    def __init__(self,
                params):
        super().__init__()

        # Define an OFDM Resource Grid Object
        self.rg = sionna.phy.ofdm.ResourceGrid(
                            num_ofdm_symbols=params["num_ofdm_symbols"],
                            fft_size=params["fft_size"],
                            subcarrier_spacing=params["subcarrier_spacing"],
                            num_tx=1,
                            num_streams_per_tx=1,
                            cyclic_prefix_length=params["cyclic_prefix_length"],
                            pilot_pattern="kronecker",
                            pilot_ofdm_symbol_indices=params["pilot_ofdm_symbol_indices"])
              
        # Create a Stream Management object        
        self.sm = sionna.phy.mimo.StreamManagement(rx_tx_association=np.array([[1]]),
                                                   num_streams_per_tx=1)
        
        self.coderate = params["coderate"]
        self.num_bits_per_symbol = params["num_bits_per_symbol"]
        self.n = int(self.rg.num_data_symbols*self.num_bits_per_symbol) 
        self.k = int(self.n*coderate)         

        # Init layers
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.interleaver = sionna.phy.fec.interleaving.RowColumnInterleaver(
                                        row_depth=self.num_bits_per_symbol)
        self.deinterleaver = sionna.phy.fec.interleaving.Deinterleaver(self.interleaver)
        self.mapper = sionna.phy.mapping.Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper = sionna.phy.ofdm.ResourceGridMapper(self.rg)
        self.tdl = sionna.phy.channel.tr38901.TDL(
                           model="A",
                           delay_spread=params["delay_spread"],
                           carrier_frequency=params["carrier_frequency"],
                           min_speed=params["min_speed"],
                           max_speed=params["max_speed"])
        
        self.channel = sionna.phy.channel.OFDMChannel(self.tdl, self.rg, add_awgn=True, normalize_channel=True)
        self.ls_est = sionna.phy.ofdm.LSChannelEstimator(self.rg, interpolation_type="nn")
        self.lmmse_equ = sionna.phy.ofdm.LMMSEEqualizer(self.rg, self.sm)
        self.demapper = sionna.phy.mapping.Demapper(params["demapping_method"],
                                                "qam", self.num_bits_per_symbol)
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(self.encoder,
                                                    hard_out=True,
                                                    cn_update=params["cn_update"],
                                                    num_iter=params["bp_iter"])

        print("Number of pilots: {}".format(self.rg.num_pilot_symbols))        
        print("Number of data symbols: {}".format(self.rg.num_data_symbols))
        print("Number of resource elements: {}".format(
                                    self.rg.num_resource_elements))

        print("Pilot overhead: {:.2f}%".format(
                                    self.rg.num_pilot_symbols /
                                    self.rg.num_resource_elements*100))

        print("Cyclic prefix overhead: {:.2f}%".format(
                                    params["cyclic_prefix_length"] /
                                    (params["cyclic_prefix_length"]
                                    +params["fft_size"])*100))

        print("Each frame contains {} information bits".format(self.k))

    def call(self, batch_size, ebno_db):

        # Generate a batch of random bit vectors
        # We need two dummy dimension representing the number of
        # transmitters and streams per transmitter, respectively.
        b = self.binary_source([batch_size, 1, 1, self.k])

        # Encode the bits using the all-zero dummy encoder
        c = self.encoder(b)

        # Interleave the bits before mapping (BICM)
        c_int = self.interleaver(c)

        # Map bits to constellation symbols
        s = self.mapper(c_int)

        # Map symbols onto OFDM ressource grid
        x_rg = self.rg_mapper(s)

        # Transmit over noisy multi-path channel 
        no = sionna.phy.utils.ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, self.rg)
        y = self.channel(x_rg, no) 

        # LS Channel estimation with nearest pilot interpolation
        h_hat, err_var = self.ls_est (y, no)

        # LMMSE Equalization
        x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no)

        # Demap to LLRs
        llr = self.demapper(x_hat, no_eff)

        # Deinterleave before decoding
        llr_int = self.deinterleaver(llr)

        # Decode
        b_hat = self.decoder(llr_int) 

        # number of simulated bits
        nb_bits = batch_size*self.k

        # transmitted bits and the receiver's estimate after decoding
        return b, b_hat


# Let us define the system parameters for our simulation as dictionary:

# In[16]:


sys_params = {
    # Channel
    "carrier_frequency" : 3.5e9,
    "delay_spread" : 100e-9,
    "min_speed" : 3,
    "max_speed" : 3,
    "tdl_model" : "A",

    # OFDM
    "fft_size" : 256,
    "subcarrier_spacing" : 30e3,
    "num_ofdm_symbols" : 14,
    "cyclic_prefix_length" : 16,
    "pilot_ofdm_symbol_indices" : [2, 11],

    # Code & Modulation
    "coderate" : 0.5,
    "num_bits_per_symbol" : 4,
    "demapping_method" : "app",
    "cn_update" : "boxplus",
    "bp_iter" : 20
}


# ...and initialize the model:

# In[17]:


model = e2e_model(sys_params)


# As before, we can simply *call* the model to simulate the BER for the given simulation parameters.

# In[18]:


#simulation parameters
ebno_db = 10
batch_size = 200

# and call the model
b, b_hat = model(batch_size, ebno_db)

ber = sionna.phy.utils.compute_ber(b, b_hat)
nb_bits = np.size(b.numpy())

print("BER: {:.4} at Eb/No of {} dB and {} simulated bits".format(ber.numpy(), ebno_db, nb_bits))


# ### Run some Throughput Tests (Graph Mode)
# 
# Sionna is not just an easy-to-use library, but also incredibly fast.
# Let us measure the throughput of the model defined above.
# 
# We compare *eager* and *graph* execution modes (see [Tensorflow Doc](https://www.tensorflow.org/guide/intro_to_graphs) for details), as well
# as *eager with XLA* (see https://www.tensorflow.org/xla#enable_xla_for_tensorflow_models).
# 
# **Tip**: change the `batch_size` to see how the batch parallelism enhances the throughput.
# Depending on your machine, the `batch_size` may be too large.

# In[19]:


import time # this block requires the timeit library

batch_size = 200
ebno_db = 5 # evalaute SNR point
repetitions = 4 # throughput is averaged over multiple runs

def get_throughput(batch_size, ebno_db, model, repetitions=1):
    """ Simulate throughput in bit/s per ebno_db point.

    The results are average over `repetition` trials.

    Input
    -----
    batch_size: int or tf.int32
        Batch-size for evaluation.

    ebno_db: float or tf.float32
        A tensor containing the SNR points be evaluated    

    model:
        Function or model that yields the transmitted bits `u` and the
        receiver's estimate `u_hat` for a given ``batch_size`` and
        ``ebno_db``.

    repetitions: int
        An integer defining how many trails of the throughput 
        simulation are averaged.

    """


    # call model once to be sure it is compile properly 
    # otherwise time to build graph is measured as well.
    u, u_hat = model(tf.constant(batch_size, tf.int32),    
                     tf.constant(ebno_db, tf.float32))

    t_start = time.perf_counter()
    # average over multiple runs
    for _ in range(repetitions):
        u, u_hat = model(tf.constant(batch_size, tf.int32),
                            tf.constant(ebno_db, tf. float32))
    t_stop = time.perf_counter()

    # throughput in bit/s
    throughput = np.size(u.numpy())*repetitions / (t_stop - t_start)

    return throughput

# eager mode - just call the model
def run_eager(batch_size, ebno_db):
    return model(batch_size, ebno_db)
    
time_eager = get_throughput(batch_size, ebno_db, run_eager, repetitions=4)

# the decorator "@tf.function" enables the graph mode
@tf.function
def run_graph(batch_size, ebno_db):
    return model(batch_size, ebno_db)

time_graph = get_throughput(batch_size, ebno_db, run_graph, repetitions=4)

# the decorator "@tf.function(jit_compile=True)" enables the graph mode with XLA
@tf.function(jit_compile=True)
def run_graph_xla(batch_size, ebno_db):
    return model(batch_size, ebno_db)

time_graph_xla = get_throughput(batch_size, ebno_db, run_graph_xla, repetitions=4)

print(f"Throughput in eager execution: {time_eager/1e6:.2f} Mb/s")
print(f"Throughput in graph execution: {time_graph/1e6:.2f} Mb/s")
print(f"Throughput in graph execution with XLA: {time_graph_xla/1e6:.2f} Mb/s")


# Obviously, *graph* execution (with XLA) yields much higher throughputs (at least if a fast GPU is available).
# Thus, for exhaustive training and Monte-Carlo simulations the *graph* mode (with XLA and GPU acceleration) is the preferred choice.

# ### Bit-Error Rate (BER) Monte-Carlo Simulations
# 
# Monte-Carlo simulations are omnipresent in todays communications research and development.
# Due its performant implementation, Sionna can be directly used to simulate BER at a performance that competes with compiled languages -- but still keeps the flexibility of a script language.

# In[20]:


ebno_dbs = np.arange(0, 15, 1.)
batch_size = 200 # reduce in case you receive an out-of-memory (OOM) error

max_mc_iter = 1000 # max number of Monte-Carlo iterations before going to next SNR point
num_target_block_errors = 500 # continue with next SNR point after target number of block errors

ber_mc,_ = sionna.phy.utils.sim_ber(run_graph_xla, # you can also evaluate the model directly
                                    ebno_dbs,
                                    batch_size=batch_size, 
                                    num_target_block_errors=num_target_block_errors,
                                    max_mc_iter=max_mc_iter,
                                    verbose=True) # print status and summary


# Let's look at the results:

# In[21]:


sionna.phy.utils.plot_ber(ebno_dbs,
                          ber_mc,
                          legend="E2E Model",
                          ylabel="Coded BER");


# ### Conclusion
# 
# We hope you are excited about Sionna - there is much more to be discovered:
# 
# - TensorBoard debugging available
# - Scaling to multi-GPU simulation is simple
# - See the [available tutorials](https://nvlabs.github.io/sionna/phy/tutorials.html) for more advanced examples.
# 
# And if something is still missing - the project is [open-source](https://github.com/nvlabs/sionna/):  you can modify, add, and extend any component at any time.
-e 
# --- End of Discover_Sionna.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # From GSM to 5G - The Evolution of Forward Error Correction
# 
# This notebook compares the different FEC schemes from GSM via UMTS and LTE to 5G NR.
# Please note that a *fair* comparison of different coding schemes depends on many aspects such as:
# 
#  - Decoding complexity, latency, and scalability
# 
# - Level of parallelism of the decoding algorithm and memory access patterns
# 
# - Error-floor behavior
# 
# - Rate adaptivity and flexibility
# 
# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [System Model](#System-Model)
# * [Error Rate Simulations](#Error-Rate-Simulations)
# * [Results for Longer Codewords](#Results-for-Longer-Codewords)

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

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder
from sionna.phy.utils import ebnodb2no, hard_decisions, PlotBER
from sionna.phy.channel import AWGN


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ## System Model

# In[3]:


class System_Model(Block):
    """System model for channel coding BER simulations.
    
    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to 
    initialize the model.
    
    Parameters
    ----------
        k: int
            number of information bits per codeword.
        
        n: int 
            codeword length.
        
        num_bits_per_symbol: int
            number of bits per QAM symbol.
            
        encoder: Sionna Block
            A Sionna Block that encodes information bit tensors.
            
        decoder: Sionna Block
            A Sionna Block that decodes llr tensors.
            
        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".
            
        sim_esno: bool  
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.
            
    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        
        ebno_db: float or tf.float
            A float defining the simulation SNR.
            
    Output
    ------
        (u, u_hat):
            Tuple:
        
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.           

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.           
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,                 
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False):

        super().__init__()
        
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        
        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol

        # init components
        self.source = BinarySource()
       
        # initialize mapper and demapper for constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)
        
        # the channel can be replaced by more sophisticated models
        self.channel = AWGN()

        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder

    @tf.function(jit_compile=True) # enable graph mode for increased throughputs
    def call(self, batch_size, ebno_db):
        return self.call_no_xla(batch_size, ebno_db)

    # Polar codes cannot be executed with XLA
    @tf.function(jit_compile=False) # enable graph mode 
    def call_no_xla(self, batch_size, ebno_db):
        
        u = self.source([batch_size, self.k]) # generate random data
        
        if self.encoder is None:
            # uncoded transmission
            c = u
        else:
            c = self.encoder(u) # explicitly encode

        # calculate noise variance
        if self.sim_esno:
            no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else: 
            if self.encoder is None:
                # uncoded transmission
                coderate = 1
            else:
                coderate = self.k/self.n

            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=coderate)            
        
        x = self.mapper(c) # map c to symbols x
        
        y = self.channel(x, no) # transmit over AWGN channel

        llr_ch = self.demapper(y, no) # demapp y to LLRs
        
        if self.decoder is None:
            # uncoded transmission
            u_hat = hard_decisions(llr_ch) 
        else:
            u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)
        return u, u_hat


# ## Error Rate Simulations
# 
# We now compare the different schemes for a codeword length of $n=1024$ and coderate $r=0.5$.
# 
# Let us define the codes to be simulated.

# In[4]:


# code parameters
k = 512 # number of information bits per codeword
n = 1024 # desired codeword length
codes_under_test = []

# Uncoded transmission
enc = None
dec = None
name = "Uncoded QPSK"
codes_under_test.append([enc, dec, name])

# Conv. code with Viterbi decoding 
enc = ConvEncoder(rate=1/2, constraint_length=5)
dec = ViterbiDecoder(gen_poly=enc.gen_poly, method="soft_llr")
name = "GSM: Convolutional Codes"
codes_under_test.append([enc, dec, name])

# Turbo codes
enc = TurboEncoder(rate=1/2, constraint_length=4, terminate=True)
dec = TurboDecoder(encoder=enc, num_iter=8)
name = "UMTS/LTE: Turbo Codes"
codes_under_test.append([enc, dec, name])

# LDPC codes
enc = LDPC5GEncoder(k, n)
dec = LDPC5GDecoder(encoder=enc, num_iter=40)
name = "5G: LDPC"
codes_under_test.append([enc, dec, name])

# Polar codes
enc = Polar5GEncoder(k, n)
dec = Polar5GDecoder(enc, dec_type="hybSCL", list_size=32)
name = "5G: Polar+CRC"
codes_under_test.append([enc, dec, name])


# Generate a new BER plot figure to save and plot simulation results efficiently.

# In[5]:


ber_plot = PlotBER("")


# And run the BER simulation for each code.

# In[6]:


num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0., 8, 0.2) # sim SNR range 

# run ber simulations for each code we have added to the list
for code in codes_under_test:
    print("\n Running: " + code[2])
    
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1],
                         sim_esno=False)
    
    # run the Polar code in a separate call, as currently no XLA is supported
    if not code[2]=="5G: Polar+CRC":
        ber_plot.simulate(model, # the function have defined previously
                        ebno_dbs=ebno_db, # SNR to simulate
                        legend=code[2], # legend string for plotting
                        max_mc_iter=1000, # run 1000 Monte Carlo runs per SNR point
                        num_target_block_errors=2000, # continue with next SNR point after 1000 bit errors
                        target_bler=3e-4,
                        batch_size=10000, # batch-size per Monte Carlo run
                        soft_estimates=False, # the model returns hard-estimates
                        early_stop=True, # stop simulation if no error has been detected at current SNR point
                        show_fig=False, # we show the figure after all results are simulated
                        add_bler=True, # in case BLER is also interesting
                        forward_keyboard_interrupt=False);
    else:
        # run model in non_xla mode        
        ber_plot.simulate(model.call_no_xla, # no XLA
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=code[2], # legend string for plotting
                         max_mc_iter=10000, # we use more iterations with smaller batches
                         num_target_block_errors=200, # continue with next SNR point after 1000 bit errors
                         target_bler=3e-4,
                         batch_size=1000, # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         forward_keyboard_interrupt=False);        


# And show the final performance

# In[7]:


# remove "(BLER)" labels from legend
for idx, l in enumerate(ber_plot.legend):
    ber_plot.legend[idx] = l.replace(" (BLER)", "")
    
# and plot the BLER
ber_plot(xlim=[0, 7], ylim=[3.e-4, 1], show_ber=False)


# In[8]:


# BER
ber_plot(xlim=[0, 7], ylim=[2.e-7, 1], show_bler=False)


# ## Results for Longer Codewords
# 
# In particular for the data channels, longer codewords are usually required.
# For these applications, LDPC and Turbo codes are the workhorse of 5G and LTE, respectively. 
# 
# Let's compare LDPC and Turbo codes for $k=6144$ information bits and coderate $r=1/3$.

# In[9]:


# code parameters
k = 2048 # number of information bits per codeword
n = 6156 # desired codeword length (including termination bits)
codes_under_test = []

# Uncoded QPSK
enc = None
dec = None
name = "Uncoded QPSK"
codes_under_test.append([enc, dec, name])

#Turbo. codes
enc = TurboEncoder(rate=1/3, constraint_length=4, terminate=True)
dec = TurboDecoder(encoder=enc, num_iter=8)
name = "UMTS/LTE: Turbo Codes"
codes_under_test.append([enc, dec, name])

# LDPC
enc = LDPC5GEncoder(k, n)
dec = LDPC5GDecoder(encoder=enc, num_iter=40)
name = "5G: LDPC"
codes_under_test.append([enc, dec, name])


# In[10]:


ber_plot_long = PlotBER(f"Error Rate Performance (k={k}, n={n})")


# In[11]:


num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(-1, 1.8, 0.1) # sim SNR range 

# run ber simulations for each code we have added to the list
for code in codes_under_test:
    print("\n Running: " + code[2])
    
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1],
                         sim_esno=False)
    
    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot_long.simulate(model, # the function have defined previously
                     ebno_dbs=ebno_db, # SNR to simulate
                     legend=code[2], # legend string for plotting
                     max_mc_iter=1000, # run 100 Monte Carlo runs per SNR point
                     num_target_block_errors=500, # continue with next SNR point after 2000 bit errors
                     target_ber=6e-7,
                     batch_size=10000, # batch-size per Monte Carlo run
                     soft_estimates=False, # the model returns hard-estimates
                     early_stop=True, # stop simulation if no error has been detected at current SNR point
                     show_fig=False, # we show the figure after all results are simulated
                     add_bler=True, # in case BLER is also interesting
                     forward_keyboard_interrupt=False); # should be True in a loop


# In[12]:


# and show the figure
ber_plot_long(xlim=[-1., 1.7],ylim=(6e-7, 1)) # we set the ylim to 1e-5 as otherwise more extensive simualtions would be required for accurate curves.


# A comparison of short length codes can be found in the tutorial notebook [5G Channel Coding Polar vs. LDPC Codes](5G_Channel_Coding_Polar_vs_LDPC_Codes.ipynb).

# ## Final Figure
# 
# Combine results from the two simulations.

# In[13]:


snrs = list(np.compress(a=ber_plot._snrs, condition=ber_plot._is_bler, axis=0))
bers = list(np.compress(a=ber_plot._bers, condition=ber_plot._is_bler, axis=0))
legends = list(np.compress(a=ber_plot._legends, condition=ber_plot._is_bler, axis=0))
is_bler = list(np.compress(a=ber_plot._is_bler, condition=ber_plot._is_bler, axis=0))

ylabel = "BLER"

# generate two subplots
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,10))

ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)

   
# Part A 
xlim=[0, 6]
ylim=[1e-4, 1]

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)

for idx, b in enumerate(bers):
    ax1.semilogy(snrs[idx], b, "--", linewidth=2)

ax1.grid(which="both")
ax1.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
ax1.set_ylabel(ylabel, fontsize=25)
ax1.legend(legends, fontsize=20, loc="upper right");
ax1.set_title("$k=512, n=1024$", fontsize=20)


# remove "(BLER)" labels from legend
for idx, l in enumerate(ber_plot_long.legend):
    ber_plot_long.legend[idx] = l.replace(" (BLER)", "")
    
snrs = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._snrs, axis=0))
bers = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._bers, axis=0))
legends = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._legends, axis=0))
is_bler = list(np.compress(condition=ber_plot_long._is_bler, a=ber_plot_long._is_bler, axis=0))


# Part B
xlim=[-1, 2]
ylim=[1e-4, 1]

ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_title("$k=2048, n=6156$", fontsize=20)

# return figure handle
#for idx, b in enumerate(bers):

ax2.semilogy(snrs[0], bers[0], "--", linewidth=2, color="orange")
ax2.semilogy(snrs[1], bers[1], "--", linewidth=2, color="green")
ax2.semilogy(snrs[2], bers[2], "--", linewidth=2, color="blue")

ax2.grid(which="both")
ax2.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
ax2.set_ylabel(ylabel, fontsize=25)
plt.legend(legends, fontsize=20, loc="upper right");

-e 
# --- End of Evolution_of_FEC.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # “Hello, world!”

# Import Sionna:

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

# IPython "magic function" for inline plots
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# Let us first create a [BinarySource](https://nvlabs.github.io/sionna/phy/api/mapping.html#sionna.phy.mapping.BinarySource) to generate a random batch of bit vectors that we can map to constellation symbols:

# In[2]:


batch_size = 1000 # Number of symbols we want to generate
num_bits_per_symbol = 4 # 16-QAM has four bits per symbol
binary_source = sionna.phy.mapping.BinarySource()
b = binary_source([batch_size, num_bits_per_symbol])
b


# Next, let us create a [Constellation](https://nvlabs.github.io/sionna/phy/api/mapping.html#sionna.phy/mapping.Constellation) and visualize it:

# In[3]:


constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol)
constellation.show();


# We now need a [Mapper](https://nvlabs.github.io/sionna/phy/api/mapping.html#sionna.phy.mapping.Mapper) that maps each row of b to the constellation symbols according to the bit labeling shown above.

# In[4]:


mapper = sionna.phy.mapping.Mapper(constellation=constellation)
x = mapper(b)
x[:10]


# Let us now make things a bit more interesting a send our symbols over and [AWGN channel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.phy.channel.AWGN):

# In[5]:


awgn = sionna.phy.channel.AWGN()
ebno_db = 15 # Desired Eb/No in dB
no = sionna.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1)
y = awgn(x, no)

# Visualize the received signal
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.scatter(np.real(y), np.imag(y));
ax.set_aspect("equal", adjustable="box")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, which="both", axis="both")
plt.title("Received Symbols");

-e 
# --- End of Hello_World.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Introduction to Iterative Detection and Decoding
# In this notebook, you will learn how to set-up an iterative detection and decoding (IDD) scheme (first presented in [1]) by combining multiple available components in Sionna.
# 
# For a gentle introduction to MIMO simulations, we refer to the notebooks ["Simple MIMO Simulations"](https://nvlabs.github.io/sionna/phy/tutorials/Simple_MIMO_Simulation.html) and ["MIMO OFDM Transmissions over CDL"](https://nvlabs.github.io/sionna/phy/tutorials/MIMO_OFDM_Transmissions_over_CDL.html).
# 
# You will evaluate the performance of IDD with OFDM MIMO detection and soft-input soft-output (SISO) LDPC decoding and compare it againts several non-iterative detectors, such as soft-output LMMSE, K-Best, and expectation propagation (EP), as well as iterative SISO MMSE-PIC detection [2].
# 
# For the non-IDD models, the signal processing pipeline looks as follows:

# ![block_diagram.png](attachment:block_diagram.png)

# ## Iterative Detection and Decoding
# The IDD MIMO receiver iteratively exchanges soft-information between the data detector and the channel decoder, which works as follows:

# ![idd_diagram.png](attachment:idd_diagram.png)

# 
# Originally, IDD was proposed with a resetting (Turbo) decoder [1]. However, state-of-the-art IDD with LDPC message passing decoding showed better performance with a non-resetting decoder [3], particularly for a low number of decoding iterations. Therefore, we will forward the decoder state (i.e., the check node to variable node messages) from each IDD iteration to the next.

# ## Table of contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Simulation Parameters](#Simulation-Parameters)
# * [Setting-up an End-to-end Block](#Setting-up-an-end-to-end-Block)
# * [Non-IDD versus IDD Benchmarks](#Non-IDD-versus-IDD-Benchmarks)
# * [Discussion-Optimizing IDD with Machine Learning](#Discussion-Optimizing-IDD-with-Machine-Learning)
# * [Comments](#Comments)
# * [List of References](#List-of-References)

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
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import sim_ber, ebnodb2no, expand_to_rank
from sionna.phy.mapping import Mapper, Constellation, BinarySource
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
                            LinearDetector, KBestDetector, EPDetector, \
                            RemoveNulledSubcarriers, MMSEPICDetector
from sionna.phy.channel import OFDMChannel, RayleighBlockFading, gen_single_sector_topology
from sionna.phy.channel.tr38901 import UMa, Antenna, PanelArray
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation


# ## Simulation Parameters
# In the following, we set the simulation parameters. Please modify at will; adapting the batch size to your hardware setup might be beneficial.
# 
# The standard configuration implements a coded 5G inspired MU-MIMO OFDM uplink transmission over 3GPP UMa channels, with 4 single-antenna UEs, 16-QAM modulation, and a 16 element dual-polarized uniform planar antenna array (UPA) at the gNB. We implement least squares channel estimation with linear interpolation. Alternatively, we implement iid Rayleigh fading channels and perfect channel state information (CSI), which can be controlled by the model parameter `perfect_csi_rayleigh`.
# As channel code, we apply a rate-matched 5G LDPC code at rate 1/2.

# In[2]:


SIMPLE_SIM = False   # reduced simulation time for simple simulation if set to True
if SIMPLE_SIM:
    batch_size = int(1e1)  # number of OFDM frames to be analyzed per batch
    num_iter = 5  # number of Monte Carlo Iterations (total number of Monte Carlo trials is num_iter*batch_size)
    num_steps = 6
    tf.config.run_functions_eagerly(True)   # run eagerly for better debugging
else:
    batch_size = int(64)  # number of OFDM frames to be analyzed per batch
    num_iter = 128  # number of Monte Carlo Iterations (total number of Monte Carlo trials is num_iter*batch_size)
    num_steps = 11

ebno_db_min_perf_csi = -10  # min EbNo value in dB for perfect csi benchmarks
ebno_db_max_perf_csi = 0
ebno_db_min_cest = -10
ebno_db_max_cest = 10


NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 12*4 # 4 PRBs
SUBCARRIER_SPACING = 30e3 # Hz
CARRIER_FREQUENCY = 3.5e9 # Hz
SPEED = 3. # m/s
num_bits_per_symbol = 4 # 16 QAM
n_ue = 4 # 4 UEs
NUM_RX_ANT = 16 # 16 BS antennas
num_pilot_symbols = 2

# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
UT_ANTENNA = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni', # Omnidirectional antenna pattern
                     carrier_frequency=CARRIER_FREQUENCY)

# The base station is equipped with an antenna
# array of 8 cross-polarized antennas,
# resulting in a total of 16 antenna elements.
BS_ARRAY = PanelArray(num_rows_per_panel=2,
                      num_cols_per_panel=4,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901', # 3GPP 38.901 antenna pattern
                      carrier_frequency=CARRIER_FREQUENCY)

# 3GPP UMa channel model is considered
channel_model_uma = UMa(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=UT_ANTENNA,
                    bs_array=BS_ARRAY,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)

channel_model_rayleigh = RayleighBlockFading(num_rx=1, num_rx_ant=NUM_RX_ANT, num_tx=n_ue, num_tx_ant=1)

constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)

rx_tx_association = np.ones([1, n_ue])
sm = StreamManagement(rx_tx_association, 1)

# Parameterize the OFDM channel
rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS, pilot_ofdm_symbol_indices = [2, 11],
                  fft_size=FFT_SIZE, num_tx=n_ue,
                  pilot_pattern = "kronecker",
                  subcarrier_spacing=SUBCARRIER_SPACING)

rg.show()
plt.show()

# Parameterize the LDPC code
R = 0.5  # rate 1/2
N = int(FFT_SIZE * (NUM_OFDM_SYMBOLS - 2) * num_bits_per_symbol)
# N = int((FFT_SIZE) * (NUM_OFDM_SYMBOLS - 2) * num_bits_per_symbol)
# code length; - 12 because of 11 guard carriers and 1 DC carrier, - 2 becaues of 2 pilot symbols
K = int(N * R)  # number of information bits per codeword


# ## Setting-up an End-to-end Block
# 
# Now, we define the baseline models for benchmarking. Let us start with the non-IDD models.

# In[3]:


class NonIddModel(Block):
    def __init__(self, num_bp_iter=12, detector='lmmse', cest_type="LS", interp="lin", perfect_csi_rayleigh=False):
        super().__init__()
        self._num_bp_iter = int(num_bp_iter)
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(K, N, num_bits_per_symbol=num_bits_per_symbol)
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(rg)

        # Channel
        if perfect_csi_rayleigh:
            self._channel_model = channel_model_rayleigh
        else:
            self._channel_model = channel_model_uma

        self._channel = OFDMChannel(channel_model=self._channel_model,
                                    resource_grid=rg,
                                    add_awgn=True, normalize_channel=True, return_channel=True)

        # Receiver
        self._cest_type = cest_type
        self._interp = interp

        # Channel estimation
        self._perfect_csi_rayleigh = perfect_csi_rayleigh
        if self._perfect_csi_rayleigh:
            self._removeNulledSc = RemoveNulledSubcarriers(rg)
        elif cest_type == "LS":
            self._ls_est = LSChannelEstimator(rg, interpolation_type=interp)
        else:
            raise NotImplementedError('Not implemented:' + cest_type)

        # Detection
        if detector == "lmmse":
            self._detector = LinearDetector("lmmse", 'bit', "maxlog", rg, sm, constellation_type="qam",
                                            num_bits_per_symbol=num_bits_per_symbol, hard_out=False)
        elif detector == "k-best":
            k = 64
            self._detector = KBestDetector('bit', n_ue, k, rg, sm, constellation_type="qam",
                                           num_bits_per_symbol=num_bits_per_symbol, hard_out=False)
        elif detector == "ep":
            l = 10
            self._detector = EPDetector('bit', rg, sm, num_bits_per_symbol, l=l, hard_out=False)

        # Forward error correction (decoder)
        self._decoder = LDPC5GDecoder(self._encoder, return_infobits=True, hard_out=True, num_iter=num_bp_iter, cn_update='minsum')

    def new_topology(self, batch_size):
        """Set new topology"""
        if isinstance(self._channel_model, UMa):
            # sensible values according to 3GPP standard, no mobility by default
            topology = gen_single_sector_topology(batch_size,
                                                  n_ue, max_ut_velocity=SPEED,
                                                  scenario="uma")
            self._channel_model.set_topology(*topology)

    @tf.function  # We don't use jit_compile=True to ensure better numerical stability
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)

        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=num_bits_per_symbol,
                       coderate=R)  # normalize in OFDM freq. domain
        b = self._binary_source([batch_size, n_ue, 1, K])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel(x_rg, no_)

        ######################################
        ## Receiver
        if self._perfect_csi_rayleigh:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat),
                                    dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est(y, no)

        llr_ch = self._detector(y, h_hat, chan_est_var, no)  # detector
        b_hat = self._decoder(llr_ch)
        return b, b_hat


# Next, we implement the IDD model with a non-resetting LDPC decoder, as in [3], i.e., we forward the LLRs and decoder state from one IDD iteration to the following.

# In[4]:


class IddModel(NonIddModel):  # inherited from NonIddModel
    def __init__(self, num_idd_iter=3, num_bp_iter_per_idd_iter=12, cest_type="LS", interp="lin", perfect_csi_rayleigh=False):
        super().__init__(num_bp_iter=num_bp_iter_per_idd_iter, detector="lmmse", cest_type=cest_type,
                         interp=interp, perfect_csi_rayleigh=perfect_csi_rayleigh)
        # first IDD detector is LMMSE as MMSE-PIC with zero-prior bils down to soft-output LMMSE
        self._num_idd_iter = num_idd_iter
        self._siso_detector = MMSEPICDetector(output="bit", resource_grid=rg, stream_management=sm,
                                              demapping_method='maxlog', constellation=constellation, num_iter=1,
                                              hard_out=False)
        self._siso_decoder = LDPC5GDecoder(self._encoder, return_infobits=False,
                                           num_iter=num_bp_iter_per_idd_iter, return_state=True, hard_out=False, cn_update='minsum')
        self._decoder = LDPC5GDecoder(self._encoder, return_infobits=True, return_state=True, hard_out=True, num_iter=num_bp_iter_per_idd_iter, cn_type='minsum')
        # last decoder must also be statefull

    @tf.function  # We don't use jit_compile=True to ensure better numerical stability
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)

        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db=ebno_db, num_bits_per_symbol=num_bits_per_symbol,
                       coderate=R)  # normalize in OFDM freq. domain
        b = self._binary_source([batch_size, n_ue, 1, K])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel(x_rg, no_)

        ######################################
        ## Receiver
        if self._perfect_csi_rayleigh:
            h_hat = self._removeNulledSc(h)
            chan_est_var = tf.zeros(tf.shape(h_hat),
                                    dtype=tf.float32)  # No channel estimation error when perfect CSI knowledge is assumed
        else:
            h_hat, chan_est_var = self._ls_est(y, no)

        llr_ch = self._detector(y, h_hat, chan_est_var, no)  # soft-output LMMSE detection
        msg_v2c = None

        if self._num_idd_iter >= 2:
            # perform first iteration outside the while_loop to initialize msg_v2c
            [llr_dec, msg_v2c] = self._siso_decoder(llr_ch, msg_v2c=msg_v2c)
            # forward a posteriori information from decoder

            llr_ch = self._siso_detector(y, h_hat, llr_dec, chan_est_var, no)
            # forward extrinsic information

            def idd_iter(llr_ch, msg_v2c, it):
                [llr_dec, msg_v2c] = self._siso_decoder(llr_ch, msg_v2c=msg_v2c)
                # forward a posteriori information from decoder
                llr_ch = self._siso_detector(y, h_hat, llr_dec, chan_est_var, no)
                # forward extrinsic information from detector

                it += 1
                return llr_ch, msg_v2c, it

            def idd_stop(llr_ch, msg_v2c, it):
                return tf.less(it, self._num_idd_iter - 1)

            it = tf.constant(1)     # we already performed initial detection and one full iteration
            llr_ch, msg_v2c, it = tf.while_loop(idd_stop, idd_iter, (llr_ch, msg_v2c, it), parallel_iterations=1,
                                               maximum_iterations=self._num_idd_iter - 1)
        else:
            # non-idd
            pass

        [b_hat, _] = self._decoder(llr_ch, msg_v2c=msg_v2c)    # final hard-output decoding (only returning information bits)
        return b, b_hat


# ## Non-IDD versus IDD Benchmarks

# In[5]:


# Range of SNR (dB)
snr_range_cest = np.linspace(ebno_db_min_cest, ebno_db_max_cest, num_steps)
snr_range_perf_csi = np.linspace(ebno_db_min_perf_csi, ebno_db_max_perf_csi, num_steps)

def run_idd_sim(snr_range, perfect_csi_rayleigh):
    lmmse = NonIddModel(detector="lmmse", perfect_csi_rayleigh=perfect_csi_rayleigh)
    k_best = NonIddModel(detector="k-best", perfect_csi_rayleigh=perfect_csi_rayleigh)
    ep = NonIddModel(detector="ep", perfect_csi_rayleigh=perfect_csi_rayleigh)
    idd2 = IddModel(num_idd_iter=2, perfect_csi_rayleigh=perfect_csi_rayleigh)
    idd3 = IddModel(num_idd_iter=3, perfect_csi_rayleigh=perfect_csi_rayleigh)

    ber_lmmse, bler_lmmse = sim_ber(lmmse,
                                    snr_range,
                                    batch_size=batch_size,
                                    max_mc_iter=num_iter,
                                    num_target_block_errors=int(batch_size * num_iter * 0.1))

    ber_ep, bler_ep = sim_ber(ep,
                              snr_range,
                              batch_size=batch_size,
                              max_mc_iter=num_iter,
                              num_target_block_errors=int(batch_size * num_iter * 0.1))

    ber_kbest, bler_kbest = sim_ber(k_best,
                                    snr_range,
                                    batch_size=batch_size,
                                    max_mc_iter=num_iter,
                                    num_target_block_errors=int(batch_size * num_iter * 0.1))

    ber_idd2, bler_idd2 = sim_ber(idd2,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1)
                                  )

    ber_idd3, bler_idd3 = sim_ber(idd3,
                                  snr_range,
                                  batch_size=batch_size,
                                  max_mc_iter=num_iter,
                                  num_target_block_errors=int(batch_size * num_iter * 0.1))

    return bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3


BLER = {}

# Perfect CSI
bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3 = run_idd_sim(snr_range_perf_csi, perfect_csi_rayleigh=True)
BLER['Perf. CSI / LMMSE'] = bler_lmmse
BLER['Perf. CSI / EP'] = bler_ep
BLER['Perf. CSI / K-Best'] = bler_kbest
BLER['Perf. CSI / IDD2'] = bler_idd2
BLER['Perf. CSI / IDD3'] = bler_idd3

# Estimated CSI
bler_lmmse, bler_ep, bler_kbest, bler_idd2, bler_idd3 = run_idd_sim(snr_range_cest, perfect_csi_rayleigh=False)
BLER['Ch. Est. / LMMSE'] = bler_lmmse
BLER['Ch. Est. / EP'] = bler_ep
BLER['Ch. Est. / K-Best'] = bler_kbest
BLER['Ch. Est. / IDD2'] = bler_idd2
BLER['Ch. Est. / IDD3'] = bler_idd3


# Finally, we plot the simulation results and observe that IDD outperforms the non-iterative methods by about 1 dB in the scenario with iid Rayleigh fading channels and perfect CSI. In the scenario with 3GPP UMa channels and estimated CSI, IDD performs slightly better than K-best, at considerably lower runtime.

# In[6]:


fig, ax = plt.subplots(1,2, figsize=(16,7))
fig.suptitle(f"{n_ue}x{NUM_RX_ANT} MU-MIMO UL | {2**num_bits_per_symbol}-QAM")

## Perfect CSI Rayleigh
ax[0].set_title("Perfect CSI iid. Rayleigh")
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / LMMSE'], 'x-', label='LMMSE', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / EP'], 'o--', label='EP', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / K-Best'], 's-.', label='K-Best', c='C0')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[0].semilogy(snr_range_perf_csi, BLER['Perf. CSI / IDD3'], 'd:', label=r'IDD $I=3$', c='C2')

ax[0].set_xlabel(r"$E_b/N0$")
ax[0].set_ylabel("BLER")
ax[0].set_ylim((1e-4, 1.0))
ax[0].legend()
ax[0].grid(True)

## Estimated CSI Rayleigh
ax[1].set_title("Estimated CSI 3GPP UMa")
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / LMMSE'], 'x-', label='LMMSE', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / EP'], 'o--', label='EP', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / K-Best'], 's-.', label='K-Best', c='C0')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD2'], 'd:', label=r'IDD $I=2$', c='C1')
ax[1].semilogy(snr_range_cest, BLER['Ch. Est. / IDD3'], 'd:', label=r'IDD $I=3$', c='C2')

ax[1].set_xlabel(r"$E_b/N0$")
ax[1].set_ylabel("BLER")
ax[1].set_ylim((1e-3, 1.0))
ax[1].legend()
ax[1].grid(True)

plt.show()


# ## Discussion-Optimizing IDD with Machine Learning

# ## Comments
# 
# - As discussed in [3], IDD receivers with a non-resetting decoder converge faster than with resetting decoders. However, a resetting decoder (which does not forward `msg_vn`) might perform slightly better for a large number of message passing decoding iterations. Among other quantities, a scaling of the forwarded decoder state is optimized in the DUIDD receiver [4].
# - With estimated channels, we observed that the MMSE-PIC output LLRs become large, much larger as with non-iterative receive processing.

# ## List of References
# 
# [1] B. Hochwald and S. Ten Brink, [*"Achieving near-capacity on a multiple-antenna channel,"*](https://ieeexplore.ieee.org/abstract/document/1194444) IEEE Trans. Commun., vol. 51, no. 3, pp. 389–399, Mar. 2003.
# 
# [2] C. Studer, S. Fateh, and D. Seethaler, [*"ASIC implementation of soft-input soft-output MIMO detection
# using MMSE parallel interference cancellation,"*](https://ieeexplore.ieee.org/abstract/document/5779722) IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, Jul. 2011.
# 
# [3] W.-C. Sun, W.-H. Wu, C.-H. Yang, and Y.-L. Ueng, [*"An iterative detection and decoding receiver for LDPC-coded MIMO systems,"*](https://ieeexplore.ieee.org/abstract/document/7272776) IEEE Trans. Circuits Syst. I, vol. 62, no. 10, pp. 2512–2522, Oct. 2015.
# 
# [4] R. Wiesmayr, C. Dick, J. Hoydis, and C. Studer, [*"DUIDD: Deep-unfolded interleaved detection and decoding for MIMO wireless systems,"*](https://arxiv.org/abs/2212.07816) in Asilomar Conf. Signals, Syst., Comput., Oct. 2022.
-e 
# --- End of Introduction_to_Iterative_Detection_and_Decoding.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Link-level simulations with Sionna RT

# In this notebook, you will use ray-traced channels for link-level simulations instead of stochastic channel models

# ## Background Information

# Ray tracing is a technique to simulate environment-specific and physically accurate channel realizations for a given scene and user position. Sionna RT is a ray tracing extension for radio propagation modeling which is built on top of [Mitsuba 3](https://www.mitsuba-renderer.org/). Like all of Sionna's components, it is differentiable.
# 
# For an introduction about how to use Sionna RT, please see the [corresponding tutorials](https://nvlabs.github.io/sionna/rt/tutorials.html).
# The [EM Primer](https://nvlabs.github.io/sionna/rt/em_primer.html) provides further details on the theoretical background of ray tracing of wireless channels.
# 
# In this notebook, we will use Sionna RT for site-specific link-level simulations. For this, we evaluate the BER performance for a MU-MIMO 5G NR system in the uplink direction based on ray traced CIRs for random user positions.
# 
# We use the 5G NR PUSCH transmitter and receiver from the [5G NR PUSCH Tutorial notebook](https://nvlabs.github.io/sionna/phy/tutorials/5G_NR_PUSCH.html). Note that also the systems from the MIMO [OFDM Transmissions over the CDL Channel Model](https://nvlabs.github.io/sionna/phy/tutorials/MIMO_OFDM_Transmissions_over_CDL.html) or the [Neural Receiver for OFDM SIMO Systems](https://nvlabs.github.io/sionna/phy/tutorials/Neural_Receiver.html) tutorials could be used instead.
# 
# There are different ways to implement uplink scenarios in Sionna RT. In this example, we configure the basestation as transmitter and the user equipments (UEs) as receivers which simplifies the ray tracing. Due to channel reciprocity, one can *reverse* the direction of the ray traced channels afterwards. For the ray tracer itself, the direction (uplink/downlink) does not change the simulated paths.

# ## Imports

# In[1]:


import os # Configure which GPU 
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import or install Sionna
try:
    import sionna.phy
    import sionna.rt
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

# For link-level simulations
from sionna.phy.channel import OFDMChannel, CIRDataset
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.utils import ebnodb2no, PlotBER
from sionna.phy.ofdm import KBestDetector, LinearDetector
from sionna.phy.mimo import StreamManagement

# Import Sionna RT components
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization


# ## Setting up the Ray Tracer

# Let's start by defining some constants that control the system we want to simulate.

# In[2]:


# System parameters
subcarrier_spacing = 30e3 # Hz
num_time_steps = 14 # Total number of ofdm symbols per slot

num_tx = 4 # Number of users
num_rx = 1 # Only one receiver considered
num_tx_ant = 4 # Each user has 4 antennas
num_rx_ant = 16 # The receiver is equipped with 16 antennas

# batch_size for CIR generation
batch_size_cir = 1000


# We then set up the radio propagation environment. We start by loading a scene and then add a transmitter that acts as a base station. We will later use channel reciprocity to simulate the uplink direction.

# In[3]:


# Load an integrated scene.
# You can try other scenes, such as `sionna.rt.scene.etoile`. Note that this would require
# updating the position of the transmitter (see below in this cell).
scene = load_scene(sionna.rt.scene.munich)

# Transmitter (=basestation) has an antenna pattern from 3GPP 38.901
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=num_rx_ant//2, # We want to transmitter to be equiped with 16 antennas
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="cross")

# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27],
                 look_at=[45,90,1.5], # optional, defines view direction
                 display_radius=3.) # optinal, radius of the sphere for visualizing the device
scene.add(tx)

# Create new camera
bird_cam = Camera(position=[0,80,500], orientation=np.array([0,np.pi/2,-np.pi/2]))


# We then compute a radio map for the instantiated transmitter.

# In[4]:


max_depth = 5

# Radio map solver
rm_solver = RadioMapSolver()

# Compute the radio map
rm = rm_solver(scene,
               max_depth=5,
               cell_size=(1., 1.),
               samples_per_tx=10**7)


# Let's visualize the computed radio map.

# In[5]:


if no_preview:
    # Render an image
    scene.render(camera=bird_cam,
                 radio_map=rm,
                 rm_vmin=-110,
                 clip_at=12.); # Clip the scene at rendering for visualizing the refracted field
else:
    # Show preview
    scene.preview(radio_map=rm,
                  rm_vmin=-110,
                  clip_at=12.); # Clip the scene at rendering for visualizing the refracted field


# The function `RadioMap.sample_positions()` allows sampling of random user positions from a radio map. It ensures that only positions that have a path gain of at least `min_gain_dB` dB and at `most max_gain_dB` dB are sampled, i.e., it ignores positions without a connection to the transmitter. Further, one can set the distances `min_dist` and `max_dist` to sample only points within a certain distance range from the transmitter.

# In[6]:


min_gain_db = -130 # in dB; ignore any position with less than -130 dB path gain
max_gain_db = 0 # in dB; ignore strong paths

# Sample points in a 5-400m range around the receiver
min_dist = 5 # in m
max_dist = 400 # in m

# Sample batch_size random user positions from the radio map
ue_pos, _ = rm.sample_positions(num_pos=batch_size_cir,
                                metric="path_gain",
                                min_val_db=min_gain_db,
                                max_val_db=max_gain_db,
                                min_dist=min_dist,
                                max_dist=max_dist)


# We now add new receivers (=UEs) at the sampled positions.
# 
# *Remark:* This is an example for 5G NR PUSCH (uplink direction), we will reverse the direction of the channel for
# later BER simulations.

# In[7]:


# Configure antenna array for all receivers (=UEs)
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=num_tx_ant//2, # Each receiver is equipped with 4 antennas
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="cross")

# Create batch_size receivers
for i in range(batch_size_cir):
    scene.remove(f"rx-{i}") # Remove old receiver if any
    rx = Receiver(name=f"rx-{i}",
                  position=ue_pos[0][i], # Position sampled from radio map
                  velocity=(3.,3.,0),
                  display_radius=1., # optional, radius of the sphere for visualizing the device
                  color=(1,0,0) # optional, color for visualizing the device
                  )
    scene.add(rx)

# And visualize the scene
if no_preview:
    # Render an image
    scene.render(camera=bird_cam,
                 radio_map=rm,
                 rm_vmin=-110,
                 clip_at=12.); # Clip the scene at rendering for visualizing the refracted field
else:
    # Show preview
    scene.preview(radio_map=rm,
                  rm_vmin=-110,
                  clip_at=12.); # Clip the scene at rendering for visualizing the refracted field


# Each dot represents a receiver position drawn from the random sampling function of the radio map.
# This allows to efficiently sample batches of random channel realizations even in complex scenarios.
# 
# ## Creating a CIR Dataset
# 
# We can now simulate the CIRs for many different positions which will be used
# later on as a dataset for link-level simulations.
# 
# *Remark:* Running the cells below can take some time depending on the requested number of CIRs.

# In[8]:


target_num_cirs = 5000 # Defines how many different CIRs are generated.
# Remark: some path are removed if no path was found for this position

max_depth = 5
min_gain_db = -130 # in dB / ignore any position with less than -130 dB path gain
max_gain_db = 0 # in dB / ignore any position with more than 0 dB path gain

# Sample points within a 10-400m radius around the transmitter
min_dist = 10 # in m
max_dist = 400 # in m

# List of channel impulse reponses
a_list = []
tau_list = []

# Maximum number of paths over all batches of CIRs.
# This is used later to concatenate all CIRs.
max_num_paths = 0

# Path solver
p_solver = PathSolver()

# Each simulation returns batch_size_cir results
num_runs = int(np.ceil(target_num_cirs/batch_size_cir))
for idx in range(num_runs):
    print(f"Progress: {idx+1}/{num_runs}", end="\r")

    # Sample random user positions
    ue_pos, _ = rm.sample_positions(
                        num_pos=batch_size_cir,
                        metric="path_gain",
                        min_val_db=min_gain_db,
                        max_val_db=max_gain_db,
                        min_dist=min_dist,
                        max_dist=max_dist,
                        seed=idx) # Change the seed from one run to the next to avoid sampling the same positions

    # Update all receiver positions
    for rx in range(batch_size_cir):
        scene.receivers[f"rx-{rx}"].position = ue_pos[0][rx]

    # Simulate CIR
    paths = p_solver(scene, max_depth=max_depth, max_num_paths_per_src=10**7)

    # Transform paths into channel impulse responses
    a, tau = paths.cir(sampling_frequency=subcarrier_spacing,
                         num_time_steps=14,
                         out_type='numpy')
    a_list.append(a)
    tau_list.append(tau)

    # Update maximum number of paths over all batches of CIRs
    num_paths = a.shape[-2]
    if num_paths > max_num_paths:
        max_num_paths = num_paths

# Concatenate all the CIRs into a single tensor along the num_rx dimension.
# First, we need to pad the CIRs to ensure they all have the same number of paths.
a = []
tau = []
for a_,tau_ in zip(a_list, tau_list):
    num_paths = a_.shape[-2]
    a.append(np.pad(a_, [[0,0],[0,0],[0,0],[0,0],[0,max_num_paths-num_paths],[0,0]], constant_values=0))
    tau.append(np.pad(tau_, [[0,0],[0,0],[0,max_num_paths-num_paths]], constant_values=0))
a = np.concatenate(a, axis=0) # Concatenate along the num_rx dimension
tau = np.concatenate(tau, axis=0)

# Let's now convert to uplink direction, by switing the receiver and transmitter
# dimensions
a = np.transpose(a, (2,3,0,1,4,5))
tau = np.transpose(tau, (1,0,2))

# Add a batch_size dimension
a = np.expand_dims(a, axis=0)
tau = np.expand_dims(tau, axis=0)

# Exchange the num_tx and batchsize dimensions
a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
tau = np.transpose(tau, [2, 1, 0, 3])

# Remove CIRs that have no active link (i.e., a is all-zero)
p_link = np.sum(np.abs(a)**2, axis=(1,2,3,4,5,6))
a = a[p_link>0.,...]
tau = tau[p_link>0.,...]

print("Shape of a:", a.shape)
print("Shape of tau: ", tau.shape)


# Note that transmitters and receivers have been reversed, i.e., the transmitter now denotes the UE (with 4 antennas each) and the receiver is the base station (with 16 antennas).
# 
# *Remark:* We have removed all positions for which the resulting CIR had zero gain, i.e., there was no path between the transmitter and the receiver. This comes from the fact that the `RadioMap.sample_positions()` function samples from a radio map subdivided into cells and randomizes the position within the cells. Therefore, randomly sampled positions may have no paths connecting them to the transmitter.
# 
# Let us now define a data generator that samples random UEs from the dataset and yields the previously simulated CIRs.

# In[9]:


class CIRGenerator:
    """Creates a generator from a given dataset of channel impulse responses

    The generator samples ``num_tx`` different transmitters from the given path
    coefficients `a` and path delays `tau` and stacks the CIRs into a single tensor.

    Note that the generator internally samples ``num_tx`` random transmitters
    from the dataset. For this, the inputs ``a`` and ``tau`` must be given for
    a single transmitter (i.e., ``num_tx`` =1) which will then be stacked
    internally.

    Parameters
    ----------
    a : [batch size, num_rx, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients per transmitter

    tau : [batch size, num_rx, 1, num_paths], float
        Path delays [s] per transmitter

    num_tx : int
        Number of transmitters

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    def __init__(self,
                 a,
                 tau,
                 num_tx):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]

        self._num_tx = num_tx

    def __call__(self):

        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample 4 random users and stack them together
            idx,_,_ = tf.random.uniform_candidate_sampler(
                            tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                            num_true=self._dataset_size,
                            num_sampled=self._num_tx,
                            unique=True,
                            range_max=self._dataset_size)

            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension
            a = tf.transpose(a, (3,1,2,0,4,5,6))
            tau = tf.transpose(tau, (2,1,0,3))

            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)

            yield a, tau


# We use Sionna's built-in [CIRDataset](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.phy.channel.CIRDataset) to initialize a channel model that can be directly used in Sionna's [OFDMChannel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.phy.channel.OFDMChannel) layer.

# In[10]:


batch_size = 20 # Must be the same for the BER simulations as CIRDataset returns fixed batch_size

# Init CIR generator
cir_generator = CIRGenerator(a,
                             tau,
                             num_tx)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model = CIRDataset(cir_generator,
                           batch_size,
                           num_rx,
                           num_rx_ant,
                           num_tx,
                           num_tx_ant,
                           max_num_paths,
                           num_time_steps)


# ## PUSCH Link-Level Simulations
# 
# Let's now define an end-to-end model that simulates a PUSCH transmission.

# In[11]:


class Model:
    """Simulate PUSCH transmissions

    This model runs BER simulations for a multi-user MIMO uplink channel
    compliant with the 5G NR PUSCH specifications.
    You can pick different scenarios, i.e., channel models, perfect or
    estimated CSI, as well as different MIMO detectors (LMMSE or KBest).

    Parameters
    ----------
    channel_model : :class:`~sionna.channel.ChannelModel` object
        An instance of a :class:`~sionna.channel.ChannelModel` object, such as
        :class:`~sionna.channel.RayleighBlockFading` or
        :class:`~sionna.channel.tr38901.UMi` or
        :class:`~sionna.channel.CIRDataset`.

    perfect_csi : bool
        Determines if perfect CSI is assumed or if the CSI is estimated

    detector : str, one of ["lmmse", "kbest"]
        MIMO detector to be used. Note that each detector has additional
        parameters that can be configured in the source code of the _init_ call.

    Input
    -----
    batch_size : int
        Number of simultaneously simulated slots

    ebno_db : float
        Signal-to-noise-ratio

    Output
    ------
    b : [batch_size, num_tx, tb_size], tf.float
        Transmitted information bits

    b_hat : [batch_size, num_tx, tb_size], tf.float
        Decoded information bits
    """
    def __init__(self,
                 channel_model,
                 perfect_csi, # bool
                 detector,    # "lmmse", "kbest"
                ):
        super().__init__()

        self._channel_model = channel_model
        self._perfect_csi = perfect_csi

        # System configuration
        self._num_prb = 16
        self._mcs_index = 14
        self._num_layers = 1
        self._mcs_table = 1
        self._domain = "freq"

        # Below parameters must equal the Path2CIR parameters
        self._num_tx_ant = 4
        self._num_tx = 4
        self._subcarrier_spacing = 30e3 # must be the same as used for Path2CIR

        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing/1000
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_tx_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table

        # Create PUSCHConfigs for the other transmitters by cloning of the first PUSCHConfig
        # and modifying the used DMRS ports.
        pusch_configs = [pusch_config]
        for i in range(1, self._num_tx):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(range(i*self._num_layers, (i+1)*self._num_layers))
            pusch_configs.append(pc)

        # Create PUSCHTransmitter
        self._pusch_transmitter = PUSCHTransmitter(pusch_configs, output_domain=self._domain)

        # Create PUSCHReceiver
        rx_tx_association = np.ones([1, self._num_tx], bool)
        stream_management = StreamManagement(rx_tx_association,
                                             self._num_layers)

        assert detector in["lmmse", "kbest"], "Unsupported MIMO detector"
        if detector=="lmmse":
            detector = LinearDetector(equalizer="lmmse",
                                      output="bit",
                                      demapping_method="maxlog",
                                      resource_grid=self._pusch_transmitter.resource_grid,
                                      stream_management=stream_management,
                                      constellation_type="qam",
                                      num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        elif detector=="kbest":
            detector = KBestDetector(output="bit",
                                     num_streams=self._num_tx*self._num_layers,
                                     k=64,
                                     resource_grid=self._pusch_transmitter.resource_grid,
                                     stream_management=stream_management,
                                     constellation_type="qam",
                                     num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)

        if self._perfect_csi:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 channel_estimator="perfect")
        else:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain)


        # Configure the actual channel
        self._channel = OFDMChannel(
                            self._channel_model,
                            self._pusch_transmitter.resource_grid,
                            normalize_channel=True,
                            return_channel=True)

    # XLA currently not supported by the CIRDataset function
    @tf.function(jit_compile=False)
    def __call__(self, batch_size, ebno_db):

        x, b = self._pusch_transmitter(batch_size)
        no = ebnodb2no(ebno_db,
                       self._pusch_transmitter._num_bits_per_symbol,
                       self._pusch_transmitter._target_coderate,
                       self._pusch_transmitter.resource_grid)
        y, h = self._channel(x, no)
        if self._perfect_csi:
            b_hat = self._pusch_receiver(y, no, h)
        else:
            b_hat = self._pusch_receiver(y, no)
        return b, b_hat


# We now initialize the end-to-end model that uses the `CIRDataset`.

# In[12]:


ebno_db = 10.
e2e_model = Model(channel_model,
                  perfect_csi=False, # bool
                  detector="lmmse")  # "lmmse", "kbest" 

# We can draw samples from the end-2-end link-level simulations
b, b_hat = e2e_model(batch_size, ebno_db)


# Now, let's run the BER evaluation for different system configurations.
# 
# *Remark:* Running the cell below can take some time.

# In[13]:


ebno_db = np.arange(-3, 18, 2) # sim SNR range
ber_plot = PlotBER(f"Site-Specific MU-MIMO 5G NR PUSCH")

for detector in ["lmmse", "kbest"]:
    for perf_csi in [True, False]:
        e2e_model = Model(channel_model,
                          perfect_csi=perf_csi,
                          detector=detector)
        # define legend
        csi = "Perf. CSI" if perf_csi else "Imperf. CSI"
        det = "K-Best" if detector=="kbest" else "LMMSE"
        l = det + " " + csi
        ber_plot.simulate(
                    e2e_model,
                    ebno_dbs=ebno_db, # SNR to simulate
                    legend=l, # legend string for plotting
                    max_mc_iter=500,
                    num_target_block_errors=2000,
                    batch_size=batch_size, # batch-size per Monte Carlo run
                    soft_estimates=False, # the model returns hard-estimates
                    early_stop=True,
                    show_fig=False,
                    add_bler=True,
                    forward_keyboard_interrupt=True);


# In[14]:


ber_plot(show_ber=False)

-e 
# --- End of Link_Level_Simulations_with_RT.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # MIMO OFDM Transmissions over the CDL Channel Model
# In this notebook, you will learn how to setup a realistic simulation of a MIMO
# point-to-point link between a mobile user terminal (UT) and a base station (BS).
# Both, uplink and downlink directions are considered.
# Here is a schematic diagram of the system model with all required components:
# 
# 
# 
# The setup includes:
# 
# * 5G LDPC FEC
# * QAM modulation
# * OFDM resource grid with configurabel pilot pattern
# * Multiple data streams
# * 3GPP 38.901 CDL channel models and antenna patterns
# * ZF Precoding with perfect channel state information
# * LS Channel estimation with nearest-neighbor interpolation as well as perfect CSI
# * LMMSE MIMO equalization
# 
# You will learn how to simulate the channel in the time and frequency domains
# and understand when to use which option.
# 
# In particular, you will investigate:
# 
# * The performance over different CDL models
# * The impact of imperfect CSI
# * Channel aging due to mobility
# * Inter-symbol interference due to insufficient cyclic prefix length
# 
# We will first walk through the configuration of all components of the system model, before simulating
# some simple transmissions in the time and frequency domain.
# Then, we will build a general end-to-end model which will allow us to run efficiently simulations with different
# parameter settings.
# 
# This is a notebook demonstrating a fairly advanced use of the Sionna library.
# It is recommended that you familiarize yourself with the API documentation of the [Channel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html) module and understand the difference between time- and frequency-domain modeling. Some of the simulations
# take some time, especially when you have no GPU available. For this reason, we provide the simulation
# results within the cells generating the figures. If you want to visualize your own results, just comment
# the corresponding line.

# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [System Setup](#System-Setup)
#     * [Stream Management](#Stream-Management)
#     * [OFDM Resource Grid & Pilot Pattern](#OFDM-Resource-Grid-&-Pilot-Pattern)
#     * [Antenna Arrays](#Antenna-Arrays)
#     * [CDL Channel Model](#CDL-Channel-Model)
#         * [CIR Sampling Process](#CIR-Sampling-Process)
#         * [Generate the Channel Frequency Response](#Generate-the-Channel-Frequency-Response)
#         * [Generate the Discrete-Time Channel Impulse Response](#Generate-the-Discrete-Time-Channel-Impulse-Response)
#     * [Other Physical Layer Components](#Other-Physical-Layer-Components)
# * [Simulations](#Simulations)
#     * [Uplink Transmission in the Frequency Domain](#Uplink-Transmission-in-the-Frequency-Domain)
#     * [Uplink Transmission in the Time Domain](#Uplink-Transmission-in-the-Time-Domain)
#     * [Downlink Transmission in the Frequency Domain](#Downlink-Transmission-in-the-Frequency-Domain)
#     * [Understand the Difference Between the CDL Models](#Understand-the-Difference-Between-the-CDL-Models)
#     * [Create an End-to-End Model](#Create-an-End-to-End-Model)
#     * [Compare Uplink Performance Over Different CDL Models](#Compare-Uplink-Performance-Over-the-Different-CDL-Models)
#     * [Compare Downlink Performance Over Different CDL Models](#Compare-Downlink-Performance-Over-the-Different-CDL-Models)
#     * [Evaluate the Impact of Mobility](#Evaluate-the-Impact-of-Mobility)
#     * [Evaluate the Impact of Insufficient Cyclic Prefix Length](#Evaluate-the-Impact-of-Insufficient-Cyclic-Prefix-Length)

# ### GPU Configuration and Imports <a class="anchor" id="GPU-Configuration-and-Imports"></a>

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
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
sionna.phy.config.seed = 42


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import time

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
                               time_lag_discrete_time_channel, ApplyOFDMChannel, ApplyTimeChannel, \
                               OFDMChannel, TimeChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, sim_ber, compute_ber


# ## System Setup <a class="anchor" id="System-Setup"></a>
# 
# We will now configure all components of the system model step-by-step.

# ### Stream Management <a class="anchor" id="Stream-Management"></a>
# 
# For any type of MIMO simulations, it is useful to setup a [StreamManagement](https://nvlabs.github.io/sionna/phy/api/mimo.html#stream-management) object.
# It determines which transmitters and receivers communicate data streams with each other.
# In our scenario, we will configure a single UT and BS with multiple antennas each.
# Whether the UT or BS is considered as a transmitter depends on the `direction`, which can be
# either uplink or downlink. The [StreamManagement](https://nvlabs.github.io/sionna/phy/api/mimo.html#stream-management) has many properties that are used by other components,
# such as precoding and equalization.
# 
# We will configure the system here such that the number of streams per transmitter (in both uplink and donwlink)
# is equal to the number of UT antennas.

# In[3]:


# Define the number of UT and BS antennas.
# For the CDL model, that will be used in this notebook, only
# a single UT and BS are supported.
num_ut = 1
num_bs = 1
num_ut_ant = 4
num_bs_ant = 8

# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = num_ut_ant

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)


# ### OFDM Resource Grid & Pilot Pattern <a class="anchor" id="OFDM-Resource-Grid-&-Pilot-Pattern"></a>

# Next, we configure an OFDM [ResourceGrid](https://nvlabs.github.io/sionna/phy/api/ofdm.html#resource-grid) spanning multiple OFDM symbols.
# The resource grid contains data symbols and pilots and is equivalent to a
# *slot* in 4G/5G terminology. Although it is not relevant for our simulation, we null the DC subcarrier
# and a few guard carriers to the left and right of the spectrum. Also a cyclic prefix is added.
# 
# During the creation of the [ResourceGrid](https://nvlabs.github.io/sionna/phy/api/ofdm.html#resource-grid), a [PilotPattern](https://nvlabs.github.io/sionna/phy/api/ofdm.html#pilot-pattern) is automatically generated.
# We could have alternatively created a [PilotPattern](https://nvlabs.github.io/sionna/phy/api/ofdm.html#pilot-pattern) first and then provided it as initialization parameter.

# In[4]:


rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=76,
                  subcarrier_spacing=15e3,
                  num_tx=1,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=6,
                  num_guard_carriers=[5,6],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
rg.show();


# As can be seen in the figure above, the resource grid spans 76 subcarriers over 14 OFDM symbols. A DC guard carrier as well as some guard carriers to the left and right of the spectrum are nulled. The third and twelfth OFDM symbol are dedicated to pilot transmissions. 
# 
# Let us now have a look at the pilot pattern used by the transmitter. 

# In[5]:


rg.pilot_pattern.show();


# The pilot patterns are defined over the resource grid of *effective subcarriers* from which the nulled DC and guard carriers have been removed. This leaves us in our case with 76 - 1 (DC) - 5 (left guards) - 6 (right guards) = 64 effective subcarriers.
# 
# 
# Let us now have a look at the actual pilot sequences for all streams which consists of random QPSK symbols.
# By default, the pilot sequences are normalized, such that the average power per pilot symbol is
# equal to one. As only every fourth pilot symbol in the sequence is used, their amplitude is scaled by a factor of two.

# In[6]:


plt.figure()
plt.title("Real Part of the Pilot Sequences")
for i in range(num_streams_per_tx):
    plt.stem(np.real(rg.pilot_pattern.pilots[0, i]),
             markerfmt="C{}.".format(i), linefmt="C{}-".format(i),
             label="Stream {}".format(i))
plt.legend()
print("Average energy per pilot symbol: {:1.2f}".format(np.mean(np.abs(rg.pilot_pattern.pilots[0,0])**2)))


# ### Antenna Arrays <a class="anchor" id="Antenna-Arrays"></a>

# Next, we need to configure the antenna arrays used by the UT and BS.
# 
# We will assume here that UT and BS antenna arrays are composed of dual cross-polarized antenna elements with an antenna pattern defined in the 3GPP 38.901 specification. By default, the antenna elements are spaced half of a wavelength apart in both vertical and horizontal directions. You can define your own antenna geometries an radiation patterns if needed.
# 
# An [AntennaArray](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#antennaarray) is always defined in the y-z plane. It's final orientation will be determined by the orientation of the UT or BS. This parameter can be configured in the [ChannelModel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#channel-model-interface) that we will create later.

# In[7]:


carrier_frequency = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

ut_array = AntennaArray(num_rows=1,
                        num_cols=int(num_ut_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
ut_array.show()

bs_array = AntennaArray(num_rows=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
bs_array.show()


# We can also visualize the radiation pattern of an individual antenna element:

# In[8]:


ut_array.show_element_radiation_pattern()


# ### CDL Channel Model <a class="anchor" id="CDL-Channel-Model"></a>
# 
# Now, we will create an instance of the CDL channel model.

# In[9]:


delay_spread = 300e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value.

direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.
cdl_model = "B"       # Suitable values are ["A", "B", "C", "D", "E"]

speed = 10            # UT speed [m/s]. BSs are always assumed to be fixed.
                      # The direction of travel will chosen randomly within the x-y plane.

# Configure a channel impulse reponse (CIR) generator for the CDL model.
# cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)


# #### CIR Sampling Process <a class="anchor" id="CIR-Sampling-Process"></a>

# The instance `cdl` of the [CDL](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#clustered-delay-line-cdl) [ChannelModel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#channel-model-interface) can be used to generate batches of random realizations of continuous-time
# channel impulse responses, consisting of complex gains `a` and delays `tau` for each path. 
# To account for time-varying channels, a channel impulse responses is sampled at the `sampling_frequency` for `num_time_samples` samples.
# For more details on this, please have a look at the [API documentation](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html) of the channel models.
# 
# In order to model the channel in the frequency domain, we need `num_ofdm_symbols` samples that are taken once per `ofdm_symbol_duration`, which corresponds to the length of an OFDM symbol plus the cyclic prefix.

# In[10]:


a, tau = cdl(batch_size=32, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)


# The path gains `a` have shape\
# `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`\
# and the delays `tau` have shape\
# `[batch_size, num_rx, num_tx, num_paths]`.

# In[11]:


print("Shape of the path gains: ", a.shape)
print("Shape of the delays:", tau.shape)


# The delays are assumed to be static within the time-window of interest. Only the complex path gains change over time.
# The following two figures depict the channel impulse response at a particular time instant and the time-evolution of the gain of one path, respectively.

# In[12]:


plt.figure()
plt.title("Channel impulse response realization")
plt.stem(tau[0,0,0,:]/1e-9, np.abs(a)[0,0,0,0,0,:,0])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")


plt.figure()
plt.title("Time evolution of path gain")
plt.plot(np.arange(rg.num_ofdm_symbols)*rg.ofdm_symbol_duration/1e-6, np.real(a)[0,0,0,0,0,0,:])
plt.plot(np.arange(rg.num_ofdm_symbols)*rg.ofdm_symbol_duration/1e-6, np.imag(a)[0,0,0,0,0,0,:])
plt.legend(["Real part", "Imaginary part"])

plt.xlabel(r"$t$ [us]")
plt.ylabel(r"$a$");


# #### Generate the Channel Frequency Response <a class="anchor" id="Generate-the-Channel-Frequency-Response"></a>
# 
# If we want to use the continuous-time channel impulse response to simulate OFDM transmissions
# under ideal conditions, i.e., no inter-symbol interference, inter-carrier interference, etc.,
# we need to convert it to the frequency domain. 
# 
# This can be done with the function [cir_to_ofdm_channel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#cir-to-ofdm-channel) that computes the Fourier transform of the
# continuous-time channel impulse response at a set of `frequencies`, corresponding to the
# different subcarriers. The frequencies can be obtained with the help of the convenience function
# [subcarrier_frequencies](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#subcarrier-frequencies).

# In[13]:


frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)


# Let us have a look at the channel frequency response at a given time instant:

# In[14]:


plt.figure()
plt.title("Channel frequency response")
plt.plot(np.real(h_freq[0,0,0,0,0,0,:]))
plt.plot(np.imag(h_freq[0,0,0,0,0,0,:]))
plt.xlabel("OFDM Symbol Index")
plt.ylabel(r"$h$")
plt.legend(["Real part", "Imaginary part"]);


# We can apply the channel frequency response to a given input with the [ApplyOFDMChannel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#applyofdmchannel) block. This block can also add additive white Gaussian noise (AWGN) to the channel output.

# In[15]:


# Function that will apply the channel frequency response to an input signal
channel_freq = ApplyOFDMChannel(add_awgn=True)


# #### Generate the Discrete-Time Channel Impulse Response <a class="anchor" id="Generate-the-Discrete-Time-Channel-Impulse-Response"></a>
# 
# In the same way as we have created the frequency channel impulse response from the continuous-time
# response, we can use the latter to compute a discrete-time impulse response. This can then be used
# to model the channel in the time-domain through discrete convolution with an input signal.
# Time-domain channel modeling is necessary whenever we want to deviate from the perfect OFDM scenario,
# e.g., OFDM without cyclic prefix, inter-subcarrier interference due to carrier-frequency offsets,
# phase noise, or very high Doppler spread scenarios, as well as other single or multicarrier waveforms
# (OTFS, FBMC, UFMC, etc).
# 
# A discrete-time impulse response can be obtained with the help of the function [cir_to_time_channel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#cir-to-time-channel)
# that requires a `bandwidth` parameter. This function first applies a perfect low-pass filter of
# the provided `bandwith` to the continuous-time channel impulse response and then samples the filtered
# response at the Nyquist rate. The resulting  discrete-time impulse response is then truncated to
# finite length, depending on the delay spread. `l_min` and `l_max` denote truncation boundaries and the
# resulting channel has `l_tot=l_max-l_min+1` filter taps. A detailed mathematical description of this process
# is provided in the API documentation of the channel models. You can freely chose both parameters if you
# do not want to rely on the default values.
# 
# In order to model the channel in the domain, the continuous-time channel impulse response must be sampled
# at the Nyquist rate. We also need now `num_ofdm_symbols x (fft_size + cyclic_prefix_length) + l_tot-1` samples
# in contrast to `num_ofdm_symbols` samples for modeling in the frequency domain. This implies that the 
# memory requirements of time-domain channel modeling is significantly higher. We therefore
# recommend to only use this feature if it is really necessary. Simulations with many transmitters, receivers,
# and/or large antenna arrays become otherwise quickly prohibitively complex.

# In[16]:


# The following values for truncation are recommended.
# Please feel free to tailor them to you needs.
l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
l_tot = l_max-l_min+1

a, tau = cdl(batch_size=2, num_time_steps=rg.num_time_samples+l_tot-1, sampling_frequency=rg.bandwidth)


# In[17]:


h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)


# In[18]:


plt.figure()
plt.title("Discrete-time channel impulse response")
plt.stem(np.abs(h_time[0,0,0,0,0,0]))
plt.xlabel(r"Time step $\ell$")
plt.ylabel(r"$|\bar{h}|$");


# We can apply the discrete-time impulse response to a given input with the [ApplyTimeChannel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#applytimechannel) block. This block can also add additive white Gaussian noise (AWGN) to the channel output.

# In[19]:


# Function that will apply the discrete-time channel impulse response to an input signal
channel_time = ApplyTimeChannel(rg.num_time_samples, l_tot=l_tot, add_awgn=True)


# ### Other Physical Layer Components <a class="anchor" id="Other-Physical-Layer-Components"></a>
# 
# Finally, we create instances of all other physical layer components we need. Most of these blocks are self-explanatory.
# For more information, please have a look at the API documentation.

# In[20]:


num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # Code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits

# The binary source will create batches of information bits
binary_source = BinarySource()

# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)

# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)

# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)

# The zero forcing precoder precodes the transmit stream towards the intended antennas
zf_precoder = RZFPrecoder(rg, sm, return_effective_channel=True)

# OFDM modulator and demodulator
modulator = OFDMModulator(rg.cyclic_prefix_length)
demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)

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


# ## Simulations <a class="anchor" id="Simulations"></a>

# ### Uplink Transmission in the Frequency Domain <a class="anchor" id="Uplink-Transmission-in-the-Frequency-Domain"></a>
# 
# Now, we will simulate our first uplink transmission! Inspect the code to understand how perfect CSI at the receiver can be simulated.

# In[21]:


batch_size = 32 # Depending on the memory of your GPU (or system when a CPU is used),
                # you can in(de)crease the batch size. The larger the batch size, the
                # more memory is required. However, simulations will also run much faster.
ebno_db = 40
perfect_csi = False # Change to switch between perfect and imperfect CSI

# Compute the noise power for a given Eb/No value.
# This takes not only the coderate but also the overheads related pilot
# transmissions and nulled carriers
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)

# As explained above, we generate random batches of CIR, transform them
# in the frequency domain and apply them to the resource grid in the
# frequency domain.
cir = cdl(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
y = channel_freq(x_rg, h_freq, no)

if perfect_csi:
    # For perfect CSI, the receiver gets the channel frequency response as input
    # However, the channel estimator only computes estimates on the non-nulled
    # subcarriers. Therefore, we need to remove them here from `h_freq`.
    # This step can be skipped if no subcarriers are nulled.
    h_hat, err_var = remove_nulled_scs(h_freq), 0.
else:
    h_hat, err_var = ls_est (y, no)

x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))


# An alternative approach to simulations in the frequency domain is to use the
# convenience function [OFDMChannel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#ofdmchannel) that jointly generates and applies
# the channel frequency response. Using this function, we could have used
# the following code:

# In[22]:


ofdm_channel = OFDMChannel(cdl, rg, add_awgn=True, normalize_channel=True, return_channel=True)
y, h_freq = ofdm_channel(x_rg, no)


# ### Uplink Transmission in the Time Domain <a class="anchor" id="Uplink-Transmission-in-the-Time-Domain"></a>

# In the previous example, OFDM modulation/demodulation were not needed as the entire system was simulated
# in the frequency domain. However, this modeling approach is not able to capture many realistic effects. 
# 
# With the following modifications, the system can be modeled in the time domain.
# 
# Have a careful look at how perfect CSI of the channel frequency response is simulated here.

# In[23]:


batch_size = 4 # We pick a small batch_size as executing this code in Eager mode could consume a lot of memory
ebno_db = 30
perfect_csi = True

no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)

# The CIR needs to be sampled every 1/bandwith [s].
# In contrast to frequency-domain modeling, this implies
# that the channel can change over the duration of a single
# OFDM symbol. We now also need to simulate more
# time steps.
cir = cdl(batch_size, rg.num_time_samples+l_tot-1, rg.bandwidth)

# OFDM modulation with cyclic prefix insertion
x_time = modulator(x_rg)

# Compute the discrete-time channel impulse reponse
h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min, l_max, normalize=True)

# Compute the channel output
# This computes the full convolution between the time-varying
# discrete-time channel impulse reponse and the discrete-time
# transmit signal. With this technique, the effects of an
# insufficiently long cyclic prefix will become visible. This
# is in contrast to frequency-domain modeling which imposes
# no inter-symbol interfernce.
y_time = channel_time(x_time, h_time, no)

# OFDM demodulation and cyclic prefix removal
y = demodulator(y_time)

if perfect_csi:

    a, tau = cir

    # We need to sub-sample the channel impulse reponse to compute perfect CSI
    # for the receiver as it only needs one channel realization per OFDM symbol
    a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)]
    a_freq = a_freq[...,:rg.num_ofdm_symbols]

    # Compute the channel frequency response
    h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)

    h_hat, err_var = remove_nulled_scs(h_freq), 0.
else:
    h_hat, err_var = ls_est (y, no)

x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))


# An alternative approach to simulations in the time domain is to use the
# convenience function [TimeChannel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#timechannel) that jointly generates and applies
# the discrete-time channel impulse response. Using this function, we could have used
# the following code:

# In[24]:


time_channel = TimeChannel(cdl, rg.bandwidth, rg.num_time_samples,
