#!/usr/bin/env python
# coding: utf-8

# # 5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes
# 
# *"For block lengths of about 500, an IBM 7090 computer requires about 0.1 seconds per iteration to decode a block by probabilistic decoding scheme. Consequently, many hours of computation time are necessary to evaluate even a* $P(e)$ *in the order of* ${10^{-4}}$ *."* Robert G. Gallager, 1963 [7]
# 
# In this notebook, you will learn about the different coding schemes in 5G NR and how rate-matching works (cf. 3GPP TS 38.212 [3]).
# The coding schemes are compared under different length/rate settings and for different decoders.
# 
# You will learn about the following components:
# 
# * 5G low-density parity-checks (LDPC) codes [7]. These codes support - without further segmentation - up to *k=8448* information bits per codeword [3] for a wide range of coderates.
# 
# * Polar codes [1] including CRC concatenation and rate-matching for 5G compliant en-/decoding is implemented for the Polar uplink control channel (UCI) [3]. Besides Polar codes, Reed-Muller (RM) codes and several decoders are available:
#     - Successive cancellation (SC) decoding [1]
#     - Successive cancellation list (SCL) decoding [2]
#     - Hybrid SC / SCL decoding for enhanced throughput
#     - Iterative belief propagation (BP) decoding [6]
# 
# 
# Further, we will demonstrate the basic functionality of the Sionna forward error correction (FEC) module which also includes support for:
# 
# * Convolutional codes with non-recursive encoding and Viterbi/BCJR decoding
# 
# * Turbo codes and iterative BCJR decoding
# 
# * Ordered statistics decoding (OSD) for any binary, linear code
# 
# * Interleaving and scrambling
# 
# For additional technical background we refer the interested reader to [4,5,8].
# 
# Please note that block segmentation is not implemented as it only concatenates multiple code blocks without increasing the effective codewords length (from decoder's perspective).
# 
# Some simulations in this notebook require severe simulation time, in particular if parameter sweeps are involved (e.g., different length comparisons).
# Please keep in mind that each cell in this notebook already contains the pre-computed outputs and no new execution is required to understand the examples.
# 
# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [BER Performance of 5G Coding Schemes](#BER-Performance-of-5G-Coding-Schemes)
# * [A Deeper Look into the Polar Code Module](#A-Deeper-Look-into-the-Polar-Code-Module)
# * [Rate-matching](#Rate-Matching-and-Rate-Recovery)
# * [Throughput and Decoding Complexity](#Throughput-and-Decoding-Complexity)
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

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder
from sionna.phy.fec.linear import OSDecoder
from sionna.phy.utils import count_block_errors, ebnodb2no, PlotBER
from sionna.phy.channel import AWGN


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import time # for throughput measurements


# ## BER Performance of 5G Coding Schemes
# 
# Let us first focus on short length coding, e.g., for internet of things (IoT) and ultra-reliable low-latency communications (URLLC).
# We aim to reproduce similar results as in [9] for the coding schemes supported by Sionna.
# 
# For a detailed explanation of the `PlotBER` class, we refer to the example notebook on [Bit-Interleaved Coded Modulation](https://nvlabs.github.io/sionna/phy/tutorials/Bit_Interleaved_Coded_Modulation.html).
# 
# The Sionna API allows to pass an encoder object/layer to the decoder initialization for the 5G decoders. This means that the decoder is directly *associated* to a specific encoder and *knows* all relevant code parameters.
# Please note that - of course - no data or information bits are exchanged between these two associated components. It just simplifies handling of the code parameters, in particular, if rate-matching is used.

# Let us define the system model first. We use encoder and decoder as input parameter such that the model remains flexible w.r.t. the coding scheme.

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
            A Sionna Block layer that decodes llr tensors.
            
        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".
            
        sim_esno: bool  
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.

         cw_estiamtes: bool  
            A boolean defaults to False. If true, codewords instead of information estimates are returned.
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
                 sim_esno=False,
                 cw_estimates=False):

        super().__init__()
        
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        self.cw_estimates=cw_estimates # if true codewords instead of info bits are returned
        
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

    @tf.function() # enable graph mode for increased throughputs
    def call(self, batch_size, ebno_db):

        # calculate noise variance
        if self.sim_esno:
                no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else: 
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=self.k/self.n)            

        u = self.source([batch_size, self.k]) # generate random data
        c = self.encoder(u) # explicitly encode
        
        x = self.mapper(c) # map c to symbols x

        y = self.channel(x, no) # transmit over AWGN channel

        llr_ch = self.demapper(y, no) # demap y to LLRs

        u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)

        if self.cw_estimates:
            return c, u_hat
            
        return u, u_hat


#  And let us define the codes to be simulated.

# In[4]:


# code parameters
k = 64 # number of information bits per codeword
n = 128 # desired codeword length

# Create list of encoder/decoder pairs to be analyzed.
# This allows automated evaluation of the whole list later.
codes_under_test = []

# 5G LDPC codes with 20 BP iterations
enc = LDPC5GEncoder(k=k, n=n)
dec = LDPC5GDecoder(enc, num_iter=20)
name = "5G LDPC BP-20"
codes_under_test.append([enc, dec, name])

# Polar Codes (SC decoding)
enc = Polar5GEncoder(k=k, n=n)
dec = Polar5GDecoder(enc, dec_type="SC")
name = "5G Polar+CRC SC"
codes_under_test.append([enc, dec, name])

# Polar Codes (SCL decoding) with list size 8.
# The CRC is automatically added by the layer.
enc = Polar5GEncoder(k=k, n=n)
dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8)
name = "5G Polar+CRC SCL-8"
codes_under_test.append([enc, dec, name])

### non-5G coding schemes

# RM codes with SCL decoding
f,_,_,_,_ = generate_rm_code(3,7) # equals k=64 and n=128 code
enc = PolarEncoder(f, n)
dec = PolarSCLDecoder(f, n, list_size=8)
name = "Reed Muller (RM) SCL-8"
codes_under_test.append([enc, dec, name])

# Conv. code with Viterbi decoding 
enc = ConvEncoder(rate=1/2, constraint_length=8)
dec = ViterbiDecoder(gen_poly=enc.gen_poly, method="soft_llr")
name = "Conv. Code Viterbi (constraint length 8)"
codes_under_test.append([enc, dec, name])

# Turbo. codes
enc = TurboEncoder(rate=1/2, constraint_length=4, terminate=False) # no termination used due to the rate loss
dec = TurboDecoder(enc, num_iter=8)
name = "Turbo Code (constraint length 4)"
codes_under_test.append([enc, dec, name])


# *Remark*: some of the coding schemes are not 5G relevant, but are included in this comparison for the sake of completeness.
# 
# Generate a new BER plot figure to save and plot simulation results efficiently.

# In[5]:


ber_plot128 = PlotBER(f"Performance of Short Length Codes (k={k}, n={n})")


# And run the BER simulation for each code.

# In[6]:


num_bits_per_symbol = 2 # QPSK
ebno_db = np.arange(0, 5, 0.5) # sim SNR range 

# run ber simulations for each code we have added to the list
for code in codes_under_test:
    print("\nRunning: " + code[2])
    
    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1])
    
    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot128.simulate(model, # the function have defined previously
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=code[2], # legend string for plotting
                         max_mc_iter=100, # run 100 Monte Carlo runs per SNR point
                         num_target_block_errors=1000, # continue with next SNR point after 1000 bit errors
                         batch_size=10000, # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         forward_keyboard_interrupt=True); # should be True in a loop

# and show the figure
ber_plot128(ylim=(1e-5, 1), show_bler=False) # we set the ylim to 1e-5 as otherwise more extensive simulations would be required for accurate curves.


# And let's also look at the block-error-rate.

# In[7]:


ber_plot128(ylim=(1e-5, 1), show_ber=False)


# Please keep in mind that the decoding complexity differs significantly and should be also included in a fair comparison as shown in Section [Throughput and Decoding Complexity](#Throughput-and-Decoding-Complexity).

# ### Performance under Optimal Decoding
# 
# The achievable error-rate performance of a coding scheme depends on the strength of the code construction and the performance of the actual decoding algorithm.
# We now approximate the maximum-likelihood performance of all previous coding schemes by using the ordered statistics decoder (OSD) [12].

# In[8]:


# overwrite existing legend entries for OSD simulations
legends = ["5G LDPC", "5G Polar+CRC", "5G Polar+CRC", "RM", "Conv. Code", "Turbo Code"]

# run ber simulations for each code we have added to the list
for idx, code in enumerate(codes_under_test):

    if idx==2: # skip second polar code (same code only different decoder)
        continue 

    print("\nRunning: " + code[2])
    
    # initialize encoder
    encoder = code[0]
    # encode dummy bits to init conv encoders (otherwise k is not defined)
    encoder(tf.zeros((1, k))) 

    # OSD can be directly associated to an encoder
    decoder = OSDecoder(encoder=encoder, t=4) 

    # generate a new model with the given encoder/decoder
    model = System_Model(k=k,
                         n=n,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=encoder,
                         decoder=decoder, 
                         cw_estimates=True) # OSD returns codeword estimates and not info bit estimates
    
    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot128.simulate(tf.function(model, jit_compile=True), 
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=legends[idx]+f" OSD-{decoder.t} ", # legend string for plotting
                         max_mc_iter=1000, # run 100 Monte Carlo runs per SNR point
                         num_target_block_errors=1000, # continue with next SNR point after 1000 bit errors
                         batch_size=32, # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         forward_keyboard_interrupt=True); # should be True in a loop


# And let's plot the results. 
# 
# *Remark*: we define a custom plotting function to enable a nicer visualization of OSD vs. non-OSD results.

# In[9]:


# for simplicity, we only plot a subset of the simulated curves 
# focus on BLER
plots_to_show = ['5G LDPC BP-20 (BLER)', '5G LDPC OSD-4  (BLER)', '5G Polar+CRC SCL-8 (BLER)', '5G Polar+CRC OSD-4  (BLER)', 'Reed Muller (RM) SCL-8 (BLER)', 'RM OSD-4  (BLER)', 'Conv. Code Viterbi (constraint length 8) (BLER)', 'Conv. Code OSD-4  (BLER)', 'Turbo Code (constraint length 4) (BLER)', 'Turbo Code OSD-4  (BLER)']

# find indices of relevant curves
idx = []
for p in plots_to_show:
    for i,l in enumerate(ber_plot128._legends):
        if p==l:
            idx.append(i)

# generate new figure
fig, ax = plt.subplots(figsize=(16,12))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(f"Performance under Ordered Statistic Decoding (k={k},n={n})", fontsize=25)
plt.grid(which="both")
plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
plt.ylabel(r"BLER", fontsize=25)

# plot pairs of BLER curves (non-osd vs. osd)
for i in range(int(len(idx)/2)):

    # non-OSD
    plt.semilogy(ebno_db,
                 ber_plot128._bers[idx[2*i]],
                 c='C%d'%(i),
                 label=ber_plot128._legends[idx[2*i]].replace(" (BLER)", ""), #remove "(BLER)" from label
                 linewidth=2)
    # OSD
    plt.semilogy(ebno_db,
                 ber_plot128._bers[idx[2*i+1]],
                 c='C%d'%(i),
                 label= ber_plot128._legends[idx[2*i+1]].replace(" (BLER)", ""), #remove "(BLER)" from label
                 linestyle = "--",
                 linewidth=2)

plt.legend(fontsize=20)
plt.xlim([0, 4.5])
plt.ylim([1e-4, 1]);


# As can be seen, the performance of Polar and Convolutional codes is in practice close to their ML performance.
# For other codes such as LDPC codes, there is a practical performance gap under BP decoding which tends to be smaller for longer codes.

# ### Performance of Longer LDPC Codes
# 
# Now, let us have a look at the performance gains due to longer codewords. 
# For this, we scale the length of the LDPC code and compare the results (same rate, same decoder, same channel).

# In[10]:


# init new figure
ber_plot_ldpc = PlotBER(f"BER/BLER Performance of LDPC Codes @ Fixed Rate=0.5")


# In[11]:


# code parameters to simulate
ns = [128, 256, 512, 1000, 2000, 4000, 8000, 16000]  # number of codeword bits per codeword
rate = 0.5 # fixed coderate

# create list of encoder/decoder pairs to be analyzed
codes_under_test = []

# 5G LDPC codes
for n in ns:
    k = int(rate*n) # calculate k for given n and rate
    enc = LDPC5GEncoder(k=k, n=n)
    dec = LDPC5GDecoder(enc, num_iter=20)
    name = f"5G LDPC BP-20 (n={n})"
    codes_under_test.append([enc, dec, name, k, n])


# In[12]:


# and simulate the results
num_bits_per_symbol = 2 # QPSK

ebno_db = np.arange(0, 5, 0.25) # sim SNR range
# note that the waterfall for long codes can be steep and requires a fine
# SNR quantization

# run ber simulations for each case
for code in codes_under_test:
    print("Running: " + code[2])
    model = System_Model(k=code[3],
                         n=code[4],
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1])
    
    # the first argument must be a callable (function) that yields u and u_hat 
    # for given batch_size and ebno
    # we fix the target number of BLOCK errors instead of the BER to 
    # ensure that same accurate results for each block lengths is simulated
    ber_plot_ldpc.simulate(model, # the function have defined previously
                           ebno_dbs=ebno_db,
                           legend=code[2],
                           max_mc_iter=100,
                           num_target_block_errors=500, # we fix the target block errors
                           batch_size=1000,
                           soft_estimates=False, 
                           early_stop=True,
                           show_fig=False,
                           forward_keyboard_interrupt=True); # should be True in a loop

# and show figure
ber_plot_ldpc(ylim=(1e-5, 1))


# ## A Deeper Look into the Polar Code Module
# 
# A Polar code can be defined by a set of `frozen bit` and `information bit` positions [1].
# The package `sionna.fec.polar.utils` supports 5G-compliant Polar code design, but also Reed-Muller (RM) codes are available and can be used within the same encoder/decoder layer.
# If required, rate-matching and CRC concatenation are handled by the class `sionna.fec.polar.Polar5GEncoder` and `sionna.fec.polar.Polar5GDecoder`, respectively.
# 
# Further, the following decoders are available:
# 
# * Successive cancellation (SC) decoding [1]
#     * Fast and low-complexity
#     * Sub-optimal error-rate performance
# * Successive cancellation list (SCL) decoding [2]
#     * Excellent error-rate performance
#     * High-complexity
#     * CRC-aided decoding possible
# * Hybrid SCL decoder (combined SC and SCL decoder)
#     * Pre-decode with SC and only apply SCL iff CRC fails
#     * Excellent error-rate performance
#     * Needs outer CRC (e.g., as done in 5G)
#     * CPU-based implementation and, thus, no XLA support (+ increased decoding latency)
# * Iterative belief propagation (BP) decoding [6]
#     * Produces soft-output estimates
#     * Sub-optimal error-rate performance

# Let us now generate a new Polar code.

# In[13]:


code_type = "5G" # try also "RM"

# Load the 5G compliant polar code
if code_type=="5G":    
    k = 32
    n = 64
    # load 5G compliant channel ranking [3]
    frozen_pos, info_pos = generate_5g_ranking(k,n)
    print("Generated Polar code of length n = {} and k = {}".format(n, k))
    print("Frozen codeword positions: ", frozen_pos)

# Alternatively Reed-Muller code design is also available
elif code_type=="RM":    
    r = 3
    m = 7
    frozen_pos, info_pos, n, k, d_min = generate_rm_code(r, m)
    print("Generated ({},{}) Reed-Muller code of length n = {} and k = {} with minimum distance d_min = {}".format(r, m, n, k, d_min))
    print("Frozen codeword positions: ", frozen_pos)

else:
    print("Code not found")


# Now, we can initialize the encoder and a `BinarySource` to generate random Polar codewords.

# In[14]:


# init polar encoder
encoder_polar = PolarEncoder(frozen_pos, n)

# init binary source to generate information bits
source = BinarySource()
# define a batch_size
batch_size = 1

# generate random info bits
u = source([batch_size, k])
# and encode
c = encoder_polar(u)

print("Information bits: ", u.numpy())
print("Polar encoded bits: ", c.numpy())


# As can be seen, the length of the resulting code must be a power of 2. 
# This brings us to the problem of rate-matching and we will now have a closer look how we can adapt the length of the code.

# ## Rate-Matching and Rate-Recovery
# 
# The general task of rate-matching is to enable flexibility of the code w.r.t. the codeword length $n$ and information bit input size $k$ and, thereby, the rate $r = \frac{k}{n}$. 
# In modern communication standards such as 5G NR, these parameters can be adjusted on a bit-level granularity without - in a wider sense - redefining the (mother) code itself.
# This is enabled by a powerful rate-matching and the corresponding rate-recovery block which will be explained in the following.
# 
# 
# The principle idea is to select a mother code as close as possible to the desired properties from a set of possible mother codes.
# For example for Polar codes, the codeword length must be a power of 2, i.e., $n = 32, 64, ..., 512, 1024$.
# For LDPC codes the codeword length is more flexible (due to the different *lifting* factors), however, does not allow bit-wise granularity neither.
# Afterwards, the bit-level granularity is provided by shortening, puncturing and repetitions.
# 
# To summarize, the rate-matching procedure consists of:
# 
# 1. ) 5G NR defines multiple *mother* codes with similar properties (e.g., via base-graph lifting of LDPC code or sub-codes for Polar codes)
# 2. ) Puncturing, shortening and repetitions of bits to allow bit-level rate adjustments
# 
# The following figure summarizes the principle for the 5G NR Polar code uplink control channel (UCI). The Fig. is inspired by Fig. 6 in [9].

# 

# For bit-wise length adjustments, the following techniques are commonly used:
# 
# 1. ) *Puncturing:* A ($k,n$) mother code is punctured by *not* transmitting $p$ punctured codeword bits. Thus, the rate increases to $r_{\text{pun}} = \frac{k}{n-p} > \frac{k}{n} \quad \forall p > 0$.
# At the decoder these codeword bits are treated as erasure ($\ell_{\text{ch}} = 0$).
# 
# 2. ) *Shortening:* A ($k,n$) mother code is shortened by setting $s$ information bits to a fixed (=known) value. Assuming systematic encoding, these $s$ positions are not transmitted leading to a new code of rate $r_{\text{short}} =  \frac{k-s}{n-s}<\frac{k}{n}$. At the decoder these codeword bits are treated as known values ($\ell_{\text{ch}} = \infty$).
# 
# 3. ) *Repetitions* can be used to lower the effective rate.
# For details we refer the interested reader to [11].
# 
# 

# We will now simulate the performance of rate-matched 5G Polar codes for different lengths and rates. For this, we are interested in the required SNR to achieve a target BLER at $10^{-3}$.
# Please note that this is a reproduction of the results from [Fig.13a, 4].
# 
# **Note**: This needs a bisection search as we usually simulate the BLER at fixed SNR and, thus, this is simulation takes some time. Please only execute the cell below if you have enough simulation capabilities.

# In[15]:


# find the EsNo in dB to achieve target_bler
def find_threshold(model, # model to be tested 
                   batch_size=1000, 
                   max_batch_iter=10, # simulate cws up to batch_size * max_batch_iter
                   max_block_errors=100,  # number of errors before stop       
                   target_bler=1e-3): # target error rate to simulate (same as in[4])  
        """Bisection search to find required SNR to reach target SNR."""
        
        # bisection parameters        
        esno_db_min = -15 # smallest possible search SNR
        esno_db_max = 15 # largest possible search SNR
        esno_interval = (esno_db_max-esno_db_min)/4 # initial search interval size
        esno_db = 2*esno_interval + esno_db_min # current test SNR
        max_iters = 12 # number of iterations for bisection search

        # run bisection
        for i in range(max_iters):
            num_block_error = 0
            num_cws = 0
            for j in range(max_batch_iter):
                # run model and evaluate BLER
                u, u_hat = model(tf.constant(batch_size, tf.int32),
                                 tf.constant(esno_db, tf.float32))
                num_block_error += count_block_errors(u, u_hat) 
                num_cws += batch_size
                # early stop if target number of block errors is reached
                if num_block_error>max_block_errors:
                    break
            bler = num_block_error/num_cws
            # increase SNR if BLER was great than target
            # (larger SNR leads to decreases BLER)
            if bler>target_bler:
                esno_db += esno_interval        
            else: # and decrease SNR otherwise
                esno_db -= esno_interval 
            esno_interval = esno_interval/2
        
        # return final SNR after max_iters
        return esno_db


# In[16]:


# run simulations for multiple code parameters
num_bits_per_symbol = 2 # QPSK
# we sweep over multiple values for k and n
ks = np.array([12, 16, 32, 64, 128, 140, 210, 220, 256, 300, 400, 450, 460, 512, 800, 880, 940])
ns = np.array([160, 240, 480, 960])

# we use EsNo instead of EbNo to have the same results as in [4]
esno = np.zeros([len(ns), len(ks)])

for j,n in enumerate(ns):  
    for i,k in enumerate(ks):        
        if k<n: # only simulate if code parameters are feasible (i.e., r < 1)
            print(f"Finding threshold of k = {k}, n = {n}")

            # initialize new encoder / decoder pair
            enc = Polar5GEncoder(k=k, n=n)
            dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8) 
            #build model
            model = System_Model(k=k,
                                 n=n,
                                 num_bits_per_symbol=num_bits_per_symbol,
                                 encoder=enc,
                                 decoder=dec,
                                 sim_esno=True) # no rate adjustment 
            # and find threshold via bisection search
            esno[j, i] = find_threshold(model)
            print("Found threshold at: ", esno[j, i])


# In[17]:


# plot the results
leg_str = []
for j,n in enumerate(ns): 
    plt.plot(np.log2(ks[ks<n]), esno[j, ks<n])
    leg_str.append("n = {}".format(n))


# define labels manually
x_tick_labels = np.power(2, np.arange(3,11))
plt.xticks(ticks=np.arange(3,11),labels=x_tick_labels, fontsize=18)

# adjusted layout of figure
plt.grid("both")
plt.ylim([-10, 15])
plt.xlabel("Number of information bits $k$", fontsize=20)
plt.yticks(fontsize=18)
plt.ylabel("$E_s/N_0^*$ (dB)", fontsize=20)
plt.legend(leg_str, fontsize=18);
fig = plt.gcf() # get handle to current figure
fig.set_size_inches(15,10)


# This figure equals [Fig. 13a, 4] with a few small exception for extreme low-rate codes. This can be explained by the fact that the 3 explicit parity-bits bits are not implemented, however, these bits are only relevant for for $12\leq k \leq20$.
# It also explains the degraded performance of the n=960, k=16 code.

# ## Throughput and Decoding Complexity
# 
# In the last part of this notebook, you will compare the different computational complexity of the different codes and decoders.
# In theory the complexity is given as:
# 
# - Successive cancellation list (SCL) decoding of Polar codes scales with $\mathcal{O}(L \cdot n \cdot \operatorname{log} n)$ (with $L=1$ for SC decoding)
# - Iterative belief propagation (BP) decoding of LDPC codes scales with $\mathcal{O}(n)$.
# However, in particular for short codes a complexity comparison should be supported by empirical results.
# 
# We want to emphasize that the results strongly depend on the exact implementation and may differ for different implementations/optimizations.

# In[18]:


def get_throughput(batch_size, ebno_dbs, model, repetitions=1):
    """ Simulate throughput in bit/s per ebno_dbs point.
    
    The results are average over `repetition` trials.
    
    Input
    -----
    batch_size: tf.int32
        Batch-size for evaluation.

    ebno_dbs: tf.float32
        A tensor containing SNR points to be evaluated.    
    
    model:
        Function or model that yields the transmitted bits `u` and the
        receiver's estimate `u_hat` for a given ``batch_size`` and
        ``ebno_db``.
        
    repetitions: int
        An integer defining how many trails of the throughput 
        simulation are averaged.
    
    """
    throughput = np.zeros_like(ebno_dbs)

    # call model once to be sure it is compile properly 
    # otherwise time to build graph is measured as well.
    u, u_hat = model(tf.constant(batch_size, tf.int32),    
                     tf.constant(0., tf.float32))

    for idx, ebno_db in enumerate(ebno_dbs):

        t_start = time.perf_counter()
        # average over multiple runs
        for _ in range(repetitions):
            u, u_hat = model(tf.constant(batch_size, tf.int32),
                             tf.constant(ebno_db, tf. float32))
        t_stop = time.perf_counter()
        # throughput in bit/s
        throughput[idx] = np.size(u.numpy())*repetitions / (t_stop - t_start)

    return throughput


# In[19]:


# plot throughput and ber together for ldpc codes
# and simulate the results
num_bits_per_symbol = 2 # QPSK

ebno_db = [5] # SNR to simulate
num_bits_per_batch = 5e6 # must be reduced in case of out-of-memory errors
num_repetitions = 20 # average throughput over multiple runs

# run throughput simulations for each code
throughput = np.zeros(len(codes_under_test))
code_length = np.zeros(len(codes_under_test))
for idx, code in enumerate(codes_under_test):
    print("Running: " + code[2])
    
    # save codeword length for plotting
    code_length[idx] = code[4] 
    
    # init new model for given encoder/decoder
    model = System_Model(k=code[3],
                         n=code[4],
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=code[0],
                         decoder=code[1])
    
    # scale batch_size such that same number of bits is simulated for all codes
    batch_size = int(num_bits_per_batch / code[4])
    # and measure throughput of the model
    throughput[idx] = get_throughput(batch_size,
                                     ebno_db,
                                     model,
                                     repetitions=num_repetitions)


# In[20]:


# plot results
plt.figure(figsize=(16,10))

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.title("Throughput LDPC BP Decoding @ rate=0.5", fontsize=25)
plt.xlabel("Codeword length", fontsize=25)
plt.ylabel("Throughput (Mbit/s)", fontsize=25)
plt.grid(which="both")

# and plot results (logarithmic scale in x-dim)
x_tick_labels = code_length.astype(int)
plt.xticks(ticks=np.log2(code_length),labels=x_tick_labels, fontsize=18)
plt.plot(np.log2(code_length), throughput/1e6)


# As expected the throughput of BP decoding is (relatively) constant as the complexity scales linearly with $\mathcal{O}(n)$ and, thus, the complexity *per* decoded bit remains constant.
# It is instructive to realize that the above plot is in the log-domain for the x-axis.
# 
# Let us have a look at what happens for different SNR values.

# In[21]:


# --- LDPC ---
n = 1000
k = 500
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder)

# init a new model
model = System_Model(k=k,
                     n=n,
                     num_bits_per_symbol=num_bits_per_symbol,
                     encoder=encoder,
                     decoder=decoder)

# run throughput tests at 2 dB and 5 dB
ebno_db = [2, 5]
batch_size = 10000
throughput = get_throughput(batch_size,
                            ebno_db, # snr point
                            model,
                            repetitions=num_repetitions)

# and print the results
for idx, snr_db in enumerate(ebno_db):
    print(f"Throughput @ {snr_db:.1f} dB: {throughput[idx]/1e6:.2f} Mbit/s")


# For most Sionna decoders the throughput is not SNR dependent as early stopping of individual samples within a batch is difficult to realize.
# 
# However, the `hybrid SCL` decoder uses an internal NumPy SCL decoder only if the SC decoder failed similar to [10].
# We will now benchmark this decoder for different SNR values.

# In[22]:


# --- Polar ---
n = 256
k = 128
encoder = Polar5GEncoder(k, n)
decoder = Polar5GDecoder(encoder, "hybSCL")

# init a new model
model = System_Model(k=k,
                     n=n,
                     num_bits_per_symbol=num_bits_per_symbol,
                     encoder=encoder,
                     decoder=decoder)

ebno_db = np.arange(0, 5, 0.5) # EbNo to evaluate
batch_size = 1000
throughput = get_throughput(batch_size,
                            ebno_db, # snr point
                            model,
                            repetitions=num_repetitions)

# and print the results
for idx, snr_db in enumerate(ebno_db):
    print(f"Throughput @ {snr_db:.1f} dB: {throughput[idx]/1e6:.3f} Mbit/s")


# We can overlay the throughput with the BLER of the SC decoder. 
# This can be intuitively explained by the fact that he `hybrid SCL` decoder consists of two decoding stages:
# 
# - SC decoding for all received codewords.
# - SCL decoding *iff* the CRC does not hold, i.e., SC decoding did not yield the correct codeword.
# 
# Thus, the throughput directly depends on the BLER of the internal SC decoder.

# In[23]:


ber_plot_polar = PlotBER("Polar SC/SCL Decoding")

ber_plot_polar.simulate(model, # the function have defined previously
                        ebno_dbs=ebno_db,
                        legend="hybrid SCL decoding",
                        max_mc_iter=100,
                        num_target_block_errors=100, # we fix the target bler
                        batch_size=1000,
                        soft_estimates=False, 
                        early_stop=True,
                        add_ber=False,
                        add_bler=True,
                        show_fig=False,
                        forward_keyboard_interrupt=False);

# and add SC decoding
decoder2 = Polar5GDecoder(encoder, "SC")

model = System_Model(k=k,
                     n=n,
                     num_bits_per_symbol=num_bits_per_symbol,
                     encoder=encoder,
                     decoder=decoder2)

ber_plot_polar.simulate(model, # the function have defined previously
                        ebno_dbs=ebno_db,
                        legend="SC decoding",
                        max_mc_iter=100,
                        num_target_block_errors=100, # we fix the target bler
                        batch_size=1000,
                        soft_estimates=False, 
                        early_stop=True,
                        add_ber=False, # we only focus on BLER
                        add_bler=True,
                        show_fig=False,
                        forward_keyboard_interrupt=False);


# Let us visualize the results.

# In[24]:


ber_plot_polar()
ax2 = plt.gca().twinx()  # new axis 
ax2.plot(ebno_db, throughput, 'g', label="Throughput hybSCL-8")
ax2.legend(fontsize=20)
ax2.set_ylabel("Throughput (bit/s)", fontsize=25);
ax2.tick_params(labelsize=25)


# You can also try:
# 
# - Analyze different rates
# - What happens for different batch-sizes? Can you explain what happens?
# - What happens for higher order modulation. Why is the complexity increased?

# ## References
# 
# [1] E. Arikan, "Channel polarization: A method for constructing capacity-achieving codes for symmetric binary-input memoryless channels," IEEE Transactions on Information Theory, 2009.
# 
# [2] Ido Tal and Alexander Vardy, "List Decoding of Polar Codes." IEEE Transactions on Information Theory, 2015.
# 
# [3] ETSI 3GPP TS 38.212 "5G NR Multiplexing and channel coding", v.16.5.0, 2021-03.
# 
# [4] V. Bioglio, C. Condo, I. Land, "Design of Polar Codes in 5G New Radio." IEEE Communications Surveys & Tutorials, 2020.
# 
# [5] D. Hui, S. Sandberg, Y. Blankenship, M. Andersson, L. Grosjean "Channel coding in 5G new radio: A Tutorial Overview and Performance Comparison with 4G LTE." IEEE Vehicular Technology Magazine, 2018.
# 
# [6] E. Arikan, “A Performance Comparison of Polar Codes and Reed-Muller Codes,” IEEE Commun. Lett., vol. 12, no. 6, pp. 447–449, Jun. 2008.
# 
# [7] R. G. Gallager, Low-Density Parity-Check Codes, M.I.T. Press Classic Series, Cambridge MA, 1963.
# 
# [8] T. Richardson and S. Kudekar. "Design of low-density parity check codes for 5G new radio," IEEE Communications Magazine 56.3, 2018.
# 
# [9] G. Liva, L. Gaudio, T. Ninacs, T. Jerkovits, "Code design for short blocks: A survey," arXiv preprint arXiv:1610.00873, 2016.
# 
# [10] S. Cammerer, B. Leible, M. Stahl, J. Hoydis, and S ten Brink, "Combining Belief Propagation and Successive Cancellation List Decoding of Polar Codes on a GPU Platform," IEEE ICASSP, 2017.
# 
# [11] V. Bioglio, F. Gabry, I. Land, "Low-complexity puncturing and shortening of polar codes," IEEE Wireless Communications and Networking Conference Workshops (WCNCW), 2017.
# 
# [12] M. Fossorier, S. Lin, "Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics", IEEE Transactions on Information Theory, vol. 41, no. 5, 1995.
-e 
# --- End of 5G_Channel_Coding_Polar_vs_LDPC_Codes.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # 5G NR PUSCH Tutorial
# 
# 
# 
# You will
# 
# - Get an understanding of the different components of a PUSCH configuration, such as the carrier, DMRS, and transport block,
# - Learn how to rapidly simulate PUSCH transmissions for multiple transmitters,
# - Modify the PUSCHReceiver to use a custom MIMO Detector.
# 
# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [A "Hello World!" Example](#A-Hello-World-Example)
# * [Carrier Configuration](#Carrier-Configuration)
# * [Understanding the DMRS Configuration](#Understanding-the-DMRS-Configuration)
#     * [Configuring Multiple Layers](#Configuring-Multiple-Layers)
#     * [Controlling the Number of DMRS Symbols in a Slot](#Controlling-the-Number-of-DMRS-Symbols-in-a-Slot)
#     * [How to control the number of available DMRS ports?](#How-to-control-the-number-of-available-DMRS-ports?)
# * [Transport Blocks and MCS](#Transport-Blocks-and-MCS)
# * [Looking into the PUSCHTransmitter](#Looking-into-the-PUSCHTransmitter)
# * [Components of the PUSCHReceiver](#Components-of-the-PUSCHReceiver)
# * [End-to-end PUSCH Simulations](#End-to-end-PUSCH-Simulations)
# 

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

sionna.phy.config.seed = 42 # Set seed for reproducible results

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.channel import AWGN, RayleighBlockFading, OFDMChannel, \
                               TimeChannel, time_lag_discrete_time_channel
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi, UMa, RMa
from sionna.phy.utils import compute_ber, ebnodb2no, sim_ber
from sionna.phy.ofdm import KBestDetector, LinearDetector
from sionna.phy.mimo import StreamManagement


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import time


# ## A Hello World Example

# Let us start with a simple "Hello, World!" example in which we will simulate PUSCH transmissions from a single transmitter to a single receiver over an AWGN channel.

# In[3]:


# Create a PUSCH configuration with default settings
pusch_config = PUSCHConfig()

# Instantiate a PUSCHTransmitter from the PUSCHConfig
pusch_transmitter = PUSCHTransmitter(pusch_config)

# Create a PUSCHReceiver using the PUSCHTransmitter
pusch_receiver = PUSCHReceiver(pusch_transmitter)

# AWGN channel
channel = AWGN()

# Simulate transmissions over the AWGN channel
batch_size = 16
no = 0.1 # Noise variance

x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits

y = channel(x, no) # Simulate channel output

b_hat = pusch_receiver(y, no) # Recover the info bits

# Compute BER
print("BER:", compute_ber(b, b_hat).numpy())


# Although the above code snippet seems rather simple, you have actually carried out standard-compliant simulations of the NR PUSCH!
# 
# To better understand what is actually going on under the hood, we can inspect the OFDM resource grid that is generated by the transmitter with the following command:

# In[4]:


pusch_transmitter.resource_grid.show();


# The above figure tells us that we are simulating a slot of 14 OFDM symbols spanning 48 subcarriers, which correspond to four physical resource blocks (PRBs) in 5G terminology. The third OFDM symbol is reserved for pilot transmissions, so-called demodulation reference signals (DMRS), and the rest is used for data. 

# ## Carrier Configuration
# 
# When you create a PUSCHConfig instance, it automatically creates a CarrierConfig instance with default settings.
# You can inspect this configuration with the following command:

# In[5]:


pusch_config.carrier.show()


# Most of these parameters cannot be controlled as they are simply derived from others. For example, the cyclic prefix length depends on the subcarrier spacing.
# Let us see what happens, when we choose larger subcarrier spacing:

# In[6]:


pusch_config.carrier.subcarrier_spacing = 60
pusch_config.carrier.show()


# The cyclic prefix has shrunk from $5.2 \mu s$ to $1.69 \mu s$ and the number of slots per frame has increased from $10$ to $40$.
# 
# If we change to the extended cyclic prefix, the number of OFDM symbols per slot will decrease from 14 to 12.

# In[7]:


pusch_config_ext = pusch_config.clone()
pusch_config_ext.carrier.cyclic_prefix = "extended"
pusch_config_ext.carrier.show()


# Please have a look at the API documentation of [PUSCHCarrierConfig](https://nvlabs.github.io/sionna/phy/api/nr.html#sionna.phy.nr.PUSCHConfig) for more detail.

# ## Understanding the DMRS Configuration
# 
# We can learn more about the structure of the resoure grid by having a look at the pilot pattern in the next section.

# In[8]:


pusch_transmitter.pilot_pattern.show();


# From the figure above, we can see that there is a single transmitter sending a single stream (or so-called layer).
# DMRS are only sent on even subcarriers while odd subcarriers are masked, i.e., blocked for data transmission.
# This corresponds to the DMRS Configuration Type 1 with the parameter `NumCDMGroupsWithoutData` set to 2. We will explain what that means later.
# 
# In 5G NR, one can configure many different pilot patterns to adapt to different channel conditions and to allow for spatial multiplexing of up to twelve layers. Each transmitted layer is identified by a DMRS port, i.e., a distinct pilot pattern. In our running example, the transmitter uses the DMRS port 0.
# 
# With the current PUSCH configuration, four different DMRS ports 0,1,2,3 are available.
# This can be verified with the following command:

# In[9]:


pusch_config.dmrs.allowed_dmrs_ports


# Next, we configure three other transmitters using each one of the remaing ports. Then, we create a new PUSCHTransmitter instance from the list of PUSCH configurations which is able to generate transmit signals for all four transmitters in parallel.

# In[10]:


# Clone the original PUSCHConfig and change the DMRS port set
pusch_config_1 = pusch_config.clone()
pusch_config_1.dmrs.dmrs_port_set = [1]
pusch_config_2 = pusch_config.clone()
pusch_config_2.dmrs.dmrs_port_set = [2]
pusch_config_3 = pusch_config.clone()
pusch_config_3.dmrs.dmrs_port_set = [3]

# Create a PUSCHTransmitter from the list of PUSCHConfigs
pusch_transmitter_multi = PUSCHTransmitter([pusch_config, pusch_config_1, pusch_config_2, pusch_config_3])

# Generate a batch of random transmit signals
x, b  = pusch_transmitter_multi(batch_size)

# x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
print("Shape of x:", x.shape)


# Inspecting the shape of x reveals that we have indeed four single-antenna transmitters. Let us now have a look at the resuling pilot pattern for each of them:

# In[11]:


pusch_transmitter_multi.pilot_pattern.show();


# As before, all transmitters send pilots only on the third OFDM symbol. Transmitter 0 and 1 (using DMRS port 0 and 1, respectively) send pilots on all even subcarriers, while Transmitter 2 and 3 (using DMRS port 2 and 3, respectively), send pilots on the odd subcarriers.
# This means that the pilots signals of DMRS port 0 and 1 (as well as 2 and 3) interfere with each other as they occupy the same resource elements.
# So how can we estimate the channel coefficients for both transmitters individually without pilot contamination? 
# 
# The solution to this problem are the so-called code division multiplexing (CDM) groups in 5G NR.
# DMRS ports 0,1 belong to CDM group 0, while DMRS ports 2,3 belong to CDM group 1.
# 
# The pilot signals belonging to the same CDM group are multiplied by orthogonal cover codes which allow separating them during channel estimation.
# The way this works is as follows. Denote by $\mathbf{p_0} = [s_1, s_2]^\textsf{T}$ a pair of two adjacent pilot symbols, e.g., those on subcarrier 0 and 2, of DMRS port 0. DMRS port 1 will simply send $\mathbf{p_1} = [s_1, -s_2]^\textsf{T}$. If we assume that the channel is constant over both subcarriers, we get the following received pilot signal at the receiver (we look only at a single antenna here):
# 
# \begin{align}
# \mathbf{y} = h_0\mathbf{p}_0 + h_1\mathbf{p}_1 + \mathbf{n}
# \end{align}
# 
# where $\mathbf{y}\in\mathbb{C}^2$ is the received signal on both subcarriers, $h_0, h_1$ are the channel coefficients for both users, and $\mathbf{n}\in\mathbb{C}^2$ is a noise vector.
# 
# We can now obtain channel estimates for both transmitters by projecting $\mathbf{y}$ onto their respective pilot sequences:
# 
# \begin{align}
# \hat{h}_0 &= \frac{\mathbf{p}_0^\mathsf{H}}{\lVert \mathbf{p}_0 \rVert|^2} \mathbf{y} = h_0 + \frac{|s_1|^2-|s_2|^2}{\lVert \mathbf{p}_0 \rVert|^2} h_1 + \frac{\mathbf{p}_0^\mathsf{H}}{\lVert \mathbf{p}_0 \rVert|^2} \mathbf{n} = h_0 + n_0 \\
# \hat{h}_1 &= \frac{\mathbf{p}_1^\mathsf{H}}{\lVert \mathbf{p}_1 \rVert|^2} \mathbf{y} = \frac{|s_1|^2-|s_2|^2}{\lVert \mathbf{p}_1 \rVert|^2} h_0 + h_1 +\frac{\mathbf{p}_1^\mathsf{H}}{\lVert \mathbf{p}_1 \rVert|^2} \mathbf{n} = h_1 + n_1.
# \end{align}
# 
# Since the pilot symbols have the same amplitude, we have $|s_1|^2-|s_2|^2=0$, i.e., the interference between both pilot sequence is zero. Moreover, due to an implict averaging of the channel estimates for both subcarriers, the effective noise variance is reduced by a factor of 3dB since
# 
# \begin{align}
# \mathbb{E}\left[ |n_0|^2 \right] = \mathbb{E}\left[ |n_1|^2 \right] = \frac{\sigma^2}{\lVert \mathbf{p}_1 \rVert|^2} = \frac{\sigma^2}{2 |s_0|^2}.
# \end{align}
# 
# We can access the actual pilot sequences that are transmitted as follows:

# In[12]:


# pilots has shape [num_tx, num_layers, num_pilots]
pilots = pusch_transmitter_multi.pilot_pattern.pilots
print("Shape of pilots:", pilots.shape)

# Select only the non-zero subcarriers for all sequence
p_0 = pilots[0,0,::2] # Pilot sequence of TX 0 on even subcarriers
p_1 = pilots[1,0,::2] # Pilot sequence of TX 1 on even subcarriers
p_2 = pilots[2,0,1::2] # Pilot sequence of TX 2 on odd subcarriers
p_3 = pilots[3,0,1::2] # Pilot sequence of TX 3 on odd subcarriers


# Each pilot pattern consists of 48 symbols that are transmitted on the third OFDM symbol with 4PRBs, i.e., 48 subcarriers. 
# Let us now verify that pairs of two adjacent pilot symbols in `p_0` and `p_1` as well as in `p_2` and `p_3` are orthogonal.

# In[13]:


print(np.sum(np.reshape(p_0, [-1,2]) * np.reshape(np.conj(p_1), [-1,2]), axis=1))
print(np.sum(np.reshape(p_2, [-1,2]) * np.reshape(np.conj(p_3), [-1,2]), axis=1))


# Let us now come back to the masked resource elements in each pilot pattern. 
# The parameter `NumCDMGroupsWithoutData` mentioned earlier determines which resource elements in a DMRS-carrying OFDM symbol are masked for data transmissions. This is to avoid inference with pilots from other DMRS groups.
# 
# In our example, `NumCDMGroupsWithoutData` is set to two. This means that no data can be transmitted on any of the resource elements occupied by both DMRS groups. However, if we would have set `NumCDMGroupsWithoutData` equal to one, data and pilots would be frequency multiplexed. 
# This can be useful, if we only schedule transmissions from DMRS ports in the same CDM group.
# 
# Here is an example of such a configuration:

# In[14]:


pusch_config = PUSCHConfig()
pusch_config.dmrs.num_cdm_groups_without_data = 1
pusch_config.dmrs.dmrs_port_set = [0]

pusch_config_1 = pusch_config.clone()
pusch_config_1.dmrs.dmrs_port_set = [1]

PUSCHTransmitter([pusch_config, pusch_config_1]).pilot_pattern.show();


# The DRMS ports 0 and 1 belong both to CDM group 0 so that the resource elements of CDM group 1 do not need to be masked and can be used for data transmission. One can see in the above figure that data and pilots are now indeed multiplexed in the frequency domain.

# ### Configuring Multiple Layers

# In 5G NR, a transmitter can be equipped with 1,2, or 4 antenna ports, i.e., physical antennas that are fed with an individual transmit signal.
# It can transmit 1,2,3 or 4 layers, i.e., spatial streams, as long as the number of layers does not exceed the number of antenna ports. 
# Using codebook-based precoding, a number of layers can be mapped onto a larger number of antenna ports, e.g., 2 layers using 4 antenna ports. If no precoding is used, each layer is simply mapped to one of the antenna ports.
# 
# It is important to understand that each layer is transmitted using a different DMRS port. That means that the number of DMRS ports is independent of the number of antenna ports.
# 
# In the next cell, we will configure a single transmitter with four antenna ports, sending two layers on DMRS ports 0 and 1.
# We can then choose among different precoding matrices with the help of the transmit precoding matrix identifier (TPMI). 

# In[15]:


pusch_config = PUSCHConfig()
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 2
pusch_config.dmrs.dmrs_port_set = [0,1]
pusch_config.precoding = "codebook"
pusch_config.tpmi = 7

# Show the precoding matrix
pusch_config.precoding_matrix


# In[16]:


PUSCHTransmitter(pusch_config).pilot_pattern.show();


# We can see from the pilot patterns above, that we have now a single transmitter sending two streams. Both streams will be precoded and transmit over four antenna ports. From a channel estimation perspective at the receiver, however, this scenario is identical to the previous one with two single-antenna transmitters. The receiver will simply estimate the effective channel (including precoding) for every configured DMRS port.

# ### Controlling the Number of DMRS Symbols in a Slot
# 
# How can we add additional DMRS symbols to the resource grid to enable channel estimation for high-speed scenarios? 
# 
# This can be controlled with the parameter ``DMRS.additional_position``.
# In the next cell, we configure one additional DMRS symbol to the pattern and visualize it. You can try setting it to different values and see the impact.

# In[17]:


pusch_config.dmrs.additional_position = 1

# In order to reduce the number of figures, we here only show
# the pilot pattern of the first stream
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);


# ### How to control the number of available DMRS ports?
# 
# There are two factors that determine the available number of DMRS ports, i.e., layers, that can be transmitted. 
# The first is the DMRS Configuration and the second the length of a DMRS symbol. Both parameters can take two values so that there are four options in total.
# In the previous example, the DMRS Configuration Type 1 was used. In this case, there are two CDM groups and each group uses either odd or even subcarriers. This leads to four available DMRS ports.
# With DMRS Configuration Type 2, there are three CDM groups and each group uses two pairs of adjacent subcarriers per PRB, i.e., four pilot-carrying subcarriers. That means that there are six available DMRS ports.
# 
# 
# 
# 

# In[18]:


pusch_config.dmrs.config_type = 2
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);
print("Available DMRS ports:", pusch_config.dmrs.allowed_dmrs_ports)


# In the above figure, you can see that the pilot pattern has become sparser in the frequency domain. However, there are still only four available DMRS ports. This is because we now need to mask also the resource elements that are used by the third CDM group. This can be done by setting the parameter `NumCDMGroupsWithoutData` equal to three.

# In[19]:


pusch_config.dmrs.num_cdm_groups_without_data = 3
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);
print("Available antenna ports:", pusch_config.dmrs.allowed_dmrs_ports)


# The second parameter that controls the number of available DMRS ports is the ``length``, which can be equal to either one or two.
# Let's see what happens when we change it to two.

# In[20]:


pusch_config.n_size_bwp = 1 # We reduce the bandwidth to one PRB for better visualization
pusch_config.dmrs.length = 2
PUSCHTransmitter(pusch_config).pilot_pattern.show(stream_ind = 0);
print("Available DMRS ports:", pusch_config.dmrs.allowed_dmrs_ports)


# The pilot pattern is now composed of four 2x2 blocks within a PRB. These blocks are used by the four DMRS ports within the same CDM group. This means that we can now support up to twelve layers! 
# 
# Let's create a setup with three transmitters, each sending four layers using four antenna ports. We choose the DMRS ports for each transmitters such that they belong to the CDM group. This is not necessary and you are free to choose any desired allocation. 
# It is however important to understand, that for channel estimation to work, the channel is supposed to be static over 2x2 blocks of resource elements. This is in general the case for low mobility scenarios and channels with not too large delay spread. You can see from the results below that the pilot sequences of the DMRS ports in the same CDM group are indeed orthogonal over the 2x2 blocks.

# In[21]:


pusch_config = PUSCHConfig()
pusch_config.n_size_bwp = 1
pusch_config.dmrs.config_type = 2
pusch_config.dmrs.length = 2
pusch_config.dmrs.additional_position = 1
pusch_config.dmrs.num_cdm_groups_without_data = 3
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 4
pusch_config.dmrs.dmrs_port_set = [0,1,6,7]
pusch_config.precoding = "codebook"
pusch_config.tpmi = 4

pusch_config_1 = pusch_config.clone()
pusch_config_1.dmrs.dmrs_port_set = [2,3,8,9]
pusch_config_2 = pusch_config.clone()
pusch_config_2.dmrs.dmrs_port_set = [4,5,10,11]

pusch_transmitter_multi = PUSCHTransmitter([pusch_config, pusch_config_1, pusch_config_2])


# In[22]:


# Extract the first 2x2 block of pilot symbols for all DMRS ports of the first transmitter
p = pusch_transmitter_multi.pilot_pattern.pilots[0].numpy()
p = np.matrix(p[:, [0,1,12,13]])

# Test that these pilot sequences are mutually orthogonal
# The result should be a boolean identity matrix
np.abs(p*p.getH())>1e-6


# There are several other parameters that impact the pilot patterns. The full DMRS configuration can be displayed with the following command.
# We refer to the [API documentation of the PUSCHDMRSConfig class](https://nvlabs.github.io/sionna/phy/api/nr.html#puschdmrsconfig) for further details.

# In[23]:


pusch_config.dmrs.show()


# ## Transport Blocks and MCS
# 
# The modulation and coding scheme (MCS) is set in 5G NR via the MCS index and MCS table which are properties of transport block configuration [TBConfig](https://nvlabs.github.io/sionna/phy/api/nr.html#sionna.nr.TBConfig). When you create an instance of ``PUSCHConfig``, a default instance of ``TBConfig`` is created. It can be accessed via the following command:

# In[24]:


pusch_config = PUSCHConfig()
pusch_config.tb.show()


# 
# We can change the MCS index and table as follows:

# In[25]:


pusch_config.tb.mcs_index = 26
pusch_config.tb.mcs_table = 2
pusch_config.tb.show()


# The transport block segmentation allows the PUSCH transmitter to fill resource grids of almost arbitrary size and with any of the possible DMRS configurations.
# The number of information bits transmitted in a single slot is given by the property ``tb_size`` of the ``PUSCHConfig``.
# 

# In[26]:


# Adding more PRBs will increase the TB size
pusch_config.carrier.n_size_grid = 273
pusch_config.tb_size


# In[27]:


# Adding more layers will increase the TB size
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 4
pusch_config.tb_size


# For more details about how the transportblock encoding/decoding works, we refer to the API documentation of the [TBEncoder](https://nvlabs.github.io/sionna/phy/api/nr.html#sionna.nr.TBEncoder).

# ## Looking into the PUSCHTransmitter
# 
# We have used the ``PUSCHTransmitter`` class already multiple times without speaking about what it actually does. In short, it generates for every configured transmitter a batch of random information bits of length ``pusch_config.tb_size`` and outputs either a frequency to time-domain representation of the transmitted OFDM waveform from each of the antenna ports of each transmitter.
# 
# However, under the hood it implements the sequence of layers shown in the following figure: 

# ![Picture 1.jpg](attachment:45dafb96-b379-4430-82c8-9df578084676.jpg)

# x
# . If ``output_domain`` is chosen to be “time”, the resource grids are transformed into time-domain signals by the [OFDMModulator](https://nvlabs.github.io/sionna/phy/api/ofdm.html#sionna.ofdm.OFDMModulator).

# Let us configure a ``PUSCHTransmitter`` from a list of two ``PUSCHConfig`` and inspect the output shapes: 

# In[28]:


pusch_config = PUSCHConfig()
pusch_config.num_antenna_ports = 4
pusch_config.num_layers = 2
pusch_config.dmrs.dmrs_port_set = [0,1]
pusch_config.precoding = "codebook"
pusch_config.tpmi = 7

pusch_config_1 = pusch_config.clone()
pusch_config.dmrs.dmrs_port_set = [2,3]

pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1])

batch_size = 32
x, b = pusch_transmitter(batch_size)

# b has shape [batch_size, num_tx, tb_size]
print("Shape of b:", b.shape)

# x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
print("Shape of x:", x.shape)


# If you want to transmit a custom payload, you simply need to deactive the ``return_bits`` flag when creating the transmitter:

# In[29]:


pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1], return_bits=False)
x_2 = pusch_transmitter(b)
assert np.array_equal(x, x_2) # Check that we get the same output for the payload b generated above


# By default, the ``PUSCHTransmitter`` generates frequency-domain outputs. If you want to make time-domain simulations, you need to configure the ``output_domain`` during the initialization:

# In[30]:


pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1], output_domain="time", return_bits=False)
x_time = pusch_transmitter(b)

# x has shape [batch_size, num_tx, num_tx_ant, num_time_samples]
print("Shape of x:", x_time.shape)


# The last dimension of the output signal correspond to the total number of time-domain samples which can be computed in the following way:

# In[31]:


(pusch_transmitter.resource_grid.cyclic_prefix_length  \
 + pusch_transmitter.resource_grid.fft_size) \
* pusch_transmitter.resource_grid.num_ofdm_symbols


# ## Components of the PUSCHReceiver
# 
# The [PUSCHReceiver](https://nvlabs.github.io/sionna/phy/api/nr.html#sionna.nr.PUSCHReceiver) is the counter-part to the [PUSCHTransmitter](https://nvlabs.github.io/sionna/phy/api/nr.html#sionna.nr.PUSCHTransmitter) as it *simply* recovers the transmitted information bits from received waveform. It combines multiple processing blocks in a single layer as shown in the following figure:

# ![Picture 2.jpg](attachment:f7ad545c-20f0-47d6-8f02-6473dce4f964.jpg)

# If the ``input_domain`` equals “time”, the inputs $\mathbf{y}$
#  
# If we instantiate a ``PUSCHReceiver`` as done in the next cell, default implementations of all blocks as described in the [API documentation](https://nvlabs.github.io/sionna/phy/api/nr.html#sionna.nr.PUSCHReceiver) are used.  
# 

# In[32]:


pusch_receiver = PUSCHReceiver(pusch_transmitter)
pusch_receiver._mimo_detector


# We can also provide custom implementations for each block by providing them as keyword arguments during initialization.
# In the folllwing code snippet, we first create an instance of the [KBestDetector](https://nvlabs.github.io/sionna/phy/api/ofdm.html#sionna.ofdm.KBestDetector), which is then used as MIMO detector in the ``PUSCHReceiver``.

# In[33]:


# Create a new PUSCHTransmitter
pusch_transmitter = PUSCHTransmitter([pusch_config, pusch_config_1])

# Create a StreamManagement instance
rx_tx_association = np.ones([1, pusch_transmitter.resource_grid.num_tx], bool)
stream_management = StreamManagement(rx_tx_association,
                                     pusch_config.num_layers)

# Get relevant parameters for the detector
num_streams = pusch_transmitter.resource_grid.num_tx \
              * pusch_transmitter.resource_grid.num_streams_per_tx

k = 32 # Number of canditates for K-Best detection

k_best = KBestDetector("bit", num_streams, k,
                       pusch_transmitter.resource_grid,
                       stream_management,
                       "qam", pusch_config.tb.num_bits_per_symbol)

# Create a PUSCHReceiver using the KBest detector
pusch_receiver = PUSCHReceiver(pusch_transmitter, mimo_detector=k_best)


# Next, we test if this receiver works over a simple Rayleigh block fading channel:

# In[34]:


num_rx_ant = 16
rayleigh = RayleighBlockFading(num_rx=1,
                               num_rx_ant=num_rx_ant,
                               num_tx=pusch_transmitter.resource_grid.num_tx,
                               num_tx_ant=pusch_config.num_antenna_ports)

channel = OFDMChannel(rayleigh,
                      pusch_transmitter.resource_grid,
                      add_awgn=True,
                      normalize_channel=True)

x, b = pusch_transmitter(32)
no = 0.1
y = channel(x, no)
b_hat = pusch_receiver(y, no)
print("BER:", compute_ber(b, b_hat).numpy())


# ## End-to-end PUSCH Simulations
# 
# We will now implement an end-to-end model that is capable of running PUSCH simulations for many different configurations.
# You can use it as a boilerplate template for your own experiments.

# In[35]:


class Model(Block):
    """Simulate PUSCH transmissions over a 3GPP 38.901 model

    This model runs BER simulations for a multiuser MIMO uplink channel
    compliant with the 5G NR PUSCH specifications.
    You can pick different scenarios, i.e., channel models, perfect or
    estimated CSI, as well as different MIMO detectors (LMMSE or KBest).
    You can chosse to run simulations in either time ("time") or frequency ("freq")
    domains and configure different user speeds.

    Parameters
    ----------
    scenario : str, one of ["umi", "uma", "rma"]
        3GPP 38.901 channel model to be used

    perfect_csi : bool
        Determines if perfect CSI is assumed or if the CSI is estimated

    domain :  str, one of ["freq", "time"]
        Domain in which the simulations are carried out.
        Time domain modelling is typically more complex but allows modelling
        of realistic effects such as inter-symbol interference of subcarrier
        interference due to very high speeds.

    detector : str, one of ["lmmse", "kbest"]
        MIMO detector to be used. Note that each detector has additional
        parameters that can be configured in the source code of the _init_ call.

    speed: float
        User speed (m/s)

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
                 scenario,    # "umi", "uma", "rma"
                 perfect_csi, # bool
                 domain,      # "freq", "time"
                 detector,    # "lmmse", "kbest"
                 speed        # float
                ):
        super().__init__()
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        self._domain = domain
        self._speed = speed

        self._carrier_frequency = 3.5e9
        self._subcarrier_spacing = 30e3
        self._num_tx = 4
        self._num_tx_ant = 4
        self._num_layers = 2
        self._num_rx_ant = 16
        self._mcs_index = 14
        self._mcs_table = 1
        self._num_prb = 16

        # Create PUSCHConfigs

        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing/1000
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_tx_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 3
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
        self._l_min, self._l_max = time_lag_discrete_time_channel(self._pusch_transmitter.resource_grid.bandwidth)


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
                                                 channel_estimator="perfect",
                                                 l_min = self._l_min)
        else:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 l_min = self._l_min)

        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=int(self._num_tx_ant/2),
                                 polarization="dual",
                                 polarization_type="cross",
                                 antenna_pattern="omni",
                                 carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_rx_ant/2),
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

        # Configure the actual channel
        if domain=="freq":
            self._channel = OFDMChannel(
                                self._channel_model,
                                self._pusch_transmitter.resource_grid,
                                normalize_channel=True,
                                return_channel=True)
        else:
            self._channel = TimeChannel(
                                self._channel_model,
                                self._pusch_transmitter.resource_grid.bandwidth,
                                self._pusch_transmitter.resource_grid.num_time_samples,
                                l_min=self._l_min,
                                l_max=self._l_max,
                                normalize_channel=True,
                                return_channel=True)

    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_tx,
                                self._scenario,
                                min_ut_velocity=self._speed,
                                max_ut_velocity=self._speed)

        self._channel_model.set_topology(*topology)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
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


# We will now compare the PUSCH BLER performance over the 3GPP 38.901 UMi channel model with different detectors and either perfect or imperfect CSI.
# Note that these simulations might take some time depending or you available hardware. You can reduce the `batch_size` if the model does not fit into the memory of your GPU. Running the simulations in the time domain will significantly increase the complexity and you might need to decrease the `batch_size` further. The code will also run on CPU if not GPU is available. 
# 
# If you do not want to run the simulation yourself, you can skip the next cell and visualize the results in the next cell.

# In[36]:


PUSCH_SIMS = {
    "scenario" : ["umi"],
    "domain" : ["freq"],
    "perfect_csi" : [True, False],
    "detector" : ["kbest", "lmmse"],
    "ebno_db" : list(range(-2,11)),
    "speed" : 3.0,
    "batch_size_freq" : 128,
    "batch_size_time" : 28, # Reduced batch size from time-domain modeling
    "bler" : [],
    "ber" : []
    }

start = time.time()

for scenario in PUSCH_SIMS["scenario"]:
    for domain in PUSCH_SIMS["domain"]:
        for perfect_csi in PUSCH_SIMS["perfect_csi"]:
            batch_size = PUSCH_SIMS["batch_size_freq"] if domain=="freq" else PUSCH_SIMS["batch_size_time"]
            for detector in PUSCH_SIMS["detector"]:
                model = Model(scenario, perfect_csi, domain, detector, PUSCH_SIMS["speed"])
                ber, bler = sim_ber(model,
                            PUSCH_SIMS["ebno_db"],
                            batch_size=batch_size,
                            max_mc_iter=1000,
                            num_target_block_errors=200)
                PUSCH_SIMS["ber"].append(list(ber.numpy()))
                PUSCH_SIMS["bler"].append(list(bler.numpy()))

PUSCH_SIMS["duration"] = time.time() - start


# In[37]:


# Uncomment to show precomputed results
print("Simulation duration: {:1.2f} [h]".format(PUSCH_SIMS["duration"]/3600))

plt.figure()
plt.title("5G NR PUSCH over UMi Channel Model (8x16)")
plt.xlabel("SNR (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.xlim([PUSCH_SIMS["ebno_db"][0], PUSCH_SIMS["ebno_db"][-1]])
plt.ylim([1e-5, 1.0])

i = 0
legend = []
for scenario in PUSCH_SIMS["scenario"]:
    for domain in PUSCH_SIMS["domain"]:
        for perfect_csi in PUSCH_SIMS["perfect_csi"]:
            for detector in PUSCH_SIMS["detector"]:
                plt.semilogy(PUSCH_SIMS["ebno_db"], PUSCH_SIMS["bler"][i])
                i += 1
                csi = "Perf. CSI" if perfect_csi else "Imperf. CSI"
                det = "K-Best" if detector=="kbest" else "LMMSE"
                legend.append(det + " " + csi)
plt.legend(legend);


# Hopefully you have enjoyed this tutorial on Sionna's 5G NR PUSCH module!
# 
# Please have a look at the [API documentation](https://nvlabs.github.io/sionna/phy/api/phy.html) of the various components or the other available [tutorials](https://nvlabs.github.io/sionna/phy/tutorials.html) to learn more.
-e 
# --- End of 5G_NR_PUSCH.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # End-to-end Learning with Autoencoders

# In this notebook, you will learn how to implement an end-to-end communication system as an autoencoder [1].
# The implemented system is shown in the figure below.
# An additive white Gaussian noise (AWGN) channel is considered.
# On the transmitter side, joint training of the constellation geometry and bit-labeling is performed, as in [2].
# On the receiver side, a neural network-based demapper that computes log-likelihood ratios (LLRs) on the transmitted bits from the received samples is optimized.
# The considered autoencoder is benchmarked against a quadrature amplitude modulation (QAM) with Gray labeling and the optimal AWGN demapper.

# 

# Two algorithms for training the autoencoder are implemented in this notebook:
# 
# * Conventional stochastic gradient descent (SGD) with backpropagation, which assumes a differentiable channel model and therefore optimizes the end-to-end system by backpropagating the gradients through the channel (see, e.g., [1]).
# * The training algorithm from [3], which does not assume a differentiable channel model, and which trains the end-to-end system by alternating between conventional training of the receiver and reinforcement learning (RL)-based training of the transmitter. Compared to [3], an additional step of fine-tuning of the receiver is performed after alternating training.
# 
# **Note:** For an introduction to the implementation of differentiable communication systems and their optimization through SGD and backpropagation with Sionna, please refer to [the Part 2 of the Sionna tutorial for Beginners](https://nvlabs.github.io/sionna/phy/tutorials/Sionna_tutorial_part2.html).

# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Simulation Parameters](#Simulation-Parameters)
# * [Neural Demapper](#Neural-Demapper)
# * [Trainable End-to-end System: Conventional Training](#Trainable-End-to-end-System:-Conventional-Training)
# * [Trainable End-to-end System: RL-based Training](#Trainable-End-to-end-System:-RL-based-Training)
# * [Evaluation](#Evaluation)
# * [Visualizing the Learned Constellations](#Visualizing-the-Learned-Constellations)
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

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense

from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, log10, expand_to_rank
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.utils import sim_ber

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pickle


# ## Simulation Parameters

# In[2]:


###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 4.0
ebno_db_max = 8.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.5 # Coderate for the outer code
n = 1500 # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword

###############################################
# Training configuration
###############################################
num_training_iterations_conventional = 10000 # Number of training iterations for conventional training
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
num_training_iterations_rl_alt = 7000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(128, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_conventional_training = "awgn_autoencoder_weights_conventional_training" # Filename to save the autoencoder weights once conventional training is done
model_weights_path_rl_training = "awgn_autoencoder_weights_rl_training" # Filename to save the autoencoder weights once RL-based training is done

###############################################
# Evaluation configuration
###############################################
results_filename = "awgn_autoencoder_results" # Location to save the results


# ## Neural Demapper

# The neural network-based demapper shown in the figure above is made of three dense layers with ReLU activation.
# 
# The input of the demapper consists of a received sample $y \in \mathbb{C}$ and the noise power spectral density $N_0$ in log-10 scale to handle different orders of magnitude for the SNR.
# 
# As the neural network can only process real-valued inputs, these values are fed as a 3-dimensional vector
# 
# $$\left[ \mathcal{R}(y), \mathcal{I}(y), \log_{10}(N_0) \right]$$
# 
# where $\mathcal{R}(y)$ and $\mathcal{I}(y)$ refer to the real and imaginary component of $y$, respectively.
# 
# The output of the neural network-based demapper consists of LLRs on the `num_bits_per_symbol` bits mapped to a constellation point. Therefore, the last layer consists of ``num_bits_per_symbol`` units.
# 
# **Note**: The neural network-based demapper processes the received samples $y$ forming a block individually. The [neural receiver notebook](https://nvlabs.github.io/sionna/phy/tutorials/Neural_Receiver.html) provides an example of a more advanced neural network-based receiver that jointly processes a resource grid of received symbols.

# In[3]:


class NeuralDemapper(Layer):

    def __init__(self):
        super().__init__()

        self._dense_1 = Dense(128, 'relu')
        self._dense_2 = Dense(128, 'relu')
        self._dense_3 = Dense(num_bits_per_symbol, None) # The feature correspond to the LLRs for every bits carried by a symbol

    def call(self, y, no):

        # Using log10 scale helps with the performance
        no_db = log10(no)

        # Stacking the real and imaginary components of the complex received samples
        # and the noise variance
        no_db = tf.tile(no_db, [1, num_symbols_per_codeword]) # [batch size, num_symbols_per_codeword]
        z = tf.stack([tf.math.real(y),
                      tf.math.imag(y),
                      no_db], axis=2) # [batch size, num_symbols_per_codeword, 3]
        llr = self._dense_1(z)
        llr = self._dense_2(llr)
        llr = self._dense_3(llr) # [batch size, num_symbols_per_codeword, num_bits_per_symbol]

        return llr


# ## Trainable End-to-end System: Conventional Training

# The following cell defines an end-to-end communication system that transmits bits modulated using a trainable constellation over an AWGN channel.
# 
# The receiver uses the previously defined neural network-based demapper to compute LLRs on the transmitted (coded) bits.
# 
# As in [1], the constellation and neural network-based demapper are jointly trained through SGD and backpropagation using the binary cross entropy (BCE) as loss function.
# 
# Training on the BCE is known to be equivalent to maximizing an achievable information rate [2].
# 
# The following model can be instantiated either for training (`training = True`) or evaluation (`training = False`).
# 
# In the former case, the BCE is returned and no outer code is used to reduce computational complexity and as it does not impact the training of the constellation or demapper.
# 
# When setting `training` to `False`, an LDPC outer code from 5G NR is applied.

# In[4]:


class E2ESystemConventionalTraining(Model):

    def __init__(self, training):
        super().__init__()

        self._training = training

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            # num_bits_per_symbol is required for the interleaver
            self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)

        # Trainable constellation
        # We initialize a custom constellation with qam points
        qam_points = Constellation("qam", num_bits_per_symbol).points
        self.constellation = Constellation("custom",
                                           num_bits_per_symbol,
                                           points=qam_points,
                                           normalize=True,
                                           center=True)
        # To make the constellation trainable, we need to create seperate
        # variables for the real and imaginary parts
        self.points_r = self.add_weight(shape=qam_points.shape,
                                        initializer="zeros")
        self.points_i = self.add_weight(shape=qam_points.shape,
                                        initializer="zeros")
        self.points_r.assign(tf.math.real(qam_points))
        self.points_i.assign(tf.math.imag(qam_points))

        self._mapper = Mapper(constellation=self.constellation)

        ################
        ## Channel
        ################
        self._channel = AWGN()

        ################
        ## Receiver
        ################
        # We use the previously defined neural network for demapping
        self._demapper = NeuralDemapper()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

        #################
        # Loss function
        #################
        if self._training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):
        # Set the constellation points equal to a complex tensor constructed
        # from two real-valued variables
        points = tf.complex(self.points_r, self.points_i)
        self.constellation.points = points

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, n])
        else:
            b = self._binary_source([batch_size, k])
            c = self._encoder(b)
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]

        ################
        ## Channel
        ################
        y = self._channel(x, no) # [batch size, num_symbols_per_codeword]

        ################
        ## Receiver
        ################
        llr = self._demapper(y, no)
        llr = tf.reshape(llr, [batch_size, n])
        # If training, outer decoding is not performed and the BCE is returned
        if self._training:
            loss = self._bce(c, llr)
            return loss
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation


# A simple training loop is defined in the next cell, which performs `num_training_iterations_conventional` training iterations of SGD. Training is done over a range of SNR, by randomly sampling a batch of SNR values at each iteration.

# **Note:** For an introduction to the implementation of differentiable communication systems and their optimization through SGD and backpropagation with Sionna, please refer to [the Part 2 of the Sionna tutorial for Beginners](https://nvlabs.github.io/sionna/phy/tutorials/Sionna_tutorial_part2.html).

# In[5]:


def conventional_training(model):
    # Optimizer used to apply gradients
    optimizer = tf.keras.optimizers.Adam()

    @tf.function(jit_compile=True)
    def train_step():
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            loss = model(training_batch_size,  ebno_db)
        # Computing and applying gradients
        weights = model.trainable_variables
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        return loss

    for i in range(num_training_iterations_conventional):
        loss = train_step()
        # Printing periodically the progress
        if i % 100 == 0:
            print('Iteration {}/{}  BCE: {:.4f}'.format(i, num_training_iterations_conventional, loss.numpy()), end='\r')


# The next cell defines a utility function for saving the weights using [pickle](https://docs.python.org/3/library/pickle.html).

# In[6]:


def save_weights(model, model_weights_path):
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)


# In the next cell, an instance of the model defined previously is instantiated and trained.

# In[7]:


# Instantiate and train the end-to-end system
model = E2ESystemConventionalTraining(training=True)
conventional_training(model)
# Save weights
save_weights(model, model_weights_path_conventional_training)


# ## Trainable End-to-end System: RL-based Training

# The following cell defines the same end-to-end system as before, but stop the gradients after the channel to simulate a non-differentiable channel.
# 
# To jointly train the transmitter and receiver over a non-differentiable channel, we follow [3], which key idea is to alternate between:
# 
# - Training of the receiver on the BCE using conventional backpropagation and SGD.
# - Training of the transmitter by applying (known) perturbations to the transmitter output to enable estimation of the gradient of the transmitter weights with respect to an approximation of the loss function.
# 
# When `training` is set to `True`, both losses for training the receiver and the transmitter are returned.

# In[8]:


class E2ESystemRLTraining(Model):

    def __init__(self, training):
        super().__init__()

        self._training = training

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)

        # Trainable constellation
        # We initialize a custom constellation with qam points
        qam_points = Constellation("qam", num_bits_per_symbol).points
        self.constellation = Constellation("custom",
                                           num_bits_per_symbol,
                                           points=qam_points,
                                           normalize=True,
                                           center=True)
        # To make the constellation trainable, we need to create seperate
        # variables for the real and imaginary parts
        self.points_r = self.add_weight(shape=qam_points.shape,
                                        initializer="zeros")
        self.points_i = self.add_weight(shape=qam_points.shape,
                                        initializer="zeros")
        self.points_r.assign(tf.math.real(qam_points))
        self.points_i.assign(tf.math.imag(qam_points))

        self._mapper = Mapper(constellation=self.constellation)

        ################
        ## Channel
        ################
        self._channel = AWGN()

        ################
        ## Receiver
        ################
        # We use the previously defined neural network for demapping
        self._demapper = NeuralDemapper()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not self._training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def call(self, batch_size, ebno_db, perturbation_variance=tf.constant(0.0, tf.float32)):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, n])
        else:
            b = self._binary_source([batch_size, k])
            c = self._encoder(b)
        # Modulation
        # Set the constellation points equal to a complex tensor constructed
        # from two real-valued variables
        points = tf.complex(self.points_r, self.points_i)
        self.constellation.points = points
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]

        # Adding perturbation
        # If ``perturbation_variance`` is 0, then the added perturbation is null
        epsilon_r = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon_i = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon = tf.complex(epsilon_r, epsilon_i) # [batch size, num_symbols_per_codeword]
        x_p = x + epsilon # [batch size, num_symbols_per_codeword]

        ################
        ## Channel
        ################
        y = self._channel(x_p, no) # [batch size, num_symbols_per_codeword]
        y = tf.stop_gradient(y) # Stop gradient here

        ################
        ## Receiver
        ################
        llr = self._demapper(y, no)

        # If training, outer decoding is not performed
        if self._training:
            # Average BCE for each baseband symbol and each batch example
            c = tf.reshape(c, [-1, num_symbols_per_codeword, num_bits_per_symbol])
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(c, llr), axis=2) # Avergare over the bits mapped to a same baseband symbol
            # The RX loss is the usual average BCE
            rx_loss = tf.reduce_mean(bce)
            # From the TX side, the BCE is seen as a feedback from the RX through which backpropagation is not possible
            bce = tf.stop_gradient(bce) # [batch size, num_symbols_per_codeword]
            x_p = tf.stop_gradient(x_p)
            p = x_p-x # [batch size, num_symbols_per_codeword] Gradient is backpropagated through `x`
            tx_loss = tf.square(tf.math.real(p)) + tf.square(tf.math.imag(p)) # [batch size, num_symbols_per_codeword]
            tx_loss = -bce*tx_loss/rl_perturbation_var # [batch size, num_symbols_per_codeword]
            tx_loss = tf.reduce_mean(tx_loss)
            return tx_loss, rx_loss
        else:
            llr = tf.reshape(llr, [-1, n]) # Reshape as expected by the outer decoder
            b_hat = self._decoder(llr)
            return b,b_hat


# The next cell implements the training algorithm from [3], which alternates between conventional training of the neural network-based receiver, and RL-based training of the transmitter.

# In[9]:


def rl_based_training(model):
    # Optimizers used to apply gradients
    optimizer_tx = tf.keras.optimizers.Adam() # For training the transmitter
    optimizer_rx = tf.keras.optimizers.Adam() # For training the receiver

    # Function that implements one transmitter training iteration using RL.
    @tf.function(jit_compile=True)
    def train_tx():
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            # Keep only the TX loss
            tx_loss, _ = model(training_batch_size, ebno_db,
                               tf.constant(rl_perturbation_var, tf.float32)) # Perturbation are added to enable RL exploration
        ## Computing and applying gradients
        weights = tape.watched_variables()
        grads = tape.gradient(tx_loss, weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer_tx.apply_gradients(zip(grads, weights))

    # Function that implements one receiver training iteration
    @tf.function(jit_compile=True)
    def train_rx():
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            # Keep only the RX loss
            _, rx_loss = model(training_batch_size, ebno_db) # No perturbation is added
        ## Computing and applying gradients
        weights = tape.watched_variables()
        grads = tape.gradient(rx_loss, weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer_rx.apply_gradients(zip(grads, weights))
        # The RX loss is returned to print the progress
        return rx_loss

    # Training loop.
    for i in range(num_training_iterations_rl_alt):
        # 10 steps of receiver training are performed to keep it ahead of the transmitter
        # as it is used for computing the losses when training the transmitter
        for _ in range(10):
            rx_loss = train_rx()
        # One step of transmitter training
        train_tx()
        # Printing periodically the progress
        if i % 100 == 0:
            print('Iteration {}/{}  BCE {:.4f}'.format(i, num_training_iterations_rl_alt, rx_loss.numpy()), end='\r')
    print() # Line break

    # Once alternating training is done, the receiver is fine-tuned.
    print('Receiver fine-tuning... ')
    for i in range(num_training_iterations_rl_finetuning):
        rx_loss = train_rx()
        if i % 100 == 0:
            print('Iteration {}/{}  BCE {:.4f}'.format(i, num_training_iterations_rl_finetuning, rx_loss.numpy()), end='\r')


# In the next cell, an instance of the model defined previously is instantiated and trained.

# In[10]:


# Instantiate and train the end-to-end system
model = E2ESystemRLTraining(training=True)
rl_based_training(model)
# Save weights
save_weights(model, model_weights_path_rl_training)


# ## Evaluation

# The following cell implements a baseline which uses QAM with Gray labeling and conventional demapping for AWGN channel.

# In[11]:


class Baseline(Model):

    def __init__(self):
        super().__init__()

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)

        ################
        ## Channel
        ################
        self._channel = AWGN()

        ################
        ## Receiver
        ################
        self._demapper = Demapper("app", constellation=constellation)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def call(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        b = self._binary_source([batch_size, k])
        c = self._encoder(b)
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]

        ################
        ## Channel
        ################
        y = self._channel(x, no) # [batch size, num_symbols_per_codeword]

        ################
        ## Receiver
        ################
        llr = self._demapper(y, no)
        # Outer decoding
        b_hat = self._decoder(llr)
        return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation


# In[12]:


# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step


# In[13]:


# Utility function to load and set weights of a model
def load_weights(model, model_weights_path):
    model(tf.cast(1, tf.int32), tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    points = tf.complex(model.points_r, model.points_i)
    model.constellation.points = points


# The next cell evaluate the baseline and the two autoencoder-based communication systems, trained with different method.
# The results are stored in the dictionary ``BLER``.

# In[14]:


# Dictionary storing the results
BLER = {}

model_baseline = Baseline()
_,bler = sim_ber(model_baseline, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=1000, graph_mode="xla")
BLER['baseline'] = bler.numpy()

model_conventional = E2ESystemConventionalTraining(training=False)
load_weights(model_conventional, model_weights_path_conventional_training)
_,bler = sim_ber(model_conventional, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=1000, graph_mode="xla")
BLER['autoencoder-conv'] = bler.numpy()

model_rl = E2ESystemRLTraining(training=False)
load_weights(model_rl, model_weights_path_rl_training)
_,bler = sim_ber(model_rl, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=1000, graph_mode="xla")
BLER['autoencoder-rl'] = bler.numpy()

with open(results_filename, 'wb') as f:
    pickle.dump((ebno_dbs, BLER), f)


# In[15]:


plt.figure(figsize=(10,8))
# Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c=f'C0', label=f'Baseline')
# Autoencoder - conventional training
plt.semilogy(ebno_dbs, BLER['autoencoder-conv'], 'x-.', c=f'C1', label=f'Autoencoder - conventional training')
# Autoencoder - RL-based training
plt.semilogy(ebno_dbs, BLER['autoencoder-rl'], 'o-.', c=f'C2', label=f'Autoencoder - RL-based training')

plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()


# ## Visualizing the Learned Constellations

# In[16]:


model_conventional = E2ESystemConventionalTraining(training=True)
load_weights(model_conventional, model_weights_path_conventional_training)
fig = model_conventional.constellation.show()
fig.suptitle('Conventional training');


# In[17]:


model_rl = E2ESystemRLTraining(training=False)
load_weights(model_rl, model_weights_path_rl_training)
fig = model_rl.constellation.show()
fig.suptitle('RL-based training');


# In[18]:


get_ipython().run_line_magic('rm', 'awgn_autoencoder_weights_conventional_training awgn_autoencoder_weights_rl_training awgn_autoencoder_results')


# ## References

# [1] T. O’Shea and J. Hoydis, "An Introduction to Deep Learning for the Physical Layer," in IEEE Transactions on Cognitive Communications and Networking, vol. 3, no. 4, pp. 563-575, Dec. 2017, doi: 10.1109/TCCN.2017.2758370.
# 
# [2] S. Cammerer, F. Ait Aoudia, S. Dörner, M. Stark, J. Hoydis and S. ten Brink, "Trainable Communication Systems: Concepts and Prototype," in IEEE Transactions on Communications, vol. 68, no. 9, pp. 5489-5503, Sept. 2020, doi: 10.1109/TCOMM.2020.3002915.
# 
# [3] F. Ait Aoudia and J. Hoydis, "Model-Free Training of End-to-End Communication Systems," in IEEE Journal on Selected Areas in Communications, vol. 37, no. 11, pp. 2503-2516, Nov. 2019, doi: 10.1109/JSAC.2019.2933891.
-e 
# --- End of Autoencoder.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Bit-Interleaved Coded Modulation (BICM)
# 
# In this notebook you will learn about the principles of bit interleaved coded modulation (BICM) and focus on the interface between LDPC decoding and demapping for higher order modulation.
# Further, we will discuss the idea of *all-zero codeword* simulations that enable bit-error rate simulations without having an explicit LDPC encoder available.
# In the last part, we analyze what happens for mismatched demapping, e.g., if the SNR is unknown and show how min-sum decoding can have practical advantages in such cases.
# 
# *"From the coding viewpoint, the modulator, waveform channel, and demodulator together constitute a discrete channel with* $q$ *input letters and* $q'$ *output letters. [...] the real goal of the modulation system is to create the “best” discrete memoryless channel (DMC) as seen by the coding system."*
# James L. Massey, 1974 [4, cf. preface in 5].
# 
# The fact that we usually separate modulation and coding into two individual tasks is strongly connected to the concept of bit-interleaved coded modulation (BICM) [1,2,5]. However, the joint optimization of coding and modulation has a long history, for example by Gottfried Ungerböck's *Trellis coded modulation* (TCM) [3] and we refer the interested reader to [1,2,5,6] for these *principles of coded modulation* [5].
# Nonetheless, BICM has become the *de facto* standard in virtually any modern communication system due to its engineering simplicity.
# 
# In this notebook, you will use the following components:
# 
# * Mapper / demapper and the constellation class
# * LDPC5GEncoder / LDPC5GDecoder
# * AWGN channel
# * BinarySource and GaussianPriorSource
# * Interleaver / deinterleaver
# * Scrambler / descrambler
# 
# ## Table of Contents
# * [System Block Diagram](#System-Block-Diagram)
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [A Simple BICM System](#A-Simple-BICM-System)
# * [All-zero Codeword Simulations](#All-zero-Codeword-Simulations)
# * [EXIT-Charts](#EXIT-Charts)
# * [Mismatched Demapping and the Advantages of Min-sum Decoding](#Mismatched-Demapping-and-the-Advantages-of-Min-sum-Decoding)
# * [References](#References)
# 
# ## System Block Diagram
# 
# We introduce the following terminology:
# 
# - `u` denotes the `k` uncoded information bits
# - `c` denotes the `n` codewords bits
# - `x` denotes the complex-valued symbols after mapping `m` bits to one symbol
# - `y` denotes the (noisy) channel observations
# - `l_ch` denotes the demappers llr estimate on each bit `c`
# - `u_hat` denotes the estimated information bits at the decoder output


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

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder, EXITCallback
from sionna.phy.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.phy.fec.scrambling import Scrambler, Descrambler
from sionna.phy.fec.utils import GaussianPriorSource, load_parity_check_examples, \
                                 get_exit_analytic, plot_exit_chart, plot_trajectory
from sionna.phy.utils import ebnodb2no, hard_decisions, PlotBER
from sionna.phy.channel import AWGN

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation


# ## A Simple BICM System
# 
# The principle idea of higher order modulation is to map *m* bits to one (complex-valued) symbol *x*.
# As each received symbol now contains information about *m* transmitted bits, the demapper produces *m* bit-wise LLR estimates (one per transmitted bit) where each LLR contains information about an individual bit.
# This scheme allows a simple binary interface between demapper and decoder. 
# 
# From a decoder's perspective, the transmission of all *m* bits - mapped onto one symbol - could be modeled as if they have been transmitted over *m* different *surrogate* channels with certain properties as shown in the figure below.

# 

# In the following, we are now interested in the LLR distribution at the decoder input (= demapper output) for each of these *surrogate* channels (denoted as *bit-channels* in the following).
# Please note that in some scenario these surrogate channels can share the same statistical properties, e.g., for QPSK, both bit-channels behave equally due to symmetry.
# 
# Advanced note: the *m* binary LLR values are treated as independent estimates which is not exactly true for higher order modulation. As a result, the sum of the *bitwise* mutual information of all *m* transmitted bits does not exactly coincide with the *symbol-wise* mutual information describing the relation between channel input / output from a symbol perspective.
# However, in practice the (small) losses are usually neglected if a QAM with a rectangular grid and Gray labeling is used.

# ### Constellations and Bit-Channels
# 
# Let us first look at some higher order constellations. 

# In[2]:


# show QPSK constellation
constellation = Constellation("qam", num_bits_per_symbol=2)
constellation.show();


# Assuming an AWGN channel and QPSK modulation all symbols behave equally due to the symmetry (all constellation points are located on a circle). However, for higher order modulation such as 16-QAM the situation changes and the LLRs after demapping are not equally distributed anymore.

# In[3]:


# generate 16QAM with Gray labeling
constellation = Constellation("qam", num_bits_per_symbol=4)
constellation.show();


# We can visualize this by applying *a posteriori propability* (APP) demapping and plotting of the corresponding LLR distributions for each of the *m* transmitted bits per symbol individually. 
# As each bit could be either *0* or *1*, we flip the signs of the LLRs *after* demapping accordingly. Otherwise, we would observe two symmetric distributions per bit *b_i* for *b_i=0* and *b_i=1*, respectively.
# See [10] for a closed-form approximation and further details.

# In[4]:


# simulation parameters
batch_size = int(1e6) # number of symbols to be analyzed
num_bits_per_symbol = 4 # bits per modulated symbol, i.e., 2^4 = 16-QAM
ebno_db = 4 # simulation SNR

# init system components
source = BinarySource() # generates random info bits

# we use a simple AWGN channel
channel = AWGN()

# calculate noise var for given Eb/No (no code used at the moment)
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1) 

# and generate bins for the histogram
llr_bins = np.arange(-20,20,0.1)

# initialize mapper and demapper for constellation object
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)
mapper = Mapper(constellation=constellation)

# APP demapper
demapper = Demapper("app", constellation=constellation)

# Binary source that generates random 0s/1s
b = source([batch_size, num_bits_per_symbol])

# init mapper, channel and demapper
x = mapper(b)
y = channel(x, no)
llr = demapper(y, no)

# we flip the sign of all LLRs where b_i=0
# this ensures that all positive LLRs mark correct decisions
# all negative LLR values would lead to erroneous decisions
llr_b = tf.multiply(llr, (2.*b-1.))

# calculate LLR distribution for all bit-channels individually
llr_dist = []
for i in range(num_bits_per_symbol):
    
    llr_np = tf.reshape(llr_b[:,i],[-1]).numpy()    
    t, _ = np.histogram(llr_np, bins=llr_bins, density=True);
    llr_dist.append(t)

# and plot the results
plt.figure(figsize=(20,8))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(which="both")
plt.xlabel("LLR value", fontsize=25)
plt.ylabel("Probability density", fontsize=25)
for idx, llr_hist in enumerate(llr_dist):
    leg_str = f"Demapper output for bit_channel {idx} (sign corrected)".format()
    plt.plot(llr_bins[:-1], llr_hist, label=leg_str)
plt.title("LLR distribution after demapping (16-QAM / AWGN)", fontsize=25)
plt.legend(fontsize=20);


# This also shows up in the bit-wise BER without any forward-error correction (FEC). <a class="anchor" id="ber_per_bit"></a>

# In[5]:


# calculate bitwise BERs

b_hat = hard_decisions(llr) # hard decide the LLRs

# each bit where b != b_hat is defines a decision error
# cast to tf.float32 to allow tf.reduce_mean operation
errors = tf.cast(tf.not_equal(b, b_hat), tf.float32)

# calculate ber PER bit_channel
# axis = 0 is the batch-dimension, i.e. contains individual estimates
# axis = 1 contains the m individual bit channels
ber_per_bit = tf.reduce_mean(errors, axis=0)

print("BER per bit-channel: ", ber_per_bit.numpy())


# So far, we have not applied any outer channel coding. However, from the previous histograms it is obvious that the quality of the received LLRs depends bit index within a symbol. Further, LLRs may become correlated and each symbol error may lead to multiple erroneous received bits (mapped to the same symbol). 
# The principle idea of BICM is to *break* the local dependencies by adding an interleaver between channel coding and mapper (or demapper and decoder, respectively).
# 
# For sufficiently long codes (and well-suited interleavers), the channel decoder effectively *sees* one channel.
# This separation enables the - from engineering's perspective - simplified and elegant design of channel coding schemes based on binary bit-metric decoding while following Massey's original spirit that *the real goal of the modulation system is to create the “best” discrete memoryless channel (DMC) as seen by the coding system"* [1].
# 
# 

# ### Simple BER Simulations
# 
# We are now interested to simulate the BER of the BICM system including LDPC codes.
# For this, we use the class `PlotBER` which essentially provides convenience functions for BER simulations.
# It internally calls `sim_ber()` to simulate each SNR point until reaching a pre-defined target number of errors.
# 
# **Note**: a custom BER simulation is always possible. However, without early stopping the simulations can take significantly more simulation time and `PlotBER` directly stores the results internally for later comparison.

# In[6]:


# generate new figure
ber_plot_allzero = PlotBER("BER Performance of All-zero Codeword Simulations")

# and define baseline
num_bits_per_symbol = 2 # QPSK
num_bp_iter = 20 # number of decoder iterations

# LDPC code parameters
k = 600 # number of information bits per codeword
n = 1200 # number of codeword bits

# and the initialize the LDPC encoder / decoder
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder, # connect encoder (for shared code parameters)
                        cn_update="boxplus-phi", # use the exact boxplus function
                        num_iter=num_bp_iter) 

# initialize a random interleaver and corresponding deinterleaver
interleaver = RandomInterleaver()
deinterleaver = Deinterleaver(interleaver)

# mapper and demapper
constellation = Constellation("qam", num_bits_per_symbol=num_bits_per_symbol)
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation) # APP demapper

# define system 
@tf.function() # we enable graph mode for faster simulations
def run_ber(batch_size, ebno_db):
    # calculate noise variance
    no = ebnodb2no(ebno_db,
                   num_bits_per_symbol=num_bits_per_symbol,
                   coderate=k/n)    
    u = source([batch_size, k]) # generate random bit sequence to transmit
    c = encoder(u) # LDPC encode (incl. rate-matching and CRC concatenation)
    c_int = interleaver(c)
    x = mapper(c_int) # map to symbol (QPSK)
    y = channel(x, no) # transmit over AWGN channel
    llr_ch = demapper(y, no) # demapp
    llr_deint = deinterleaver(llr_ch)
    u_hat = decoder(llr_deint) # run LDPC decoder (incl. de-rate-matching)

    return u, u_hat


# We simulate the BER at each SNR point in `ebno_db` for a given `batch_size` of samples.
# In total, per SNR point `max_mc_iter` batches are simulated.
# 
# To improve the simulation throughput, several optimizations are available:
# 
# 1. ) Continue with next SNR point if `num_target_bit_errors` is reached (or `num_target_block_errors`).
# 2. ) Stop simulation if current SNR point returned no error (usually the BER is monotonic w.r.t. the SNR, i.e., a higher SNR point will also return BER=0)
# 
# **Note**: by setting `forward_keyboard_interrupt`=False, the simulation can be interrupted at any time and returns the intermediate results.

# In[7]:


# the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
ber_plot_allzero.simulate(run_ber, # the function have defined previously
                         ebno_dbs=np.arange(0, 5, 0.25), # sim SNR range 
                         legend="Baseline (with encoder)",
                         max_mc_iter=50,
                         num_target_bit_errors=1000,
                         batch_size=1000,
                         soft_estimates=False,
                         early_stop=True,
                         show_fig=True,
                         forward_keyboard_interrupt=False);


# ## All-zero Codeword Simulations
# 
# In this section you will learn about how to simulate accurate BER curves without the need for having an actual encoder in-place.
# We compare each step with the ground truth from the Sionna encoder:
# 
# 1. ) Simulate baseline with encoder as done above.
# 2. ) Remove encoder: Simulate QPSK with all-zero codeword transmission.
# 3. ) Gaussian approximation (for BPSK/QPSK): Remove (de-)mapping and mimic the LLR distribution for the all-zero codeword.
# 4. ) Learn that a scrambler is required for higher order modulation schemes.
# 
# An important property of linear codes is that each codewords has - in average - the same behavior. Thus, for BER simulations the all-zero codeword is sufficient.
# 
# **Note**: strictly speaking, this requires *symmetric* decoders in a sense that the decoder is not biased towards positive or negative LLRs (e.g., by interpreting $\ell_\text{ch}=0$ as positive value). However, in practice this can be either avoided or is often neglected.
# 
# Recall that we have simulated the following setup as baseline.
# Note: for simplicity and readability, the interleaver is omitted in the following.
# 


# Let us implement a Sionna Block that can be re-used and configured for the later experiments.

# In[8]:


class LDPC_QAM_AWGN(Block):
    """System model for channel coding BER simulations

    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. It can enable/disable multiple options to analyse all-zero codeword simulations.

    If active, the system uses the 5G LDPC encoder/decoder module.

    Parameters
    ----------
    k: int
        number of information bits per codeword.

    n: int 
        codeword length.

    num_bits_per_symbol: int
        number of bits per QAM symbol.

    demapping_method: str
        A string defining the demapping method. Can be either "app" or "maxlog".

    cn_update: str
        A string defining the check node update function type of the LDPC decoder.

    use_allzero: bool  
        A boolean defaults to False. If True, no encoder is used and all-zero codewords are sent.

    use_scrambler: bool
        A boolean defaults to False. If True, a scrambler after the encoder and a descrambler before the decoder
        is used, respectively.

    use_ldpc_output_interleaver: bool
        A boolean defaults to False. If True, the output interleaver as 
        defined in 3GPP 38.212 is applied after rate-matching.

    no_est_mismatch: float
        A float defaults to 1.0. Defines the SNR estimation mismatch of the demapper such that the effective demapping
        noise variance estimate is the scaled by ``no_est_mismatch`` version of the true noise_variance

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
                 demapping_method="app",
                 cn_update="boxplus",
                 use_allzero=False,
                 use_scrambler=False,
                 use_ldpc_output_interleaver=False,
                 no_est_mismatch=1.):
        super().__init__()
        self.k = k
        self.n = n

        self.num_bits_per_symbol = num_bits_per_symbol

        self.use_allzero = use_allzero
        self.use_scrambler = use_scrambler

        # adds noise to SNR estimation at demapper
        # see last section "mismatched demapping"
        self.no_est_mismatch = no_est_mismatch 

        # init components
        self.source = BinarySource()
       
        # initialize mapper and demapper with constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)

        self.channel = AWGN()

        # LDPC encoder / decoder
        if use_ldpc_output_interleaver:
            # the output interleaver needs knowledge of the modulation order
            self.encoder = LDPC5GEncoder(self.k, self.n, num_bits_per_symbol)
        else:
            self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, cn_update=cn_update)

        self.scrambler = Scrambler()        
        # connect descrambler to scrambler
        self.descrambler = Descrambler(self.scrambler, binary=False) 

    @tf.function() # enable graph mode for higher throughputs
    def call(self, batch_size, ebno_db):

        # calculate noise variance
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=self.k/self.n)            

        if self.use_allzero:
            u = tf.zeros([batch_size, self.k]) # only needed for 
            c = tf.zeros([batch_size, self.n]) # replace enc. with all-zero codeword
        else:
            u = self.source([batch_size, self.k])
            c = self.encoder(u) # explicitly encode
        
        # scramble codeword if actively required
        if self.use_scrambler:
            c = self.scrambler(c)

        x = self.mapper(c) # map c to symbols

        y = self.channel(x, no) # transmit over AWGN channel

        # add noise estimation mismatch for demapper (see last section)
        # set to 1 per default -> no mismatch
        no_est = no * self.no_est_mismatch
        llr_ch = self.demapper(y, no_est) # demapp

        if self.use_scrambler:
            llr_ch = self.descrambler(llr_ch)

        u_hat = self.decoder(llr_ch) # run LDPC decoder (incl. de-rate-matching)

        return u, u_hat


# ### Remove Encoder: Simulate QPSK with All-zero Codeword Transmission
# 
# We now simulate the same system *without* encoder and transmit constant *0*s.
# 
# Due to the symmetry of the QPSK, no scrambler is required. You will learn about the effect of the scrambler in the last section.


# In[9]:


model_allzero = LDPC_QAM_AWGN(k,
                              n,
                              num_bits_per_symbol=2,
                              use_allzero=True, # disable encoder
                              use_scrambler=False) # we do not use a scrambler for the moment (QPSK!)

# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the 
# Monte Carlo simulation
ber_plot_allzero.simulate(model_allzero,
                          ebno_dbs=np.arange(0, 5, 0.25),
                          legend="All-zero / QPSK (no encoder)",
                          max_mc_iter=50,
                          num_target_bit_errors=1000,
                          batch_size=1000,
                          soft_estimates=False,
                          show_fig=True,
                          forward_keyboard_interrupt=False);


# As expected, the BER curves are identical (within the accuracy of the Monte Carlo simulation).

# ### Remove (De-)Mapping: Approximate the LLR Distribution of the All-zero Codeword (and BPSK/QPSK)
# 
# For the all-zero codeword, the BPSK mapper generates the *all-one* signal (as each *0* is mapped to a *1*).
# 
# Assuming an AWGN channel with noise variance $\sigma_\text{ch}^2$, it holds that the output of the channel $y$ is Gaussian distributed with mean $\mu=1$ and noise variance $\sigma_\text{ch}^2$.
# Demapping of the BPSK symbols is given as $\ell_\text{ch} = -\frac{2}{\sigma_\text{ch}^2}y$.
# 
# This leads to the effective LLR distribution of $\ell_\text{ch} \sim \mathcal{N}(-\frac{2}{\sigma_\text{ch}^2},\frac{4}{\sigma_\text{ch}^2})$ and, thereby, allows to mimic the mapper, AWGN channel and demapper by a Gaussian distribution.
# The `GaussianPriorSource` block provides such a source for arbitrary shapes.
# 
# The same derivation holds for QPSK. Let us quickly verify the correctness of these results by a Monte Carlo simulation.
# 
# **Note**: the negative sign for the BPSK demapping rule comes from the (in communications) unusual definition of logits $\ell = \operatorname{log} \frac{p(x=1)}{p(x=0)}$.


# In[10]:


num_bits_per_symbol = 2 # we use QPSK
ebno_db = 4 # choose any SNR
batch_size = 100000 # we only simulate 1 symbol per batch

# calculate noise variance
no = ebnodb2no(ebno_db,
               num_bits_per_symbol=num_bits_per_symbol,
               coderate=k/n)  

# generate bins for the histogram
llr_bins = np.arange(-20, 20, 0.2)

c = tf.zeros([batch_size, num_bits_per_symbol]) # all-zero codeword
x = mapper(c) # mapped to constant symbol
y = channel(x, no)
llr = demapper(y, no) # and generate LLRs

llr_dist, _ = np.histogram(llr.numpy() , bins=llr_bins, density=True);

# negative mean value due to different logit/llr definition 
# llr = log[(x=1)/p(x=0)]
mu_llr = -2 / no
no_llr = 4 / no

# generate Gaussian pdf
llr_pred = 1/np.sqrt(2*np.pi*no_llr) * np.exp(-(llr_bins-mu_llr)**2/(2*no_llr))

# and compare the results
plt.figure(figsize=(20,8))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(which="both")
plt.xlabel("LLR value", fontsize=25)
plt.ylabel("Probability density", fontsize=25)
plt.plot(llr_bins[:-1], llr_dist, label="Measured LLR distribution")
plt.plot(llr_bins, llr_pred, label="Analytical LLR distribution (GA)")
plt.title("LLR distribution after demapping", fontsize=25)
plt.legend(fontsize=20);


# In[11]:


num_bits_per_symbol = 2 # QPSK 

# initialize LLR source
ga_source = GaussianPriorSource()

@tf.function() # enable graph mode
def run_ber_ga(batch_size, ebno_db):
    # calculate noise variance
    no = ebnodb2no(ebno_db,
                   num_bits_per_symbol=num_bits_per_symbol,
                   coderate=k/n)    
    
    u = tf.zeros([batch_size, k]) # only needed for ber calculations
    
    llr_ch = ga_source([batch_size, n], no) # generate LLRs directly
    u_hat = decoder(llr_ch) # run LDPC decoder (incl. de-rate-matching)

    return u, u_hat

# and simulate the new curve
ber_plot_allzero.simulate(run_ber_ga,
                          ebno_dbs=np.arange(0, 5, 0.25), # simulation SNR,
                          max_mc_iter=50,
                          num_target_bit_errors=1000,
                          legend="Gaussian Approximation of LLRs",
                          batch_size = 10000,
                          soft_estimates=False,
                          show_fig=True,
                          forward_keyboard_interrupt=False);


# ### The Role of the Scrambler
# 
# So far, we have seen that the all-zero codeword yields the same error-rates as any other sequence.
# Intuitively, the *all-zero codeword trick* generates a constant stream of *0*s at the input of the mapper.
# However, if the channel is not symmetric we need to ensure that we capture the *average* behavior of all possible symbols equally.
# Mathematically this symmetry condition can be expressed as $p(Y=y|c=0)=p(Y=-y|c=1)$
# 
# As shown in the previous experiments, for QPSK both *bit-channels* have the same behavior but for 16-QAM systems this does not hold anymore and our simulated BER does not represent the average decoding performance of the original system. 
# 
# One possible solution is to scramble the all-zero codeword before transmission and descramble the received LLRs before decoding (i.e., flip the sign accordingly). 
# This ensures that the mapper/demapper (+channel) operate on (pseudo-)random data, but from decoder's perspective the the all-zero codeword assumption is still valid. This avoids the need for an actual encoder. For further details, we refer to *i.i.d. channel adapters* in [9].
# 
# **Note**: another example is that the recorded LLRs can be even used to evaluate different codes as the all-zero codeword is a valid codeword for all linear codes. Going one step further, one can even simulate codes of different rates with the same pre-recorded LLRs.


# In[12]:


# we generate a new plot
ber_plot_allzero16qam = PlotBER("BER Performance for 64-QAM")


# In[13]:


# simulate a new baseline for 16-QAM
model_baseline_16 = LDPC_QAM_AWGN(k,
                                  n,
                                  num_bits_per_symbol=4,
                                  use_allzero=False, # baseline without all-zero
                                  use_scrambler=False)


# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the 
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_baseline_16,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="Baseline 16-QAM",
                               max_mc_iter=50,
                               num_target_bit_errors=2000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);


# We now apply the *all-zero trick* as above and simulate ther BER performance without scrambling.

# In[14]:


# and repeat the experiment for a 16QAM WITHOUT scrambler 
model_allzero_16_no_sc = LDPC_QAM_AWGN(k,
                                       n,
                                       num_bits_per_symbol=4,
                                       use_allzero=True, # all-zero codeword
                                       use_scrambler=False) # no scrambler used


# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the 
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_allzero_16_no_sc,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="All-zero / 16-QAM (no scrambler!)",
                               max_mc_iter=50,
                               num_target_bit_errors=1000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);


# As expected the results are wrong as we have transmitted all bits over the *less reliable* channel (cf [BER per bit-channel](#Constellations-and-Bit-Channels)).
# 
# Let us repeat this experiment with scrambler and descrambler at the correct position.

# In[15]:


# and repeat the experiment for a 16QAM WITHOUT scrambler 
model_allzero_16_sc = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=True, # all-zero codeword
                                    use_scrambler=True) # activate scrambler


# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the 
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_allzero_16_sc,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="All-zero / 16-QAM (with scrambler)",
                               max_mc_iter=50,
                               num_target_bit_errors=1000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);


# The 5G standard defines an additional output interleaver after the rate-matching (see Sec. 5.4.2.2 in [11]).
# 
# We now activate this additional interleaver to enable additional BER gains.

# In[16]:


# activate output interleaver
model_output_interleaver= LDPC_QAM_AWGN(k,
                                        n,
                                        num_bits_per_symbol=4,
                                        use_ldpc_output_interleaver=True, 
                                        use_allzero=False, 
                                        use_scrambler=False)


# and simulate the new curve
# Hint: as the model is callable, we can directly pass it to the 
# Monte Carlo simulation
ber_plot_allzero16qam.simulate(model_output_interleaver,
                               ebno_dbs=np.arange(0, 5, 0.25),
                               legend="16-QAM with 5G FEC interleaver",
                               max_mc_iter=50,
                               num_target_bit_errors=1000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);


# ## EXIT Charts
# 
# You now learn about how the convergence behavior of iterative receivers can be visualized. 
# 
# Extrinsic Information Transfer (EXIT) charts [7] are a widely adopted tool to analyze the convergence behavior of iterative receiver algorithms. The principle idea is to treat each component decoder (or demapper etc.) as individual entity with its own EXIT characteristic.
# EXIT charts not only allow to predict the decoding behavior (*open decoding tunnel*) but also enable LDPC code design (cf. [8]). However, this is beyond the scope of this notebook.
# 
# We can analytically derive the EXIT characteristic for check node (CN) and variable node (VN) decoder for a given code with `get_exit_analytic`.
# Further, if the `LDPCBPDecoder` is initialized with option `track_exit`=True, it internally stores the average extrinsic mutual information after each iteration at the output of the VN/CN decoder.
# 
# Please note that this is only an approximation for the AWGN channel and assumes infinite code length. However, it turns out that the results are often accurate enough and 

# In[17]:


# parameters
ebno_db = 2.3
batch_size = 10000
num_bits_per_symbol = 2

pcm_id = 4 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-LDPC 4: LDPC 802.11n
pcm, k_exit, n_exit, coderate = load_parity_check_examples(pcm_id, verbose=True)

# init callbacks for tracking of EXIT charts
num_iter=20
cb_exit_vn = EXITCallback(num_iter)
cb_exit_cn = EXITCallback(num_iter)

# init components
decoder_exit = LDPCBPDecoder(pcm, 
                             hard_out=False, 
                             cn_update="boxplus", 
                             track_exit=True,
                             num_iter=num_iter,
                             v2c_callbacks=[cb_exit_vn,],
                             c2v_callbacks=[cb_exit_cn,])


# generates fake llrs as if the all-zero codeword was transmitted over an AWNG channel with BPSK modulation (see early sections)
llr_source = GaussianPriorSource()

noise_var = ebnodb2no(ebno_db=ebno_db,
                      num_bits_per_symbol=num_bits_per_symbol,
                      coderate=coderate)

# use fake llrs from GA
llr = llr_source([batch_size, n_exit], noise_var)

# simulate free runing trajectory
decoder_exit(llr)

# calculate analytical EXIT characteristics
# Hint: these curves assume asymptotic code length, i.e., may become inaccurate in the short length regime
Ia, Iev, Iec = get_exit_analytic(pcm, ebno_db)

# and plot the EXIT curves
plt = plot_exit_chart(Ia, Iev, Iec)

# however, as track_exit=True, the decoder logs the actual exit trajectory during decoding. This can be accessed by decoder.ie_v/decoder.ie_c after the simulation

# and add simulated trajectory to plot
plot_trajectory(plt, cb_exit_vn.mi.numpy(), cb_exit_cn.mi.numpy(), ebno_db)


# As can be seen, the simulated trajectory of the decoder matches (relatively) well with the predicted
# EXIT functions of the VN and CN decoder, respectively.
# 
# A few things to try:
# 
# - Change the SNR; which curves change? Why is one curve constant? Hint: does every component directly *see* the channel?
# - What happens for other codes?
# - Can you predict the *threshold* of this curve (i.e., the minimum SNR required for successful decoding)
# - Verify the correctness of this threshold via BER simulations (hint: the codes are relatively short, thus the prediction is less accurate)

# ## Mismatched Demapping and the Advantages of Min-sum Decoding
# 
# So far, we have demapped with exact knowledge of the underlying noise distribution (including the exact SNR).
# However, in practice estimating the SNR can be a complicated task and, as such, the estimated SNR used for demapping can be inaccurate.
# 
# In this part, you will learn about the advantages of min-sum decoding and we will see that it is more robust against mismatched demapping.

# In[18]:


# let us first remove the non-scrambled result from the previous experiment
ber_plot_allzero16qam.remove(idx=1) # remove curve with index 1


# In[19]:


# simulate with mismatched noise estimation
model_allzero_16_no = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=False, # full simulation
                                    no_est_mismatch=0.15) # noise variance estimation mismatch (no scaled by 0.15 )

ber_plot_allzero16qam.simulate(model_allzero_16_no,
                               ebno_dbs=np.arange(0, 7, 0.5),
                               legend="Mismatched Demapping / 16-QAM",
                               max_mc_iter=50,
                               num_target_bit_errors=1000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);


# In[20]:


# simulate with mismatched noise estimation
model_allzero_16_ms = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=False, # full simulation
                                    cn_update="minsum", # activate min-sum decoding
                                    no_est_mismatch=1.) # no mismatch
                                     
ber_plot_allzero16qam.simulate(model_allzero_16_ms,
                               ebno_dbs=np.arange(0, 7, 0.5),
                               legend="Min-sum decoding / 16-QAM (no mismatch)",
                               max_mc_iter=50,
                               num_target_bit_errors=1000,
                               batch_size=1000,
                               soft_estimates=False,
                               show_fig=True,
                               forward_keyboard_interrupt=False);


# In[21]:


# simulate with mismatched noise estimation
model_allzero_16_ms = LDPC_QAM_AWGN(k,
                                    n,
                                    num_bits_per_symbol=4,
                                    use_allzero=False, # full simulation
                                    cn_update="minsum", # activate min-sum decoding
                                    no_est_mismatch=0.15) # noise_var mismatch at demapper
                                     
ber_plot_allzero16qam.simulate(model_allzero_16_ms,
                            ebno_dbs=np.arange(0, 7, 0.5),
                            legend="Min-sum decoding / 16-QAM (with mismatch)",
                            max_mc_iter=50,
                            num_target_bit_errors=1000,
                            batch_size=1000,
                            soft_estimates=False,
                            show_fig=True,
                            forward_keyboard_interrupt=False);


# Interestingly, *min-sum* decoding is more robust w.r.t. inaccurate LLR estimations. 
# It is worth mentioning that *min-sum* decoding itself causes a performance loss. However, more advanced min-sum-based decoding approaches (offset-corrected min-sum) can operate close to *full BP* decoding.
# 
# You can also try:
# 
# - What happens with max-log demapping?
# - Implement offset corrected min-sum decoding
# - Have a closer look at the error-floor behavior
# - Apply the concept of [Weighted BP](https://nvlabs.github.io/sionna/phy/tutorials/Weighted_BP_Algorithm.html) to mismatched demapping

# ## References
# 
# [1] E. Zehavi, "8-PSK Trellis Codes for a Rayleigh Channel," IEEE Transactions on Communications, vol. 40, no. 5, 1992.
# 
# [2] G. Caire, G. Taricco and E. Biglieri, "Bit-interleaved Coded Modulation," IEEE Transactions on Information Theory, vol. 44, no. 3, 1998.
# 
# [3] G. Ungerböck, "Channel Coding with Multilevel/Phase Signals."IEEE Transactions on Information Theory, vol. 28, no. 1, 1982.
# 
# [4] J. L. Massey, “Coding and modulation in digital communications,” in Proc. Int. Zurich Seminar Commun., 1974.
# 
# [5] G. Böcherer, "Principles of Coded Modulation," Habilitation thesis, Tech. Univ. Munich, Munich, Germany, 2018.
# 
# [6] F. Schreckenbach, "Iterative Decoding of Bit-Interleaved Coded Modulation", PhD thesis, Tech. Univ. Munich, Munich, Germany, 2007.
# 
# [7] S. ten Brink, “Convergence Behavior of Iteratively Decoded Parallel Concatenated Codes,” IEEE Transactions on Communications, vol. 49, no. 10, pp. 1727-1737, 2001.
# 
# [8] S. ten Brink, G. Kramer, and A. Ashikhmin, “Design of low-density parity-check codes for modulation and detection,” IEEE Trans. Commun., vol. 52, no. 4, pp. 670–678, Apr. 2004.
# 
# [9] J. Hou, P. H. Siegel, L. B. Milstein, and H. D. Pfister, “Capacity-approaching bandwidth-efficient coded modulation schemes based on low-density parity-check codes,” IEEE Trans. Inform. Theory, vol. 49, no. 9, pp. 2141–2155, 2003.
# 
# [10] A. Alvarado, L. Szczecinski, R. Feick, and L. Ahumada, "Distribution of L-values in Gray-mapped M 2-QAM: Closed-form approximations and applications," IEEE Transactions on Communications, vol. 57, no. 7, pp. 2071-2079, 2009.
# 
# [11] ETSI 3GPP TS 38.212 “5G NR Multiplexing and channel coding”, v.16.5.0, 2021-03.
-e 
# --- End of Bit_Interleaved_Coded_Modulation.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Channel Models from Datasets

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
                           l_min=l_min, l_max=l_max, normalize_channel=True,
                           add_awgn=True, return_channel=True)

y_time, h_time = time_channel(x_time, no)


# Next, we will compare the perfect CSI that we computed above using the ideal
# channel frequency response and the estimated channel response that we obtain 
# from pilots with nearest-neighbor interpolation based on simulated transmissions
# in the time domain.

# In[25]:


# In the example above, we assumed perfect CSI, i.e.,
# h_hat correpsond to the exact ideal channel frequency response.
h_perf = h_hat[0,0,0,0,0,0]

# We now compute the LS channel estimate from the pilots.
h_est, _ = ls_est (y, no)
h_est = h_est[0,0,0,0,0,0]


# In[26]:


plt.figure()
plt.plot(np.real(h_perf))
plt.plot(np.imag(h_perf))
plt.plot(np.real(h_est), "--")
plt.plot(np.imag(h_est), "--")
plt.xlabel("Subcarrier index")
plt.ylabel("Channel frequency response")
plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
plt.title("Comparison of channel frequency responses");


# ### Downlink Transmission in the Frequency Domain <a class="anchor" id="Downlink-Transmission-in-the-Frequency-Domain"></a>
# 
# We will now simulate a simple downlink transmission in the frequency domain.
# In contrast to the uplink, the transmitter is now assumed to precode independent data
# streams to each antenna of the receiver based on perfect CSI.
# 
# The receiver can either estimate the channel or get access to the effective channel
# after precoding.
# 
# The first thing to do, is to change the `direction` within the CDL model. This makes the BS the transmitter and the UT the receiver.

# In[27]:


direction = "downlink"
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)


# The following code shows the other necessary modifications:

# In[28]:


perfect_csi = True # Change to switch between perfect and imperfect CSI
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
cir = cdl(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)

# Precode the transmit signal in the frequency domain
# It is here assumed that the transmitter has perfect knowledge of the channel
# One could here reduce this to perfect knowledge of the channel for the first
# OFDM symbol, or a noisy version of it to take outdated transmit CSI into account.
# `g` is the post-beamforming or `effective channel` that can be
# used to simulate perfect CSI at the receiver.
x_rg, g = zf_precoder(x_rg, h_freq)

y = channel_freq(x_rg, h_freq, no)

if perfect_csi:
    # The receiver gets here the effective channel after precoding as CSI
    h_hat, err_var = g, 0.
else:
    h_hat, err_var = ls_est (y, no)

x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))


# We do not explain here on purpose how to model the downlink transmission in the time domain
# as it is a good exercise for the reader to do it her/himself. The key steps are:
# 
# * Sample the channel impulse response at the Nyquist rate.
# * Downsample it to the OFDM symbol (+ cyclic prefix) rate (look at the uplink example).
# * Convert the downsampled CIR to the frequency domain.
# * Give this CSI to the transmitter for precoding.
# * Convert the CIR to discrete-time to compute the channel output in the time domain.

# ### Understand the Difference Between the CDL Models <a class="anchor" id="Understand-the-Difference-Between-the-CDL-Models"></a>
# 
# Before we proceed with more advanced simulations, it is important to understand the differences
# between the different CDL models. The models "A", "B", and "C" are non-line-of-sight (NLOS) models,
# while "D" and "E" are LOS. In the following code snippet, we compute the empirical cummulative
# distribution function (CDF) of the condition number of the channel frequency response matrix
# between all receiver and transmit antennas.

# In[29]:


def fun(cdl_model):
    """Generates a histogram of the channel condition numbers"""

    # Setup a CIR generator
    cdl = CDL(cdl_model, delay_spread, carrier_frequency,
              ut_array, bs_array, "uplink", min_speed=0)

    # Generate random CIR realizations
    # As we nned only a single sample in time, the sampling_frequency
    # does not matter.
    cir = cdl(2000, 1, 1)

    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)

    # Reshape to [batch_size, fft_size, num_rx_ant, num_tx_ant]
    h = tf.squeeze(h)
    h = tf.transpose(h, [0,3,1,2])

    # Compute condition number
    c = np.reshape(np.linalg.cond(h), [-1])

    # Compute normalized histogram
    hist, bins = np.histogram(c, 150, (1, 150))
    hist = hist/np.sum(hist)
    return bins[:-1], hist

plt.figure()
for cdl_model in ["A", "B", "C", "D", "E"]:
    bins, hist = fun(cdl_model)
    plt.plot(bins, np.cumsum(hist))
plt.xlim([0,150])
plt.legend(["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"]);
plt.xlabel("Channel Condition Number")
plt.ylabel("CDF")
plt.title("CDF of the condition number of 8x4 MIMO channels");


# From the figure above, you can observe that the CDL-B and CDL-C models
# are substantially better conditioned than the other models. This makes 
# them more suitable for MIMO transmissions as we will observe in the next
# section.

# ### Create an End-to-End Model <a class="anchor" id="Create-an-End-to-End-Model"></a>
# 
# For longer simulations, it is often convenient to pack all code into a single
# model that outputs batches of transmitted and received information bits
# at a given Eb/No point. The following code defines a very general model that can
# simulate uplink and downlink transmissions with time or frequency domain modeling
# over the different CDL models. It allows to configure perfect or imperfect CSI,
# UT speed, cyclic prefix length, and the number of OFDM symbols for pilot transmissions.

# In[30]:


class Model(Block):
    """This block simulates OFDM MIMO transmissions over the CDL model.

    Simulates point-to-point transmissions between a UT and a BS.
    Uplink and downlink transmissions can be realized with either perfect CSI
    or channel estimation. ZF Precoding for downlink transmissions is assumed.
    The receiver (in both uplink and downlink) applies LS channel estimation
    and LMMSE MIMO equalization. A 5G LDPC code as well as QAM modulation are
    used.

    Parameters
    ----------
    domain : One of ["time", "freq"], str
        Determines if the channel is modeled in the time or frequency domain.
        Time-domain simulations are generally slower and consume more memory.
        They allow modeling of inter-symbol interference and channel changes
        during the duration of an OFDM symbol.

    direction : One of ["uplink", "downlink"], str
        For "uplink", the UT transmits. For "downlink" the BS transmits.

    cdl_model : One of ["A", "B", "C", "D", "E"], str
        The CDL model to use. Note that "D" and "E" are LOS models that are
        not well suited for the transmissions of multiple streams.

    delay_spread : float
        The nominal delay spread [s].

    perfect_csi : bool
        Indicates if perfect CSI at the receiver should be assumed. For downlink
        transmissions, the transmitter is always assumed to have perfect CSI.

    speed : float
        The UT speed [m/s].

    cyclic_prefix_length : int
        The length of the cyclic prefix in number of samples.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    subcarrier_spacing : float
        The subcarrier spacing [Hz]. Defaults to 15e3.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of independent Mote Carlo simulations
        to be performed at once. The larger this number, the larger the memory
        requiremens.

    ebno_db : float
        The Eb/No [dB]. This value is converted to an equivalent noise power
        by taking the modulation order, coderate, pilot and OFDM-related
        overheads into account.

    Output
    ------
    b : [batch_size, 1, num_streams, k], tf.float32
        The tensor of transmitted information bits for each stream.

    b_hat : [batch_size, 1, num_streams, k], tf.float32
        The tensor of received information bits for each stream.
    """

    def __init__(self,
                 domain,
                 direction,
                 cdl_model,
                 delay_spread,
                 perfect_csi,
                 speed,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing = 15e3
                ):
        super().__init__()

        # Provided parameters
        self._domain = domain
        self._direction = direction
        self._cdl_model = cdl_model
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # System parameters
        self._carrier_frequency = 2.6e9
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 72
        self._num_ofdm_symbols = 14
        self._num_ut_ant = 4 # Must be a multiple of two as dual-polarized antennas are used
        self._num_bs_ant = 8 # Must be a multiple of two as dual-polarized antennas are used
        self._num_streams_per_tx = self._num_ut_ant
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 2
        self._coderate = 0.5

        # Required system components
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        self._ut_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_ut_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        if self._domain == "freq":
            self._channel_freq = ApplyOFDMChannel(add_awgn=True)

        elif self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
            self._l_tot = self._l_max - self._l_min + 1
            self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                  l_tot=self._l_tot,
                                                  add_awgn=True)
            self._modulator = OFDMModulator(self._cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        if self._direction == "downlink":
            self._zf_precoder = RZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

    @tf.function # Run in graph mode. See the following guide: https://www.tensorflow.org/guide/function
    def call(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        if self._domain == "time":
            # Time-domain simulations

            a, tau = self._cdl(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                         l_min=self._l_min, l_max=self._l_max, normalize=True)

            # As precoding is done in the frequency domain, we need to downsample
            # the path gains `a` to the OFDM symbol rate prior to converting the CIR
            # to the channel frequency response.
            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            x_time = self._modulator(x_rg)
            y_time = self._channel_time(x_time, h_time, no)

            y = self._demodulator(y_time)

        elif self._domain == "freq":
            # Frequency-domain simulations

            cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1/self._rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            y = self._channel_freq(x_rg, h_freq, no)

        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = self._remove_nulled_scs(h_freq)
            elif self._direction =="downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est (y, no)

        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)

        return b, b_hat


# ### Compare Uplink Performance Over the Different CDL Models <a class="anchor" id="Compare-Uplink-Performance-Over-the-Different-CDL-Models"></a>
# 
# We will now compare the uplink performance over the various CDL models assuming perfect CSI at the receiver. 
# Note that these simulations might take some time depending or you available hardware.
# You can reduce the `batch_size` if the model does not fit into the memory of your GPU. 
# The code will also run on CPU if not GPU is available.
# 
# If you do not want to run the simulation your self, you skip the next cell and simply look at the result in the next cell.

# In[31]:


UL_SIMS = {
    "ebno_db" : list(np.arange(-5, 20, 4.0)),
    "cdl_model" : ["A", "B", "C", "D", "E"],
    "delay_spread" : 100e-9,
    "domain" : "freq",
    "direction" : "uplink",
    "perfect_csi" : True,
    "speed" : 0.0,
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for cdl_model in UL_SIMS["cdl_model"]:

    model = Model(domain=UL_SIMS["domain"],
                  direction=UL_SIMS["direction"],
                  cdl_model=cdl_model,
                  delay_spread=UL_SIMS["delay_spread"],
                  perfect_csi=UL_SIMS["perfect_csi"],
                  speed=UL_SIMS["speed"],
                  cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"])

    ber, bler = sim_ber(model,
                        UL_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=1000,
                        target_bler=1e-3)

    UL_SIMS["ber"].append(list(ber.numpy()))
    UL_SIMS["bler"].append(list(bler.numpy()))

UL_SIMS["duration"] = time.time() - start


# In[32]:


print("Simulation duration: {:1.2f} [h]".format(UL_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("8x4 MIMO Uplink - Frequency Domain Modeling");
plt.ylim([1e-3, 1.1])
legend = []
for i, bler in enumerate(UL_SIMS["bler"]):
    plt.semilogy(UL_SIMS["ebno_db"], bler)
    legend.append("CDL-{}".format(UL_SIMS["cdl_model"][i]))
plt.legend(legend);


# ### Compare Downlink Performance Over the Different CDL Models <a class="anchor" id="Compare-Downlink-Performance-Over-the-Different-CDL-Models"></a>
# 
# We will now compare the downlink performance over the various CDL models assuming perfect CSI at the receiver. 
# 
# If you do not want to run the simulation your self, you skip the next cell and simply look at the result in the next cell.

# In[33]:


DL_SIMS = {
    "ebno_db" : list(np.arange(-5, 20, 4.0)),
    "cdl_model" : ["A", "B", "C", "D", "E"],
    "delay_spread" : 100e-9,
    "domain" : "freq",
    "direction" : "downlink",
    "perfect_csi" : True,
    "speed" : 0.0,
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for cdl_model in DL_SIMS["cdl_model"]:

    model = Model(domain=DL_SIMS["domain"],
                  direction=DL_SIMS["direction"],
                  cdl_model=cdl_model,
                  delay_spread=DL_SIMS["delay_spread"],
                  perfect_csi=DL_SIMS["perfect_csi"],
                  speed=DL_SIMS["speed"],
                  cyclic_prefix_length=DL_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=DL_SIMS["pilot_ofdm_symbol_indices"])

    ber, bler = sim_ber(model,
                        DL_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=1000,
                        target_bler=1e-3)

    DL_SIMS["ber"].append(list(ber.numpy()))
    DL_SIMS["bler"].append(list(bler.numpy()))

DL_SIMS["duration"] = time.time() -  start


# In[34]:


print("Simulation duration: {:1.2f} [h]".format(DL_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("8x4 MIMO Downlink - Frequency Domain Modeling");
plt.ylim([1e-3, 1.1])
legend = []
for i, bler in enumerate(DL_SIMS["bler"]):
    plt.semilogy(DL_SIMS["ebno_db"], bler)
    legend.append("CDL-{}".format(DL_SIMS["cdl_model"][i]))
plt.legend(legend);


# ### Evaluate the Impact of Mobility <a class="anchor" id="Evaluate-the-Impact-of-Mobility"></a>
# 
# Let us now have a look at the impact of the UT speed on the uplink performance.
# We compare the scenarios of perfect and imperfect CSI and 0 m/s and 20 m/s speed.
# To amplify the detrimental effects of high mobility, we only configure a single
# OFDM symbol for pilot transmissions at the beginning of the resource grid.
# With perfect CSI, mobility plays hardly any role. However, once channel estimation is 
# taken into acount, the BLER saturates.
# 
# If you do not want to run the simulation your self, you skip the next cell and simply look at the result in the next cell.

# In[35]:


MOBILITY_SIMS = {
    "ebno_db" : list(np.arange(0, 32, 2.0)),
    "cdl_model" : "D",
    "delay_spread" : 100e-9,
    "domain" : "freq",
    "direction" : "uplink",
    "perfect_csi" : [True, False],
    "speed" : [0.0, 20.0],
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [0],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for perfect_csi in MOBILITY_SIMS["perfect_csi"]:
    for speed in MOBILITY_SIMS["speed"]:

        model = Model(domain=MOBILITY_SIMS["domain"],
                  direction=MOBILITY_SIMS["direction"],
                  cdl_model=MOBILITY_SIMS["cdl_model"],
                  delay_spread=MOBILITY_SIMS["delay_spread"],
                  perfect_csi=perfect_csi,
                  speed=speed,
                  cyclic_prefix_length=MOBILITY_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=MOBILITY_SIMS["pilot_ofdm_symbol_indices"])

        ber, bler = sim_ber(model,
                        MOBILITY_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=1000,
                        target_bler=1e-3)

        MOBILITY_SIMS["ber"].append(list(ber.numpy()))
        MOBILITY_SIMS["bler"].append(list(bler.numpy()))

MOBILITY_SIMS["duration"] = time.time() - start


# In[36]:


print("Simulation duration: {:1.2f} [h]".format(MOBILITY_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("CDL-D MIMO Uplink - Impact of UT mobility")

i = 0
for perfect_csi in MOBILITY_SIMS["perfect_csi"]:
    for speed in MOBILITY_SIMS["speed"]:
        style = "{}".format("-" if perfect_csi else "--")
        s = "{} CSI {}[m/s]".format("Perf." if perfect_csi else "Imperf.", speed)
        plt.semilogy(MOBILITY_SIMS["ebno_db"],
                     MOBILITY_SIMS["bler"][i],
                      style, label=s,)
        i += 1
plt.legend();
plt.ylim([1e-3, 1]);


# ### Evaluate the Impact of Insufficient Cyclic Prefix Length <a class="anchor" id="Evaluate-the-Impact-of-Insufficient-Cyclic-Prefix-Length"></a>
# 
# As a final example, let us have a look at how to simulate OFDM with an insufficiently long cyclic prefix.
# 
# It is important to notice, that ISI cannot be simulated in the frequency domain as the [OFDMChannel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#channel-with-ofdm-waveform) 
# implicitly assumes perfectly synchronized and ISI-free transmissions. Having no cyclic
# prefix translates simply into an improved Eb/No as no energy for its transmission is used.
# 
# Simulating a channel in the time domain requires significantly more memory and compute which
# might limit the scenarios for which it can be used.
# 
# If you do not want to run the simulation your self, you skip the next cell and simply look at the result in the next cell.If you do not want to run the simulation your self, you skip the next cell and visualize the result in the next cell.

# In[37]:


CP_SIMS = {
    "ebno_db" : list(np.arange(0, 17, 2.0)),
    "cdl_model" : "C",
    "delay_spread" : 100e-9,
    "subcarrier_spacing" : 15e3,
    "domain" : ["freq", "time"],
    "direction" : "uplink",
    "perfect_csi" : False,
    "speed" : 3.0,
    "cyclic_prefix_length" : [20, 2],
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration": None
}

start = time.time()

for cyclic_prefix_length in CP_SIMS["cyclic_prefix_length"]:
    for domain in CP_SIMS["domain"]:
        model = Model(domain=domain,
                  direction=CP_SIMS["direction"],
                  cdl_model=CP_SIMS["cdl_model"],
                  delay_spread=CP_SIMS["delay_spread"],
                  perfect_csi=CP_SIMS["perfect_csi"],
                  speed=CP_SIMS["speed"],
                  cyclic_prefix_length=cyclic_prefix_length,
                  pilot_ofdm_symbol_indices=CP_SIMS["pilot_ofdm_symbol_indices"],
                  subcarrier_spacing=CP_SIMS["subcarrier_spacing"])

        ber, bler = sim_ber(model,
                        CP_SIMS["ebno_db"],
                        batch_size=64,
                        max_mc_iter=1000,
                        num_target_block_errors=1000,
                        target_bler=1e-3)

        CP_SIMS["ber"].append(list(ber.numpy()))
        CP_SIMS["bler"].append(list(bler.numpy()))

CP_SIMS["duration"] = time.time() - start


# In[38]:


print("Simulation duration: {:1.2f} [h]".format(CP_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("CDL-B MIMO Uplink - Impact of Cyclic Prefix Length")

i = 0
for cyclic_prefix_length in CP_SIMS["cyclic_prefix_length"]:
    for domain in CP_SIMS["domain"]:
        s = "{} Domain, CP length: {}".format("Freq" if domain=="freq" else "Time",
                                               cyclic_prefix_length)
        plt.semilogy(CP_SIMS["ebno_db"],
                     CP_SIMS["bler"][i],
                     label=s)
        i += 1
plt.legend();
plt.ylim([1e-3, 1]);


# One can make a few important observations from the figure above:
# 
# 1. The length of the cyclic prefix has no impact on the performance if
#    the system is simulated in the frequency domain.   
#    The reason why the two curves for both frequency-domain simulations
#    do not overlap is that the cyclic prefix length affects the way the Eb/No is computed.
# 2. With a sufficiently large cyclic prefix (in our case ``cyclic_prefix_length = 20 >= l_tot = 17`` ), the performance of
#    time and frequency-domain simulations are identical.
# 3. With a too small cyclic prefix length, the performance degrades. At high SNR,  inter-symbol interference 
#    (from multiple streams) becomes the dominating source of interference. 
-e 
# --- End of MIMO_OFDM_Transmissions_over_CDL.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Neural Receiver for OFDM SIMO Systems

# In this notebook, you will learn how to train a neural receiver that implements OFDM detection.
# The considered setup is shown in the figure below.
# As one can see, the neural receiver substitutes channel estimation, equalization, and demapping.
# It takes as input the post-DFT (discrete Fourier transform) received samples, which form the received resource grid, and computes log-likelihood ratios (LLRs) on the transmitted coded bits.
# These LLRs are then fed to the outer decoder to reconstruct the transmitted information bits.


# Two baselines are considered for benchmarking, which are shown in the figure above.
# Both baselines use linear minimum mean square error (LMMSE) equalization and demapping assuming additive white Gaussian noise (AWGN).
# They differ by how channel estimation is performed:
# 
# - **Pefect CSI**: Perfect channel state information (CSI) knowledge is assumed.
# - **LS estimation**: Uses the transmitted pilots to perform least squares (LS) estimation of the channel with nearest-neighbor interpolation.
# 
# All the considered end-to-end systems use an LDPC outer code from the 5G NR specification, QPSK modulation, and a 3GPP CDL channel model simulated in the frequency domain.

# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Simulation Parameters](#Simulation-Parameters)
# * [Neural Receiver](#Neural-Receiver)
# * [End-to-end System](#End-to-end-System)
# * [End-to-end System as a Sionna Block](#End-to-end-System-as-a-Sionna-Block)
# * [Evaluation of the Baselines](#Evaluation-of-the-Baselines)
# * [Training the Neural Receiver](#Training-the-Neural-Receiver)
# * [Evaluation of the Neural Receiver](#Evaluation-of-the-Neural-Receiver)
# * [Pre-computed Results](#Pre-computed-Results)
# * [References](#References)

# ## GPU Configuration and Imports <a class="anchor" id="GPU-Configuration-and-Imports"></a>

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
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

from sionna.phy import Block
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.phy.channel import OFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
                            LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
from sionna.phy.utils import  ebnodb2no, insert_dims, log10, expand_to_rank
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import sim_ber


# ## Simulation Parameters <a class="anchor" id="Simulation-Parameters"></a>

# In[3]:


############################################
## Channel configuration
carrier_frequency = 3.5e9 # Hz
delay_spread = 100e-9 # s
cdl_model = "C" # CDL model to use
speed = 10.0 # Speed for evaluation and training [m/s]
# SNR range for evaluation and training [dB]
ebno_db_min = -5.0
ebno_db_max = 10.0

############################################
## OFDM waveform configuration
subcarrier_spacing = 30e3 # Hz
fft_size = 128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

############################################
## Modulation and coding configuration
num_bits_per_symbol = 2 # QPSK
coderate = 0.5 # Coderate for LDPC code

############################################
## Neural receiver configuration
num_conv_channels = 128 # Number of convolutional channels for the convolutional layers forming the neural receiver

############################################
## Training configuration
num_training_iterations = 30000 # Number of training iterations
training_batch_size = 128 # Training batch size
model_weights_path = "neural_receiver_weights" # Location to save the neural receiver weights once training is done

############################################
## Evaluation configuration
results_filename = "neural_receiver_results" # Location to save the results


# The `StreamManagement` class is used to configure the receiver-transmitter association and the number of streams per transmitter.
# A SIMO system is considered, with a single transmitter equipped with a single non-polarized antenna.
# Therefore, there is only a single stream, and the receiver-transmitter association matrix is $[1]$.
# The receiver is equipped with an antenna array.

# In[4]:


stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                                  1)               # One stream per transmitter


# The `ResourceGrid` class is used to configure the OFDM resource grid. It is initialized with the parameters defined above.

# In[5]:


resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = 1,
                             num_streams_per_tx = 1,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)


# Outer coding is performed such that all the databits carried by the resource grid with size `fft_size`x`num_ofdm_symbols` form a single codeword.

# In[6]:


# Codeword length. It is calculated from the total number of databits carried by the resource grid, and the number of bits transmitted per resource element
n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
# Number of information bits per codeword
k = int(n*coderate)


# The SIMO link is setup by considering an uplink transmission with one user terminal (UT) equipped with a single non-polarized antenna, and a base station (BS) equipped with an antenna array.
# One can try other configurations for the BS antenna array.

# In[7]:


ut_antenna = Antenna(polarization="single",
                     polarization_type="V",
                     antenna_pattern="38.901",
                     carrier_frequency=carrier_frequency)

bs_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)


# ## Neural Receiver <a class="anchor" id="Neural-Receiver"></a>

# The next cell defines the Keras layers that implement the neural receiver.
# As in [1] and [2], a neural receiver using residual convolutional layers is implemented. Convolutional layers are leveraged to efficienly process the 2D resource grid, that is fed as an input to the neural receiver.
# Residual (skip) connections are used to avoid gradient vanishing [3].
# 
# For convinience, a Keras layer that implements a *residual block* is first defined. The Keras layer that implements the neural receiver is built by stacking such blocks. The following figure shows the architecture of the neural receiver.


# In[8]:


class ResidualBlock(Layer):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Input of the layer

    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Output of the layer
    """

    def build(self, input_shape):

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_1 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=num_conv_channels,
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
    r"""
    Keras layer implementing a residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.

    As the neural receiver is fed with the entire resource grid, including the guard bands and pilots, it also computes LLRs for these resource elements.
    They must be discarded to only keep the LLRs corresponding to the data-carrying resource elements.

    Input
    ------
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
        Received post-DFT samples.

    no : [batch size], tf.float32
        Noise variance. At training, a different noise variance value is sampled for each batch example.

    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
        LLRs on the transmitted bits.
        LLRs computed for resource elements not carrying data (pilots, guard bands...) must be discarded.
    """

    def build(self, input_shape):

        # Input convolution
        self._input_conv = Conv2D(filters=num_conv_channels,
                                  kernel_size=[3,3],
                                  padding='same',
                                  activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = Conv2D(filters=num_bits_per_symbol,
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=None)

    def call(self, y, no):

        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)

        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
        no = insert_dims(no, 3, 1)
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

        return z


# ## End-to-end System <a class="anchor" id="End-to-end-System"></a>

# The following cell defines the end-to-end system.
# 
# Training is done on the bit-metric decoding (BMD) rate which is computed from the transmitted bits and LLRs:
# 
# \begin{equation}
# R = 1 - \frac{1}{SNMK} \sum_{s = 0}^{S-1} \sum_{n = 0}^{N-1} \sum_{m = 0}^{M-1} \sum_{k = 0}^{K-1} \texttt{BCE} \left( B_{s,n,m,k}, \texttt{LLR}_{s,n,m,k} \right)
# \end{equation}
# 
# where
# 
# * $S$ is the batch size
# * $N$ the number of subcarriers
# * $M$ the number of OFDM symbols
# * $K$ the number of bits per symbol
# * $B_{s,n,m,k}$ the $k^{th}$ coded bit transmitted on the resource element $(n,m)$ and for the $s^{th}$ batch example
# * $\texttt{LLR}_{s,n,m,k}$ the LLR (logit) computed by the neural receiver corresponding to the $k^{th}$ coded bit transmitted on the resource element $(n,m)$ and for the $s^{th}$ batch example
# * $\texttt{BCE} \left( \cdot, \cdot \right)$ the binary cross-entropy in log base 2
# 
# Because no outer code is required at training, the outer encoder and decoder are not used at training to reduce computational complexity.
# 
# The BMD rate is known to be an achievable information rate for BICM systems, which motivates its used as objective function [4].

# In[9]:


## Transmitter
binary_source = BinarySource()
mapper = Mapper("qam", num_bits_per_symbol)
rg_mapper = ResourceGridMapper(resource_grid)

## Channel
cdl = CDL(cdl_model, delay_spread, carrier_frequency,
          ut_antenna, bs_array, "uplink", min_speed=speed)
channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

## Receiver
neural_receiver = NeuralReceiver()
rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements


# The following cell performs one forward step through the end-to-end system:

# In[10]:


batch_size = 64
ebno_db = tf.fill([batch_size], 5.0)
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)


## Transmitter
# Generate codewords
c = binary_source([batch_size, 1, 1, n])
print("c shape: ", c.shape)
# Map bits to QAM symbols
x = mapper(c)
print("x shape: ", x.shape)
# Map the QAM symbols to a resource grid
x_rg = rg_mapper(x)
print("x_rg shape: ", x_rg.shape)

######################################
## Channel
# A batch of new channel realizations is sampled and applied at every inference
no_ = expand_to_rank(no, tf.rank(x_rg))
y,_ = channel(x_rg, no_)
print("y shape: ", y.shape)

######################################
## Receiver
# The neural receiver computes LLRs from the frequency domain received symbols and N0
y = tf.squeeze(y, axis=1)
llr = neural_receiver(y, no)
print("llr shape: ", llr.shape)
# Reshape the input to fit what the resource grid demapper is expected
llr = insert_dims(llr, 2, 1)
# Extract data-carrying resource elements. The other LLRs are discarded
llr = rg_demapper(llr)
llr = tf.reshape(llr, [batch_size, 1, 1, n])
print("Post RG-demapper LLRs: ", llr.shape)


# The BMD rate is computed from the LLRs and transmitted bits as follows:

# In[11]:


bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
bce = tf.reduce_mean(bce)
rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
print(f"Rate: {rate:.2E} bit")


# The rate is very poor (negative values means 0 bit) as the neural receiver is not trained.

# ## End-to-end System as a Sionna Block

# The following Sionna block implements the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver).
# 
# When instantiating the end-to-end model, the parameter ``system`` is used to specify the system to setup, and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated. The ``training`` parameter is only relevant when the neural receiver is used.
# 
# At each call of this model:
# 
# * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
# * A batch of channel realizations is randomly sampled and applied to the channel inputs
# * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
#   Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends on the specified ``system`` parameter.
# * If not training, the outer decoder is applied to reconstruct the information bits
# * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits
# 

# In[12]:


class E2ESystem(Block):
    r"""
    Sionna Block that implements the end-to-end system

    As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
    the link components (transmitter, channel model, outer code...), they are implemented using the same end-to-end model.

    When instantiating the Sionna block, the parameter ``system`` is used to specify the system to setup,
    and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
    The ``training`` parameter is only relevant when the neural

    At each call of this model:
    * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
    * A batch of channel realizations is randomly sampled and applied to the channel inputs
    * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
      Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
      on the specified ``system`` parameter.
    * If not training, the outer decoder is applied to reconstruct the information bits
    * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits

    Parameters
    -----------
    system : str
        Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'

    training : bool
        Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
        If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
        This significantly reduces the computational complexity of training.
        If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
        an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.

    Input
    ------
    batch_size : int
        Batch size

    no : scalar or [batch_size], tf.float
        Noise variance.
        At training, a different noise variance should be sampled for each batch example.

    Output
    -------
    If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
    should be used as objective for training.
    If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
    compute the block/bit error rate.
    """

    def __init__(self, system, training=False):
        super().__init__()
        self._system = system
        self._training = training

        ######################################
        ## Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(k, n)
        self._mapper = Mapper("qam", num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)

        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
        cdl = CDL(cdl_model, delay_spread, carrier_frequency,
                  ut_antenna, bs_array, "uplink", min_speed=speed)
        self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
            self._demapper = Demapper("app", "qam", num_bits_per_symbol)
        elif system == "neural-receiver": # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    @tf.function
    def call(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        ######################################
        ## Transmitter
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, 1, 1, n])
        else:
            b = self._binary_source([batch_size, 1, 1, k])
            c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y,h = self._channel(x_rg, no_)

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est(y, no) # LS channel estimation with nearest-neighbor
            x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper(x_hat, no_eff_) # Demapping
        elif self._system == "neural-receiver":
            # The neural receiver computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            llr = self._neural_receiver(y, no)
            llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            return rate
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b,b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation


# ## Evaluation of the Baselines <a class="anchor" id="Evaluation-of-the-Baselines"></a>

# We evaluate the BERs achieved by the baselines in the next cell.
# 
# **Note:** Evaluation of the two systems can take a while. Therefore, we provide pre-computed results at the end of this notebook.

# In[13]:


# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     0.5) # Step


# In[14]:


# Dictionary storing the evaluation results
BLER = {}

model = E2ESystem('baseline-perfect-csi')
_,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
BLER['baseline-perfect-csi'] = bler.numpy()

model = E2ESystem('baseline-ls-estimation')
_,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
BLER['baseline-ls-estimation'] = bler.numpy()


# ## Training the Neural Receiver <a class="anchor" id="Training-the-Neural-Receiver"></a>

# In the next cell, one forward pass is performed within a *gradient tape*, which enables the computation of gradient and therefore the optimization of the neural network through stochastic gradient descent (SGD).

# **Note:** For an introduction to the implementation of differentiable communication systems and their optimization through SGD and backpropagation with Sionna, please refer to [the Part 2 of the Sionna tutorial for Beginners](https://nvlabs.github.io/sionna/phy/tutorials/Sionna_tutorial_part2.html).

# In[15]:


# The end-to-end system equipped with the neural receiver is instantiated for training.
# When called, it therefore returns the estimated BMD rate
model = E2ESystem('neural-receiver', training=True)

# Sampling a batch of SNRs
ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
# Forward pass
with tf.GradientTape() as tape:
    rate = model(training_batch_size, ebno_db)
    # Tensorflow optimizers only know how to minimize loss function.
    # Therefore, a loss function is defined as the additive inverse of the BMD rate
    loss = -rate


# Next, one can perform one step of stochastic gradient descent (SGD).
# The Adam optimizer is used

# In[16]:


optimizer = tf.keras.optimizers.Adam()

# Computing and applying gradients
weights = tape.watched_variables()
grads = tape.gradient(loss, weights)
optimizer.apply_gradients(zip(grads, weights))


# Training consists in looping over SGD steps. The next cell implements a training loop.
# 
# At each iteration:
# - A batch of SNRs $E_b/N_0$ is sampled
# - A forward pass through the end-to-end system is performed within a gradient tape
# - The gradients are computed using the gradient tape, and applied using the Adam optimizer
# - The achieved BMD rate is periodically shown
# 
# After training, the weights of the models are saved in a file
# 
# **Note:** Training can take a while. Therefore, [we have made pre-trained weights available](https://drive.google.com/file/d/1W9WkWhup6H_vXx0-CojJHJatuPmHJNRF/view?usp=sharing). Do not execute the next cell if you don't want to train the model from scratch. 

# In[17]:


training = False # Change to True to train your own model
if training:
    model = E2ESystem('neural-receiver', training=True)

    optimizer = tf.keras.optimizers.Adam()

    for i in range(num_training_iterations):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            rate = model(training_batch_size, ebno_db)
            # Tensorflow optimizers only know how to minimize loss function.
            # Therefore, a loss function is defined as the additive inverse of the BMD rate
            loss = -rate
        # Computing and applying gradients
        weights = tape.watched_variables()
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Periodically printing the progress
        if i % 100 == 0:
            print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')

    # Save the weights in a file
    weights = model._neural_receiver.weights
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)


# ## Evaluation of the Neural Receiver <a class="anchor" id="Evaluation-of-the-Neural-Receiver"></a>

# The next cell evaluates the neural receiver.
# 
# **Note:** Evaluation of the system can take a while and requires having the trained weights of the neural receiver. Therefore, we provide pre-computed results at the end of this notebook.

# In[18]:


model = E2ESystem('neural-receiver')

# Run one inference to build the layers and loading the weights
model(1, tf.constant(10.0, tf.float32))
with open(model_weights_path, 'rb') as f:
    weights = pickle.load(f)

for i, w in enumerate(weights):
    model._neural_receiver.weights[i].assign(w)

# Evaluations
_,bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
BLER['neural-receiver'] = bler.numpy()


# Finally, we plots the BLERs

# In[19]:


plt.figure(figsize=(10,6))
# Baseline - Perfect CSI
plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
# Baseline - LS Estimation
plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
# Neural receiver
plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-.', c=f'C2', label=f'Neural receiver')
#
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()


# ## Pre-computed Results <a class="anchor" id="Pre-computed-Results"></a>

# In[20]:


BLER = eval(pre_computed_results)


# ## References <a class="anchor" id="References"></a>

# [1] M. Honkala, D. Korpi and J. M. J. Huttunen, "DeepRx: Fully Convolutional Deep Learning Receiver," in IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 3925-3940, June 2021, doi: 10.1109/TWC.2021.3054520.
# 
# [2] F. Ait Aoudia and J. Hoydis, "End-to-end Learning for OFDM: From Neural Receivers to Pilotless Communication," in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2021.3101364.
# 
# [3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778
# 
# [4] G. Böcherer, "Achievable Rates for Probabilistic Shaping", arXiv:1707.01134, 2017.
-e 
# --- End of Neural_Receiver.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # OFDM MIMO Channel Estimation and Detection

# In this notebook, we will evaluate some of the OFDM channel estimation and MIMO
# detection algorithms available in Sionna PHY.
# 
# We will start by evaluating the mean square error (MSE) preformance of various channel estimation and interpolation methods.
# 
# Then, we will compare some of the MIMO detection algorithms under both perfect and imperfect channel state information (CSI) in terms of uncoded symbol error rate (SER) and coded bit error rate (BER).
# 
# The developed end-to-end models in this notebook are a great tool for benchmarking of MIMO receivers under realistic conditions.  They can be easily extended to new channel estimation methods or MIMO detection algorithms.

# For MSE evaluations, the block diagram of the system looks as follows:


# where the channel estimation module is highlighted as it is the focus of this evaluation. The channel covariance matrices are required for linear minimum mean square error (LMMSE) channel interpolation.

# For uncoded SER evaluations, the block diagram of the system looks as follows:


# where the channel estimation and detection modules are highlighted as they are the focus of this evaluation.

# Finally, for coded BER evaluations, the block diagram of the system looks as follows:


# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Simulations parameters](#Simulation-parameters)
# * [Estimation of the channel time, frequency, and spatial covariance matrices](#Estimation-of-the-channel-time,-frequency,-and-spatial-covariance-matrices)
# * [Loading the channel covariance matrices](#Loading-the-channel-covariance-matrices)
# * [Comparison of OFDM estimators](#Comparison-of-OFDM-estimators)
# * [Comparison of MIMO detectors](#Comparison-of-MIMO-detectors)

# ## GPU Configuration and Imports <a class="anchor" id="GPU-Configuration-and-Imports"></a>

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

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import sim_ber, ebnodb2no
from sionna.phy.mapping import Mapper, QAMSource, BinarySource
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
                            LMMSEInterpolator, LinearDetector, KBestDetector, \
                            EPDetector, MMSEPICDetector
from sionna.phy.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.phy.channel.tr38901 import UMi, Antenna, PanelArray
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder


# ## Simulation parameters

# The next cell defines the simulation parameters used throughout this notebook.
# 
# This includes the OFDM waveform parameters, [antennas geometries and patterns](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.channel.tr38901.PanelArray), and the [3GPP UMi channel model](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.channel.tr38901.UMi).

# In[3]:


NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 12*4 # 4 PRBs
SUBCARRIER_SPACING = 30e3 # Hz
CARRIER_FREQUENCY = 3.5e9 # Hz
SPEED = 3. # m/s

# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
UT_ANTENNA = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni', # Omnidirectional antenna pattern
                     carrier_frequency=CARRIER_FREQUENCY)

# The base station is equipped with an antenna
# array of 8 cross-polarized antennas,
# resulting in a total of 16 antenna elements.
NUM_RX_ANT = 16
BS_ARRAY = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901', # 3GPP 38.901 antenna pattern
                      carrier_frequency=CARRIER_FREQUENCY)

# 3GPP UMi channel model is considered
CHANNEL_MODEL = UMi(carrier_frequency=CARRIER_FREQUENCY,
                    o2i_model='low',
                    ut_array=UT_ANTENNA,
                    bs_array=BS_ARRAY,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)


# ## Estimation of the channel time, frequency, and spatial covariance matrices

# The linear minimum mean square (LMMSE) interpolation method requires knowledge of the time (i.e., across OFDM symbols), frequency (i.e., across sub-carriers), and spatial (i.e., across receive antennas) covariance matrices of the channel frequency response.
# 
# These are estimated in this section using Monte Carlo sampling.
# 
# We explain below how this is achieved for the frequency covariance matrix. The same approach is used for the time and spatial covariance matrices.
# 
# Let $N$ be the number of sub-carriers.
# The first step for estimating the frequency covariance matrix is to sample the channel model in order to build a set of frequency-domain channel realizations $\left\{ \mathbf{h}_k \right\}, 1 \leq k \leq K$, where $K$ is the number of samples and $\mathbf{h}_k  \in \mathbb{C}^{N}$ are complex-valued samples of the channel frequency response.
# 
# The frequency covariance matrix $\mathbf{R}^{(f)} \in \mathbb{C}^{N \times N}$ is then estimated by
# 
# \begin{equation}
# \mathbf{R}^{(f)} \approx \frac{1}{K} \sum_{k = 1}^K \mathbf{h}_k \mathbf{h}_k^{\mathrm{H}}
# \end{equation}
# 
# where we assume that the frequency-domain channel response has zero mean.
# 
# The following cells implement this process for all three dimensions (frequency, time, and space).
# 

# The next cell defines a [resource grid](https://nvlabs.github.io/sionna/phy/api/ofdm.html#resource-grid) and an [OFDM channel generator](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.phy.channel.GenerateOFDMChannel) for sampling the channel in the frequency domain.

# In[4]:


rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                  fft_size=FFT_SIZE,
                  subcarrier_spacing=SUBCARRIER_SPACING)
channel_sampler = GenerateOFDMChannel(CHANNEL_MODEL, rg)


# Then, a function that samples the channel is defined.
# It randomly samples a network topology for every batch and for every batch example using the [appropriate utility function](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.phy.channel.gen_single_sector_topology).

# In[5]:


def sample_channel(batch_size):
    # Sample random topologies
    topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
    CHANNEL_MODEL.set_topology(*topology)

    # Sample channel frequency responses
    # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    h_freq = channel_sampler(batch_size)
    # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
    h_freq = h_freq[:,0,:,0,0]

    return h_freq


# We now define a function that estimates the frequency, time, and spatial covariance matrcies using Monte Carlo sampling.

# In[6]:


@tf.function(jit_compile=True) # Use XLA for speed-up
def estimate_covariance_matrices(num_it, batch_size):
    freq_cov_mat = tf.zeros([FFT_SIZE, FFT_SIZE], tf.complex64)
    time_cov_mat = tf.zeros([NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS], tf.complex64)
    space_cov_mat = tf.zeros([NUM_RX_ANT, NUM_RX_ANT], tf.complex64)
    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = sample_channel(batch_size)

        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0,1,3,2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

        ###############################
        # Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0,2,1,3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(NUM_OFDM_SYMBOLS*num_it, tf.float32), 0.0)
    time_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(FFT_SIZE*num_it, tf.float32), 0.0)

    return freq_cov_mat, time_cov_mat, space_cov_mat


# We then compute the estimates by executing the function defined in the previous cell.
# 
# The batch size and number of iterations determine the total number of samples, i.e.,
# 
# ```
# number of samples = batch_size x num_iterations
# ```
# 
# and hence control the tradeoff between the accuracy of the estimates and the time needed for their computation.

# In[7]:


batch_size = 1000
num_iterations = 100
FREQ_COV_MAT, TIME_COV_MAT, SPACE_COV_MAT = estimate_covariance_matrices(batch_size, num_iterations)


# Finally, the estimated matrices are saved (as numpy arrays) for future use.

# In[8]:


# FREQ_COV_MAT : [fft_size, fft_size]
# TIME_COV_MAT : [num_ofdm_symbols, num_ofdm_symbols]
# SPACE_COV_MAT : [num_rx_ant, num_rx_ant]

np.save('freq_cov_mat', FREQ_COV_MAT.numpy())
np.save('time_cov_mat', TIME_COV_MAT.numpy())
np.save('space_cov_mat', SPACE_COV_MAT.numpy())


# ## Loading the channel covariance matrices

# The next cell loads saved estimates of the time, frequency, and space covariance matrices.

# In[9]:


FREQ_COV_MAT = np.load('freq_cov_mat.npy')
TIME_COV_MAT = np.load('time_cov_mat.npy')
SPACE_COV_MAT = np.load('space_cov_mat.npy')


# We then visualize the loaded matrices.
# 
# As one can see, the frequency correlation slowly decays with increasing spectral distance.
# 
# The time-correlation is much stronger as the mobility low. The covariance matrix is hence very badly conditioned with rank almost equal to one.
# 
# The spatial covariance matrix has a regular structure which is determined by the array geometry and polarization of its elements.

# In[10]:


fig, ax = plt.subplots(3,2, figsize=(10,12))
fig.suptitle("Time and frequency channel covariance matrices")

ax[0,0].set_title("Freq. cov. Real")
im = ax[0,0].imshow(FREQ_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[0,1].set_title("Freq. cov. Imag")
im = ax[0,1].imshow(FREQ_COV_MAT.imag, vmin=-0.3, vmax=1.8)

ax[1,0].set_title("Time cov. Real")
im = ax[1,0].imshow(TIME_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[1,1].set_title("Time cov. Imag")
im = ax[1,1].imshow(TIME_COV_MAT.imag, vmin=-0.3, vmax=1.8)

ax[2,0].set_title("Space cov. Real")
im = ax[2,0].imshow(SPACE_COV_MAT.real, vmin=-0.3, vmax=1.8)
ax[2,1].set_title("Space cov. Imag")
im = ax[2,1].imshow(SPACE_COV_MAT.imag, vmin=-0.3, vmax=1.8)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax);


# ## Comparison of OFDM estimators

# This section focuses on comparing the available OFDM channel estimators in Sionna for the considered setup.

# OFDM channel estimation consists of two steps:
# 
# 1. Channel estimation at pilot-carrying resource elements using [least-squares (LS)](https://nvlabs.github.io/sionna/phy/api/ofdm.html#sionna.phy.ofdm.LSChannelEstimator).
# 
# 2. Interpolation for data-carrying resource elements, for which three methods are available in Sionna:
# 
# - [Nearest-neighbor](https://nvlabs.github.io/sionna/phy/api/ofdm.html#sionna.phy.ofdm.NearestNeighborInterpolator), which uses the channel estimate of the nearest pilot
# - [Linear](https://nvlabs.github.io/sionna/phy/api/ofdm.html#sionna.phy.ofdm.LinearInterpolator), with optional averaging over the OFDM symbols (time dimension) for low mobility scenarios
# - [LMMSE](https://nvlabs.github.io/sionna/phy/api/ofdm.html#sionna.phy.ofdm.LMMSEInterpolator), which requires knowledge of the time and frequency covariance matrices
# 
# The LMMSE interpolator also features optional spatial smoothin, which requires the spatial covarance matrix. The [API documentation](https://nvlabs.github.io/sionna/phy/api/ofdm.html#sionna.phy.ofdm.LMMSEInterpolator) explains in more detail how this interpolator operates.

# ### End-to-end model

# In the next cell, we will create a Sionna Block which uses the interpolation method specified at initialization.
# 
# It computes the mean square error (MSE) for a specified batch size and signal-to-noise ratio (SNR) (in dB).
# 
# The following interpolation methods are available (set through the `int_method` parameter):
# 
# - `"nn"` : Nearest-neighbor interpolation
# - `"lin"` : Linear interpolation
# - `"lmmse"` : LMMSE interpolation
# 
# When LMMSE interpolation is used, it is required to specified the order in which interpolation and optional spatial smoothing is performed.
# This is achieved using the `lmmse_order` parameter. For example, setting this parameter to `"f-t"` leads to frequency interpolation being performed first followed by time interpolation, and no spatial smoothing.
# Setting it to `"t-f-s"` leads to time interpolation being performed first, followed by frequency interpolation, and finally spatial smoothing. 

# In[11]:


class MIMOOFDMLink(Block):

    def __init__(self, int_method, lmmse_order=None, **kwargs):
        super().__init__(kwargs)

        assert int_method in ('nn', 'lin', 'lmmse')


        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=1,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg

        # Stream management
        # Only a sinlge UT is considered for channel estimation
        sm = StreamManagement([[1]], 1)

        ##################################
        # Transmitter
        ##################################

        self.qam_source = QAMSource(num_bits_per_symbol=2) # Modulation order does not impact the channel estimation. Set to QPSK
        self.rg_mapper = ResourceGridMapper(rg)

        ##################################
        # Channel
        ##################################

        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
        time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
        space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
        if int_method == 'nn':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='nn')
        elif int_method == 'lin':
            self.channel_estimator = LSChannelEstimator(rg, interpolation_type='lin')
        elif int_method == 'lmmse':
            lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order=lmmse_order)
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)

    @tf.function
    def call(self, batch_size, snr_db):


        ##################################
        # Transmitter
        ##################################

        x = self.qam_source([batch_size, 1, 1, self.rg.num_data_symbols])
        x_rg = self.rg_mapper(x)

        ##################################
        # Channel
        ##################################

        no = tf.pow(10.0, -snr_db/10.0)
        topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel(x_rg, no)

        ###################################
        # Channel estimation
        ###################################

        h_hat,_ = self.channel_estimator(y_rg,no)

        ###################################
        # MSE
        ###################################

        mse = tf.reduce_mean(tf.square(tf.abs(h_freq-h_hat)))

        return mse


# The next cell defines a function for evaluating the mean square error (MSE) of a `model` over a range of SNRs (`snr_dbs`).
# 
# The `batch_size` and `num_it` parameters control the number of samples used to compute the MSE for each SNR value.

# In[12]:


def evaluate_mse(model, snr_dbs, batch_size, num_it):

    # Casting model inputs to TensorFlow types to avoid
    # re-building of the graph
    snr_dbs = tf.cast(snr_dbs, tf.float32)
    batch_size = tf.cast(batch_size, tf.int32)

    mses = []
    for snr_db in snr_dbs:

        mse_ = 0.0
        for _ in range(num_it):
            mse_ += model(batch_size, snr_db).numpy()
        # Averaging over the number of iterations
        mse_ /= float(num_it)
        mses.append(mse_)

    return mses


# The next cell defines the evaluation parameters.

# In[13]:


# Range of SNR (in dB)
SNR_DBs = np.linspace(-10.0, 20.0, 20)

# Number of iterations and batch size.
# These parameters control the number of samples used to compute each SNR value.
# The higher the number of samples is, the more accurate the MSE estimation is, at
# the cost of longer compute time.
BATCH_SIZE = 512
NUM_IT = 10

# Interpolation/filtering order for the LMMSE interpolator.
# All valid configurations are listed.
# Some are commented to speed-up simulations.
# Uncomment configurations to evaluate them!
ORDERS = ['s-t-f', # Space - time - frequency
          #'s-f-t', # Space - frequency - time
          #'t-s-f', # Time - space - frequency
          't-f-s', # Time - frequency - space
          #'f-t-s', # Frequency - time - space
          #'f-s-t', # Frequency - space- time
          #'f-t',   # Frequency - time (no spatial smoothing)
          't-f'   # Time - frequency (no spatial smoothing)
          ]


# The next cell evaluates the nearest-neighbor, linear, and LMMSE interpolator.
# For the LMMSE interpolator, we loop through the configuration listed in `ORDERS`.

# In[14]:


MSES = {}

# Nearest-neighbor interpolation
e2e = MIMOOFDMLink("nn")
MSES['nn'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)

# Linear interpolation
e2e = MIMOOFDMLink("lin")
MSES['lin'] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)

# LMMSE
for order in ORDERS:
    e2e = MIMOOFDMLink("lmmse", order)
    MSES[f"lmmse: {order}"] = evaluate_mse(e2e, SNR_DBs, BATCH_SIZE, NUM_IT)


# Finally, we plot the MSE.

# In[15]:


plt.figure(figsize=(8,6))

for est_label in MSES:
    plt.semilogy(SNR_DBs, MSES[est_label], label=est_label)

plt.xlabel(r"SNR (dB)")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)


# Unsurprisingly, the LMMSE interpolator leads to more accurate estimates compared to the two other methods, as it leverages knowledge of the the channel statistics.
# Moreover, the order in which the LMMSE interpolation steps are performed strongly impacts the accuracy of the estimator. This is because the LMMSE interpolation operates in one dimension at a time which is not equivalent to full-blown LMMSE estimation across all dimensions at one.
# 
# Also note that the order that leads to the best accuracy depends on the channel statistics. As a rule of thumb, it might be good to start with the dimension that is most strongly correlated (i.e., time in our example).

# ## Comparison of MIMO detectors

# An OFDM MIMO receiver consists of two stages: **OFDM channel estimation** and **MIMO detection**.
# 
# While the previous section focused on OFDM channel estimation, this section focuses now on MIMO detection.
# 
# The following MIMO detection algorithms, all available out-of-the-box in Sionna, are considered:
# 
# - [LMMSE equalization followed by APP demapping](https://nvlabs.github.io/sionna/phy/api/mimo.html#sionna.phy.mimo.LinearDetector)
# - [K-Best detection](https://nvlabs.github.io/sionna/phy/api/mimo.html#sionna.phy.mimo.KBestDetector)
# - [EP detection](https://nvlabs.github.io/sionna/phy/api/mimo.html#sionna.phy.mimo.EPDetector)
# - [MMSE-PIC detection](https://nvlabs.github.io/sionna/phy/api/mimo.html#sionna.phy.mimo.MMSEPICDetector)
# 
# Both perfect and imperfect channel state information is considered in the simulations.
# LS estimation combined with LMMSE interpolation is used, with time-frequency-space smoothing (in this order, i.e.,  `order='t-f-s'`).

# ### End-to-end model

# An end-to-end model is created in the next cell as a Sionna Block, which uses the detection method specified at initialization.
# 
# It computes either the coded bit error rate (BER) or the uncoded symbol error rate (SER), for a specified batch size, $E_b/N_0$ (in dB), and QAM modulation with a specified modulation order.
# When computing the BER, a 5G LDPC code is used with the specified coderate.
# 
# The following MIMO detection methods are considered (set through the `det_param` parameter):
# 
# - `"lmmse"` : No parameter needed
# - `"k-best"` : List size `k`, defaults to 64
# - `"ep"` : Number of iterations `l`, defaults to 10
# - `"mmse-pic"` : Number of self-iterations `num_it`, defaults to 4
# 
# The `det_param` parameter corresponds to either `k`, `l`, or `num_it`, for K-Best, EP, or MMSE-PIC, respectively. If set to `None`, a default value is used according to the selected detector.
# 
# The `perf_csi` parameter controls whether perfect CSI is assumed or not. If set to `False`, then LS combined with LMMSE interpolation is used to estimate the channel.
# 
# You can easily add your own MIMO detector and channel estimator to this model for a fair and realistic benchmark.

# In[16]:


class MIMOOFDMLink(Block):

    def __init__(self, output, det_method, perf_csi, num_tx, num_bits_per_symbol, det_param=None, coderate=0.5, **kwargs):
        super().__init__(kwargs)

        assert det_method in ('lmmse', 'k-best', 'ep', 'mmse-pic'), "Unknown detection method"

        self._output = output
        self.num_tx = num_tx
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.det_method = det_method
        self.perf_csi = perf_csi

        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=num_tx,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg

        # Stream management
        sm = StreamManagement(np.ones([1,num_tx], int), 1)

        # Codeword length and number of information bits per codeword
        n = int(rg.num_data_symbols*num_bits_per_symbol)
        k = int(coderate*n)
        self.n = n
        self.k = k

        # If output is symbol, then no FEC is used and hard decision are output
        hard_out = (output == "symbol")
        coded = (output == "bit")
        self.hard_out = hard_out
        self.coded = coded

        ##################################
        # Transmitter
        ##################################

        self.binary_source = BinarySource()
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, return_indices=True)
        self.rg_mapper = ResourceGridMapper(rg)
        if coded:
            self.encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=num_bits_per_symbol)

        ##################################
        # Channel
        ##################################

        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if not self.perf_csi:
            freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
            time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
            space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
            lmmse_int_time_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_time_first)

        # Detection
        if det_method == "lmmse":
            self.detector = LinearDetector("lmmse", output, "app", rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == 'k-best':
            if det_param is None:
                k = 64
            else:
                k = det_param
            self.detector = KBestDetector(output, num_tx, k, rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == "ep":
            if det_param is None:
                l = 10
            else:
                l = det_param
            self.detector = EPDetector(output, rg, sm, num_bits_per_symbol, l=l, hard_out=hard_out)
        elif det_method == 'mmse-pic':
            if det_param is None:
                l = 4
            else:
                l = det_param
            self.detector = MMSEPICDetector(output, 'app', rg, sm, num_iter=l, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)

        if coded:
            self.decoder = LDPC5GDecoder(self.encoder, hard_out=False)

    @tf.function
    def call(self, batch_size, ebno_db):


        ##################################
        # Transmitter
        ##################################

        if self.coded:
            b = self.binary_source([batch_size, self.num_tx, 1, self.k])
            c = self.encoder(b)
        else:
            c = self.binary_source([batch_size, self.num_tx, 1, self.n])
        bits_shape = tf.shape(c)
        x,x_ind = self.mapper(c)
        x_rg = self.rg_mapper(x)

        ##################################
        # Channel
        ##################################

        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, resource_grid=self.rg)
        topology = gen_single_sector_topology(batch_size, self.num_tx, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel(x_rg, no)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if self.perf_csi:
            h_hat = h_freq
            err_var = 0.0
        else:
            h_hat,err_var = self.channel_estimator(y_rg,no)

        # Detection
        if self.det_method == "mmse-pic":
            if self._output == "bit":
                prior_shape = bits_shape
            elif self._output == "symbol":
                prior_shape = tf.concat([tf.shape(x), [self.num_bits_per_symbol]], axis=0)
            prior = tf.zeros(prior_shape)
            det_out = self.detector(y_rg,h_hat,prior,err_var,no)
        else:
            det_out = self.detector(y_rg,h_hat,err_var,no)

        # (Decoding) and output
        if self._output == "bit":
            llr = tf.reshape(det_out, bits_shape)
            b_hat = self.decoder(llr)
            return b, b_hat
        elif self._output == "symbol":
            x_hat = tf.reshape(det_out, tf.shape(x_ind))
            return x_ind, x_hat


# The following function is used to evaluate all of the considered detectors for a given setup: It instantiates the end-to-end systems, runs the simulations, and returns the BER or SER.

# In[17]:


def run_sim(num_tx, num_bits_per_symbol, output, ebno_dbs, perf_csi, det_param=None):

    lmmse = MIMOOFDMLink(output, "lmmse", perf_csi, num_tx, num_bits_per_symbol, det_param)
    k_best = MIMOOFDMLink(output, "k-best", perf_csi, num_tx, num_bits_per_symbol, det_param)
    ep = MIMOOFDMLink(output, "ep", perf_csi, num_tx, num_bits_per_symbol, det_param)
    mmse_pic = MIMOOFDMLink(output, "mmse-pic", perf_csi, num_tx, num_bits_per_symbol, det_param)

    if output == "symbol":
        soft_estimates = False
        ylabel = "Uncoded SER"
    else:
        soft_estimates = True
        ylabel = "Coded BER"

    er_lmmse,_ = sim_ber(lmmse,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);

    er_ep,_ = sim_ber(ep,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);

    er_kbest,_ = sim_ber(k_best,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);

    er_mmse_pic,_ = sim_ber(mmse_pic,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);

    return er_lmmse, er_ep, er_kbest, er_mmse_pic


# The next cell defines the simulation parameters.

# In[18]:


# Range of SNR (dB)
EBN0_DBs = np.linspace(-10., 20.0, 10)

# Number of transmitters
NUM_TX = 4

# Modulation order (number of bits per symbol)
NUM_BITS_PER_SYMBOL = 4 # 16-QAM


# We start by evaluating the uncoded SER. The next cell runs the simulations with perfect CSI and channel estimation. Results are stored in the `SER` dictionary.

# In[19]:


SER = {} # Store the results

# Perfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, True)
SER['Perf. CSI / LMMSE'] = ser_lmmse
SER['Perf. CSI / EP'] = ser_ep
SER['Perf. CSI / K-Best'] = ser_kbest
SER['Perf. CSI / MMSE-PIC'] = ser_mmse_pic

# Imperfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, False)
SER['Ch. Est. / LMMSE'] = ser_lmmse
SER['Ch. Est. / EP'] = ser_ep
SER['Ch. Est. / K-Best'] = ser_kbest
SER['Ch. Est. / MMSE-PIC'] = ser_mmse_pic


# Next, we evaluate the coded BER. The cell below runs the simulations with perfect CSI and channel estimation. Results are stored in the `BER` dictionary.

# In[20]:


BER = {} # Store the results

# Perfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, True)
BER['Perf. CSI / LMMSE'] = ber_lmmse
BER['Perf. CSI / EP'] = ber_ep
BER['Perf. CSI / K-Best'] = ber_kbest
BER['Perf. CSI / MMSE-PIC'] = ber_mmse_pic

# Imperfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, False)
BER['Ch. Est. / LMMSE'] = ber_lmmse
BER['Ch. Est. / EP'] = ber_ep
BER['Ch. Est. / K-Best'] = ber_kbest
BER['Ch. Est. / MMSE-PIC'] = ber_mmse_pic


# Finally, we plot the results.

# In[21]:


fig, ax = plt.subplots(1,2, figsize=(16,7))
fig.suptitle(f"{NUM_TX}x{NUM_RX_ANT} UMi | {2**NUM_BITS_PER_SYMBOL}-QAM")

## SER

ax[0].set_title("Symbol error rate")
# Perfect CSI
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / LMMSE'], 'x-', label='Perf. CSI / LMMSE', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / EP'], 'o--', label='Perf. CSI / EP', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / K-Best'], 's-.', label='Perf. CSI / K-Best', c='C0')
ax[0].semilogy(EBN0_DBs, SER['Perf. CSI / MMSE-PIC'], 'd:', label='Perf. CSI / MMSE-PIC', c='C0')

# Imperfect CSI
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / LMMSE'], 'x-', label='Ch. Est. / LMMSE', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / EP'], 'o--', label='Ch. Est. / EP', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / K-Best'], 's-.', label='Ch. Est. / K-Best', c='C1')
ax[0].semilogy(EBN0_DBs, SER['Ch. Est. / MMSE-PIC'], 'd:', label='Ch. Est. / MMSE-PIC', c='C1')

ax[0].set_xlabel(r"$E_b/N0$")
ax[0].set_ylabel("SER")
ax[0].set_ylim((1e-4, 1.0))
ax[0].legend()
ax[0].grid(True)

## SER

ax[1].set_title("Bit error rate")
# Perfect CSI
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / LMMSE'], 'x-', label='Perf. CSI / LMMSE', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / EP'], 'o--', label='Perf. CSI / EP', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / K-Best'], 's-.', label='Perf. CSI / K-Best', c='C0')
ax[1].semilogy(EBN0_DBs, BER['Perf. CSI / MMSE-PIC'], 'd:', label='Perf. CSI / MMSE-PIC', c='C0')

# Imperfect CSI
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / LMMSE'], 'x-', label='Ch. Est. / LMMSE', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / EP'], 'o--', label='Ch. Est. / EP', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / K-Best'], 's-.', label='Ch. Est. / K-Best', c='C1')
ax[1].semilogy(EBN0_DBs, BER['Ch. Est. / MMSE-PIC'], 'd:', label='Ch. Est. / MMSE-PIC', c='C1')

ax[1].set_xlabel(r"$E_b/N0$")
ax[1].set_ylabel("BER")
ax[1].set_ylim((1e-4, 1.0))
ax[1].legend()
ax[1].grid(True)


# For this setup, the non-linear detection algorithms K-Best, EP, and MMSE-PIC, outperform the linear MMSE detection method.
# It is remarkable that K-Best and EP with imperfect CSI achieve lower BER than LMMSE detection with perfect CSI.
# 
# However, one should keep in mind that:
# 
# - EP is prone to numerical imprecision and could therefore achieve better BER/SER with double precision (`dtype=tf.complex128`). The number of iterations `l` as well as the update smoothing parameter `beta` impact performance.
# 
# - For K-Best, there is not a unique way to compute soft information and better performance could be achieved with improved methods for computing soft information from a list of candidates (see [list2llr](https://nvlabs.github.io/sionna/phy/api/mimo.html#sionna.phy.mimo.List2LLR)). Increasing the list size `k` results in improved accuracy at the cost of higher complexity.
# 
# - MMSE-PIC can be easily combined with a decoder to implement iterative detection and decoding, as it takes as input soft prior information on the bits/symbols.
-e 
# --- End of OFDM_MIMO_Detection.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Optical Channel with Lumped Amplification

# In this notebook, you will learn how to simulate the transmission of a single Gaussian impulse over a
# lumped amplification channel model consisting of multiple fiber spans and optical amplifiers, so-called Erbium Doped Fiber Amplifiers (EDFA), as shown in the Figure below. We assume a *standard single mode fiber* (S-SMF) and denote the fiber length between two amplifiers by $\ell_\text{span}$.


# Let $G$ denote the amplifier gain and $F$ the noise figure of each EDFA.
# 
# As we focus on the optical channel and not the corresponding signal processing, 
# the transmitter directly generates the optical signal. Hence, all
# components that, in practice, are required to
# generate the optical signal given an electrical control voltage
# (e.g., the Mach-Zehnder-Modulator (MZM)) are assumed to be ideal or are neglected. The same holds on the receiver side, where the photodiode that would add shot noise is neglected.
# 
# To provide a better understanding of the implemented channel impairments (attenuation, noise, dispersion,
# nonlinearity) introduced during propagation, those are successively
# enabled, starting with attenuation.

# ## Table of Contents
# - [Setup](#Setup)
# - [Attenuation](#Attenuation)
# - [Amplified Spontaneous Emission Noise](#Amplified-Spontaneous-Emission-Noise)
# - [Chromatic Dispersion](#Chromatic-Dispersion)
# - [Kerr Nonlinearity](#Kerr-Nonlinearity)
# - [Split-Step Fourier Method](#Split-Step-Fourier-Method)
# - [References](#References)

# ## Setup

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

import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Set random seed for reproducibility
from sionna.phy import dtypes, config
from sionna.phy.channel import utils
config.seed = 42


# ## Impulse Generation
# 
# Before diving into the first channel model, the simulation shall be parametrized and the
# initial Gaussian impulse
# 
# $$g(t)=\sqrt{P_0\cdot 2^{\left(-\left(\frac{2t}{T_0}\right)^2\right)}}$$
# 
# is generated. The impulse shall have peak power $P_0$ and a pulse duration of $T_0$.
# Note that the Gaussian filter is infinitely long, such that $T_0$ is the
# full width at half-maximum (FWHM) pulse duration.
# 
# Further, the simulation window is set to $T_\mathrm{sim}=1000\,\mathrm{ps}$ and the sample duration
# is set to $\Delta_t=1\,\mathrm{ps}$.

# In[2]:


config.precision="double"
config.tf_cdtype


# In[3]:


# Simulation parameters
config.precision = "double"
t_sim = int(1e4)  # (ps) Simulation time window
n_sim = int(1e4)  # Number of simulation samples

# Channel parameters
n_span = 3

# Impulse parameters
p_0 = 3e-2  # (W) Peak power of the Gaussian pulse
t_0 = 50  # (ps) Norm. temporal scaling of the Gaussian pulse

# Support
dt = t_sim / n_sim  # (s) sample duration
t, f = utils.time_frequency_vector(n_sim, dt)  # (ps), (THz) Time and frequency vector

# Generate Gaussian impulse
g_0 = np.sqrt(p_0 * 2**(-((2.0*t / t_0) ** 2.0)))
g_0 = tf.cast(g_0, dtype=config.tf_cdtype)
G_0 = tf.signal.fftshift(
        tf.abs(
            tf.cast(dt, config.tf_cdtype) *
            tf.signal.fft(g_0) /
            tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
        ) ** 2
)


# In[4]:


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t.numpy().flatten(), np.abs(g_0.numpy().flatten())**2, '-')
ax1.set_xlim(-150, 150)
ax1.set_xlabel(r"$t$ in (ps)")
ax1.set_ylabel(r"$|g(t)|^2$ in (W)")
ax1.grid()

ax2.plot(
    f.numpy().flatten(),
    (G_0.numpy().flatten())/np.max(G_0.numpy().flatten()),
    '-')
ax2.set_xlim(-0.015, 0.015)
ax2.set_xlabel(r"$f-f_c$ in (THz)")
ax2.set_ylabel(r"$\frac{|G(f-f_c)|^2}{|G_\mathrm{max}|^2}$")
ax2.grid()

ax1.legend(['transmitted'])
plt.tight_layout()
plt.show()


# ## Attenuation
# 
# Attenuation is present in all media including optical fibers. A typical value
# of $\alpha=0.046\,\mathrm{km}^{-1}$ is used in this notebook.
# To compensate for this, Erbium doped fiber amplifiers (EDFAs) are required, as shown in
# the figure at the beginning of this notebook.

# ## Amplified Spontaneous Emission Noise
# 
# An optical channel model contains several sources of noise, e.g., amplified spontaneous
# emission (ASE) and Rayleigh scattering. However, for this experiment, only ASE
# noise is implemented. It was shown in [1] that this is the most dominant source of noise.
# 
# As we assume a discrete lumped amplification, ASE noise is introduced only due to the amplification by
# the EDFAs. The noise power is given as
# 
# $$P_\mathrm{ASE}=\rho_\mathrm{ASE}\cdot f_\text{sim}=\frac{1}{2}G F h f_\text{c}\cdot f_\mathrm{sim}$$
# 
# and, hence, depends on the gain $G$, the (linear) noise figure $F$, the carrier frequency $f_\text{c}$,
# and the simulation bandwidth $f_\mathrm{sim}$. The intermediate quantitiy $\rho_\mathrm{ASE}$ denotes the noise
# spectral density of the EDFAs and $h$ is Planck's constant. Usually, not the simulation bandwidth but the captured
# bandwidth $W$ of the receiver is used. Here, for demonstration purpose, our receiver has an infinite
# bandwidth that is only limited by the simulation $W=f_\mathrm{sim}$.
# 
# **Note** that Sionna also provides ideally distributed Raman amplification, where the noise is introduced in
# a fiber span. This can be enabled by setting `with_amplification` to `True` when instantiating the `SSFM` layer.

# ### Channel Configuration

# The fiber (i.e., *SSFM*) is implemented using normalized units for distance and time.
# Hence, it is required that the units of the same kind (time, distance, power) for all
# parameters ($\alpha$, $\beta_2$, ...) have to be given with the same unit prefix,
# e.g., for time use always $\mathrm{ps}$. This does not only simplify the usage of the SSFM but also prevents from dealing with different
# orders of magnitude within the SSFM.
# 
# For our first experiment only ASE noise is considered and, thus, nonlinearity and chromatic dispersion (CD) are disabled.
# Attenuation is kept enabled such that the amplifiers are required and introduce the noise.
# The gain is chosen such that the link becomes transparent (input power equals output power).

# In[5]:


# Normalization
t_norm = 1e-12  # (s) -> (ps) Time normalization
z_norm = 1e3  # (m) -> (km) Distance normalization

# Fiber parameters
f_c = 193.55e12  # (Hz) Abs. Carrier frequency
length_sp = 80.0  # (km) Norm. fiber span length
alpha = 0.046  # (1/km) Norm. fiber attenuation

# EDFA parameters
g_edfa = tf.exp(alpha * length_sp)
f_edfa = 10**(5/10)  # (1) Noise figure


# In[6]:


span = sionna.phy.channel.optical.SSFM(
            alpha=alpha,
            f_c=f_c,
            length=length_sp,
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=True,
            with_dispersion=False,
            with_nonlinearity=False,
            t_norm=t_norm)

amplifier = sionna.phy.channel.optical.EDFA(
             g=g_edfa,
             f=f_edfa,
             f_c=f_c,
             dt=dt * t_norm  # t_norm is in absolute (not normalized) units
            )

def lumped_amplification_channel(inputs):
    (u_0) = inputs

    u = u_0
    for _ in range(n_span):
        u = span(u)
        u = amplifier(u)

    return u


# ### Transmission
# 
# Next, the impulse is transmitted over the channel and the output is visualized.

# In[7]:


x = g_0
y = lumped_amplification_channel(x)

X = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(x) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)

Y = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(y) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)


# In[8]:


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t.numpy().flatten(), np.abs(x.numpy().flatten())**2, '-')
ax1.plot(t.numpy().flatten(), np.abs(y.numpy().flatten())**2, '--')
ax1.set_xlim(-150, 150)
ax1.set_xlabel(r"$t$ in (ps)")
ax1.set_ylabel(r"$|g(t)|^2$ in (W)")
ax1.grid()

ax2.plot(
    f.numpy().flatten(),
    (X.numpy().flatten())/np.max(X.numpy().flatten()),
    '-')
ax2.plot(
    f.numpy().flatten(),
    (Y.numpy().flatten())/np.max(Y.numpy().flatten()),
    '--')
ax2.set_xlim(-0.015, 0.015)
ax2.set_xlabel(r"$f-f_c$ in (THz)")
ax2.set_ylabel(r"$\frac{|G(f-f_c)|^2}{|G_\mathrm{max}|^2}$")
ax2.grid()

ax1.legend(['transmitted', 'received'])
plt.tight_layout()

plt.show()


# As can be seen, attenuation is completely compensated by the EDFAs. However, they introduce significant noise.
# 

# ## Chromatic Dispersion
# 
# After having seen how the noise distorts the original Gaussian impulse, we can now
# enable the next linear effect, which is chromatic dispersion (CD). Regarding the nonlinear Schrödinger
# equation that describes the propagation of an optical signal, the impact of CD is parametrized
# by the group velocity dispersion (GVD) parameter $\beta_2$, where $\beta_2=-21.67\,\mathrm{ps}^2\mathrm{km}^{-1}$ is a typical choice.
# 
# ### Channel Configuration
# 
# Besides the present parameters we now set $\beta_2$. For a better understanding of
# CD we disable the noise (`EDFA.f = 0`) from the previous section.

# In[9]:


beta_2 = -21.67  # (ps^2/km) Norm. group velocity dispersion


# In[10]:


span_cd = sionna.phy.channel.optical.SSFM(
            alpha=alpha,
            beta_2=beta_2,
            f_c=f_c,
            length=length_sp,
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=True,
            with_dispersion=True,
            with_nonlinearity=False,
            t_norm=t_norm)

amplifier_cd = sionna.phy.channel.optical.EDFA(
                                    g=g_edfa,
                                    f=0,
                                    f_c=f_c,
                                    dt=dt * t_norm)


def lumped_amplification_channel_cd(inputs):
    (u_0) = inputs

    u = u_0
    for _ in range(n_span):
        u = span_cd(u)
        u = amplifier_cd(u)

    return u


# ### Transmission
# 
# We now transmit the previously generated Gaussian impulse over the optical fiber and compare the received signal with the transmitted impulse.
# 

# In[11]:


x = g_0  # previously generated Gaussian impulse
y = lumped_amplification_channel_cd(x)

X = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(x) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)

Y = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(y) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)

X_angle = tf.math.angle(tf.signal.fftshift(tf.signal.fft(x)))
Y_angle = tf.math.angle(tf.signal.fftshift(tf.signal.fft(y)))


# In[12]:


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t.numpy().flatten(), np.abs(x.numpy().flatten())**2, '-')
ax1.plot(t.numpy().flatten(), np.abs(y.numpy().flatten())**2, '--')
ax1.set_xlim(-250, 250)
ax1.set_xlabel(r"$t$ in (ps)")
ax1.set_ylabel(r"$|g(t)|^2$ in (W)")
ax1.grid()

ax2.plot(
    f.numpy().flatten(),
    (X.numpy().flatten())/np.max(X.numpy().flatten()),
    '-')
ax2.plot(
    f.numpy().flatten(),
    (Y.numpy().flatten())/np.max(Y.numpy().flatten()),
    '--')
ax2.set_xlim(-0.015, 0.015)
ax2.set_xlabel(r"$f-f_c$ in (THz)")
ax2.set_ylabel(r"$\frac{|G(f-f_c)|^2}{|G_\mathrm{max}|^2}$")
ax2.grid()

ax1.legend(['transmitted', 'received'])
plt.tight_layout()

plt.show()


# Compared to the transmit impulse the received one has significantly broadened in time.
# The absolute value of the spectrum, nevertheless, stayed the same.
# By plotting the phase of the received signal one can see the typical parabolic shift.

# In[13]:


fig, (ax1) = plt.subplots(1, 1, tight_layout=True)

ax1.plot(t.numpy().flatten(), np.angle(x.numpy().flatten()), '-')
ax1.plot(t.numpy().flatten(), np.angle(y.numpy().flatten()), '--')
ax1.set_xlim(-750, 750)
ax1.set_xlabel(r"$t$ in (ps)")
ax1.set_ylabel("$\u2220 x(t), \u2220 y(t)$")
ax1.grid()

ax1.legend(['transmitted', 'received'])
plt.tight_layout()

plt.show()


# ## Kerr Nonlinearity
# 
# Last, we depict the Kerr nonlinearity and, for a better understanding, disable all previous impairments.
# This nonlinear effect applies a phase shift to the transmitted signal depending on its
# instantaneous power. Hence, we should see a phase that, in contrast to the phase of the original signal which is zero, follows the (inverse) absolute value of the impulse.
# 
# **Note:** Only the interaction between Kerr nonlinearity and CD requires an
# SSFM for fiber simulation. Otherwise (as done so far), the transfer function
# of the individual effect is just a single multiplication (in time- or Fourier-domain,
# respectively).
# 
# ### Channel configuration
# 
# Similarly to the definition of CD, we specify a typical value for $\gamma=1.27\,\mathrm{\frac{1}{km W}}$.

# In[14]:


gamma = 1.27  # (1/W/km) Nonlinearity coefficient


# In[15]:


span_nl = sionna.phy.channel.optical.SSFM(
            alpha=alpha,
            beta_2=beta_2,
            f_c=f_c,
            length=length_sp,
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=True,
            with_dispersion=False,
            with_nonlinearity=True,
            t_norm=t_norm)

amplifier_nl = sionna.phy.channel.optical.EDFA(
                                g=g_edfa,
                                f=0,
                                f_c=f_c,
                                dt=dt * t_norm)


def lumped_amplification_channel_nl(inputs):
    (u_0) = inputs

    u = u_0
    for _ in range(n_span):
        u = span_nl(u)
        u = amplifier_nl(u)

    return u


# ### Transmission
# 
# We now transmit the same Gaussian impulse again over the optical fiber where only Kerr nonlinearity is activated.
# 

# In[16]:


x = g_0
y = lumped_amplification_channel_nl(x)

X = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(x) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)

Y = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(y) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)

X_angle = tf.math.angle(tf.signal.fftshift(tf.signal.fft(x)))
Y_angle = tf.math.angle(tf.signal.fftshift(tf.signal.fft(y)))


# In[17]:


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t.numpy().flatten(), np.abs(x.numpy().flatten())**2, '-')
ax1.plot(t.numpy().flatten(), np.abs(y.numpy().flatten())**2, '--')
ax1.set_xlim(-150, 150)
ax1.set_xlabel(r"$t$ in (ps)")
ax1.set_ylabel("$\u2220 x(t), \u2220 y(t)$")
ax1.grid()

ax2.plot(
    f.numpy().flatten(),
    (X.numpy().flatten())/np.max(X.numpy().flatten()),
    '-')
ax2.plot(
    f.numpy().flatten(),
    (Y.numpy().flatten())/np.max(Y.numpy().flatten()),
    '--')
ax2.set_xlim(-0.015, 0.015)
ax2.set_xlabel(r"$f-f_c$ in (THz)")
ax2.set_ylabel(r"$\frac{|G(f-f_c)|^2}{|G_\mathrm{max}|^2}$")
ax2.grid()

ax1.legend(['transmitted', 'received'])
plt.tight_layout()


plt.show()


# As shown in the previous plot the (isolated) Kerr nonlinearity does not affect the
# absolute value of the signal's amplitude but only shifts the phase (see below).
# 
# Further, the bandwidth of the transmit signal was slightly increased.
# 
# **Hint**: Increasing the peak power $p_0$ of the transmitted impuls increases the impact of the Kerr nonlinearity.
# 

# In[18]:


fig, (ax1) = plt.subplots(1, 1, tight_layout=True)

ax1.plot(t.numpy().flatten(), np.angle(x.numpy().flatten()), '-')
ax1.plot(t.numpy().flatten(), np.angle(y.numpy().flatten()), '--')
ax1.set_xlim(-750, 750)
ax1.set_xlabel("$t$ in (ps)")
ax1.set_ylabel("$\u2220 x(t), \u2220 y(t)$")
ax1.grid()

ax1.legend(['transmitted', 'received'])
plt.tight_layout()

plt.show()


# ## Split-Step Fourier Method
# 
# Last, we perform the true SSFM to simulate the impairments (ASE noise, CD, and Kerr
# nonlinearity) jointly. As this is computationally complex, we compile
# the channel model before its execution by adding the `tf.function` decorator.
# 
# ### Channel Configuration
# 
# Keeping the former configuration, we only have to increase the number of SSFM simulation steps.

# In[19]:


n_ssfm = 160  # number of SSFM simulation steps


# In[20]:


span_ssfm = sionna.phy.channel.optical.SSFM(
            alpha=alpha,
            beta_2=beta_2,
            gamma=gamma,
            f_c=f_c,
            length=length_sp,
            sample_duration=dt,
            n_ssfm=n_ssfm,
            with_amplification=False,
            with_attenuation=True,
            with_dispersion=True,
            with_nonlinearity=True,
            t_norm=t_norm)

amplifier_ssfm = sionna.phy.channel.optical.EDFA(
                        g=g_edfa,
                        f=0,
                        f_c=f_c,
                        dt=dt * t_norm)

@tf.function
def lumped_amplification_channel_ssfm(inputs):
    (u_0) = inputs

    u = u_0
    for _ in range(1):
        u = span_ssfm(u)
        u = amplifier_ssfm(u)

    return u


# ### Transmission
# 
# We transmit the Gaussian impulse over the optical fiber. However, we have now enabled ASE noise, CD, and Kerr
# nonlinearity.

# In[21]:


x = g_0
y = lumped_amplification_channel_ssfm(x)

X = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(x) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)

Y = tf.signal.fftshift(
    tf.abs(
        tf.cast(dt, config.tf_cdtype) *
        tf.signal.fft(y) /
        tf.cast(tf.math.sqrt(2 * np.pi), config.tf_cdtype)
    ) ** 2
)

X_angle = tf.math.angle(tf.signal.fftshift(tf.signal.fft(x)))
Y_angle = tf.math.angle(tf.signal.fftshift(tf.signal.fft(y)))


# In[22]:


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t.numpy().flatten(), np.abs(x.numpy().flatten())**2, '-')
ax1.plot(t.numpy().flatten(), np.abs(y.numpy().flatten())**2, '--')
ax1.set_xlim(-150, 150)
ax1.set_xlabel(r"$t$ in (ps)")
ax1.set_ylabel(r"$|g(t)|^2$ in (W)")
ax1.grid()

ax2.plot(
    f.numpy().flatten(),
    (X.numpy().flatten()/np.max(X.numpy().flatten())),
    '-')
ax2.plot(
    f.numpy().flatten(),
    (Y.numpy().flatten()/np.max(Y.numpy().flatten())),
    '--')
ax2.set_xlim(-0.015, 0.015)
ax2.set_xlabel(r"$f-f_c$ in (THz)")
ax2.set_ylabel(r"$\frac{|G(f-f_c)|^2}{|G_\mathrm{max}|^2}$")
ax2.grid()

ax1.legend(['transmitted', 'received'])
plt.tight_layout()

plt.show()


# In[23]:


fig, (ax1) = plt.subplots(1, 1, tight_layout=True)

ax1.plot(t.numpy().flatten(), np.angle(x.numpy().flatten()), '-')
ax1.plot(t.numpy().flatten(), np.angle(y.numpy().flatten()), '--')
ax1.set_xlim(-500, 500)
ax1.set_xlabel("$t$ in (ps)")
ax1.set_ylabel("$\u2220 x(t), \u2220 y(t)$")
ax1.grid()

ax1.legend(['transmitted', 'received'])
plt.tight_layout()

plt.show()


# The most interesting observation that one can make here is that the
# spectrum of the received signal is compressed. This is in contrast
# to the expected Kerr nonlinearity-induced spectral broadening, and shows
# that joint application of the fiber effects may result in completely
# different observations compared to the isolated investigation.
# 
# What we can see here, however, is that the Gaussian input impulse is
# transformed to a higher-order Soliton during propagation. Those require
# a joint CD and Kerr nonlinearity to exist.
# 

# ## References
# 
# [1] René-Jean Essiambre, Gerhard Kramer, Peter J. Winzer, Gerard J. Foschini, and Bernhard Goebel.
# „Capacity Limits of Optical Fiber Networks“. Journal of Lightwave Technology 28, Nr. 4, pp 662–701, February 2010.
# 
# 
# 
-e 
# --- End of Optical_Lumped_Amplification_Channel.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Pulse-shaping Basics
# In this tutorial notebook, you will learn about various components of Sionna PHY's signal module, such as pulse-shaping filters, windowing functions, as well as layers for up- and down-sampling.
# 
# Below is a schematic diagram of the used components and how they connect. For simplicity, we have not added any channel between the pulse-shaping filter and the matched filter.


# You will learn how to:
# 
# * Use filters for pulse-shaping and matched filtering
# * Visualize impulse and magnitude responses
# * Compute the empirical power spectral density (PSD) and adjacent channel leakage power ratio (ACLR)
# * Apply the Upsampling and Downsampling layers
# * Add windowing to filters for improved spectral characteristics
# 
# 

# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [Pulse-shaping of a sequence of QAM symbols](#Pulse-shaping-of-a-sequence-of-QAM-symbols)
# * [Recovering the QAM symbols through matched filtering and downsampling](#Recovering-the-QAM-symbols-through-matched-filtering-and-downsampling)
# * [Investigating the ACLR](#Investigating-the-ACLR)
# * [Windowing](#Windowing)

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

# Set random seed for reproducibility
sionna.phy.config.seed = 42


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

from sionna.phy.mapping import QAMSource
from sionna.phy.signal import Upsampling, Downsampling, \
                              RootRaisedCosineFilter, empirical_psd, \
                              empirical_aclr


# ## Pulse-shaping of a sequence of QAM symbols

# We start by creating a root-raised-cosine filter with a roll-off factor of 0.22, spanning 32 symbols, with an oversampling factor of four.

# In[3]:


beta = 0.22 # Roll-off factor
span_in_symbols = 32 # Filter span in symbold
samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
rrcf = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)


# All filters have a function to visualize their impulse response $h(t)$ and magnitude response $H(f)$, i.e., the absolute value of the Fourier transform of $h(t)$. The symbol duration is denoted $T$ and the bandwidth $W$. The *normalized time* and *normalized frequency* are then defined as $t/T$ and $f/W$, respectively.

# In[4]:


rrcf.show("impulse")
rrcf.show("magnitude", "db") # Logarithmic scale
rrcf.show("magnitude", "lin") # Linear scale


# In Sionna, filters have always an odd number of samples. This is despite the fact that the product *span_in_symbols* $\times$ *samples_per_symbol* can be even. Let us verify the length property of our root-raised-cosine filter:

# In[5]:


print("Filter length:", rrcf.length)


# Next, we will use this filter to pulse shape a sequence of QAM symbols. This requires upsampling of the sequence to the desired sampling rate. The sampling rate is defined as the number of samples per symbol $k$, and upsampling simply means that $k-1$ zeros are inserted after every QAM symbol.

# In[6]:


# Configure QAM source
num_bits_per_symbol = 4 # The modulation order of the QAM constellation, i.e., 16QAM
qam = QAMSource(num_bits_per_symbol) # Layer to generate batches of QAM symbols

# Generate batch of QAM symbol sequences
batch_size = 128
num_symbols = 1000
x = qam([batch_size, num_symbols])
print("Shape of x", x.shape)

# Create instance of the Upsampling layer
us = Upsampling(samples_per_symbol)

# Upsample the QAM symbol sequence
x_us = us(x)
print("Shape of x_us", x_us.shape)

# Inspect the first few elements of one row of x_us
plt.stem(np.abs(x_us)[0,:20]);
plt.xlabel(r"Sample index $i$")
plt.ylabel(r"|$x_{us}[i]$|");


# After upsampling, we can apply the filter:

# In[7]:


# Filter the upsampled sequence
x_rrcf = rrcf(x_us)


# ## Recovering the QAM symbols through matched filtering and downsampling

# 
# 
# 

# In[8]:


# Apply the matched filter
x_mf = rrcf(x_rrcf)

# Instantiate a downsampling layer
ds = Downsampling(samples_per_symbol, rrcf.length-1, num_symbols)

# Recover the transmitted symbol sequence
x_hat = ds(x_mf)

# Visualize the different signals
plt.figure(figsize=(12, 8))
plt.plot(np.real(x_us[0]), "x")
plt.plot(np.real(x_rrcf[0, rrcf.length//2:]))
plt.plot(np.real(x_mf[0, rrcf.length-1:]));
plt.xlim(0,100)
plt.legend([r"Oversampled sequence of QAM symbols $x_{us}$",
            r"Transmitted sequence after pulse shaping $x_{rrcf}$",
            r"Received sequence after matched filtering $x_{mf}$"]);


# 
# Give it a try and change *span_in_symbols* above to a larger number, e.g., 100. This will reduce the MSE by around 26dB.

# In[9]:


plt.figure()
plt.scatter(np.real(x_hat), np.imag(x_hat));
plt.scatter(np.real(x), np.imag(x));
plt.legend(["Transmitted", "Received"]);
plt.title("Scatter plot of the transmitted and received QAM symbols")
print("MSE between x and x_hat (dB)", 10*np.log10(np.var(x-x_hat)))


# ## Investigating the ACLR
# 
# An important metric of waveforms is the so-called adjacent channel leakage power ratio, or short ACLR. It is defined as the ratio of the out-of-band power and the in-band power. One can get a first idea of the ACLR by looking at the power spectral density (PSD) of a transmitted signal.

# In[10]:


empirical_psd(x_rrcf, oversampling=samples_per_symbol, ylim=[-100, 3]);


# The in-band is defined by the interval from [-0.5, 0.5] in normalized frequency. Due to the non-zero roll-of factor, a significant amount of energy is located out of this band. The resulting ACLR can be computed with the following convenience function:

# In[11]:


aclr_db = 10*np.log10(empirical_aclr(x_rrcf, oversampling=samples_per_symbol))
print("Empirical ACLR (db):", aclr_db)


# We can now verify that this empirical ACLR is well aligned with the theoretical ACLR that can be computed based on the magnitude response of the pulse-shaping filter. Every filter provides this value as the property *Filter.aclr*.

# In[12]:


print("Filter ACLR (dB)", 10*np.log10(rrcf.aclr))


# We can improve the ACLR by decreasing the roll-off factor $\beta$ from 0.22 to 0.1:

# In[13]:


print("Filter ACLR (dB)", 10*np.log10(RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, 0.1).aclr))


# ## Windowing
# 
# Windowing can be used to improve the spectral properties of a truncated filter. For a filter of length $L$, a window is a real-valued vector of the same length that is multiplied element-wise with the filter coefficients. This is equivalent to a convolution of the filter and the window in the frequency domain.
# 
# Let us now create a slightly shorter root-raised-cosine filter and compare its properties with and without windowing.
# One can see that windowing leads to a much reduced out-of-band attenuation. However, the passband of the filter is also broadened which leads to an even slightly increased ACLR.

# In[14]:


span_in_symbols = 8 # Filter span in symbols
samples_per_symbol = 8 # Number of samples per symbol, i.e., the oversampling factor

rrcf_short = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
rrcf_short_blackman = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window="blackman")

rrcf_short_blackman.window.show(samples_per_symbol)
rrcf_short_blackman.window.show(samples_per_symbol, domain="frequency", scale="db")

rrcf_short.show()
plt.title("Impulse response without windowing")

rrcf_short_blackman.show()
plt.title("Impulse response with windowing")

rrcf_short.show("magnitude", "db")
plt.title("Magnitude response without windowing")

rrcf_short_blackman.show("magnitude", "db")
plt.title("Magnitude response with windowing")

print("ACLR (db) without window", 10*np.log10(rrcf_short.aclr))
print("ACLR (db) with window", 10*np.log10(rrcf_short_blackman.aclr))

-e 
# --- End of Pulse_Shaping_Basics.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Realistic Multiuser MIMO OFDM Simulations
# In this notebook, you will learn how to setup realistic simulations
# of multiuser MIMO uplink transmissions. Multiple user terminals (UTs)
# are randomly distributed in a cell sector and communicate with a multi-antenna
# base station.
# 
# 

# The block-diagramm of the system model looks as follows:
# 
# 

# It includes the following components:
# 
# - 5G LDPC FEC
# - QAM modulation
# - OFDM resource grid with configurable pilot pattern
# - Multiple single-antenna transmitters and a multi-antenna receiver
# - 3GPP 38.901 UMi, UMa, and RMa channel models and antenna patterns
# - LS Channel estimation with nearest-neighbor interpolation as well as perfect CSI
# - LMMSE MIMO equalization
# 
# You will learn how to setup the topologies required to simulate such scenarios and investigate
# 
# - the performance over different models, and
# - the impact of imperfect CSI.
# 
# We will first walk through the configuration of all components of the system model, before simulating
# some simple uplink transmissions in the frequency domain. We will then simulate CDFs of the channel condition number
# and look into frequency-selectivity of the different channel models to understand the reasons for the observed performance differences.
# 
# It is recommended that you familiarize yourself with the [API documentation](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html) of the `Channel` module
# and, in particular, the 3GPP 38,901 models that require a substantial amount of configuration.  The last set of simulations in this notebook take some time, especially when you have no GPU available. For this reason, we provide the simulation
# results directly in the cells generating the figures. Simply uncomment the corresponding lines to show this results.

# ## Table of Contents
# * [GPU Configuration and Imports](#GPU-Configuration-and-Imports)
# * [System Setup](#System-Setup)
# * [Uplink Transmissions in the Frequency Domain](#Uplink-Transmissions-in-the-Frequency-Domain)
#     * [Compare Estimated and Actual Frequency Responses](#Compare-Estimated-and-Actual-Frequency-Responses)
#     * [Understand the Difference Between the Channel Models](#Understand-the-Difference-Between-the-Channel-Models)
#     * [Setup a Sionna Block for BER simulations](#Setup-a-Sionna-Block-for-BER-simulations)

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

# Set random seed for reproducibility
sionna.phy.config.seed = 42


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import time

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, \
                            LSChannelEstimator, LMMSEEqualizer, \
                            RemoveNulledSubcarriers
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi, UMa, RMa
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, \
                               ApplyOFDMChannel, OFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource, QAMSource
from sionna.phy.utils import ebnodb2no, sim_ber, compute_ber


# ## System Setup
# 
# We will now configure all components of the system model step-by-step.

# In[3]:


scenario = "umi"
carrier_frequency = 3.5e9
direction = "uplink"
num_ut = 4 
batch_size = 32


# In[4]:


# Define the UT antenna array
ut_array = Antenna(polarization="single",
                   polarization_type="V",
                   antenna_pattern="omni",
                   carrier_frequency=carrier_frequency)

# Define the BS antenna array
bs_array = AntennaArray(num_rows=1,
                        num_cols=4,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

# Create channel model
channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model="low",
                    ut_array=ut_array,
                    bs_array=bs_array,
                    direction=direction,
                    enable_pathloss=False,
                    enable_shadow_fading=False)

# Generate the topology
topology = gen_topology(batch_size, num_ut, scenario)

# Set the topology
channel_model.set_topology(*topology)

# Visualize the topology
channel_model.show_topology()


# In[5]:


# The number of transmitted streams is equal to the number of UT antennas
num_streams_per_tx = 1

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.zeros([1, num_ut])
rx_tx_association[0, :] = 1

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly simple. However, it can get complicated
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)


# In[6]:


rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=128,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=20,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
rg.show();


# In[7]:


num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # The code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits

# The binary source will create batches of information bits
binary_source = BinarySource()
qam_source = QAMSource(num_bits_per_symbol)

# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)

# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)

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

