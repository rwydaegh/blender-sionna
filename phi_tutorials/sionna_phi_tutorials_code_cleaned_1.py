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
