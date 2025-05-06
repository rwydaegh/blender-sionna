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

