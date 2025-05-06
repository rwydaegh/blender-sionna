#!/usr/bin/env python
# coding: utf-8

# # Introduction to Sionna RT

# Ray tracing is a technique to simulate environment-specific and physically accurate channel realizations for a given scene and user position.
# Please see the [EM Primer](https://nvlabs.github.io/sionna/rt/em_primer.html) for further details on the theoretical background of ray tracing of wireless channels.
# 
# Sionna RT is a open-source hardware-accelerated differentiable ray tracer for radio propagation modeling which is built on top of [Mitsuba 3](https://www.mitsuba-renderer.org/). Mitsuba 3 is a rendering system for forward and inverse light-transport simulation that makes use of the differentiable just-in-time compiler [Dr.Jit](https://drjit.readthedocs.io/en/latest/).
# 
# Thanks to Dr.Jit's automatic gradient computation, gradients of functions of channel responses or radio maps with respect to most parameters of the ray tracing process, including material properties, antenna and scattering patterns, orientations, and positions of objects, can be efficiently computed and used in various gradient-based optimization problems.
# 
# Sionna RT relies on Mitsuba 3 for the rendering and handling of scenes, e.g., its XML-file format.
# 

# ## Imports

# In[1]:


# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

# Other imports
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

no_preview = True # Toggle to False to use the preview widget

# Import relevant components from Sionna RT
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies


# ## Loading and Visualizing Scenes
# 
# Sionna RT can either load external scene files (in Mitsuba's XML file format) or it can load one of the [integrated scenes](https://nvlabs.github.io/sionna/rt/api/scene.html#examples).
# 
# In this example, we load an example scene containing the area around the Frauenkirche in Munich, Germany.

# In[2]:


# Load integrated scene
scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile


# To visualize the scene, we can use the `preview` function which opens an interactive preview of the scene.
# This only works in Jupyter notebooks.
# 
# You can use the following controls:
# 
# - Mouse left: Rotate
# - Scroll wheel: Zoom
# - Mouse right: Move
# 
# Please note that only one preview instance per scene can be opened at the same time.
# However, multiple scenes can be loaded in parallel.

# In[3]:


if not no_preview:
    scene.preview();


# It is often convenient to choose a viewpoint in the 3D preview prior to rendering it as a high-quality image.
# The next cell uses the "preview" camera which corresponds to the viewpoint of the current preview image.

# In[4]:


# Only availabe if a preview is open
if not no_preview:
    scene.render(camera="preview", num_samples=512);


# One can also render the image to a file as shown below:

# In[5]:


# Only availabe if a preview is open
if not no_preview:
    scene.render_to_file(camera="preview",
                         filename="scene.png",
                         resolution=[650,500]);


# Instead of the preview camera, one can also specify dedicated cameras with different positions and `look_at` directions.

# In[6]:


# Create new camera with different configuration
my_cam = Camera(position=[-250,250,150], look_at=[-15,30,28])

# Render scene with new camera*
scene.render(camera=my_cam, resolution=[650, 500], num_samples=512); # Increase num_samples to increase image quality


# ## Inspecting SceneObjects and Editing of Scenes

# A scene consists of multiple [SceneObjects](https://nvlabs.github.io/sionna/rt/api/scene_object.html) which can be accessed in the following way:

# In[7]:


scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
scene.objects


# In[8]:


floor = scene.get("floor")


# SceneObjects can be transformed by the following properties and methods: 
# - position
# - orientation
# - scaling
# - look_at

# In[9]:


print("Position (x,y,z) [m]: ", floor.position)
print("Orientation (alpha, beta, gamma) [rad]: ", floor.orientation)
print("Scaling: ", floor.scaling)


# More details on these functionalities can be found in the [Tutorial on Loading and Editing of Scenes](https://nvlabs.github.io/sionna/rt/tutorials/Scene-Edit.html).

# Every SceneObject has another important property, the `velocity` vector:

# In[10]:


print("Velocity (x,y,z) [m/s]: ", floor.velocity)


# This property is used during the ray tracing process to compute a Doppler shift for every propagation path. This information can then be used to synthetically compute time evolution of channel impulse responses. More details on this topic are provided in the [Tutorial on Mobility](https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html).

# The last property of SceneObjects that we discuss here is the [RadioMaterial](https://nvlabs.github.io/sionna/rt/api/radio_materials.html):

# In[11]:


floor.radio_material


# The radio material determines how an object interacts with incident radio waves. To learn more about radio materials and how they can be modified, we invited you to have a look at the Developer Guide on [Understanding Radio Materials](https://nvlabs.github.io/sionna/rt/developer/dev_custom_radio_materials.html).
# 
# Depending on the type of radio material, some of its properties might change as a function of the frequency of the incident radio wave:

# In[12]:


scene.frequency = 28e9 # in Hz; implicitly updates RadioMaterials that implement frequency dependent properties
floor.radio_material # Note that the conductivity (sigma) changes automatically


# ## Ray tracing of Propagation Paths
# 
# One a scene is loaded, we can place Transmitters and Receivers in it and compute propagation paths between them.
# All transmitters and all receivers are equipped with the same antenna arrays which are defined by the `scene` properties `scene.tx_array` and `scene.rx_array`, respectively. Antenna arrays are composed of multiple identical antennas. Antennas can have custom or pre-defined patterns and are either single- or dual-polarized. One can add multiple transmitters and receivers to a scene which need to have unique names.
# 
# More details on antenna patterns can be found in the Developer Guide [Understanding Radio Materials](https://nvlabs.github.io/sionna/rt/developer/dev_custom_radio_materials.html).

# In[13]:


scene = load_scene(sionna.rt.scene.munich, merge_shapes=True) # Merge shapes to speed-up computations

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")

# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27],
                 display_radius=2)

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
              position=[45,90,1.5],
              display_radius=2)

# Add receiver instance to scene
scene.add(rx)

tx.look_at(rx) # Transmitter points towards receiver


# Propagation paths are computed with the help of a [PathSolver](https://nvlabs.github.io/sionna/rt/api/paths_solvers.html).
# The next cell shows how such a path solver is instantiated and used. 
# 
# The parameter `max_depth` determines the maximum number of interactions between a ray and a scene objects. 
# For example, with a `max_depth` of zero, only LoS paths are considered For a `max_depth` of one, LoS as well as first-order reflections of refractions are considered.  When the argument `synthetic_array` is set to `False`, antenna arrays are explicitly modeled by finding paths between any pair of transmitting and receiving antennas in the scene. Otherwise, arrays are represented by a single antenna located in the center of the array.
# Phase shifts related to the relative antenna positions will then be applied based on a plane-wave assumption when the channel impulse responses are computed.

# In[14]:


# Instantiate a path solver
# The same path solver can be used with multiple scenes
p_solver  = PathSolver()

# Compute propagation paths
paths = p_solver(scene=scene,
                 max_depth=5,
                 los=True,
                 specular_reflection=True,
                 diffuse_reflection=False,
                 refraction=True,
                 synthetic_array=False,
                 seed=41)


# The [Paths](https://nvlabs.github.io/sionna/rt/paths.html) object contains all paths that have been found between transmitters and receivers.
# In principle, the existence of each path is determininistic for a given position and environment. Please note that due to the stochastic nature of the *shoot-and-bounce* algorithm, different runs of the path solver can lead to different paths that are found. Most importantly, diffusely reflected paths are obtained through random sampling of directions after each interaction with a scene object. You can provide the `seed` argument to the solver to ensure reproducibility.
# 
# Let us now visualize the found paths in the scene:

# In[15]:


if no_preview:
    scene.render(camera=my_cam, paths=paths, clip_at=20);
else:
    scene.preview(paths=paths, clip_at=20);


# The Paths object contains detailed information about every found path and allows us to generated channel impulse responses and apply Doppler shifts for the simulation of time evolution. For a detailed description, we refer to the developer guide [Understanding the Paths Object](https://nvlabs.github.io/sionna/rt/developer/dev_understanding_paths.html).

# ## From Paths to Channel Impulse and Frequency Responses
# 
# Once paths are computed, they can be transformed into a baseband-equivalent channel impulse response (CIR) via [Paths.cir()](https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.cir), into a discrete complex baseband-equivalent channel impulse response via [Paths.taps()](https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.taps), or into a channel frequency response (CFR) via
# [Paths.cfr()](https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.cfr). These class methods can simulate time evolution of the channel based on the computed Doppler shifts (see [Paths.doppler](https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.doppler)).
# 
# Let us first derive and visualize the baseband-equivalent channel impulse response from the paths computed above:

# In[16]:


a, tau = paths.cir(normalize_delays=True, out_type="numpy")

# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
print("Shape of a: ", a.shape)

# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
print("Shape of tau: ", tau.shape)


# The `out_type` argument can be used to convert the CIR into tensors from different frameworks, such as [Dr.Jit](https://drjit.readthedocs.io/en/latest/reference.html) ("drjit"), [Numpy](https://numpy.org) ("numpy"),
#             [Jax](https://jax.readthedocs.io/en/latest/index.html) ("jax"),
#             [TensorFlow](https://www.tensorflow.org) ("tf"),
#             and [PyTorch](https://pytorch.org) ("torch"). Please see the developer guide [Compatibility with other Frameworks](https://nvlabs.github.io/sionna/rt/developer/dev_compat_frameworks.html) for more information on the interoperability of Sionna RT with other array frameworks, including the propagation of gradients.

# In[17]:


t = tau[0,0,0,0,:]/1e-9 # Scale to ns
a_abs = np.abs(a)[0,0,0,0,:,0]
a_max = np.max(a_abs)

# And plot the CIR
plt.figure()
plt.title("Channel impulse response")
plt.stem(t, a_abs)
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$");


# Note that the delay of the first arriving path is by default normalized to zero. This behavior can be changed by setting the argument ``normalize_delays`` to `True`.
# 
# We can obtain the channel frequency response in a similar manner:

# In[18]:


# OFDM system parameters
num_subcarriers = 1024
subcarrier_spacing=30e3

# Compute frequencies of subcarriers relative to the carrier frequency
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)

# Compute channel frequency response
h_freq = paths.cfr(frequencies=frequencies,
                   normalize=True, # Normalize energy
                   normalize_delays=True,
                   out_type="numpy")

# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
print("Shape of h_freq: ", h_freq.shape)

# Plot absolute value
plt.figure()
plt.plot(np.abs(h_freq)[0,0,0,0,0,:]);
plt.xlabel("Subcarrier index");
plt.ylabel(r"|$h_\text{freq}$|");
plt.title("Channel frequency response");


# For link-level simulations in the time-domain, we often require the discrete baseband-equivalent channel impulse response or simply the channel taps.
# These are obtained by sampling the ideally low-pass filtered channel impulse response at the desired sampling frequency. By default, it is assumed that sampling is performed at the Nyquist rate.
# 
# As the underlying sinc filter has an infinitely long response, the channel taps need to be truncated at a minimum and maximum value, i.e., `l_min` and `l_max`, respectively.

# In[19]:


taps = paths.taps(bandwidth=100e6, # Bandwidth to which the channel is low-pass filtered
                  l_min=-6,        # Smallest time lag
                  l_max=100,       # Largest time lag
                  sampling_frequency=None, # Sampling at Nyquist rate, i.e., 1/bandwidth
                  normalize=True,  # Normalize energy
                  normalize_delays=True,
                  out_type="numpy")
print("Shape of taps: ", taps.shape)

plt.figure()
plt.stem(np.arange(-6, 101), np.abs(taps)[0,0,0,0,0]);
plt.xlabel(r"Tap index $\ell$");
plt.ylabel(r"|$h[\ell]|$");
plt.title("Discrete channel taps");


# Every radio device and scene object has a velocity vector associated with it. These are used to compute path-specific Doppler shifts that enable the simulation of mobility. More details can be found in the [Tutorial on Mobility](https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html).
# 
# We will now assign a non-zero velocity vector to the transmitter, recompute the propagation paths, and compute a time-varying channel impulse reponse:

# In[20]:


scene.get("tx").velocity = [10, 0, 0]

# Recompute propagation paths
paths_mob = p_solver(scene=scene,
                     max_depth=5,
                     los=True,
                     specular_reflection=True,
                     diffuse_reflection=False,
                     refraction=True,
                     synthetic_array=True,
                     seed=41)

# Compute CIR with time-evolution
num_time_steps=100
sampling_frequency = 1e4
a_mob, _ = paths_mob.cir(sampling_frequency=sampling_frequency,
                         num_time_steps=num_time_steps,
                         out_type="numpy")

# Inspect time-evolution of a single path coefficient
plt.figure()
plt.plot(np.arange(num_time_steps)/sampling_frequency*1000,
         a_mob[0,0,0,0,0].real);
plt.xlabel("Time [ms]");
plt.ylabel(r"$\Re\{a_0(t) \}$");
plt.title("Time-evolution of a path coefficient");


# ## Radio Maps
# 
# Sionna RT can compute radio maps for all transmitters in a scene. A [RadioMap](https://nvlabs.github.io/sionna/rt/api/radio_map.html) assigns a metric, such as path gain, received signal strength (RSS), or signal-to-interference-plus-noise ratio (SINR), for a specific transmitter to every point on a plane. In other words, for a given transmitter, it associates every point on a surface with the channel gain, RSS, or SINR, that a receiver with a specific orientation would observe at this point.
# 
# Like the computation of propagation paths requires a [PathSolver](https://nvlabs.github.io/sionna/rt/api/paths_solvers.html), the computation of radio maps requires a [RadioMapSolver](https://nvlabs.github.io/sionna/rt/api/radio_map_solvers.html). The following code snippet how a radio can be computed and displayed.
# 
# More information about radio maps can be found in the detailed [Tutorial on Radio Maps](https://nvlabs.github.io/sionna/rt/tutorials/Radio-Maps.html).

# In[21]:


rm_solver = RadioMapSolver()

rm = rm_solver(scene=scene, 
               max_depth=5,
               cell_size=[1,1],
               samples_per_tx=10**6)


# In[22]:


if no_preview:
    scene.render(camera=my_cam, radio_map=rm);
else:
    scene.preview(radio_map=rm);


# ## Summary
# 
# In this tutorial, you have learned the basics of Sionna RT. You now know how paths can be found in complex environments
# and how the CIR, CFR, and taps can be derived from them. You have also learned how radio maps can be created.
# 
# There is one key feature of Sionna RT that was not discussed in this notebook: Automatic gradient computation.
# Like most components of Sionna, also Sionna RT is differentiable with respect to most parameters, such as radio materials, scattering and atenna patterns, transmitter and receiver orientations, array geometries, positions, etc.
# 
# Please have a look at the [API documentation](https://nvlabs.github.io/sionna/rt/api/rt.html) of the various components and the other available [Tutorials](https://nvlabs.github.io/sionna/rt/tutorials.html) and [Developer Guides](https://nvlabs.github.io/sionna/rt/developer/developer.html).
-e 
# --- End of Introduction.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Mobility

# This notebook explains different ways in which the effects of mobility can be simulated with Sionna's [ray tracing (RT) module](https://nvlabs.github.io/sionna/rt/index.html). You will
# 
# - Use the `position` and `orientation` properties to move scene objects
# - Understand the `velocity` property of scene objects and their impact on the Doppler shift
# - Learn how to use the `doppler` property of a `Paths` object to simulate time evolution of channels

# ## Background Information
# 
# 
# In order to compute the Doppler shift for a specific path as shown in the figure below, Sionna RT relies on the [velocity vectors](https://nvlabs.github.io/sionna/rt/api/scene_object.html#sionna.rt.SceneObject.velocity) of the scene objects. 


# While traveling from the transmitter to the receiver, the path undergoes $n$ scattering processes, such as reflection, refraction, diffuse scattering, or diffraction. The object on which lies the $i$th scattering point has the velocity vector $\mathbf{v}_i$ and the outgoing ray direction at this point is denoted $\hat{\mathbf{k}}_i$. The first and last point correspond to the transmitter and receiver, respectively. 
# 
# The Doppler shift $f_\Delta$ for this path can be computed as (see the [documentation](https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.doppler) of `Paths.doppler`)
# 
# \begin{align}
# f_\Delta = \frac{1}{\lambda}\left[\mathbf{v}_{0}^\mathsf{T}\hat{\mathbf{k}}_0 - \mathbf{v}_{n+1}^\mathsf{T}\hat{\mathbf{k}}_n + \sum_{i=1}^n \mathbf{v}_{i}^\mathsf{T}\left(\hat{\mathbf{k}}_i-\hat{\mathbf{k}}_{i-1} \right) \right] \qquad \text{[Hz]}
# \end{align}
# 
# where $\lambda$ is the wavelength, and then be used to compute the time evolution of the path coefficient `Paths.a`
# as
# 
# \begin{align}
# a(t) = a e^{j2\pi f_\Delta t}.
# \end{align}
# 

# ## GPU Configuration and Imports <a class="anchor" id="GPU-Configuration-and-Imports"></a>

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      RadioMapSolver, PathSolver
from sionna.rt.utils import r_hat, subcarrier_frequencies


# ## Controlling Position and Orientation of Scene Objects
# 
# Every object in a scene has a `position` and `orientation` property that can be inspected and modified.
# To see this, let us load a scene consisting of a simple street canyon and a few cars.

# In[2]:


scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars,
                   merge_shapes=False)
cam = Camera(position=[50,0,130], look_at=[10,0,0])

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


# The list of all scene objects can be accessed as follows:

# In[3]:


scene.objects


# Let us now inspect the position and orientation of one of the cars:

# In[4]:


car_2 = scene.get("car_2")
print("Position: ", car_2.position.numpy()[:,0])
print("Orientation: ", car_2.orientation.numpy()[:,0])


# The position of an object corresponds to the center of its axis-aligned bounding box.
# By default, the orientation of every scene object is `[0,0,0]`. 
# We can now change the position and orientation of the car as follows:

# In[5]:


# Move the car 10m along the y-axis
car_2.position += [0, 10, 0]

# And rotate it by 90 degree around the z-axis
car_2.orientation = [np.pi/2, 0, 0]

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


# Next, we will visualize coverage maps for different positions of the cars in the scene, assuming that one of the cars is equipped with a transmit antenna:

# In[6]:


scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars,
                   merge_shapes=False)
cam =  Camera(position=[50,0,130], look_at=[10,0,0])

# Configure a transmitter that is located at the front of "car_2"
scene.add(Transmitter("tx", position=[22.7, 5.6, 0.75], orientation=[np.pi,0,0]))
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
scene.rx_array = scene.tx_array

# Create radio m ap solver
rm_solver = RadioMapSolver()

# Move cars along straight lines for a couple of steps
displacement_vec = [10, 0, 0]
num_displacements = 2
for _ in range(num_displacements+1):

    # Compute and render a coverage map at 0.5m above the ground
    rm = rm_solver(scene=scene,
                   samples_per_tx=20**6,
                   refraction=True,
                   max_depth=10,
                   center=[0,0,0.5],
                   orientation=[0,0,0],
                   size=[186,121],
                   cell_size=[2,2])
    scene.render(camera=cam, radio_map=rm,
                 num_samples=512, rm_show_color_bar=True,
                 rm_vmax=-40, rm_vmin=-150)

    # Move TX to next position
    scene.get("tx").position -= displacement_vec

    # Move cars driving in -x direction
    for j in range(1,6):
        scene.get(f"car_{j}").position -= displacement_vec

    # Move cars driving in x direction
    for j in range(6,9):
        scene.get(f"car_{j}").position += displacement_vec


# ## Time Evolution of Channels Via Doppler Shift

# In the previous section, we have seen how the position and orientation of objects in a scene can be modified. However, if we want to model the evolution of channels over very short time horizons, this approach becomes impractical. An alternative, consists in assigning to all moving objects
# a velocity vector $\mathbf{v}_i\in\mathbb{R}^3$ based on which path-wise Doppler shifts can be computed. Let us now load a simple scene with a single reflector and modify its velocity.

# In[7]:


# Load scene with a single reflector
scene = load_scene(sionna.rt.scene.simple_reflector,
                   merge_shapes=False)
# Inspect the velocity of this object
print("Velocity vector: ", scene.get("reflector").velocity.numpy()[:,0])

# Update velocity vector
scene.get("reflector").velocity = [0, 0, -20]
print("Velocity vector after update: ", scene.get("reflector").velocity.numpy()[:,0])


# Next, we will add a transmitter and receiver to the scene and compute the propagation paths:

# In[8]:


# Configure arrays for all transmitters and receivers in the scene
scene.tx_array = PlanarArray(num_rows=1,num_cols=1,pattern="iso", polarization="V")
scene.rx_array = scene.tx_array

# Add a transmitter and a receiver
scene.add(Transmitter("tx", [-25,0.1,50]))
scene.add(Receiver("rx",    [ 25,0.1,50]))

# Compute paths
p_solver = PathSolver()
paths = p_solver(scene=scene, max_depth=1)

# Visualize the scene and propagation paths
if no_preview:
    cam = Camera(position=[0, 100, 50], look_at=[0,0,30])
    scene.render(camera=cam, paths=paths);
else:
    scene.preview(paths=paths)


# Every path has a property `Paths.doppler` that corresponds to the aggregated Doppler shift due to the movement of objects it intersects as well as the velocity of the transmitter and receiver.

# In[9]:


print("Path interaction (0=LoS, 1=Specular reflection): ", paths.interactions.numpy())
print("Doppler shifts (Hz): ", paths.doppler.numpy())


# ### Example: Delay-Doppler Spectrum
# 
# We will now use the Doppler shifts to compute a time-varying channel impulse response and estimate its Delay-Doppler spectrum.
# To this end, we assume that 1024 subsequent symbols of an OFDM system with 1024 subcarriers can be observed, assuming a subcarrier spacing of 30kHz. This will define the following resolutions in the delay and Doppler domains:

# In[10]:


num_ofdm_symbols = 1024
num_subcarriers = 1024
subcarrier_spacing = 30e3

ofdm_symbol_duration = 1/subcarrier_spacing
delay_resolution = ofdm_symbol_duration/num_subcarriers
doppler_resolution = subcarrier_spacing/num_ofdm_symbols

print("Delay   resolution (ns): ", int(delay_resolution/1e-9))
print("Doppler resolution (Hz): ", int(doppler_resolution))


# In addition to the velocity of the reflector, we also assume that the transmitter is moving. 

# In[11]:


# Set TX velocity
tx_velocity = [30, 0, 0]
scene.get("tx").velocity = tx_velocity

# Recompute the paths
paths = p_solver(scene=scene, max_depth=1, refraction=False)

# Compute channel frequency response with time evolution
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)

h = paths.cfr(frequencies=frequencies,
              sampling_frequency=1/ofdm_symbol_duration,
              num_time_steps=num_ofdm_symbols,
              normalize_delays=False, normalize=True, out_type="numpy")


### Compute the Delay-Doppler spectrum

# Squeeze useless dimensions
# [num_time_steps, fft_size]
h = np.squeeze(h)

# Apply an FFTshift to bring subcarriers in the
# correct order for an IFFT
h = np.fft.fftshift(h, axes=1)

# Apply IFFT to subcarrier dimension to
# convert frequency to delay domain
h_delay = np.fft.ifft(h, axis=1, norm="ortho")

# Apply FFT to time-step dimension to
# convert time to Doppler domain
h_delay_doppler = np.fft.fft(h_delay, axis=0, norm="ortho")

# Apply FFTShift to bring Doppler dimension in the correct
# order for visualization
h_delay_doppler = np.fft.fftshift(h_delay_doppler, axes=0)

# Compute meshgrid for visualization of the Delay-Doppler spectrum
doppler_bins = np.arange(-num_ofdm_symbols/2*doppler_resolution,
                          num_ofdm_symbols/2*doppler_resolution,
                         doppler_resolution)

delay_bins = np.arange(0,
                       num_subcarriers*delay_resolution,
                       delay_resolution) / 1e-9

x, y = np.meshgrid(delay_bins, doppler_bins)

# Visualize Delay-Doppler spectrum
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(111, projection='3d')

# We only visualize the relevant part of the spectrum
offset = 20
x_start = int(num_subcarriers/2)-offset
x_end = int(num_subcarriers/2)+offset
y_start = 0
y_end = offset
x_grid = x[x_start:x_end,y_start:y_end]
y_grid = y[x_start:x_end,y_start:y_end]
z_grid = np.abs(h_delay_doppler[x_start:x_end,y_start:y_end])

surf = ax.plot_surface(x_grid,
                       y_grid,
                       z_grid,
                       cmap='viridis', edgecolor='none')

ax.set_xlabel('Delay (ns)')
ax.set_ylabel('Doppler (Hz)')
ax.set_zlabel('Magnitude');
ax.zaxis.labelpad=2
ax.view_init(elev=53, azim=-32)
ax.set_title("Delay-Doppler Spectrum");


# As expected, we can observe two peaks in the Delay-Doppler spectrum above. The first at a delay of around 160ns, and the second at a delay of approximately 370ns. The respective Doppler shifts are around 350Hz and -260Hz.
# 
# The exact Doppler shifts based on the equation provided in the [Background Information](#Background-Information) that should match the peaks in the Delay-Doppler spectrum:

# In[12]:


print("Delay - LoS Path (ns) :", paths.tau[0,0,0]/1e-9)
print("Doppler - LoS Path (Hz) :", paths.doppler[0,0,0])

print("Delay - Reflected Path (ns) :", paths.tau[0,0,1].numpy()/1e-9)
print("Doppler - Reflected Path (Hz) :", paths.doppler[0,0,1])


# ## Comparison of Doppler- vs Position-based Time Evolution

# We will now compare a time-varying channel frequency impulse response generated by the application of Doppler shifts against another one obtained by physically moving objects in a scene and retracing the paths.
# 
# The same scene as in the first section will be used where a transmitter is placed on a moving car. However, we now also place a receiver on another car and assume that all cars in the scene are moving along a linear trajectory.

# In[13]:


scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars,
                   merge_shapes=False)
cam = Camera(position=[50,0,130], look_at=[10,0,0])

# Parameters for ray tracing
max_depth = 3

# TX and RX have directional antennas
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
scene.rx_array = scene.tx_array

# TX and RX are installed at the front of two different cars.
# The directive antennas ensure that paths reaching an antenna from the back are close to zero.
scene.add(Transmitter("tx", position=[22.7, 5.6, 0.75], orientation=[np.pi,0,0]))
scene.add(Receiver("rx", position=[-27.8,-4.9, 0.75]))

# Configure an OFDM resource grid
num_ofdm_symbols = 32
num_subcarriers = 1024
subcarrier_spacing = 30e3
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
ofdm_symbol_duration = 1/subcarrier_spacing

# Define a velocity vector and the corresponding displacement over the duration
# of one OFDM symbol
velocity_vec = np.array([10,0,0])
displacement_vec = velocity_vec*ofdm_symbol_duration

# Assign velocity vector to cars driving in -x direction
for j in range(1,6):
    scene.get(f"car_{j}").velocity = -velocity_vec

# Assign velocity vector to cars driving in x direction
for j in range(6,9):
    scene.get(f"car_{j}").velocity = velocity_vec

# Compute paths
scene.get("tx").velocity = -velocity_vec
scene.get("rx").velocity = velocity_vec

paths = p_solver(scene=scene, max_depth=max_depth, refraction=False)

# Compute the corresponding channel frequency responses with time evolution
h_dop = paths.cfr(frequencies=frequencies,
              sampling_frequency=1/ofdm_symbol_duration,
              num_time_steps=num_ofdm_symbols,
              normalize_delays=False, out_type="numpy")
h_dop = np.squeeze(h_dop)


# Visualize the scene and propagation paths
if no_preview:
    scene.render(camera=cam, paths=paths, num_samples=512);
else:
    scene.preview(paths=paths)


# In the next cell, we compute a sequence of channel frequency responses by moving all cars as well as the transmitter and receiver in the scene. After each step, propagation paths are traced and the corresponding channel frequency response is computed.

# In[14]:


paths = p_solver(scene=scene, max_depth=max_depth, refraction=False)
h_sim = np.squeeze(paths.cfr(frequencies=frequencies,
              sampling_frequency=1/ofdm_symbol_duration,
              normalize_delays=False, out_type="numpy"), axis=(0,1,2,3))

for i in range(num_ofdm_symbols-1):
    # Move TX and RX to next position
    scene.get("tx").position -= displacement_vec
    scene.get("rx").position += displacement_vec

    # Move cars driving in -x direction to the next position
    for j in range(1,6):
        scene.get(f"car_{j}").position -= displacement_vec

    # Move cars driving in +x direction to the next position
    for j in range(6,9):
        scene.get(f"car_{j}").position += displacement_vec

    # Compute channel frequency response
    paths = p_solver(scene=scene, max_depth=max_depth, refraction=False)
    h = np.squeeze(paths.cfr(frequencies=frequencies,
              sampling_frequency=1/ofdm_symbol_duration,
              normalize_delays=False, out_type="numpy"), axis=(0,1,2,3))

    # Concatenate along the time dimensions
    h_sim = np.concatenate([h_sim, h], axis=0)


# Next, we visualize the the time evolution of a few subcarriers as well as some snapshots of the full channel frequency response.

# In[15]:


subcarriers = np.arange(0, 1024, 256)
timesteps =  np.arange(0, num_ofdm_symbols, 8)

fig, axs = plt.subplots(4, 4, figsize=(17, 13))
for i,j in enumerate(subcarriers):
    axs[0,i].plot(np.arange(num_ofdm_symbols), np.real(h_sim[:,j]))
    axs[0,i].plot(np.arange(num_ofdm_symbols), np.real(h_dop[:,j]), "--")
    axs[0,i].set_xlabel("Timestep")
    axs[0,i].set_ylabel(r"$\Re\{h(f,t)\}$")
    axs[0,i].set_title(f"Subcarrier {j}")
    axs[0,i].legend(["Movement", "Doppler"])

for i,j in enumerate(subcarriers):
    axs[1,i].plot(np.arange(num_ofdm_symbols), np.imag(h_sim[:,j]))
    axs[1,i].plot(np.arange(num_ofdm_symbols), np.imag(h_dop[:,j]), "--")
    axs[1,i].set_xlabel("Timestep")
    axs[1,i].set_ylabel(r"$\Im\{h(f,t)\}$")
    axs[1,i].set_title(f"Subcarrier {j}")
    axs[1,i].legend(["Movement", "Doppler"])


for i,j in enumerate(timesteps):
    axs[2,i].plot(np.arange(num_subcarriers), np.real(h_sim[j,:]))
    axs[2,i].plot(np.arange(num_subcarriers), np.real(h_dop[j,:]), "--")
    axs[2,i].set_xlabel("Subcarrier")
    axs[2,i].set_ylabel(r"$\Re\{h(f,t)\}$")
    axs[2,i].set_title(f"Timestep {j}")
    axs[2,i].legend(["Movement", "Doppler"])

for i,j in enumerate(timesteps):
    axs[3,i].plot(np.arange(num_subcarriers), np.imag(h_sim[j,:]))
    axs[3,i].plot(np.arange(num_subcarriers), np.imag(h_dop[j,:]), "--")
    axs[3,i].set_xlabel("Subcarrier")
    axs[3,i].set_ylabel(r"$\Im\{h(f,t)\}$")
    axs[3,i].set_title(f"Timestep {j}")
    axs[3,i].legend(["Movement", "Doppler"])

plt.tight_layout()
plt.show()


# From the figures above, we can see that there is for most subcarriers until approximately time step 10 no noticeable difference between the Doppler-based channel evolution and the one based on physically moving objects. For larger time steps, some paths (diss-)appear and the Doppler-based time-evolution becomes less accurate. 

# ## Summary
# We have discussed two different ways to simulate mobility in Sionna RT. One can either move objects in a scene and recompute paths or compute the time evolution of channels synthetically based on the Doppler shifts that are obtained from velocity vectors of the scene objects.
# 
# The former approach is computationally intensive but accurate while the latter is much faster but only accurate over short time spans during which the scene objects have moved very short distances. The accuracy also depends strongly on the scenario.
# 
# Both approaches can be combined to simulate mobility over longer periods of time.
# 
# We hope you enjoyed our dive into the simulation of mobility with Sionna RT. You may also want to explore our other [tutorials](https://nvlabs.github.io/sionna/rt/tutorials.html).
-e 
# --- End of Mobility.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Radio Maps
# 
# In this notebook, you will learn how to
# 
# - Compute and configure radio maps
# - Visualize different radio map metrics, such as path gain, received signal
#   strength (RSS), and signal-to-interference-plus-noise ratio (SINR)
# - Interpret radio map-based user-to-transmitter association
# - Understand the effects of precoding vectors on radio maps
# - Sample user positions from a radio map according to various criteria
# - Generate channel impulse responses for sampled user positions

# ## Imports

# In[1]:


import numpy as np
import drjit as dr
import mitsuba as mi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import LambertianPattern, DirectivePattern, BackscatteringPattern,\
                      load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver, cpx_abs, cpx_convert

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization

    
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, watt_to_dbm,\
                      RadioMapSolver, PathSolver


# ## Understanding radio maps
# 
# A [radio map](https://nvlabs.github.io/sionna/rt/api/radio_map.html)
# assigns a metric, such as path gain, received signal strength (RSS), or
# signal-to-interference-plus-noise ratio (SINR), for a specific transmitter to
# every point on a plane. In other words, for a given transmitter, it associates
# every point on a surface with the channel gain, RSS, or SINR, that a receiver
# with a specific orientation would observe at this point. 
# 
# A radio map depends on the transmit and receive arrays and their respective
# antenna patterns, the transmitter and receiver orientations, as well as the
# transmit precoding and receive combining vectors. Moreover, a radio map is
# not continuous but discrete, as the plane must be quantized into small
# rectangular bins, which we refer to as *cells*.
# 
# As a first example, we load an empty scene, place a single transmitter in it,
# and compute a coverage map.

# In[2]:


scene = load_scene() # Load empty scene

# Configure antenna arrays for all transmitters and receivers
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array

# Define and add a first transmitter to the scene
tx0 = Transmitter(name='tx0',
                  position=[150, -100, 20],
                  orientation=[np.pi*5/6, 0, 0],
                  power_dbm=44)
scene.add(tx0)

# Compute radio map
rm_solver = RadioMapSolver()
rm = rm_solver(scene,
               max_depth=5,           # Maximum number of ray scene interactions
               samples_per_tx=10**7 , # If you increase: less noise, but more memory required
               cell_size=(5, 5),      # Resolution of the radio map
               center=[0, 0, 0],      # Center of the radio map
               size=[400, 400],       # Total size of the radio map
               orientation=[0, 0, 0]) # Orientation of the radio map, e.g., could be also vertical


# ### Metrics

# There are several ways to visualize a radio map. The simplest option is to call the class method
# [show()](https://nvlabs.github.io/sionna/rt/api/radio_map.html#sionna.rt.RadioMap.show)
# for the desired metric.

# In[3]:


# Visualize path gain
rm.show(metric="path_gain");

# Visualize received signal strength (RSS)
rm.show(metric="rss");

# Visulaize SINR
rm.show(metric="sinr");


# The RSS depends on the transmit power which can be modified for each
# transmitter as shown below.

# In[4]:


tx0.power_dbm = 24
rm = rm_solver(scene,
               max_depth=5,           
               samples_per_tx=10**7,
               cell_size=(5, 5),     
               center=[0, 0, 0],      
               size=[400, 400],       
               orientation=[0, 0, 0]) 
rm.show(metric="rss");


# Compared to the previous cell, the RSS is now 20dB smaller.
# 
# The SINR depends not only on the RSS from other transmitters in the scene but
# also on the thermal noise power. The noise power is configured indirectly via the scene
# properties [bandwidth](https://nvlabs.github.io/sionna/rt/api/scene.html#sionna.rt.Scene.bandwidth) and [temperature](https://nvlabs.github.io/sionna/rt/api/scene.html#sionna.rt.Scene.temperature). 
# 
# Note that neither parameter affects the ray tracing process; they are only used for the computation of the noise power.

# In[5]:


print(f"Bandwidth: ", scene.bandwidth.numpy(), "[Hz]")
print(f"Temperature: ", scene.temperature.numpy(), "[K]")
print(f"Thermal noise power: ", watt_to_dbm(scene.thermal_noise_power).numpy(), "[dBm]")


# All metrics of a radio map can be directly accessed as tensors as shown in
# the next cell. This can be useful to define new metrics or visualize metrics in
# a different form, such as CDF plots, etc.

# In[6]:


# Metrics have the shape
# [num_tx, num_cells_y, num_cells_x]

print(f'{rm.path_gain.shape=}') # Path gain
print(f'{rm.rss.shape=}') # RSS
print(f'{rm.sinr.shape=}') # SINR

# The location of all cell centers in the global coordinate system of the scene 
# can be accessed via:
# [num_cells_y, num_cells_x, 3]
print(f'{rm.cell_centers.shape=}')


# ### Multiple transmitters
# 
# To make things more interesting, let us add two more transmitters
# to the scene and recompute the radio map.

# In[7]:


# Remove transmitters here so that the cell can be executed multiple times
scene.remove("tx1")
scene.remove("tx2")

tx1 = Transmitter(name='tx1',
                  position=[-150, -100, 20],
                  orientation=[np.pi/6, 0, 0],
                  power_dbm=21)
scene.add(tx1)

tx2 = Transmitter(name='tx2',
                  position=np.array([0, 150 * np.tan(np.pi/3) - 100, 20]),
                  orientation=[-np.pi/2, 0, 0],
                  power_dbm=27)
scene.add(tx2)

rm = rm_solver(scene,
               max_depth=5,           
               samples_per_tx=10**7,
               cell_size=(5, 5),     
               center=[0, 0, 0],      
               size=[400, 400],       
               orientation=[0, 0, 0]) 


# As soon as there are multiple transmitters in a scene, we can either visualize
# a metric for specific transmitter or visualize the maximum matric across all
# transmitters. The latter option is relevant if we want to inspect, e.g., the SINR across a large
# scene, assuming that a receiver always connects to the transmitter providing
# the best SINR.

# In[8]:


# Show SINR for tx0
rm.show(metric="sinr", tx=0, vmin=-25, vmax=20);

# Show maximum SINR across all transmitters
rm.show(metric="sinr", tx=None, vmin=-25, vmax=20);

# Experiment: Change the metric to "path_gain" or "rss"
#             and play around with the parameters vmin/vmax
#             that determine the range of the colormap


# We can also visualize the cumulative distribution function (CDF) of the metric of interest:

# In[9]:


# CDF of the SINR for transmitter 0
rm.cdf(metric="sinr", tx=0);

# CDF of the SINR if always the transmitter providing the best SINR is selected
rm.cdf(metric="sinr");


# Note that, at every position, the highest SINR across *all* transmitters is
# always more favorable than the SINR offered by a *specific* transmitter (in math terms, the
# former *stochastically dominates* the latter). This is clearly reflected in the
# shape of the two distributions.

# ### User association
# 
# It is also interesting to investigate which regions of a radio map are "covered" by each transmitter, i.e., where a transmitter provides the strongest metric. 
# You can obtain this information either as a tensor from the class method [tx_association()](https://nvlabs.github.io/sionna/rt/api/radio_map.html#sionna.rt.RadioMap.tx_association) or visualize it using [show_association()](https://nvlabs.github.io/sionna/rt/api/radio_map.html#sionna.rt.RadioMap.show_association).

# In[10]:


# Get for every cell the tx index providing the strongest value
# of the chosen metric
# [num_cells_y, num_cells_x]
print(f'{rm.tx_association("sinr").shape=}')

rm.show_association("sinr");


# ### Sampling of random user positions
# 
# In some cases, one may want to drop receivers at random positions in a scene
# while ensuring that the chosen positions have sufficient signal quality
# (e.g., SINR)
# and/or or are located within a certain range of a transmitter. The class
# method [sample_positions()](https://nvlabs.github.io/sionna/rt/api/radio_map.html#sionna.rt.RadioMap.sample_positions) is designed for this purpose, and you will see in the next
# cell how it can be used.
# 
# You are encouraged to understand why the two different criteria used for sampling lead to the observed results. 

# In[11]:


pos, cell_ids = rm.sample_positions(
          num_pos=100,         # Number of random positions per receiver
          metric="sinr",       # Metric on which constraints and TX association will be applied
          min_val_db=3,        # Mininum value for the chosen metric
          max_val_db=20,       # Maximum value for the chosen metric
          min_dist=10,         # Minimum distance from transmitter
          max_dist=200,        # Maximum distance from transmitter
          tx_association=True, # If True, only positions associated with a transmitter are chosen,
                               # i.e., positions where the chosen metric is the highest among all TXs
          center_pos=False)    # If True, random positions correspond to cell centers,
                               # otherwise a random offset within each cell is applied

fig = rm.show(metric="sinr");
plt.title("Random positions based on SINR, distance, and association")
# Visualize sampled positions
for tx, ids in enumerate(cell_ids.numpy()):
    fig.axes[0].plot(ids[:,1], ids[:,0],
                     marker='x',
                     linestyle='',
                     color=mpl.colormaps['Dark2'].colors[tx])


pos, cell_ids = rm.sample_positions(
          num_pos=100,          # Number of random positions per receiver
          metric="path_gain",   # Metric on which constraints will be applied
          min_val_db=-85,        # Mininum value for the chosen metric
          min_dist=50,          # Minimum distance from transmitter
          max_dist=200,         # Maximum distance from transmitter
          tx_association=False, # If False, then a user located in a sampled position 
                                # for a specific TX may perceive a higher metric from another TX!
          center_pos=False)     # If True, random positions correspond to cell centers,
                                # otherwise a random offset within each cell is applied

fig = rm.show(metric="path_gain");
plt.title("Random positions based on path gain and distance")
# Visualize sampled positions
for tx, ids in enumerate(cell_ids.numpy()):
    fig.axes[0].plot(ids[:,1], ids[:,0],
                     marker='x',
                     linestyle='',
                     color=mpl.colormaps['Dark2'].colors[tx])


# ### Directional antennas and precoding vectors
# 
# As mentioned above, radio maps heavily depend on the chosen antenna patterns and precoding vectors.
# In the next cell, we will study how their impact on a radio map via several visualizations. 
# 
# Let us start by assigning a single antenna to all transmitters and computing the
# corresponding radio map:

# In[12]:


scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",   # Change to "iso" and compare the results
                             polarization="V")

rm = rm_solver(scene,
               max_depth=5,           
               samples_per_tx=10**7,
               cell_size=(5, 5),     
               center=[0, 0, 0],      
               size=[400, 400],       
               orientation=[0, 0, 0]) 

rm.show(metric="rss", tx=2);

rm.show(metric="sinr");


# We now add more antennas to the antenna array of the transmitters and apply a
# precoding vector chosen from a Discrete Fourier Transform (DFT) beam grid.

# In[13]:


# Number of elements of the rectangular antenna array
num_rows = 2
num_cols = 4

# Configure all transmitters to have equal power
tx0.power_dbm = 23
tx1.power_dbm = 23
tx2.power_dbm = 23

# Configure tr38901 uniform rectangular antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=num_rows,
                             num_cols=num_cols,
                             pattern="tr38901",
                             polarization="V")

# Create a common precoding vector used by all transmitters
# It is also possible to assign individual
precoding_vec = [1, -1]*4 / np.sqrt(8)

# Convert to tuple of Mitsuba vectors
precoding_vec = (mi.TensorXf(precoding_vec.real),
                 mi.TensorXf(precoding_vec.imag))

# Compute the radio map
rm = rm_solver(scene,
               max_depth=5,           
               samples_per_tx=10**7,
               precoding_vec=precoding_vec,
               cell_size=(5, 5),     
               center=[0, 0, 0],      
               size=[400, 400],       
               orientation=[0, 0, 0]) 

rm.show(metric="sinr");
rm.show_association(metric="sinr");


# The use of antenna arrays and precoding vectors leads to complicated, even
# artistic looking, radio maps with sometimes counter-intuitive regions of user
# association. Nevertheless, we can still sample user positions for each transmitter.

# In[14]:


pos, cell_ids = rm.sample_positions(
          num_pos=500,         
          metric="sinr",       
          min_val_db=3,        
          min_dist=10,         
          tx_association=True)

fig = rm.show(metric="sinr");

# Visualize sampled positions
for tx, ids in enumerate(cell_ids.numpy()):
    fig.axes[0].plot(ids[:,1], ids[:,0],
                     marker='x',
                     linestyle='',
                     color=mpl.colormaps['Dark2'].colors[tx])
plt.title("Random positions based on SINR, distance, and association");


# ## Radio map for a realistic scene
# 
# Until now, we have only looked at radio maps in an empty scene. Let's spice things up a little bit
# and load a more interesting scene, place transmitters, and inspect the results.

# In[15]:


def config_scene(num_rows, num_cols):
    """Load and configure a scene"""
    scene = load_scene(sionna.rt.scene.etoile)
    scene.bandwidth=100e6
    
    # Configure antenna arrays for all transmitters and receivers
    scene.tx_array = PlanarArray(num_rows=num_rows,
                                 num_cols=num_cols,
                                 pattern="tr38901",
                                 polarization="V")
    
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 pattern="iso",
                                 polarization="V")

    # Place transmitters
    positions = np.array(
                 [[-150.3, 21.63, 42.5],
                  [-125.1, 9.58, 42.5],
                  [-104.5, 54.94, 42.5],
                  [-128.6, 66.73, 42.5],
                  [172.1, 103.7, 24],
                  [232.8, -95.5, 17],
                  [80.1, 193.8, 21]
                 ])
    look_ats = np.array(
                [[-216, -21,0],
                 [-90, -80, 0],
                 [-16.5, 75.8, 0],
                 [-164, 153.7, 0],
                 [247, 92, 0],
                 [211, -180, 0],
                 [126.3, 194.7, 0]
                ])
    power_dbms = [23, 23, 23, 23, 23, 23, 23]

    for i, position in enumerate(positions):
        scene.add(Transmitter(name=f'tx{i}',
                              position=position,
                              look_at=look_ats[i],
                              power_dbm=power_dbms[i]))

    return scene  


# In[16]:


# Load and configure scene
num_rows=8
num_cols=2
scene_etoile = config_scene(num_rows, num_cols)

# Compute the SINR map
rm_etoile = rm_solver(scene_etoile,
                      max_depth=5,           
                      samples_per_tx=10**7,
                      cell_size=(1, 1))


# To get a global view of the coverage, let us visualize the radio map in the preview (or  rendered image). These are alternatives to the [show()](https://nvlabs.github.io/sionna/rt/api/radio_map.html#sionna.rt.RadioMap.show) method we have used until now, which also visualizes the objects in a scene.

# In[17]:


if no_preview:
    # Render an image
    cam = Camera(position=[0,0,1000],
                     orientation=np.array([0,np.pi/2,-np.pi/2]))
    scene_etoile.render(camera=cam,
                        radio_map=rm_etoile,
                        rm_metric="sinr",
                        rm_vmin=-10,
                        rm_vmax=60);
else:
    # Show preview
    scene_etoile.preview(radio_map=rm_etoile,
                         rm_metric="sinr",
                         rm_vmin=-10,
                         rm_vmax=60)


# ### Channel impulse responses for random user locations

# With a radio map at hand, we can now sample random positions at which we place actual receivers and then compute channel impulse responses.

# In[18]:


rm_etoile.show_association("sinr");

pos, cell_ids = rm_etoile.sample_positions(
          num_pos=4,         
          metric="sinr",       
          min_val_db=3,        
          min_dist=10,
          max_dist=200,
          tx_association=True)

fig = rm_etoile.show(metric="sinr", vmin=-10);

# Visualize sampled positions
for tx, ids in enumerate(cell_ids.numpy()):
    fig.axes[0].plot(ids[:,1], ids[:,0],
                     marker='x',
                     linestyle='',
                     color=mpl.colormaps['Dark2'].colors[tx])


# In[19]:


[scene_etoile.remove(rx.name) for rx in scene_etoile.receivers.values()]
for i in range(rm_etoile.num_tx):
    for j in range(pos.shape[1]):
        scene_etoile.add(Receiver(name=f"rx-{i}-{j}",
                           position=pos[i,j]))

p_solver = PathSolver()
paths = p_solver(scene_etoile, max_depth=5)

# Channel impulse response
a, tau = paths.cir()


# In[20]:


if no_preview:
    scene_etoile.render(camera=cam,
                        paths=paths,
                        clip_at=5);
else:
    scene_etoile.preview(paths=paths,
                     radio_map=rm_etoile,
                     rm_metric="sinr",
                     rm_vmin=-10,
                     clip_at=5)


# ## Conclusions

# Radio maps are a highly versatile feature of Sionna RT. They are particularly
# useful for defining meaningful areas for random user drops that meet certain
# constraints on RSS or SINR, or for investigating the placement and
# configuration of transmitters in a scene.
# 
# However, we have barely scratched the surface of their potential. For example,
# observe that the metrics of a radio map are differentiable with respect to most
# scene parameters, such as transmitter orientations, transmit power, antenna
# patterns, precoding vectors, and so on. This
# opens up a wide range of possibilities for gradient-based optimization.
# 
# We hope you found this tutorial on radio maps in Sionna RT informative. We
# encourage you to get your hands on it, conduct your own experiments and deepen your understanding
# of ray tracing. There's always more to learn, so be sure to explore our other
# [tutorials](https://nvlabs.github.io/sionna/rt/tutorials.html) as well!
-e 
# --- End of Radio-Maps.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Scattering

# In this notebook, you will
# 
# - Learn what scattering is and why it is important
# - Make various ray tracing experiments to validate some theoretical results
# - Familiarize yourself with the Sionna RT API
# - Visualize the impact of scattering on channel impulse responses and radio maps

# ## Imports

# In[1]:


from math import sqrt, log10
import numpy as np
import drjit as dr
import matplotlib.pyplot as plt

# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import LambertianPattern, DirectivePattern, BackscatteringPattern,\
                      load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver, cpx_abs, cpx_convert

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization


# ## Scattering Basics

# 

# When an electromagnetic wave impinges on a surface, one part of the energy gets reflected while the other part gets refracted, i.e., it propagates into the surface. We distinguish between two types of reflection, specular and diffuse. The latter type is also called diffuse scattering, or just scattering. When a rays hits a diffuse reflection surface, it is not reflected into a single (specular) direction but rather scattered toward many different directions.
# 
# 
# The most important difference between diffuse and specular reflections for ray tracing is that an incoming ray essentially spawns infinitely many scattered rays while there is only a single specular path. In order to computed the scattered field at a particular position, one needs to integrate the scattered field over the entire surface.
# 
# Let us have a look at some common scattering patterns that are implemented in Sionna RT:

# In[2]:


LambertianPattern().show();


# In[3]:


# The stronger alpha_r, the more the pattern
# is concentrated around the specular direction.
DirectivePattern(alpha_r=10).show(show_directions=True);


# In order to develop a feeling for the difference between specular and diffuse reflections, let us load a very simple scene with a single quadratic reflector and place a transmitter and receiver in it.

# In[4]:


scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)

# Configure the transmitter and receiver arrays
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = scene.tx_array

# Add a transmitter and receiver with equal distance from the center of the surface
# at an angle of 45 degrees.
dist = 5
d = dist/sqrt(2)
scene.add(Transmitter(name="tx", position=[-d,0,d]))
scene.add(Receiver(name="rx", position=[d,0,d]))

if no_preview:
    # Add a camera for visualization
    my_cam = Camera(position=[0, -30, 20], look_at=[0,0,3])
    scene.render(camera=my_cam, num_samples=128)
else:
    scene.preview()


# Next, let us compute the specularly reflected path:

# In[5]:


p_solver = PathSolver()
paths = p_solver(scene=scene, los=False, specular_reflection=True)

if no_preview:
    scene.render(camera=my_cam, paths=paths);
else:
    scene.preview(paths=paths)


# As expected from geometrical optics (GO), the specular path goes through the center of the reflector and has indentical incoming and outgoing angles with the surface normal.

# We can compute the scattered paths in a similar way:

# In[6]:


paths = p_solver(scene=scene, los=False, specular_reflection=False, diffuse_reflection=True)
print(paths.a)


# It might be to your surpise that there is not a single diffusely reflected path. The reason for this is, however, very simple.
# The radio material of the reflector in the scene is simply not diffusely reflecting at all. We can change this behavior by changing the value of scattering coefficient $S$ to a positive value. The squared scattering coefficient $S^2$ determines which portion of the totally reflected energy (specular and diffuse combined) is diffusely reflected. For details on the precise modeling of the scattered field, we refer to the [EM Primer](https://nvlabs.github.io/sionna/rt/em_primer.html#scattering).

# In[7]:


scene.get("reflector").radio_material.scattering_coefficient = 0.5
paths = p_solver(scene=scene, los=False, specular_reflection=False,
                 diffuse_reflection=True, samples_per_src=10**6)
print(f"There are {paths.a[0].shape[-1]} scattered paths.")


# The number of rays hitting the surface is proportional to the total number of rays shot and the squared distance between the transmitter and the surface. However, the total received energy across the surface is constant as the transmitted energy is equally divided between all rays.
# If we double the number of rays that are shot from each source (i.e., transmitting antenna), the number of diffusely reflected paths should also double:

# In[8]:


paths = p_solver(scene=scene, los=False, specular_reflection=False,
                 diffuse_reflection=True, samples_per_src=2*10**6)
print(f"There are {paths.a[0].shape[-1]} scattered paths.")


# ## Scattering Patterns

# In order to study the impact of the scattering pattern, let's replace the perfectly diffuse Lambertian pattern (which all radio materials have by default) by the [DirectivePattern](https://nvlabs.github.io/rt/api/radio_materials.html#sionna.rt.DirectivePattern). The larger the integer parameter $\alpha_r$, the more the scattered field is focused around the direction of the specular reflection.

# In[9]:


scattering_pattern = DirectivePattern(1)
scene.get("reflector").radio_material.scattering_pattern = scattering_pattern
alpha_rs =[1,2,3,5,10,30,50,100]
received_powers = []
for alpha_r in alpha_rs:
    scattering_pattern.alpha_r = alpha_r
    paths = p_solver(scene=scene, los=False, specular_reflection=False, diffuse_reflection=True)
    received_powers.append(10*log10(dr.sum(cpx_abs(paths.a)**2).numpy()))
    
plt.figure()
plt.plot(alpha_rs, received_powers)
plt.xlabel(r"$\alpha_r$")
plt.ylabel("Received power (dB)");
plt.title("Impact of the Directivity of the Scattering Pattern");


# We can indeed observe that the received energy increases with $\alpha_r$. This is because the scattered paths are almost parallel to the specular path directions in this scene. If we move the receiver away from the specular direction, this effect should be reversed.

# In[10]:


# Move the receiver closer to the surface, i.e., away from the specular angle theta=45deg
scene.get("rx").position = [d, 0, 1]
received_powers = []
for alpha_r in alpha_rs:
    scattering_pattern.alpha_r = alpha_r
    paths = p_solver(scene=scene, los=False, specular_reflection=False, diffuse_reflection=True)
    received_powers.append(10*log10(dr.sum(cpx_abs(paths.a)**2).numpy()))
    
plt.figure()
plt.plot(alpha_rs, received_powers)
plt.xlabel(r"$\alpha_r$")
plt.ylabel("Received power (dB)");
plt.title("Impact of the Directivity of the Scattering Pattern");


# ## Validation Against the "Far"-Wall Approximation

# 
# $$ 
# P_r = \left(\frac{\lambda S \Gamma}{4\pi r_i r_s}\right)^2 f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s}) \cos(\theta_i) A
# $$
# 
# which simplifies for a perfect reflector ($\Gamma=1$) with Lambertian scattering pattern and unit surface area to
# 
# $$ 
# P_r = \left(\frac{\lambda S}{4\pi r_i r_s}\right)^2 \frac{\cos(\theta_i)\cos(\theta_s)}{\pi}
# $$
# 
# where $r_i$ and $r_s$ are the distances between the surface center and the transmitter and receiver, respectively. 
# 
# We have constructed our scene such that $r_i=r_s$ and $\theta_i=\theta_s=\pi/4$, so that $\cos(\theta_i)=1/\sqrt{2}$.
# Thus,
# 
# $$
# P_r = \left(\frac{\lambda S}{4\pi r_i^2 }\right)^2 \frac{1}{2\pi}
# $$
# 
# Let's validate for which distances $r_i$ this approximation holds.

# In[11]:


s = 0.7 # Scattering coefficient

# Configure the radio material
scene.get("reflector").radio_material.scattering_pattern = LambertianPattern()
scene.get("reflector").radio_material.scattering_coefficient = s

# Set the carrier frequency
scene.frequency = 3.5e9
wavelength = scene.wavelength

r_is = [0.1, 1, 2, 5, 10] # Varying distances
received_powers = []
theo_powers = []
for r_i in r_is:
    # Update the positions of TX and RX
    d = r_i/sqrt(2)
    scene.get("tx").position = [-d, 0, d]
    scene.get("rx").position = [d, 0, d]
    paths = p_solver(scene=scene, los=False, specular_reflection=False, diffuse_reflection=True)
    received_powers.append(10*log10(dr.sum(cpx_abs(paths.a)**2).numpy()))
    
    # Compute theoretically received power using the far-wall approximation
    theo_powers.append(10*log10((wavelength[0]*s/(4*dr.pi*r_i**2))**2/(2*dr.pi)))
    
plt.figure()
plt.plot(r_is, received_powers)
plt.plot(r_is, theo_powers, "--")
plt.title("Validation of the Scattered Field Power")
plt.xlabel(r"$r_i$ (m)")
plt.ylabel("Received power (dB)");
plt.legend(["Ray tracing", "\"Far\"-wall approximation"]);


# We can observe an almost perfect match between the results for ray-tracing and the "far"-wall approximation from a distance of $2\,$m on. For smaller distances, there is a significant (but expected) difference. In general, none of both approaches is valid for very short propagation distances.

# ## Radio Maps With Scattering

# By now, you have a gained a solid understanding of scattering from a single surface. Let us now make things a bit more interesting by looking at a complex scene with many scattering surfaces. This can be nicely observed with the help of radio maps. 
# 
# 
# Let us now load a slightly more interesting scene containing a couple of rectangular buildings and add a transmitter. Note that we do not need to add any receivers to compute a radio map (we will add one though as we need it later).

# In[12]:


scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 30e9
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = scene.tx_array
scene.add(Transmitter(name="tx",
                      position=[-33,11,32],
                      orientation=[0,0,0]))

# We add a receiver for later path computations
scene.add(Receiver(name="rx",
                      position=[27,-13,1.5],
                      orientation=[0,0,0]))


my_cam = Camera(position=[10,0,300], look_at=[0,0,0])
my_cam.look_at([0,0,0])


# Computing and visualizing a radio map is as simple as running the following commands:

# In[13]:


rm_solver = RadioMapSolver()
rm = rm_solver(scene, cell_size=[1,1], samples_per_tx=int(20e6), max_depth=5, refraction=False)
if no_preview:
    scene.render(camera=my_cam, radio_map=rm, rm_vmin=-200, rm_vmax=-90);
else:
    scene.preview(radio_map=rm, rm_vmin=-200, rm_vmax=-90);


# By default, radio maps are computed without diffuse reflections. We have also explicitly disabled refraction through the building walls in the code above. The parameter ``cm_cell_size`` determines the resolution of the radio map. However, the finer the resolution, the more rays (i.e., `num_samples`) must be shot. We can see from the above figure, that there are various regions which have no coverage as they cannot be reached by pure line-of-sight of specularly reflected paths. 
# 
# Let's now enable diffuse reflections and see what happens.

# In[14]:


# Configure radio materials for scattering
# By default the scattering coefficient is set to zero
for rm in scene.radio_materials.values():
    rm.scattering_coefficient = 1/sqrt(3) # Try different values in [0,1]
    rm.scattering_pattern = DirectivePattern(alpha_r=10) # Play around with different values of alpha_r

rm_scat = rm_solver(scene, cell_size=[1,1], samples_per_tx=int(20e6), max_depth=5,
                    refraction=False, diffuse_reflection=True)

if no_preview:
    scene.render(camera=my_cam, radio_map=rm_scat, rm_vmin=-200, rm_vmax=-90);
else:
    scene.preview(radio_map=rm_scat, rm_vmin=-200, rm_vmax=-90);


# Thanks to scattering, most regions in the scene have some coverage. However, the scattered field is weak compared to that of the LoS and reflected paths. Also note that the peak signal strength has slightly decreased. This is because the scattering coefficient takes away some of the specularly reflected energy.

# ## Impact on Channel Impulse Response

# As a last experiment in our tutorial on scattering, let us have a look at the discrete baseband-equivalent channel impulse responses we obtain with and without scattering. To this end, we will compute the channel impulse response of the  single receiver we have configured for the current scene, and then transform it into the complex baseband representation using method [Paths.taps()](https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.taps).

# In[15]:


# Change the scattering coefficient of all radio materials
for rm in scene.radio_materials.values():
    rm.scattering_coefficient = 1/sqrt(3)
    
bandwidth=200e6 # bandwidth of the receiver (= sampling frequency)
    
plt.figure()

# Paths without diffuse reflections
paths = p_solver(scene,
                 max_depth=5,
                 samples_per_src=10**6,
                 diffuse_reflection=False,
                 refraction=False,
                 synthetic_array=True)

# Paths with diffuse reflections
paths_diff = p_solver(scene,
                 max_depth=5,
                 samples_per_src=10**6,
                 diffuse_reflection=True,
                 refraction=False,
                 synthetic_array=True)

# Compute channel taps without scattering
taps = paths.taps(bandwidth, l_min=0, l_max=100, normalize=True, out_type="numpy")
taps = np.squeeze(taps)
tau = np.arange(taps.shape[0])/bandwidth*1e9

# Compute channel taps wit scattering
taps_diff = paths_diff.taps(bandwidth, l_min=0, l_max=100, normalize=True, out_type="numpy")
taps_diff = np.squeeze(taps_diff)

# Plot results
plt.figure();
plt.plot(tau, 20*np.log10(np.abs(taps)));
plt.plot(tau, 20*np.log10(np.abs(taps_diff)));
plt.xlabel(r"Delay $\tau$ (ns)");
plt.ylabel(r"$|h|^2$ (dB)");
plt.title("Comparison of Channel Impulse Responses");
plt.legend(["No Scattering", "With Scattering"]);


# The discrete channel impulse response looks similar for small values of $\tau$, where the field is dominated by strong LOS and reflected paths. However, in the middle and tail, there are differences of a few dB which can have a significant impact on the link-level performance.

# ## Summary

# In conclusion, scattering plays an important role for radio propagation modelling. In particular, the higher the carrier frequency, the rougher most surfaces appear compared to the wavelength. Thus, at THz-frequencies diffuse reflections might become the dominating form of radio wave propgation (apart from LoS).
# 
# We hope you enjoyed our dive into scattering with this Sionna RT tutorial. Please try out some experiments yourself and improve your grasp of ray tracing. There's more to discover, so so don't forget to check out our other [tutorials](https://nvlabs.github.io/sionna/rt/tutorials.html), too.

# ## References
# [1] Vittorio Degli-Esposti et al., [Measurement and modelling of scattering from buildings](https://ieeexplore.ieee.org/abstract/document/4052607), IEEE Trans. Antennas Propag., vol. 55, no. 1,  pp.143-153, Jan. 2007.
# 
# [2] Vittorio Degli-Esposti et al., [An advanced field prediction model including diffuse scattering](https://ieeexplore.ieee.org/abstract/document/1310631), IEEE Trans. Antennas Propag., vol. 52, no. 7, pp.1717-1728, Jul. 2004.
-e 
# --- End of Scattering.ipynb ---

#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Loading and Editing of Scenes

# This notebook explains how to load and edit scenes with Sionna's [ray tracing (RT) module](https://nvlabs.github.io/sionna/rt/index.html). You will:
# 
# - Use the `load_scene()` function to load a scene with and without merging objects
# - Learn how to add and remove objects from a scene
# - Learn how to translate, rotate, and scale objects within a scene

# ## Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, ITURadioMaterial, SceneObject


# ## Loading Scenes and Merging Objects

# Loading a scene with Sionna RT is done using the `load_scene()` function. By default, this function merges objects that share similar properties, such as radio materials. This is done because reducing the number of objects in a scene enables significant speed-ups for ray tracing.
# 
# Merging shapes can be disabled using the `merge_shapes` flag of `load_scene()`:

# In[2]:


scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=False) # Disable merging of objects


# Let's now print the objects that make up the scene and their constituent materials. We can see that the objects have not been merged,
# as radio materials appear multiple times in the composition of objects.

# In[3]:


for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')


# Let's now reload the scene with the merging of objects enabled:

# In[4]:


scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=True) # Enable merging of objects (default)

for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')


# We can see that objects sharing the same radio materials have been merged, as each radio material appears only once in the composition of objects.

# The function `load_scene()` also allows the exclusion of specific objects from the merging operation through the use of regular expressions. Please see the [Python documentation](https://docs.python.org/3/library/re.html) for details about the regular expression syntax.
# As an example, let's exclude buildings with indices smaller than 3 from the merging process:

# In[5]:


scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=True, # Enable merging of objects
                   merge_shapes_exclude_regex=r'building_[0-2]$') # Exclude from merging
                                                                  # buildings with indices < 3

for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')


# We can see that "building_1" and "building_2" have not been merged. As a result, "building_5" has not been merged either, as it has no other objects to be merged with.

# ## Editing Scenes

# Let's load a more complex scene and visualize it.

# In[6]:


scene = load_scene(sionna.rt.scene.etoile) # Objects are merged by default

cam = Camera(position=[-360,145,400], look_at=[-115,33,1.5])
if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


# Next, we will add a few objects to the scene.
# 
# In Sionna RT, adding an object to a scene is achieved by instantiating a new [`SceneObject`](https://nvlabs.github.io/sionna/rt/api/scene_object.html) from a mesh and then adding the object to the scene using `Scene.add()`. When the `SceneObject` is instantiated, the radio material constituting the object needs to be specified.
# 
# In the following example, we will add cars made of metal to the previously loaded scene.

# In[7]:


# Number of cars to add
num_cars = 10

# Radio material constituing the cars
# We use ITU metal, and use red color for visualization to
# make the cars easily discernible
car_material = ITURadioMaterial("car-material",
                                "metal",
                                thickness=0.01,
                                color=(0.8, 0.1, 0.1))

# Instantiate `num_cars` cars sharing the same mesh and material
cars = [SceneObject(fname=sionna.rt.scene.low_poly_car, # Simple mesh of a car
                    name=f"car-{i}",
                    radio_material=car_material)
        for i in range(num_cars)]

# Add the list of newly instantiated objects to the scene
scene.edit(add=cars)

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


# We can see the red cars in the scene, but because they are all located at the same position, it appears that only a single car was added to the scene.
# 
# In the next cell, we will position the cars in the scene and also set their orientations.

# In[8]:


# Positions 
# Car are positioned in a circle around the central monument
# Center of the circle
c = mi.Point3f(-127, 37, 1.5)
# Radius of the circle
r = 100
# Angles at which cars are positioned
thetas = dr.linspace(mi.Float, 0., dr.two_pi, num_cars, endpoint=False)
# Cars positions
cars_positions = c + mi.Point3f(dr.cos(thetas), dr.sin(thetas), 0.)*r

# Orientations
# Compute points the car "look-at" to set their orientation
d = dr.normalize(cars_positions - c)
# Tangent vector to the circle at the car position
look_at_dirs = mi.Vector3f(d.y, -d.x, 0.)
look_at_points = cars_positions + look_at_dirs

# Set the cars positions and orientations
for i in range(num_cars):
    cars[i].position = mi.Point3f(cars_positions.x[i], cars_positions.y[i], cars_positions.z[i])
    cars[i].look_at(mi.Point3f(look_at_points.x[i], look_at_points.y[i], look_at_points.z[i]))

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


# Objects can also be scaled. This is useful, for example, when the scale of the mesh from which the object is built does not suit the scene.
# 
# To illustrate this feature, let's scale the first car to be twice as large as the other cars.

# In[9]:


cars[0].scaling = 2.0

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


# Finally, objects can be removed from the scene using the `Scene.edit()` function.
# 
# To illustrate this, let's remove the last car we have added.

# In[10]:


scene.edit(remove=[cars[-1]])

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


# ## Path Computation with the Edited Scene

# Let's now compute radio propagation paths on the edited scene.
# 
# We start by adding a transmitter on the roof of an arbitrarily selected building, as well as a receiver on top of each car.
# We also set the transmitter and receiver arrays.

# In[11]:


# Add a transmitter on top of a building
scene.remove("tx")
scene.add(Transmitter("tx", position=[-36.59, -65.02, 25.], display_radius=2))

# Add a receiver on top of each car
for i in range(num_cars):
    scene.remove(f"rx-{i}")
    scene.add(Receiver(f"rx-{i}", position=[cars_positions.x[i],
                                            cars_positions.y[i],
                                            cars_positions.z[i] + 3],
                      display_radius=2))


# Set the transmit and receive antenna arrays
scene.tx_array = PlanarArray(num_cols=1,
                             num_rows=1,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array


# We are now ready to compute paths.

# In[12]:


p_solver = PathSolver()
paths = p_solver(scene, max_depth=5)

if no_preview:
    scene.render(camera=cam, paths=paths);
else:
    scene.preview(paths=paths);


# ## Summary

# A scene is loaded using the `load_scene()` function, which by default merges objects sharing similar properties, such as radio materials. Merging of objects can be disabled by setting the `merge_shapes` flag to `False`; however, this can incur a significant slowdown in the ray tracing process. Alternatively, specific objects can be removed from the merging process using the `merge_shapes_exclude_regex` parameter.
# 
# Scene objects can be instantiated from meshes and added to a scene using `Scene.edit()`. This function can also be used to remove objects from a scene. Note that to optimize performance and reduce processing time, it is recommended to use a single call to this function with a list of objects to add and/or remove, rather than making multiple individual calls to edit scene objects.
-e 
# --- End of Scene-Edit.ipynb ---

