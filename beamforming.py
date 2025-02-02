# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Basic Beamforming -- Generate a map of three sources.
=====================================================

Loads the simulated signals from the `three_sources.h5` file, analyzes them with Conventional Beamforming
and generates a map of the three sources.

.. note:: The `three_sources.h5` file must be generated first by running the :doc:`example_three_sources` example.
"""

from pathlib import Path
import os
import librosa as lb
import acoular as ac
import numpy as np
import scipy as sp
from pylab import axis, colorbar, figure, imshow, plot, show

def synth_pressure():
    sfreq = 22050
    duration = 1
    nsamples = duration * sfreq
    micgeofile = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'geo/3_by_3.xml')
    h5savefile = Path('drone+diesel.h5')

    m = ac.MicGeom(from_file=micgeofile)
    n1 = ac.WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=1, rms=0.5)
    #n2 = ac.WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.5)
    #n3 = ac.WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.5)
    p1 = ac.PointSource(signal=n1, mics=m, loc=(-1, -1, 1))
    #p2 = ac.PointSource(signal=n2, mics=m, loc=(0.15, 0, 2))
    #p3 = ac.PointSource(signal=n3, mics=m, loc=(0, 0.1, 3))

    drone_file = Path('sounds/B_S2_D1_067-bebop_000_cleaned.wav')
    assert drone_file.exists(), 'Drone file not found, run example_three_sources.py first'

    # diesel_file = Path('sounds/diesel_engine_16bit_44100_1s_0db.wav')
    # assert diesel_file.exists(), 'Diesel file not found, run example_three_sources.py first'

    drone_data, drone_sr = lb.load(drone_file)
    #diesel_data, diesel_sr = lb.load('diesel_engine_16bit_44100_1s_0db.mp3')

    drone_data = drone_data.reshape(1,-1)

    print(drone_data.ndim)
    print(drone_sr)

    drone_ts = ac.TimeSamples(data=drone_data, sample_freq=drone_sr)
    #diesel_ts = ac.TimeSamples(data=diesel_data, numsamples=diesel_sr)

    drone_generator = ac.GenericSignalGenerator(source=drone_ts)

    drone_ps = ac.PointSource(signal=drone_generator, mics=m, loc=(0, 1, 40))
    #diesel_ps = ac.PointSource(signal=diesel_ts, mics=m, loc=(-1, -1, 1))

    p = ac.Mixer(source=p1, sources=[drone_ps])

    wh5 = ac.WriteH5(source=p, name=h5savefile)
    wh5.save()    

synth_pressure()
micgeofile = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'geo/3_by_3.xml')
datafile = Path('drone+diesel.h5')
assert datafile.exists(), 'Data file not found, run example_three_sources.py first'

mg = ac.MicGeom(from_file=micgeofile)

ts = ac.TimeSamples(name=datafile)

ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')

rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, increment=0.01)

st = ac.SteeringVector(grid=rg, mics=mg)

bb = ac.BeamformerFunctional(freq_data=ps, steer=st)

pm = bb.synthetic(5000, 20)

Lm = ac.L_p(pm)

# Calculate distance using time differences
# delays = np.angle(st) / (2 * np.pi * ps.fftfreq()[1])
# distances = delays * 343.2  # Speed of sound in m/s

# Average distance across all microphones
# mean_distance = np.mean(np.abs(distances))

imshow(Lm.T, origin='lower', vmin=Lm.max() - 10, extent=rg.extend(), interpolation='bicubic')
colorbar()
figure(2)
plot(mg.mpos[0], mg.mpos[1], 'o')
axis('equal')
show()
