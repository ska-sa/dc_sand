#!/usr/bin/env python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


iTimeLenghTotal = 256
iTimeSteps = 32
iNChannels = 32
iNAntennas = 32
iNBeams = 2

# Simulated F-engine data in.
volume_knob = 127
AntennaData = np.random.randn(iNAntennas, iNChannels, iTimeLenghTotal, 2)*volume_knob  # Extra dimension of 2 there because complex
AntennaData = AntennaData.astype(np.int8)
AD_pin = cuda.register_host_memory(AntennaData)
AD_gpu = cuda.mem_alloc(AD_pin.nbytes)

# Simulated delay-val data.
DelayVals = np.random.randn(iNBeams, iNAntennas, 4) # Don't know yet whether or not the negative numbers generated in this way will hurt.
DelayVals = DelayVals.astype(np.float32)
DV_pin = cuda.register_host_memory(DelayVals)
DV_gpu = cuda.mem_alloc(DV_pin.nbytes)

# Transfer
cuda.memcpy_htod(AD_gpu, AD_pin) #This is the one we'll want to time, the other is going to happen less frequently so it's not as important.
cuda.memcpy_htod(DV_gpu, DV_pin)

# Allocate the right amount of output space.
Output = np.empty(iNBeams, iNChannels, dtype=np.complex64)
O_pin = cuda.register_host_memory(Output)
O_gpu = cuda.mem_alloc(O_pin.nbytes)

module = SourceModule(open("precise_beamformer.cu").read(), options=["--ptxas-options=-v",])
precise_beamformer = module.get_function("precise_beamformer")

# work out the size and number of thread blocks
;

# Execute the kernel
;

# Transfer back
cuda.memcpy_dtoh(O_pin, O_gpu)

# Verify results
;
