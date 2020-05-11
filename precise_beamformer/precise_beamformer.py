#!/usr/bin/env python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
from typing import Union


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
DelayVals = np.random.randn(iNBeams, iNAntennas, 4) # 4 because {delay, delay_rate, phase, phase_rate}
                                                    # Don't know yet whether or not the negative numbers generated in this way will hurt.
DelayVals = DelayVals.astype(np.float32)
DV_pin = cuda.register_host_memory(DelayVals)
DV_gpu = cuda.mem_alloc(DV_pin.nbytes)

#TODO simulate some time-related information. Will need to pass this to the kernel as well.

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
def gen_steering_coeffs(referenceTime: Union[float, time.struct_time],
                        currentTime: Union[float, time.struct_time],
                        samplingPeriod: float,
                        nChannels: int,
                        nAntennas: int,
                        nBeams: int,
                        lDelayVals: np.ndarray) -> np.ndarray:
    """
    This function needs to return an array of shape [CxAxB] so that it fits with the matmul assumption.
    """
    steeringCoeffs = np.empty((nChannels, nAntennas, nBeams), dtype=np.complex64)
    for a in range(nAntennas):
        for b in range(nBeams):
            delay, delay_rate, phase, phase_rate = lDelayVals[i, j]
            deltaTime = currentTime - referenceTime
            deltaDelay = delay_rate*deltaTime
            delay_N_2 = (delay + deltaDelay)*np.pi/(samplingPeriod*2)
            channels = np.arange(nChannels)
            delay_n = (delay + deltaDelay)*channels/2*np.pi/(samplingPeriod*nChannels)
            deltaPhase = phase_rate*deltaTime
            phase_0 = phase - delay_N_2 + deltaPhase
            rotation = delay_n + phase_0
            cplx_beamweights = np.cos(rotation) + 1j*sin(rotation)
            cplx_beamweights = cplx_beamweights.astype(np.complex64)
            steeringCoeffs[:,a,b] = cplx_beamweights
    return steeringCoeffs


foo = gen_steering_coeffs(ref_time,)

