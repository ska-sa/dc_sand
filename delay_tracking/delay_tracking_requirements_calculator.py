"""
Script to calculate MeerKAT Extension phase tracking requirements.

This script loads the MeerKAT Extension antenna coordinates, determines the maximum baseline length from this and then
calculates the MeerKAT Extension delay tracking requirements introduced by this distance.

The requirements that this script is meant to determine include:
1. The maximum baseline length that needs to be accounted for.
2. The maximum delay that the system needs to compensate for.
3. The maximum and minumum rate of change of delay.

TODO: Split this script into two scripts, one to determine the maximum baseline and one to determine the delay tracking
requirements from this maximum baseline.
"""

__author__ = "Gareth Callanan"

import geopy.distance
import numpy
import pandas as pd
import scipy
import scipy.constants


def calculate_distances_km(src, dest):
    """Calculate the distance between two GPS coordinates.

    Calculates the distance between two coordinates on the globe. It also projects the distance onto the equator and
    returns this. I thought this projection was important for delay tracking, but the baseline distance is more
    applicable. However the calculation has been left here due to laziness.

    src and dest are expected to each be a tuple containg latitude and longitude coordinates in the decimal degrees
    format. Example: (14.333,-28.444). The latitude is the first entry and longitude is the second entry.
    """
    horiz_dist_coords = dest[1] - src[1]
    vertical_dist_coords = dest[0] - src[0]
    absolute_distance_km = geopy.distance.GeodesicDistance(src, dest).kilometers
    angle = numpy.arctan2(vertical_dist_coords, horiz_dist_coords)
    horizontal_distance_km = abs(absolute_distance_km * numpy.cos(angle))
    return absolute_distance_km, horizontal_distance_km


print("Starting")

print()
print("Loading Antennas Coords from a CSV file.")
# The antenna data is expected to be in csv format with three columns. The first row must have column headings:
# "Name,Longitude,Latitude". Each row contains an entry for a different antenna. The latitude and longitude coordinates
# must be formatted as decimal degrees. An example of a row entry is "ANT01,14.333,-28.444."
data = pd.read_csv("meerkat_extension_preliminary_antenna_locations.csv")
latNames = numpy.array(data["Name"])
latArr = numpy.array(data["Latitude"])
longArr = numpy.array(data["Longitude"])

antCoords = list(zip(latArr, longArr))
numAnts = len(antCoords)

print()
print("Calculating Baseline Lengths:")

distances = []
for i in range(numAnts):
    for j in range(i + 1, numAnts):
        absolute_distance_km, horizontal_distance_km = calculate_distances_km(antCoords[i], antCoords[j])
        distances.append((latNames[i], latNames[j], absolute_distance_km, horizontal_distance_km))

absolute_distance_km, horizontal_distance_km = calculate_distances_km(antCoords[0], antCoords[1])

distances.sort(key=lambda x: x[2])  # Not technically needed
max_distance_absolute_tuple = max(distances, key=lambda x: x[2])
max_distance_horizontal_tuple = max(distances, key=lambda x: x[3])

print(f"Max Distance Tuple: {max_distance_absolute_tuple}")
print(f"Max Horizontal Distance Tuple: {max_distance_horizontal_tuple}")

# The maximum baseline length determines the maximum coarse delay - this corresponds to an object just on the horizon
# where the wavefront needs to travel almost directly along the baseline.
max_absolute_distance_m = 20000  # max_distance_absolute_tuple[2] * 1000
max_coarse_delay_s = max_absolute_distance_m / scipy.constants.c

print()
print("Calculating Maximum Compensation Delay Requirements:")

print(f"Max Comp Delay with distance of {max_absolute_distance_m:.2f} m: {max_coarse_delay_s*1000*1000:.2f} us")

# Technically what was done for MeerKAT is to take "max_coarse_delay_s" and double it - this is done to account for a
# virtual reference antenna. It is not absolutly necessary to double it, this is just the way we have implemented it.
# You could just as easily have added a constant of some us to get a delay of "max_coarse_delay_s + x us"
#
# We then need to account for the different propagation times in the PPS signal. This signal originates in the central
# MeerKAT data centre (The KAPB). The cable lengths this signal is sent on are not equal. The different in time that needs
# to be accounted for is the different in propagation time between the shortest and longest cable. A more than worst
# case estimate for this length is the length of the longest baseline. As such we now have
# "2 * max_coarse_delay_s + ~max_coarse_delay_s ~= 3 * max_coarse_delay_s"
print(
    f"Multiply by three to account for PPS propagation and virtual reference antenna: {max_coarse_delay_s*1000*1000*3:.2f} us"
)

print()
print("Calculating Maximum Delay Rates:")


def calculate_delay_from_source_elevation(baseline_length_m, source_elevation_degrees):
    """
    Determine the delay between two antennas in seconds based on a source's elevation.

    For the purposes of this calculation a right angle triangle is constructed. The baseline
    vector is the hypotenuse of this triangle. The wavefront from the source is the second side.
    For elevation != 90 degrees, the wavefront will intercept one antenna before the other. The
    third side of this triangle is the line perpendicular to the wavefront that extends to the
    antenna from the wavefront. This third side is the length of delay between the two antennas.
    The angle between the third side and the baseline vector is equal to this elevation angle.

    A diagram is probably more useful to understand this. One should be available in the
    git repo this script is commited in.
    """
    if source_elevation_degrees > 90 or source_elevation_degrees < 0:
        raise TypeError("source_elevation_degrees needs to be within: 0<=x<=90")

    # Calculate the delay distance using properties of the triangle constructed above
    delay_length_m = baseline_length_m * numpy.cos(source_elevation_degrees / 180 * scipy.constants.pi)
    delay_length_s = delay_length_m / scipy.constants.c

    return delay_length_s


# The model here assumes the earth remains stationary, and the source moves across the sky changing the wavefront
# angle at a uniform rate.
# Calculation is for an elevation change by 90 degrees over six hours.
elevationDeltaPerSecond = 90 / (3600 * 6)

# Maximum rate of change of delay occurs at 90 degree elevation
elevationMax = 90
delay1_s = calculate_delay_from_source_elevation(max_absolute_distance_m, elevationMax)
delay2_s = calculate_delay_from_source_elevation(max_absolute_distance_m, elevationMax - elevationDeltaPerSecond)
deltaDelayMax_nanosecondsPerSecond = (delay2_s - delay1_s) * 1000 * 1000 * 1000

# Minimum rate of change of delay occurs at 0 degree elevation. However we only consider objects 15 degrees above the
# horizon.
elevationMin = 15
delay1_s = calculate_delay_from_source_elevation(max_absolute_distance_m, elevationMin)
delay2_s = calculate_delay_from_source_elevation(max_absolute_distance_m, elevationMin + elevationDeltaPerSecond)
deltaDelayMin_nanosecondsPerSecond = (delay1_s - delay2_s) * 1000 * 1000 * 1000

print(f"Maximum Delay occuring at elevation of {elevationMax} degrees: {deltaDelayMax_nanosecondsPerSecond} ns/s")
print(f"Minimum Delay occuring at elevation of {elevationMin} degrees: {deltaDelayMin_nanosecondsPerSecond} ns/s")


print("Done")
