#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: Scanpath maps/videos comparison tools and example as main
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np
import os

def getStartPositions(fixationList):
	"""Return positions of first fixation in list of scanpaths.
	Get starting indices of individual fixation sequences.
	"""
	return np.where(fixationList[:, 0] == 0)[0]

def getScanpath(fixationList, startPositions, scanpathIdx=0):
	"""Return a scanpath in a list of scanpaths
	"""
	if scanpathIdx >= startPositions.shape[0]-1:
		range_ = np.arange(startPositions[scanpathIdx], fixationList.shape[0])
	else:
		range_ = np.arange(startPositions[scanpathIdx], startPositions[scanpathIdx+1])
	# print(range_, startPositions[scanpathIdx])
	return fixationList[range_, :].copy()

def dist_angle(vec1, vec2):
	"""Angle between two vectors - same result as orthodromic distance
	"""
	return np.arccos( np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)) )

def getValues(SP1, SP2, F1, F2):
	"""Measure distance and angle between all fixations that happened during a frame.
	Return scores normalized (0, 1). Lower is better."""
	values = []

	for a1 in F1:
		for a2 in F2:
			UVec1 = SP1[a1, :3]
			UVec = SP2[a2, :3]

			dist = dist_angle(UVec1, UVec)

			angle = .5
			if a1 > 0 and a2 > 0:
				vec1 = SP1[a1-1, 1:3] - SP1[a1, 1:3]
				vec2 = SP2[a2-1, 1:3] - SP2[a2, 1:3]

				angle = dist_angle(vec1, vec2)
			values.append([dist, angle])
	return np.array(values) / [np.pi, np.pi]

def sphere2UnitVector(sequence):
	"""Convert from longitude/latitude to 3D unit vectors
	"""
	UVec = np.zeros([sequence.shape[0], 3])
	UVec[:, 0] = np.cos(sequence[:,2]) * np.cos(sequence[:,1])
	UVec[:, 1] = np.cos(sequence[:,2]) * np.sin(sequence[:,1])
	UVec[:, 2] = np.sin(sequence[:,2])
	return UVec

def compareScanpath(fixations1, fixations2, starts1, starts2, iSC1, iSC2):
	"""Return comparison scores between two scanpaths.
	"""
	# print("Comparing scanpath #{} and #{}".format(idx1, idx2))

	# Get individual experiment trials
	scanpath1 = getScanpath(fixations1, starts1, iSC1)
	scanpath2 = getScanpath(fixations2, starts2, iSC2)
	
	# Convert latitudes/longitudes to unit vectors
	UVec1 = sphere2UnitVector(scanpath1)
	UVec2 = sphere2UnitVector(scanpath2)

	minFrame = int(np.min([scanpath1[0, 3], scanpath2[0, 3]]))
	maxFrame = int(np.min([scanpath1[-1, 4], scanpath2[-1, 4]]))
	# print("Frames - Min: {}, Max: {}".format(minFrame, maxFrame))

	comp = None
	values = []

	for iFrame in range(minFrame, maxFrame):
			# Get index of points that happened during this frame
			scanpath1_frame = np.where(np.logical_and(
						scanpath1[:, 3] <= iFrame,
						scanpath1[:, 4] >= iFrame))[0]
			scanpath2_frame = np.where(np.logical_and(
						scanpath2[:, 3] <= iFrame,
						scanpath2[:, 4] >= iFrame))[0]

			# Don't repeat comparisons
			if np.all(comp == np.array([scanpath1_frame, scanpath2_frame])):
				continue
			else:
				comp = np.array([scanpath1_frame, scanpath2_frame])

			# We need at least one pair of elements to compare
			if len(scanpath1_frame) == 0 or len(scanpath2_frame) == 0:
				continue

			# Compute similarity metrics
			val = getValues(UVec1, UVec2, scanpath1_frame, scanpath2_frame)
			# Add values to array
			values += val.tolist()

	scores = np.nanmean(values, axis=0)
	# print("N: {}, meanDist: {:2f}, meanAngle: {:2f}, totalscores: {:2f}".format(len(values), scores[0], scores[1], scores.mean()))

	return scores

if __name__ == "__main__":
	print("Broke main for modelComparison_scanpath script"); exit()
	# Head-only data
	SP_PATH = "../H/Scanpaths/"
	# Head-and-Eye data
	SP_PATH = "../HE/Scanpaths/"

	# Each scanpath file contains all observer's fixation sequences in a row.
	#	Fixations are reported sequentially with an index entry, index 0 begins a sequence
	scanpath1_path = SP_PATH + "3_PlanEnergyBioLab_fixations.csv"

	name = "_".join(scanpath1_path.split(os.sep)[-1].split("_")[:2])

	# Load fixation lists - 0_Idx, 1_Longitude, 2_Latitude, 5_Start/6_End frames
	fixations = np.loadtxt(scanpath1_path, delimiter=",", skiprows=1, usecols=(0, 1,2, 5,6))
	fixations = fixations * [1, 2*np.pi, np.pi, 1,1]

	# Get start/end indices of trials
	starts = getStartPositions(fixations)
	
	print("Let's compare together all {} scanpaths in this file.".format(len(starts)))

	with open("example_ScanpathComparisons.csv", "w") as saveFile:
		saveFile.write("stimName, iSC1_iSC2, distance, angle, score\n")
		for iSC1 in range(1, len(starts)):
			for iSC2 in range(1, len(starts)):
				if iSC1 != iSC2:
					print(" "*20, "\r{}/{}".format(iSC1 * len(starts) + iSC2, len(starts)**2), end="")
					scores = compareScanpath(fixations, iSC1, iSC2)
					saveFile.write("{}, {}, {}, {}\n".format(
						name, "{}_{}".format(iSC1, iSC2), *scores, scores.mean()
						)
					)
	print("\nDone")