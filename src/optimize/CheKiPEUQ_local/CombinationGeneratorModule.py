import math
import itertools
import numpy
import copy

def RestrictionsPassed(CombinationToCheck,):
	FailedStatus = 0  # This means it has not yet failed.  If any of the checks fail, then we change this to 1.
	# Note: the easy way to relate parameters to k is the following: take the k number and multiply it by 2. That's the parameter for Ea. Subtract 1 and that's the parameter for the pre-exponential.
	# Then subract 1 from each of those to convert it to arrays!
	# if (CombinationToCheck[3-1])*math.exp(-CombinationToCheck[4-1]/(R*T)) < (CombinationToCheck[5-1])*math.exp(-CombinationToCheck[6-1]/(R*T)): FailedStatus=1  #require k2 > k3, so fails if k2 < k3
	# if (CombinationToCheck[3-1])*math.exp(-CombinationToCheck[4-1]/(R*T)) < (CombinationToCheck[7-1])*math.exp(-CombinationToCheck[8-1]/(R*T)): FailedStatus=2 #fail if k2 < k4
	# if (CombinationToCheck[3-1])*math.exp(-CombinationToCheck[4-1]/(R*T)) < (CombinationToCheck[9-1])*math.exp(-CombinationToCheck[10-1]/(R*T)): FailedStatus=3 #fail if k2 < k5
	# if (CombinationToCheck[9-1])*math.exp(-CombinationToCheck[10-1]/(R*T)) <  100*(CombinationToCheck[11-1])*math.exp(-CombinationToCheck[12-1]/(R*T)): FailedStatus=4 #require k5 >> k6, so fails if k5 < 100 * k6
	# if (CombinationToCheck[21-1])*math.exp(-CombinationToCheck[22-1]/(R*T)) < (CombinationToCheck[5-1])*math.exp(-CombinationToCheck[6-1]/(R*T)): FailedStatus=5 #fail if k11 < k3
	# if (CombinationToCheck[21-1])*math.exp(-CombinationToCheck[22-1]/(R*T)) < (CombinationToCheck[7-1])*math.exp(-CombinationToCheck[8-1]/(R*T)): FailedStatus=6 #fail if k11 < k4
	# if (CombinationToCheck[21-1])*math.exp(-CombinationToCheck[22-1]/(R*T)) < (CombinationToCheck[9-1])*math.exp(-CombinationToCheck[10-1]/(R*T)): FailedStatus=7 #fail if k11 < k5
	# if (CombinationToCheck[21-1])*math.exp(-CombinationToCheck[22-1]/(R*T)) < (CombinationToCheck[11-1])*math.exp(-CombinationToCheck[12-1]/(R*T)): FailedStatus=8 #fail if k11 < k6
	# if (CombinationToCheck[13-1])*math.exp(-CombinationToCheck[14-1]/(R*T)) < (CombinationToCheck[5-1])*math.exp(-CombinationToCheck[6-1]/(R*T)): FailedStatus=9 #fail if k7 < k3
	# if (CombinationToCheck[13-1])*math.exp(-CombinationToCheck[14-1]/(R*T)) < (CombinationToCheck[7-1])*math.exp(-CombinationToCheck[8-1]/(R*T)): FailedStatus=10 #fail if k7 < k4
	# if (CombinationToCheck[13-1])*math.exp(-CombinationToCheck[14-1]/(R*T)) > (CombinationToCheck[15-1])*math.exp(-CombinationToCheck[16-1]/(R*T)): FailedStatus=11 #fail if k7 > k8
	# if (CombinationToCheck[13-1])*math.exp(-CombinationToCheck[14-1]/(R*T)) > (CombinationToCheck[17-1])*math.exp(-CombinationToCheck[18-1]/(R*T)): FailedStatus=12 #fail if k7 > k9
	# if (CombinationToCheck[13-1])*math.exp(-CombinationToCheck[14-1]/(R*T)) > (CombinationToCheck[19-1])*math.exp(-CombinationToCheck[20-1]/(R*T)): FailedStatus=13 #fail if k7 > k10
	# if (CombinationToCheck[17-1])*math.exp(-CombinationToCheck[18-1]/(R*T)) < (CombinationToCheck[15-1])*math.exp(-CombinationToCheck[16-1]/(R*T)): FailedStatus=14 #fail if k9 < k8
	# if (CombinationToCheck[19-1])*math.exp(CombinationToCheck[20-1]/(R*T)) < (CombinationToCheck[17-1])*math.exp(-CombinationToCheck[18-1]/(R*T)): FailedStatus=15 #fail if k10 < k9

	# THE FOLLOWING LINES ARE FOR DEBUGGING ETC
	#	print str(CombinationToCheck[9-1]) + ' ' + str(math.exp(-CombinationToCheck[10-1]/(R*T))) + ' ' + str(CombinationToCheck[11-1]) + ' ' + str(math.exp(-CombinationToCheck[12-1]/(R*T)))
	#	print str(CombinationToCheck[9-1]*math.exp(-CombinationToCheck[10-1]/(R*T))) + ' ' + str(CombinationToCheck[11-1]*math.exp(-CombinationToCheck[12-1]/(R*T)))
	#	if CombinationToCheck[9-1]*math.exp(-CombinationToCheck[10-1]/(R*T)) < 100*CombinationToCheck[11-1]*math.exp(-CombinationToCheck[12-1]/(R*T)):
	#		print "true that it failed"
	#	if CombinationToCheck[9-1]*math.exp(-CombinationToCheck[10-1]/(R*T)) > 100*CombinationToCheck[11-1]*math.exp(-CombinationToCheck[12-1]/(R*T)):
	#		print "false that it failed"
	if FailedStatus == 0:
		PassedStatus = 1
	else:
		PassedStatus = 0
	return PassedStatus


def combinationGenerator(OriginalParameters, BaseFactor, NumberOfVariationsInEachDirection,SpreadType="Addition", toFile = False, fileName = "output", headerList = "", numberOfTimesToWriteToFile = 1): #SpreadType can be "Addition" or "Multiplication"
	SpreadType.upper()	#makes userinput uppercase for error protection
	CombinationsSoFar = [OriginalParameters]

	for ParameterP in range(len(OriginalParameters)):  # I am Using array indexing, so if the length returned is "21" that means it will iterate across 22 values since 0 and 21 are both values in loop.
		# Note that we are using "ParameterP" to represent the index of the parameter. We'll use this index to extract parameter values and base multipliers etc..
		# Now that we are looping across each parameter, we will iterate across all CombinationsSoFar, then do the base multiplication on each Combination.
		for CurrentBaseCombinationIndex in range(len(CombinationsSoFar)):  # we can't use "for CurrentBaseCombination in CombinationsSoFar" because that would create an infinite loop due to append.
			# Instead, we use the length and array indexing for the loop, since the length will not be recalculated after the first time.
			CurrentBaseCombination = CombinationsSoFar[CurrentBaseCombinationIndex]
			# Now, for each BaseCombination, we need to do the multiplying in each direction for our current ParameterP index.
			if NumberOfVariationsInEachDirection[ParameterP] > 0:  # We're only going to do this for cases where Variations in each direction is non-zero.
				for i in range(1, NumberOfVariationsInEachDirection[ParameterP] + 1):  # If number of vars is 2 for example, then it'll go "1,2" on the i index.
					# Now for each variation index, we need to make the high and the low.
					# for the low case, it actually requires 2 steps: initialization and then changing the parameter currently being changed.
					NewCombinationLow = list(copy.copy(CurrentBaseCombination))
					if SpreadType == "Multiplication":
						NewCombinationLow[ParameterP] = CurrentBaseCombination[ParameterP] / (BaseFactor[ParameterP] ** i)
					elif SpreadType == "Addition":
						NewCombinationLow[ParameterP] = CurrentBaseCombination[ParameterP] - (BaseFactor[ParameterP] * i)
					RestrictionsPassedLow = RestrictionsPassed(NewCombinationLow)
					if RestrictionsPassedLow == 1:
						CombinationsSoFar.append(tuple(NewCombinationLow))
						#print(CombinationsSoFar)
					elif RestrictionsPassedLow == 0:
						CombinationsSoFar.append(tuple(CurrentBaseCombination))
					# Now for high combinations...
					NewCombinationHigh = list(copy.copy(CurrentBaseCombination))
					if SpreadType == "Multiplication":
						NewCombinationHigh[ParameterP] = CurrentBaseCombination[ParameterP] * (BaseFactor[ParameterP] ** i)
					if SpreadType == "Addition":
						NewCombinationHigh[ParameterP] = CurrentBaseCombination[ParameterP] + (BaseFactor[ParameterP] * i)
					RestrictionsPassedHigh = RestrictionsPassed(NewCombinationHigh)
					if RestrictionsPassedHigh == 1:
						CombinationsSoFar.append(tuple(NewCombinationHigh))
					if RestrictionsPassedHigh == 0:
						CombinationsSoFar.append(tuple(CurrentBaseCombination))

	if toFile == True:
		writeToFile(CombinationsSoFar,headerList, fileName, numberOfTimesToWriteToFile)

	return CombinationsSoFar

def writeToFile(comboGeneratedList, headerList, filename, numberOfTimesToWriteOutput = 1):
	filename+=".csv"
	ListToPrint = [headerList]*1
	ListToPrint.extend(comboGeneratedList)

	#print ListToPrint
	f1 = open(filename, 'w')
	for line in ListToPrint:
		for b in range(1,numberOfTimesToWriteOutput + 1):  # b is an arbitrary counter. we use +1 b/c upper end of range is omitted.
			f1.write(str(line)[1:-1] + '\n')
	f1.close()

if __name__ == '__main__':
	headerList = ["Header1", "Header2", "Header3", "Header4"]	#array must be seperated by commas to be split into a list
	print (combinationGenerator([0,0], [1,1], [10,10], SpreadType="Addition",toFile=True,headerList=headerList))	#a list using these parameters is created and returned to comboList


