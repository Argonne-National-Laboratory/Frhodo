# -*- coding: utf-8 -*-
import collections


#This makes the sum of a Nested array/list/tuple.
#For example: [-1, 2, [3, 4, 5], 6] will give 19.
# and [1, 2, [3, 4, 5], 6] will give 21.
def sumNested(arr):
    currentSum = 0
    if isinstance(arr,collections.abc.Iterable):
        for elem in arr:
            returnSum = sumNested(elem)
            currentSum = currentSum + returnSum
        return currentSum
    else:  #if the code has gotten here, the input is a number and the "sum" is just itself.
        sumArr = arr
        return sumArr

sum_2dNested = sumNested #This really should have been the name originally.
    
#This makes the sum of absolute values of a Nested array/list/tuple.
#For example: [-1, 2, [3, 4, 5], 6] will give 21.
# and [1, 2, [3, 4, 5], 6] will give 21.
def sumNestedAbsValues(arrayOrNumber):
    currentSum = 0
    if isinstance(arrayOrNumber,collections.abc.Iterable):
        for elem in arrayOrNumber:
            returnSum = sumNestedAbsValues(elem)
            currentSum = currentSum + returnSum
        return currentSum
    else: #if the code has gotten here, the input is an indivudal number so here is where we take the absolute value.
        AbsVal = abs(arrayOrNumber)
        return AbsVal

sumNestedAbsValues_2dNested = sumNestedAbsValues #This really should have been the name originally.

#isNestedOrString takes an array and checks to see if it is iterable.  If it is iterable then it loops through the array
#to see if it has any more iterable objects.  If there are no iterable values in the array, then the array is not nested.
#Otherwise it has nesting.
#Examples: 3 is not iterable and will return false
#[1,2] is iterable but neither 1 nor 2 are iterable so isNestedOrString will return false
#[1,[2,3]] is iterable, 1 is not iterable but [2,3] is so isNestedOrString will return true
def isNestedOrString(arr):
    if isinstance(arr,collections.abc.Iterable):
        for elem in arr:
            if isinstance(elem,collections.abc.Iterable):
                return True
        #If it finishes the loop then it hasn't found a non-iterable object and is not nested
        return False
    #If the object is not iterable then it can't be nested
    else:
        return False

'''
This function has an implied return by modifying the array called subtractionResult
Typically subtractionResult would be a deepcopy of one of the two arrays
If there is a tuple or any other immutable type, the subtractNested function will not work
subtractionResult = copy.deepcopy(arr1)
subtractNested(arr1,arr2,subtractionResult)
#we do allow approximate comparisons using the variables relativeTolerance and absoluteTolerance
'''
def subtractNested(arr1,arr2,subtractionResult, relativeTolerance=None, absoluteTolerance=None, softStringCompare=False):
    if isinstance(arr1,collections.abc.Iterable):
        for elemindex,elem in enumerate(arr1):
            if type(elem) == str:
                if softStringCompare == True: #if using softStringCompare
                    #The 0 or 1 returned is opposite what is normally done in Python
                    #1 is usually true and 0 is usually false 
                    if stringCompare(arr1[elemindex],arr2[elemindex]): #Determine if the strings are equal using stringCompare
                        subtractionResult[elemindex] = 0
                    else:
                        subtractionResult[elemindex] = 1
                else: #otherwise compare regularly
                    #The 0 or 1 returned is opposite what is normally done in Python
                    #1 is usually true and 0 is usually false 
                    if arr1[elemindex] == arr2[elemindex]:
                        subtractionResult[elemindex] = 0
                    else:
                        subtractionResult[elemindex] = 1
            else: 
                if isinstance(elem,collections.abc.Iterable):
                    subtractNested(arr1[elemindex],arr2[elemindex],subtractionResult[elemindex], relativeTolerance=relativeTolerance, absoluteTolerance=absoluteTolerance, softStringCompare=softStringCompare)
                else: #this is for final elements, like integers and floats.
                    #we do allow approximate comparisons using the variables relativeTolerance and absoluteTolerance
                    # there are a variety of comparison tools, https://docs.pytest.org/en/documentation-restructure/how-to/builtin.html#comparing-floating-point-numbers
                    # we are using numpy allclose because we want to have  as few dependencies as possible, and numpy is not such a bad dependency in our view.
                    if (relativeTolerance==None and absoluteTolerance==None):
                        subtractionResult[elemindex] = arr1[elemindex] - arr2[elemindex]
                    else: #Else one of the tolerances requested is not None
                        import numpy as np 
                        #we 1st need to make any tolerances that are still none into the numpy default, because we don't know if the person has selected both tolerances.
                        #and we cannot feed "None" into numpy.
                        if relativeTolerance == None:
                            relativeTolerance = 1.0E-5
                            print("Warning: Can't have absolute tolerance without relative tolerance. Setting relative tolerance to 1.0E-5.")
                        if absoluteTolerance == None:
                            absoluteTolerance = 1.0E-8
                            print("Warning: Can't have relative tolerance without absolute tolerance. Setting absolute tolerance to 1.0E-8.")
                        #now we do the comparison
                        trueIfTheyAreApproximatelyEqual = np.allclose(arr1[elemindex],arr2[elemindex], rtol = relativeTolerance, atol = absoluteTolerance)
                        if trueIfTheyAreApproximatelyEqual == True:
                            subtractionResult[elemindex] = 0
                        if trueIfTheyAreApproximatelyEqual == False: #If they are not equal, we return the actual subtraction result.
                            subtractionResult[elemindex] = arr1[elemindex] - arr2[elemindex]
        #There is an implied return of arr3 since arr3 was overwritten in the function
    
                        

#This function converts a nested Structure into a nested list.
#For example, 
# nestedTuple = (1,2,(3,4,(5)),6)
# nestedList = nested_iter_to_nested_list(nestedTuple)
# yields: [1, 2, [3, 4, 5], 6]
def nested_iter_to_nested_list(iterReceived):
    #The first two lines are justs to return the object immediately if it's not an iterable.
    #This is mostly to prevent bugs if someone tries to feed an integer, for example.
    if not isinstance(iterReceived,collections.abc.Iterable):
        return iterReceived
    list_at_this_level = list(iterReceived)
    for elemIndex, elem in enumerate(iterReceived):
        #A string is iterable and a single value in a string is also iterable
        #So check to see if it is not a string to avoid recursion error
        if isinstance(elem,collections.abc.Iterable) and type(elem) != str:
            list_at_this_level[elemIndex] = nested_iter_to_nested_list(elem)
        else:
            list_at_this_level[elemIndex] = elem
    return list_at_this_level

'''
stringCompare takes in two strings and compares a standardized version of the two to see if they match
Added: 181008
Last modified: 181008
'''
def stringCompare(firstString,secondString):
    import re
    #First store the strings into a variable that will be standardized
    standardizedFirstString = firstString
    standardizedSecondString = secondString
    #Strip the strings of any whitespace on the outsides
    standardizedFirstString = standardizedFirstString.strip()
    standardizedSecondString = standardizedSecondString.strip()
    #Make both strings lowercase
    standardizedFirstString = standardizedFirstString.lower()
    standardizedSecondString = standardizedSecondString.lower()
    #Using regex, find any style of whitespace on the inside and replace it with a standardized space
    standardizedFirstString = re.sub('\s+',' ',standardizedFirstString)
    standardizedSecondString = re.sub('\s+',' ',standardizedSecondString)
    
    #If the standardized strings match return True
    if standardizedFirstString == standardizedSecondString:
        return True
    else: #Otherwise return false
        return False

#This is designed for arrays of numbers.
def makeAtLeast_2dNested(arr):
    import numpy as np
    if (type(arr) == type('str')) or ('int' in str(type(arr))) or ('float' in str(type(arr))): #If it's a string, int, or float, it's not nested.
        nestedArray = [[arr]]
    elif type(arr) != type('str'):#in the normal case, check if it's nested.
        if isNestedOrString(arr) == False:
            nestedArray = [arr]
        else: #Else it is already nested. However, if it is a list containing zero length numpy arrays, then it will become collapsed unless we make the numpy arrays atleast_1d.
            if type(arr) == type([]) and type(arr[0]) == type(np.array(0)):
                import copy
                nestedArray = copy.deepcopy(arr) #first make a copy, then change what is inside.
                for elementIndex in range(len(arr)):
                    nestedArray[elementIndex] = np.atleast_1d(nestedArray[elementIndex])
            else: #Else is the normal case.
                nestedArray = arr
    nestedArray=np.array(nestedArray)
    return nestedArray
    

def convertInternalToNumpyArray_2dNested(inputArray): #This is **specifically** for a nested array of the form [ [1],[2,3] ] Such that it is a 1D array of arrays.
    import numpy as np
    inputArray = np.array(inputArray)
    for elementIndex in range(0,len(inputArray)):
        inputArray[elementIndex] = np.array(inputArray[elementIndex])
    return np.array(inputArray)    

#Takes objects like this: [1,2, [3,4]] and returns objects like this: [1,2,3,4]
def flatten_2dNested(arr):
    import numpy as np
    arr = np.atleast_1d(arr)
    for elemIndex, elem  in enumerate(arr):
        if elemIndex == 0:
            flattenedElem = np.array(elem).flatten()
            flattenedArray = flattenedElem
        elif elemIndex >0:
            flattenedElem = np.array(elem).flatten()
            flattenedArray = np.hstack((flattenedArray, flattenedElem))
    return flattenedArray
    
#should return true for  [[1],[1,2]] and return false for [[1,2],[1,2]]
def checkIfStaggered_2dNested(arr):
        try:
            arr = np.array(convertInternalToNumpyArray(arr))
            onesArray = np.ones(np.shape(arr))
            onesArray = np.matmul(arr.transpose(), onesArray) #This is the key line and catches staggered arrays.
            return False
        except:
            return True