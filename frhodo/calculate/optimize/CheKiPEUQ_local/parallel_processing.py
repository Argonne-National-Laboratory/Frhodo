try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    numProcessors = MPI.COMM_WORLD.Get_size()
    numSimulations = numProcessors - 1 #Normally, we will use rank 0 for controlling.
    currentProcessorNumber = rank
    #now this number can be accessed from elsewhere using parallel_processing.CurrentProcessorNumber
    if numProcessors > 1:
        using_mpi = True
    else:
        using_mpi = True
        
except:
    currentProcessorNumber=0
    numProcessors=1
    using_mpi = False



