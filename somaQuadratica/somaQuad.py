import numpy as np
from mpi4py import MPI

def square_sum(arr: np.ndarray) -> int:
  total = 0

  for elem in arr:
    elem = int(elem)
    if elem % 2 == 0:
      total += elem * elem
    else:
      total -= elem * elem
  
  return total

if __name__ == '__main__':
  # Initialize MPI environment with mpi4py
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  n_cores = comm.Get_size()

  # Rank 0 loads the full array
  # Rank 0 also split it into chunks
  if rank == 0:
    array: np.ndarray = np.load('A.npy')
    array_of_arrays = np.array_split(array, n_cores)
  else:
    array_of_arrays = None

  # Distribute the chunks of the array to all ranks
  data: np.ndarray = comm.scatter(array_of_arrays, root=0)

  # Perform computation on each process
  start = 0
  end = data.size

  local_result = square_sum(data)
  
  # Send results from all ranks back to rank 0
  result = comm.gather(local_result, root=0)

  # Rank 0 receives all results and can continue processing
  if rank == 0:
    print(sum(result))