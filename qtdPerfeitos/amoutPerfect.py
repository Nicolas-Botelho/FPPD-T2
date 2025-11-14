import numpy as np
from mpi4py import MPI

def amount_perfect(arr: np.ndarray) -> int:
  total = 0

  for elem in arr:
    if is_perfect(elem):
      total += 1
  
  return total


def is_perfect(num: int) -> bool:
  divisors = [1]
  sqrt = num**(0.5)
  i = 2

  if num < 6:
    return False

  while i < sqrt:
    if num%i == 0:
      divisors.append(i)
      divisors.append(num//i)

    i += 1
  
  return sum(divisors) == num

if __name__ == '__main__':
  # Initialize MPI environment with mpi4py
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  n_cores = comm.Get_size()

  # Rank 0 loads the full array
  # Rank 0 also split it into chunks
  if rank == 0:
    array: np.ndarray = np.load('T.npy')
    array_of_arrays = np.array_split(array, n_cores)
  else:
    array_of_arrays = None

  # Distribute the chunks of the array to all ranks
  data: np.ndarray = comm.scatter(array_of_arrays, root=0)

  # Perform computation on each process
  start = 0
  end = data.size

  local_result = amount_perfect(data)
  
  # Send results from all ranks back to rank 0
  result = comm.gather(local_result, root=0)

  # Rank 0 receives all results and can continue processing
  if rank == 0:
    print(sum(result))