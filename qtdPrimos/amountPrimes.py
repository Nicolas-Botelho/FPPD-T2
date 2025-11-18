import numpy as np
from mpi4py import MPI

def amount_prime(arr: np.ndarray) -> int:
  total = 0

  for elem in arr:
    if (is_prime(elem)):
      total += 1

  return total


def is_prime(num: int) -> bool:
  if (num < 2):
    return False
  elif (num == 2):
    return True

  limit = num ** 0.5
  i = 2

  while i < limit + 1:
    if num % i == 0:
      return False
    i += 1

  return True

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  if rank == 0:
    arr: np.ndarray = np.load('P.npy')
    chunks = np.array_split(arr, size)
  else:
    chunks = None

  chunk: np.ndarray = comm.scatter(chunks, root=0)

  local_result: int = amount_prime(chunk) 

  result = comm.gather(local_result, root=0)

  if rank == 0:
    print(sum(result))
