# MPI_MatrixMultiplication
Matrix Multiplication


- N = Square Matrix Size,  K = number of process
- Makes multiplication of A(NXN square matrix) and B(NX1) vector.
- N must be given as parameter.
- Example compile command in Windows:       mpiexec -n 2 "YourCompiledFile" 20
- Each process creates N/K row in A and B matrix
- B matrix is gathered from all processes
- Each process holds result’s N/K part
- After multiplication elapsed time and usage memory are calculated.


![Architecture](https://user-images.githubusercontent.com/26605040/100268496-2d69a500-2f66-11eb-8e35-4291a3c90661.JPG)

