# MPI_MatrixMultiplication
Matrix Multiplication


- N = Square Matrix Size,  K = number of process
- Makes multiplication of A(NXN square matrix) and B(NX1) vector.
- N must be given as parameter.Example usage: in Windows: mpiexec -n 2 "YourCompiledFile" 20
- Each process creates N/K row in A and B matrix
- B matrix is gathered from all processes
- Each process holds result’s N/K part


![](C:\Users\Anil\git\MPI_MatrixMultiplication\foto.JPG)