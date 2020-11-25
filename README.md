# MPI_MatrixMultiplication
Matrix Multiplication


- N = Square Matrix Size,  K = number of process
- Makes multiplication of A(NXN square matrix) and B(NX1) vector.
- N must be given as parameter.
- Each process creates N/K row in A and B matrix
- B matrix is gathered from all processes
- Each process holds result’s N/K part


![](foto.JPG)