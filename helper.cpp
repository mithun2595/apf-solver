/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
#include "cblock.h"
#include "lblock.h"
#ifdef _MPI_
#include <mpi.h>
#endif
#include <malloc.h>

using namespace std;

extern control_block cb;
// local block maintains variables for the current process
local_block lb;

void printMat(const char mesg[], double *E, int m, int n);


void localInit(double *E, double *E_prev, double *R, int m, int n,
	       int yNum, int xNum, int dim_y, int dim_x){
  // yNum start point of this subgrid
  // xNum start point of this subgrid
  // dim_y - dimensions of subgrid in y (row) direction (prepad) 
  // dim_x - dimensions of subgrid in x (col) direction (prepad)
  int absMidPt = (n+1)/2;
  int local_lo = xNum;
  int local_hi = local_lo + dim_x;
  int i;

  //  printf("localInit absMidPt = %d, local_lo = %d\n", absMidPt, local_lo);
  if (absMidPt > local_hi){
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      E_prev[i] = 0.0;
    }
    
  }else if (absMidPt <= local_lo){
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      E_prev[i] = 1.0;
    }

  }else{
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      E_prev[i] = R[i] = 0.0;
    }
    int local_mid = absMidPt - local_lo;
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      int colIndex = i % (dim_x+2);
      if (colIndex == 0 || colIndex == (dim_x+1) || colIndex < (local_mid+1)){
	continue;
      }
      E_prev[i] = 1.0;
    }
  }


  absMidPt = (m+1)/2;
  local_lo= yNum;
  local_hi = local_lo + dim_y;
  if (absMidPt > local_hi){
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      R[i] = 0.0;
    }
  }else if (absMidPt <= local_lo){
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      R[i] = 1.0;
    }
  }else{
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      R[i] = 0.0;
    }
    int local_mid = absMidPt - local_lo;
    for (i=0; i<(dim_x+2)*(dim_y+2); i++){
      int rowIndex = i / (dim_x+2);
      int colIndex = i % (dim_x+2);
      if ((colIndex == 0) || (colIndex == (dim_x+1)) || (rowIndex < (local_mid+1))){
	continue;
      }
      R[i] = 1.0;
    }
  }

  // We only print the meshes if they are small enough
  //  printMat("E_prev",E_prev,dim_y,dim_x);
  //  printMat("R",R,dim_y,dim_x);


}

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;

    int extra_x = n % cb.px;
    int extra_y = m % cb.py;
    int little_m = m / cb.py;
    int little_n = n / cb.px;

    // first row of the R matrix w/o ghost cells in the global problem
    int initial_i = lb.pIdx*little_m + min(lb.pIdx, extra_y);
    // distance of initial_i from the first rows of 1s
    int rem_rows = min((cb.m + 1)/2 - initial_i, lb.m); 

    for (i = 0; i < (lb.m + 2)*(lb.n + 2); i++) {

      int rIdx = i / (lb.n + 2);
      int cIdx = i % (lb.n + 2);

      if (rIdx == 0 || rIdx == lb.m + 1 || rIdx <= rem_rows || cIdx == 0 || cIdx == lb.n + 1) R[i] = 0.0;
      else R[i] = 1.0;
    }

    // first col of the E_prev matrix w/o ghost cells in the global problem
    int initial_j = lb.pIdy*little_n + min(lb.pIdy, extra_x);
    // distance of initial_j from the first cols of 1s
    int rem_cols = min((cb.n + 1)/2 - initial_j, lb.n);

    for (i = 0; i < (lb.m + 2)*(lb.n + 2); i++) {

      int rIdx = i / (lb.n + 2);
      int cIdx = i % (lb.n + 2);

      if (cIdx == 0 || cIdx == lb.n + 1 || cIdx <= rem_cols || rIdx == 0 || rIdx == lb.m + 1) E_prev[i] = 0.0;
      else E_prev[i] = 1.0;
    }

    // We only print the meshes if they are small enough
#if 0
    printMat("E_prev",E_prev,lb.m,lb.n);
    printMat("R",R,lb.m,lb.n);
#endif
}

double *alloc1D(int paddedM,int paddedN){
#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD,&lb.rank);
#endif
    
    lb.pIdx = lb.rank / cb.px;
    lb.pIdy = lb.rank % cb.px;

    int m = paddedM - 2;
    int n = paddedN - 2;

    // no of cells in y direction
    lb.m = m / cb.py + (lb.pIdx < (m % cb.py));
    // no of cells in x direction
    lb.n = n / cb.px + (lb.pIdy < (n % cb.px)); 

    lb.recv_W = new double[4*lb.m];
    lb.recv_E = lb.recv_W + lb.m;
    lb.send_W = lb.recv_E + lb.m;
    lb.send_E = lb.send_W + lb.m;
    
    double *E;
    // Ensures that allocated memory is aligned on a 16 byte boundary
    // plus 2 is the padding to accomodate ghost cells.
    assert(E= (double*) memalign(16, sizeof(double)*(lb.m+2)*(lb.n+2)));
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n)
{
  if (m > 8)
  {
    return;
  }
  printf("%s\n", mesg);

  for (int i = 0; i < (m + 2)*(n + 2); ++i)
  {
    int rIdx = i / (n + 2);
    int cIdx = i % (n + 2);

    // printing along with ghost cells. used for testing.
    if ((cIdx == 0 || cIdx == n + 1) && (rIdx == 0 || rIdx == m+1)) printf("      ");
    else printf("%1.3f ", E[i]); 
    
    if (cIdx == n+1)printf("\n");
  }
}
