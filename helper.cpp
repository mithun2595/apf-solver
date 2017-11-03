/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>

using namespace std;

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

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
#if 0
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif
}

double *alloc1D(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}
