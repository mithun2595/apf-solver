/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include "lblock.h"
#include <emmintrin.h>
#include <mpi.h>
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);

extern control_block cb;
extern local_block lb;

const int CORNER = 1;
const int PADDING = 2;

enum { TOP = 0, RIGHT, LEFT, BOTTOM };

#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void fill_ghosts(double *E_prev)
{
  int i,j,msgs = 0;
  const int TOP_EDGE = 0;
  const int LEFT_EDGE = 0;
  const int BOTTOM_EDGE = cb.py - 1;
  const int RIGHT_EDGE = cb.px - 1; 

  MPI_Request send[4];
  MPI_Request recv[4];
  MPI_Status  statuses[4];

  // top boundary
  if (lb.pIdx == TOP_EDGE) for (i = (0 + CORNER); i < lb.n + CORNER; i++) E_prev[i] = E_prev[i + 2*(lb.n + PADDING)];
  else {
    MPI_Irecv(&E_prev[CORNER], lb.n, MPI_DOUBLE, lb.rank - cb.px, BOTTOM, MPI_COMM_WORLD, recv + msgs);
    MPI_Isend(&E_prev[lb.n + PADDING+CORNER], lb.n, MPI_DOUBLE, lb.rank - cb.px, TOP, MPI_COMM_WORLD, send + 0);
    msgs++;
  }
  // bottom boundary
  if (lb.pIdx == BOTTOM_EDGE) for (i = (lb.m + CORNER)*(lb.n + PADDING) + CORNER; i < (lb.m + PADDING)*(lb.n + PADDING) - CORNER; i++) E_prev[i] = E_prev[i - 2*(lb.n + PADDING)];
  else {
    MPI_Irecv(&E_prev[(lb.m + CORNER)*(lb.n + PADDING) + CORNER], lb.n, MPI_DOUBLE, lb.rank + cb.px, TOP, MPI_COMM_WORLD, recv + msgs);
    MPI_Isend(&E_prev[lb.m*(lb.n + PADDING) + CORNER], lb.n, MPI_DOUBLE, lb.rank + cb.px, BOTTOM, MPI_COMM_WORLD, send + 1);
    msgs++;
  }
  // left boundary - pack & send.
  if (lb.pIdy == LEFT_EDGE) for (i = lb.n + PADDING; i < (lb.m + CORNER)*(lb.n + PADDING); i += (lb.n + PADDING)) E_prev[i] = E_prev[i + 2];
  else {
    for (i = lb.n + PADDING + 1, j = 0; j < lb.m; i += lb.n + PADDING, j++) lb.send_W[j] = E_prev[i];
    MPI_Irecv(lb.recv_W, lb.m, MPI_DOUBLE, lb.rank - 1, RIGHT, MPI_COMM_WORLD, recv + msgs);
    MPI_Isend(lb.send_W, lb.m, MPI_DOUBLE, lb.rank - 1, LEFT, MPI_COMM_WORLD, send + 2);
    msgs++;
  }
  // right boundary - pack & send.
  if (lb.pIdy == RIGHT_EDGE) for (i = (lb.n + CORNER) + 1*(lb.n + PADDING); i < (lb.n + CORNER) + (lb.m + CORNER)*(lb.n + PADDING); i += (lb.n + PADDING)) E_prev[i] = E_prev[i - 2];
  else {
    for (i = lb.n + (lb.n + PADDING), j = 0; j < lb.m; i += (lb.n + PADDING), j++) lb.send_E[j] = E_prev[i];
    MPI_Irecv(lb.recv_E, lb.m, MPI_DOUBLE, lb.rank + 1, LEFT, MPI_COMM_WORLD, recv + msgs);
    MPI_Isend(lb.send_E, lb.m, MPI_DOUBLE, lb.rank + 1, RIGHT, MPI_COMM_WORLD, send + 3);
    msgs++;
  }
  // waiting before unpacking for left and right
  MPI_Waitall(msgs, recv, statuses);
  if (lb.pIdy != LEFT_EDGE) for (i = lb.n + PADDING, j = 0; i < (lb.m+1)*(lb.n+2); i += lb.n + PADDING, j++) E_prev[i] = lb.recv_W[j];
  if (lb.pIdy != RIGHT_EDGE) for (i = (lb.n + CORNER) + (lb.n + PADDING), j = 0; j < lb.m; i += lb.n + PADDING, j++) E_prev[i] = lb.recv_E[j];
}

// void compute_inner(double *E, double *E_prev, double *R, int start_row, int end_row, int cols, double dt, double alpha) {
//   int i, j;
//    double *R_tmp = R;
//    double *E_tmp = E;
//    double *E_prev_tmp = E_prev;

//   #ifdef FUSED
//     // Solve for the excitation, a PDE
//     for(j = start_row; j <= end_row; j+=(cols+2)) {
//         E_tmp = E + j;
//         E_prev_tmp = E_prev + j;
//         R_tmp = R + j;
//         for(i = 0; i < cols; i++) {
//           E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(cols+2)]+E_prev_tmp[i-(cols+2)]);
//           E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
//           R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
//         }
//     }
//   #else
//     // Solve for the excitation, a PDE
//     for(j = start_row; j <= end_row; j+=(cols+2)) {
//         E_tmp = E + j;
//         E_prev_tmp = E_prev + j;
//         for(i = 0; i < cols; i++) {
//             E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(cols+2)]+E_prev_tmp[i-(cols+2)]);
//         }
//     }

//     // Solve the ODE, advancing excitation and recovery variables to the next timtestep
//     for(j = start_row; j <= end_row; j+=(cols+2)) {
//         E_tmp = E + j;
//         R_tmp = R + j;
//         for(i = 0; i < cols; i++) {
//             E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
//             R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
//         }
//     }
//   #endif
// }

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = lb.m, n=lb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

  int i, j;

 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++) {
    
    // ***** dunno what this is?
      // if (cb.debug && (niter==0)) {
      //   stats(E_prev,m,n,&mx,&sumSq);
      //   double l2norm = L2Norm(sumSq);
      //   repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
      //   if (cb.plot_freq) plotter->updatePlot(E,  -1, m+1, n+1);
      // }
    int i,j;
    if(!cb.noComm) {
      fill_ghosts(E_prev);
    }  

//////////////////////////////////////////////////////////////////////////////

// #define FUSED 1
    // int innerBlockRowStartIndex = 2*((n+2)+1);
    // int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - 2*(n+2) + 1;
    // compute_inner(E, E_prev, R, innerBlockRowStartIndex, innerBlockRowEndIndex, n-1, dt, alpha);

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        E_prev_tmp = E_prev + j;
        R_tmp = R + j;
      	for(i = 0; i < n; i++) {
          E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
          E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
          R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }
    }

    // Solve the ODE, advancing excitation and recovery variables to the next timtestep
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
        for(i = 0; i < n; i++) {
            E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   // if (cb.stats_freq){
   //   if ( !(niter % cb.stats_freq)){
   //      stats(E,m,n,&mx,&sumSq);
   //      double l2norm = L2Norm(sumSq);
   //      repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
   //  }
   // }

   // if (cb.plot_freq){
   //        if (!(niter % cb.plot_freq)){
	  //   plotter->updatePlot(E,  niter, m, n);
   //      }
   //  }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  // return the L2 and infinity norms via in-out parameters
  double reducedSq = 0.0; 

  stats(E_prev, m, n, &mx, &sumSq); 

  if(!cb.noComm) {
    MPI_Reduce(&sumSq, &reducedSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mx, &Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }  
 

  L2 = L2Norm(reducedSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}
