typedef struct _local_block {
    
    int rank;
    int pIdx, pIdy;
    int m,n;
    double *send_W, *recv_E, *recv_W, *send_E;

} local_block;