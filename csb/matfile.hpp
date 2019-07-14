#ifndef _H_MATFILE
#define _H_MATFILE

#define CODE_VERSION_S1  0
#define CODE_VERSION_S3  1
#define CODE_VERSION_NS3 2

/**
 * Read kNN graph data from MAT-file.
 */
int readMATdata( cs **C, int **perm,
                 double **lhsgold, double **rhsgold,
                 double *perplexity,
                 const char* basepath, const char *dataset,
                 const long long datasize, const long long knn,
                 const char* permName, const int VERSION,
                 const int flagSym );





#endif
