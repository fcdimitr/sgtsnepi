#ifndef _H_MATFILEIO
#define _H_MATFILEIO

#define CODE_VERSION_S1  0
#define CODE_VERSION_S3  1
#define CODE_VERSION_NS3 2

/**
 * Read kNN graph data from MAT-file.
 */
CS_INT readMATdata( cs **C, CS_INT **perm,
                 double **lhss1gold,
                 double **lhss3gold,
                 double **lhsns3gold,
                 double **rhsgold1,
                 double **rhsgold3,
                 double *perplexity,
                 const char* basepath, const char *dataset,
                 const CS_INT datasize, const CS_INT knn,
                 const char* permName,
                 const CS_INT flagSym );





#endif
