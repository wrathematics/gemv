#ifndef GEMV_TYPE_H
#define GEMV_TYPE_H


#define FLOAT 0
#define DOUBLE 1

// ------------------------------------------------------------
#define TYPE FLOAT
// ------------------------------------------------------------

#if TYPE == FLOAT
  #define REAL float
#elif TYPE == DOUBLE
  #define REAL double
#else
  #error "'REAL' in type.h must be float or double"
#endif


#endif
