AC_INIT()
AC_CONFIG_MACRO_DIR([tools/])

AX_CHECK_CUDA()

echo " "
echo "************************** Results of configure **************************"
echo "* CUDA Report:"
echo "    >> NVCC=${NVCC}"
echo "    >> CUDA_CFLAGS=${CUDA_CFLAGS}"
echo "    >> CUDA_LDFLAGS=${CUDA_LDFLAGS}"
echo "**************************************************************************"
echo " "

AC_SUBST(NVCC)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LDFLAGS)

AC_OUTPUT(Makefile)
