#Makefile
CC=gcc
CFLAGS=-lm -lgsl -lgslcblas -lfftw3 


poptrajec_lif: 
	$(CC) -DPARAM=\"param.h\" -DMODE=10 -DDt=0.001 -DDtbin=0.001 -DT_sim=10. pop_trajec.c $(CFLAGS)


mastertrajec_lif: 
	$(CC) -DPARAM=\"param.h\" -DMODE=30 -DDt=0.001 -DDtbin=0.001 -DT_sim=10. pop_trajec.c $(CFLAGS)


netwtrajec_lif: 
	$(CC) -DPARAM=\"param.h\" -DMODE=10 -DDt=0.001 -DDtbin=0.001 -DT_sim=10. netw_trajec.c $(CFLAGS)
