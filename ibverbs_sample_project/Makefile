custom_ibverbs_tx: custom_ibverbs_tx.c
	gcc -Wall -g -O2 -o custom_ibverbs_tx  custom_ibverbs_tx.c  -lpthread -libverbs -lrdmacm

custom_ibverbs_rx: custom_ibverbs_rx.c
	gcc -Wall -g -O2 -o custom_ibverbs_rx  custom_ibverbs_rx.c  -lpthread -libverbs -lrdmacm

#rping: rping.c
#	gcc -Wall -g -O2 -o rping  rping.c  -lpthread -libverbs -lrdmacm

#rc_pingpong: rc_pingpong.c pingpong.c pingpong.h
#	gcc -Wall -g -O2 -o rc_pingpong  rc_pingpong.c pingpong.c  -libverbs -lpthread -lrdmacm


clean: 
	rm -f rping rc_pingpong custom_ibverbs_rx custom_ibverbs_tx *.o

