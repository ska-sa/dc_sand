all: custom_ibverbs_tx custom_ibverbs_rx

custom_ibverbs_tx: custom_ibverbs_tx.c
	gcc -Wall -g -O2 -o custom_ibverbs_tx  custom_ibverbs_tx.c  -lpthread -libverbs -lrdmacm

custom_ibverbs_rx: custom_ibverbs_rx.c
	gcc -Wall -g -O2 -o custom_ibverbs_rx  custom_ibverbs_rx.c  -lpthread -libverbs -lrdmacm

clean: 
	rm -f custom_ibverbs_rx custom_ibverbs_tx *.o

