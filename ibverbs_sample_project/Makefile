all: custom_ibverbs_tx custom_ibverbs_rx

custom_ibverbs_tx: custom_ibverbs_tx.c network_packet.h
	gcc -Wall -g -O2 -o custom_ibverbs_tx  custom_ibverbs_tx.c -lpthread -libverbs

custom_ibverbs_rx: custom_ibverbs_rx.c network_packet.h
	gcc -Wall -g -O2 -o custom_ibverbs_rx  custom_ibverbs_rx.c -lpthread -libverbs

clean: 
	rm -f custom_ibverbs_rx custom_ibverbs_tx *.o

