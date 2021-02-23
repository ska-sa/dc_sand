all: ibverbs_tx ibverbs_rx

ibverbs_tx: ibverbs_tx.c common_functions.h
	gcc -Wall -g -O2 -o ibverbs_tx  ibverbs_tx.c -lpthread -libverbs

ibverbs_rx: ibverbs_rx.c common_functions.h
	gcc -Wall -g -O2 -o ibverbs_rx  ibverbs_rx.c -lpthread -libverbs

clean: 
	rm -f ibverbs_rx ibverbs_tx *.o

