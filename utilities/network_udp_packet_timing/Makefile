all: udp_send udp_receive

udp_send: udp_send.c network_packets.h
	gcc udp_send.c -o udp_send -std=c99

udp_receive: udp_receive.c network_packets.h
	gcc udp_receive.c -o udp_receive -std=c99

clean:
	rm -f udp_receive udp_send
