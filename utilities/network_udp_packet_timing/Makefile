all: udp_send udp_receive

udp_send: udp_send.c network_packets.h
	gcc udp_send.c -o udp_send

udp_receive: udp_receive.c network_packets.h
	gcc udp_receive.c -o udp_receive

clean:
	rm -f udp_receive udp_send
