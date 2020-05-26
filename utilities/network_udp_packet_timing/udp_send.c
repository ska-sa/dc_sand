
// Client side implementation of UDP client-server model 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h>  //For networking

#include "network_packets.h"

#define MAXLINE         1024 
#define SERVER_ADDRESS  "10.100.101.1"
#define LOCAL_ADDRESS  "127.0.0.1"
  
// Driver code 
int main() { 
    int sockfd; 
    char buffer[MAXLINE]; 
    char *hello = "Hello from client"; 
    struct sockaddr_in     servaddr; 
    int iTotalTransmitBytes = NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psSendBuffer = malloc(iTotalTransmitBytes);
    for (size_t i = 0; i < NUMBER_OF_PACKETS; i++)
    {
        psSendBuffer[i].header.i32PacketIndex = i;
    }
    
  
    // Creating socket file descriptor 
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
  
    memset(&servaddr, 0, sizeof(servaddr)); 
      
    // Filling server information 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_port = htons(UDP_TEST_PORT);
    servaddr.sin_addr.s_addr = inet_addr(LOCAL_ADDRESS);
      
    int n, len; 
      
    // sendto(sockfd, (const char *)hello, strlen(hello), 
    //     MSG_CONFIRM, (const struct sockaddr *) &servaddr,  
    //         sizeof(servaddr)); 
    // printf("Hello message sent.\n"); 
        
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < NUMBER_OF_PACKETS; i++)
    {
        int temp = sendto(sockfd, (const char *)&psSendBuffer[i], sizeof(struct UdpTestingPacket), 
        0, (const struct sockaddr *) &servaddr,  
            sizeof(servaddr)); 
        //printf("Sent Packet %ld %d.\n",i,temp); 
    }
    gettimeofday(&stop, NULL);
    float fTimeTaken_s = (stop.tv_sec - start.tv_sec) + ((float)(stop.tv_usec - start.tv_usec))/1000000;
    double fDataRate_Gibps = ((double)iTotalTransmitBytes)*8.0/fTimeTaken_s/1024.0/1024.0/1024.0;
    printf("It took %f seconds to transmit %d bytes of data(%d packets)\n", fTimeTaken_s,iTotalTransmitBytes,NUMBER_OF_PACKETS);
    printf("Data Rate: %f Gibps\n",fDataRate_Gibps); 

    n = recvfrom(sockfd, (char *)buffer, MAXLINE,  
                MSG_WAITALL, (struct sockaddr *) &servaddr, 
                &len); 
    buffer[n] = '\0'; 
    printf("Server : %s\n", buffer); 
  
    close(sockfd); 
    return 0; 
} 
