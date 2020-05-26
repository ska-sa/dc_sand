
// Server side implementation of UDP client-server model 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h> 
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h>  //For networking

#include "network_packets.h"
  
#define MAXLINE 1024 
  
// Driver code 
int main() { 
    int sockfd; 
    char buffer[MAXLINE]; 
    char *hello = "Hello from server"; 
    struct sockaddr_in servaddr, cliaddr; 

    int iTotalTransmitBytes = NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psReceiveBuffer = malloc(iTotalTransmitBytes);
      
    // Creating socket file descriptor 
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
      
    memset(&servaddr, 0, sizeof(servaddr)); 
    memset(&cliaddr, 0, sizeof(cliaddr)); 
      
    // Filling server information 
    servaddr.sin_family    = AF_INET; // IPv4 
    servaddr.sin_addr.s_addr = INADDR_ANY; 
    servaddr.sin_port = htons(UDP_TEST_PORT); 
      
    // Bind the socket with the server address 
    if ( bind(sockfd, (const struct sockaddr *)&servaddr,  
            sizeof(servaddr)) < 0 ) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
      
    int len, n; 
  
    len = sizeof(cliaddr);  //len is value/resuslt 
  
    // n = recvfrom(sockfd, (char *)buffer, MAXLINE,  
    //             MSG_WAITALL, ( struct sockaddr *) &cliaddr, 
    //             &len); 
    // buffer[n] = '\0'; 

    //printf("Original Message Received\n");

    for (size_t i = 0; i < 1000; i++)
    {
        printf("Waiting for stream\n");
        clock_t t;
        for (size_t i = 0; i < NUMBER_OF_PACKETS; i++)
        {
            if(i == 0){
                t = clock();
            }
            n = recvfrom(sockfd, (const char *)&psReceiveBuffer[i], sizeof(struct UdpTestingPacket),  
                    MSG_WAITALL, ( struct sockaddr *) &cliaddr, 
                    &len); 
            //printf("Received Packet %d %d.\n",i,psReceiveBuffer[i].header.i32PacketIndex); 
        }
        t = clock()-t;
        printf("All Messages Received\n");

        float fTimeTaken_s = ((float)t)/CLOCKS_PER_SEC; // in seconds 
        float fDataRate_Gibps = ((float)iTotalTransmitBytes)*8.0/fTimeTaken_s/1024.0/1024.0/1024.0;
        printf("It took %f seconds to transmit %d bytes of data\n", fTimeTaken_s,iTotalTransmitBytes);
        printf("Data Rate: %f Gibps\n",fDataRate_Gibps); 

        sendto(sockfd, (const char *)hello, strlen(hello),  
        MSG_CONFIRM, (const struct sockaddr *) &cliaddr, 
            len); 
        printf("Hello message sent.\n");  

        printf("\n");
    }

    
      
    return 0; 
} 
