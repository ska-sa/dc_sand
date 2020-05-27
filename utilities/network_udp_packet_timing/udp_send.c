
// Client side implementation of UDP client-server model 


#define _GNU_SOURCE //This gives access to the GNU source, specifically needed for the sendmmsg function

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h>  //For networking
#include <sys/time.h> //For timing functions

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

    //***** Create sample data to be sent *****
    int iTotalTransmitBytes = NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psSendBuffer = malloc(iTotalTransmitBytes);
    for (size_t i = 0; i < NUMBER_OF_PACKETS; i++)
    {
        psSendBuffer[i].sHeader.i32PacketIndex = i;
        psSendBuffer[i].sHeader.i32TrailingPacket = 0;
    }
    
  
    //***** Creating socket file descriptor *****
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

    //***** Send Initial Message To Server *****
    struct MetadataPacketClient sHelloPacket = {CLIENT_MESSAGE_HELLO,0};
    sendto(sockfd, (const struct MetadataPacketClient *)&sHelloPacket, sizeof(struct MetadataPacketClient), 
         MSG_CONFIRM, (const struct sockaddr *) &servaddr,  
             sizeof(servaddr)); 
    printf("Hello message sent.\n"); 

    //**** Wait For Response from Server with Configuration Information
    struct MetadataPacketMaster sConfigurationPacket;
    n = recvfrom(sockfd, (struct MetadataPacketMaster *)&sConfigurationPacket, sizeof(struct MetadataPacketMaster),  
                MSG_WAITALL, (struct sockaddr *) &servaddr, 
                &len);
    if(sConfigurationPacket.u32MetadataPacketCode != SERVER_MESSAGE_CONFIGURATION){
        printf("ERROR: Unexpected Message received from server\n");
        return 1;
    }
    printf("Configuration Message Received from Server\n");

    //***** Wait until the time specified by the server before streaming *****
    printf("Waiting Until Specified Time To Transmit Data\n");
    struct timeval stop, start;
    do
    {
        gettimeofday(&start, NULL);
    } while (start.tv_sec < sConfigurationPacket.sSpecifiedTransmitTime.tv_sec);
    
    
    //***** Stream data to server *****
    for (size_t i = 0; i < NUMBER_OF_PACKETS; i++)
    {
        gettimeofday(&psSendBuffer[i].sHeader.sTransmitTime, NULL);
        //printf("%d %d \n",psSendBuffer[i].header.cTransmitTime.tv_sec,psSendBuffer[i].header.cTransmitTime.tv_usec);
        int temp = sendto(sockfd, (const char *)&psSendBuffer[i], sizeof(struct UdpTestingPacket), 
        0, (const struct sockaddr *) &servaddr,  
            sizeof(servaddr)); 
        if(temp != sizeof(struct UdpTestingPacket)){
            printf("Error Transmitting Data: %d",temp);
            return 1;
        }
        //printf("Sent Packet %ld %d.\n",i,temp); 
    }
    gettimeofday(&stop, NULL);
    float fTimeTaken_s = (stop.tv_sec - start.tv_sec) + ((float)(stop.tv_usec - start.tv_usec))/1000000;
    double fDataRate_Gibps = ((double)iTotalTransmitBytes)*8.0/fTimeTaken_s/1024.0/1024.0/1024.0;
    printf("It took %f seconds to transmit %d bytes of data(%d packets)\n", fTimeTaken_s,iTotalTransmitBytes,NUMBER_OF_PACKETS);
    printf("Data Rate: %f Gibps\n",fDataRate_Gibps); 

    //***** Send Trailing Packets to stop server polling the receive socket *****
    sleep(1);
    struct UdpTestingPacket sTrailingPacket;
    sTrailingPacket.sHeader.i32TrailingPacket = 1;
    //Transmit a few times to be safe - this is UDP after all
    for (size_t i = 0; i < 10; i++)
    {
        int temp = sendto(sockfd, (const char *)&sTrailingPacket, sizeof(struct UdpTestingPacket), 
        0, (const struct sockaddr *) &servaddr,  
            sizeof(servaddr)); 
        if(temp != sizeof(struct UdpTestingPacket)){
            printf("Error Transmitting Data: %d",temp);
            return 1;
        }
    }
    

    // n = recvfrom(sockfd, (char *)buffer, MAXLINE,  
    //             MSG_WAITALL, (struct sockaddr *) &servaddr, 
    //             &len); 
    // buffer[n] = '\0'; 
    // printf("Server : %s\n", buffer); 
  
    close(sockfd); 
    return 0; 
} 
