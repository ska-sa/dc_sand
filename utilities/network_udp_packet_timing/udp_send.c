
// Client side implementation of UDP client-server model 


#define _GNU_SOURCE //This gives access to the GNU source, specifically needed for the sendmmsg function

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> //Useful functions like sleep,close and getopt
#include <string.h>     //For memset function
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h>  //For networking
#include <sys/time.h> //For timing functions

#include "network_packets.h"

#define SERVER_ADDRESS  "10.100.18.14"//TODO: Make a parameter
#define LOCAL_ADDRESS   "127.0.0.1"//TODO: Make default value when now paramter is provided.
  
// Driver code 
int main() 
{ 
    int iSocketFileDescriptor; 
    struct sockaddr_in     sServAddr; 

    //***** Create sample data to be sent *****
    size_t ulMaximumTransmitBytes = MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psSendBuffer = malloc(ulMaximumTransmitBytes);
    for (size_t i = 0; i < MAXIMUM_NUMBER_OF_PACKETS; i++)
    {
        psSendBuffer[i].sHeader.i32PacketIndex = i;
        psSendBuffer[i].sHeader.i32TrailingPacket = 0;
    }
    
  
    //***** Creating socket file descriptor *****
    if ( (iSocketFileDescriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) 
    { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
  
    memset(&sServAddr, 0, sizeof(sServAddr)); 
      
    // Filling server information 
    sServAddr.sin_family = AF_INET; 
    sServAddr.sin_port = htons(UDP_TEST_PORT);
    sServAddr.sin_addr.s_addr = inet_addr(SERVER_ADDRESS);
      
    int iReceivedBytes, iSockAddressLength; 

    //***** Send Initial Message To Server *****
    struct MetadataPacketClient sHelloPacket = {CLIENT_MESSAGE_HELLO,0};
    sendto(iSocketFileDescriptor, (const struct MetadataPacketClient *)&sHelloPacket, 
        sizeof(struct MetadataPacketClient), 
        MSG_CONFIRM, (const struct sockaddr *) &sServAddr,  
             sizeof(sServAddr)); 
    printf("Hello message sent.\n"); 

    //**** Wait For Response from Server with Configuration Information
    struct MetadataPacketMaster sConfigurationPacket;
    iReceivedBytes = recvfrom(iSocketFileDescriptor, (struct MetadataPacketMaster *)&sConfigurationPacket, 
                sizeof(struct MetadataPacketMaster),  
                MSG_WAITALL, (struct sockaddr *) &sServAddr, 
                &iSockAddressLength);
    if(sConfigurationPacket.u32MetadataPacketCode != SERVER_MESSAGE_CONFIGURATION)
    {
        printf("ERROR: Unexpected Message received from server\n");
        return 1;
    }
    printf("Configuration Message Received from Server\n");

    //***** Wait until the time specified by the server before streaming - this has to be repeated over a number of \
        windows *****
    printf("Waiting Until Specified Time To Transmit Data\n");
    int iNumWindows = sConfigurationPacket.uNumberOfRepeats;

    struct timeval * psStopTime = malloc(sizeof(struct timeval)*iNumWindows);
    struct timeval * psStartTime = malloc(sizeof(struct timeval)*iNumWindows);
    int *  piNumberOfPacketsSentPerWindow = malloc(sizeof(int)*iNumWindows);
    int iNumPacketsSentTotal = 0;

    double dWindowTransmitTime = sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_sec + \
            ((double)(sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_usec))/1000000.0;
    double dTimeBetweenWindows = dWindowTransmitTime + ((double)sConfigurationPacket.i32DeadTime_us) /1000000.0;
    //("%f\n",dTimeBetweenWindows);

    for (size_t i = 0; i < iNumWindows; i++)
    {
        double dTimeToStart_s = sConfigurationPacket.sSpecifiedTransmitStartTime.tv_sec + \
                ((double)sConfigurationPacket.sSpecifiedTransmitStartTime.tv_usec)/1000000.0;
        dTimeToStart_s = dTimeToStart_s + (dTimeBetweenWindows)*i*sConfigurationPacket.uNumClients;
        double dCurrentTime_s = 0;
        do
        {
            gettimeofday(&psStartTime[i], NULL);
            dCurrentTime_s = psStartTime[i].tv_sec + ((double)psStartTime[i].tv_usec)/1000000.0;
        } while(dTimeToStart_s > dCurrentTime_s);
        
        //printf("Window %d: %f\n",i,dTimeToStart_s);
        
        //***** Stream data to server *****
        //This has to take place on a number of occasions
        piNumberOfPacketsSentPerWindow[i] = 0;
        double dTransmittedTime_s = 0;
        double dEndTime = dTimeToStart_s + dWindowTransmitTime;
        //printf("%f %f %f\n",dEndTime,dTimeToStart_s,dWindowTransmitTime);
        do
        {
            gettimeofday(&psSendBuffer[iNumPacketsSentTotal].sHeader.sTransmitTime, NULL);
            psSendBuffer[iNumPacketsSentTotal].sHeader.i32TransmitWindowIndex = i;
            psSendBuffer[iNumPacketsSentTotal].sHeader.i32ClientIndex = sConfigurationPacket.i32ClientIndex;

            dTransmittedTime_s = (double)psSendBuffer[iNumPacketsSentTotal].sHeader.sTransmitTime.tv_sec + \
                    ((double)(psSendBuffer[iNumPacketsSentTotal].sHeader.sTransmitTime.tv_usec))/1000000.0;

            //printf("%d %d \n",iNumPacketsSentTotal,psSendBuffer[iNumPacketsSentTotal].sHeader.i32TrailingPacket);

            int temp = sendto(iSocketFileDescriptor, (const char *)&psSendBuffer[iNumPacketsSentTotal], 
                    sizeof(struct UdpTestingPacket), 
                    0, (const struct sockaddr *) &sServAddr,  
                    sizeof(sServAddr)); 
            piNumberOfPacketsSentPerWindow[i]++;
            iNumPacketsSentTotal++;
            if(temp != sizeof(struct UdpTestingPacket))
            {
                printf("Error Transmitting Data: %d",temp);
                return 1;
            }
            //printf("Sent Packet %ld %d.\n",i,temp); 
        }while(dTransmittedTime_s < dEndTime);

        gettimeofday(&psStopTime[i], NULL);
    }

    sleep(sConfigurationPacket.fWaitAfterStreamTransmitted_s);
    for (size_t i = 0; i < iNumWindows; i++)
    {
        double dTimeTaken_s = (psStopTime[i].tv_sec - psStartTime[i].tv_sec) + 
                ((double)(psStopTime[i].tv_usec - psStartTime[i].tv_usec))/1000000;
        int iTotalTransmitBytes = piNumberOfPacketsSentPerWindow[i]*sizeof(struct UdpTestingPacket);
        double dDataRate_Gibps = ((double)iTotalTransmitBytes)*8.0/dTimeTaken_s/1024.0/1024.0/1024.0;
        printf("Window %ld\n",i);
        printf("\tIt took %f seconds to transmit %d bytes of data(%d packets)\n", 
                dTimeTaken_s,iTotalTransmitBytes,piNumberOfPacketsSentPerWindow[i]);
        printf("\tData Rate: %f Gibps\n",dDataRate_Gibps); 
    }



    //***** Send Trailing Packets to stop server polling the receive socket *****;
    printf("Transmitting Trailing Packets to server\n");
    struct UdpTestingPacket sTrailingPacket;
    sTrailingPacket.sHeader.i32TrailingPacket = 1;
    sTrailingPacket.sHeader.i32PacketsSent = iNumPacketsSentTotal;
    sTrailingPacket.sHeader.i32ClientIndex = sConfigurationPacket.i32ClientIndex;
    //Transmit a few times to be safe - this is UDP after all
    for (size_t i = 0; i < 5; i++)
    {
        int temp = sendto(iSocketFileDescriptor, (const char *)&sTrailingPacket, sizeof(struct UdpTestingPacket), 
            0, (const struct sockaddr *) &sServAddr,  
            sizeof(sServAddr)); 
        if(temp != sizeof(struct UdpTestingPacket))
        {
            printf("Error Transmitting Data: %d",temp);
            return 1;
        }
    }
    printf("Program Done\n");
 
    free(psStopTime);
    free(psStartTime);
    free(psSendBuffer);
    close(iSocketFileDescriptor); 
    return 0; 
} 
