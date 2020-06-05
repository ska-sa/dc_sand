
// Server side implementation of UDP client-server model 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>     //Useful functions like sleep,close and getopt
#include <getopt.h>     //Useful functions for parsing command line parameters
#include <string.h>     //For memset function
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h> //For networking
#include <sys/time.h>   //For timing functions

#include "network_packets.h"
  
#define TRANSMIT_WINDOW_US_DEFAULT 1000 
#define DEAD_TIME_US_DEFAULT 100 
#define TOTAL_WINDOWS_PER_CLIENT_DEFAULT 3 
#define TOTAL_CLIENTS_DEFAULT 2 


int parse_cmd_parameters(
        int argc, 
        char *argv[],
        char ** pi8OutputFileName,
        uint32_t * u32TransmitWindowLength_us,
        uint32_t * u32DeadTime_us,
        uint32_t * u32TransmitWindowsperClient,
        uint32_t * u32TotalClients,
        uint8_t * u8NoTerminal);

int calculate_metrics(
        struct timeval sStopTime, 
        struct timeval sStartTime, 
        struct UdpTestingPacket * psReceiveBuffer, 
        struct timeval * psRxTimes ,
        int64_t i64ReceivedPacketsCount, 
        int64_t i64TotalSentPackets,
        char * pi8OutputFileName,
        uint8_t u8NoTerminal);

// Driver code 
int main(int argc, char *argv[]) 
{ 
    printf("Funnel In Test Server.\n\n");
    
    //***** Parse command line arguments and set up initial server *****
    uint32_t u32TransmitWindowLength_us = TRANSMIT_WINDOW_US_DEFAULT;
    uint32_t u32DeadTime_us = DEAD_TIME_US_DEFAULT;
    uint32_t u32TransmitWindowsperClient = TOTAL_WINDOWS_PER_CLIENT_DEFAULT;
    uint32_t u32TotalClients = TOTAL_CLIENTS_DEFAULT;
    uint8_t  u8NoTerminal = 0; 
    char * pu8OutputFileName = "FunnelInTesting";
    
    //*pu8OutputFileName="temp";

    //printf("%s\n",pu8OutputFileName);

    int iRet = parse_cmd_parameters(argc, argv, &pu8OutputFileName, &u32TransmitWindowLength_us, 
            &u32DeadTime_us, &u32TransmitWindowsperClient, &u32TotalClients, &u8NoTerminal);
    if(iRet != 0){
        return 0;
    }
    
    int iSocketFileDescriptor; 
    struct sockaddr_in sServAddr, sCliAddr; 

    //Allocate buffer of data to be transferred 
    size_t ulTotalTransmitBytes = MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psReceiveBuffer = malloc(ulTotalTransmitBytes);
    struct timeval * psRxTimes = malloc(MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket));
    
    //***** Creating socket file descriptor *****
    if ( (iSocketFileDescriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) 
    { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
    
    memset(&sServAddr, 0, sizeof(sServAddr)); 
    memset(&sCliAddr, 0, sizeof(sCliAddr)); 
    
    // Filling server information 
    sServAddr.sin_family    = AF_INET; // IPv4 
    sServAddr.sin_addr.s_addr = INADDR_ANY; 
    sServAddr.sin_port = htons(UDP_TEST_PORT); 
    
    // Bind the socket with the server address 
    if ( bind(iSocketFileDescriptor, (const struct sockaddr *)&sServAddr,  
            sizeof(sServAddr)) < 0 ) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    
    int iSockAddressLength, iReceivedBytes; 
    iSockAddressLength = sizeof(sCliAddr);
  
    //***** Waiting for initial hello messages from clients *****
    struct sockaddr_in * psCliAddrInit = malloc(u32TotalClients*sizeof(struct sockaddr_in));
    memset(psCliAddrInit, 0, sizeof(struct sockaddr_in)*u32TotalClients);
    for (size_t i = 0; i < u32TotalClients; i++)
    {
        
        printf("Waiting For Hello Message From Client %ld of %d\n",i+1,u32TotalClients);
        struct MetadataPacketClient sHelloPacket = {CLIENT_MESSAGE_EMPTY,0};
        uint8_t u8Duplicate = 1;

        while(sHelloPacket.u32MetadataPacketCode != CLIENT_MESSAGE_HELLO || u8Duplicate != 0)
        {
            iReceivedBytes = recvfrom(iSocketFileDescriptor, (struct MetadataPacketClient *)&sHelloPacket, 
                        sizeof(struct MetadataPacketClient),  
                        MSG_WAITALL, ( struct sockaddr *) &psCliAddrInit[i], 
                        &iSockAddressLength); 
            printf("Message Received\n");
            //Check that the message has not been received from a server that already exists
            u8Duplicate = 0;
            for (size_t j = 0; j < i; j++)
            {
                if(psCliAddrInit[i].sin_addr.s_addr == psCliAddrInit[j].sin_addr.s_addr){
                    printf("Hello message Already received from client with this address\n");
                    u8Duplicate = 1;
                }
                //printf("%d %d\n",psCliAddrInit[i].sin_addr.s_addr,psCliAddrInit[j].sin_addr.s_addr);
            }
        }
        printf("Hello Message Received from client %ld\n",i+1);
    }
        
    //***** Determine and send Configuration Information to client *****
    printf("Sending Configuration Message to client\n");
    struct timeval sCurrentTime;
    gettimeofday(&sCurrentTime,NULL);

    for (size_t i = 0; i < u32TotalClients; i++)
    {
        struct MetadataPacketMaster sConfigurationPacket;
        sConfigurationPacket.u32MetadataPacketCode = SERVER_MESSAGE_CONFIGURATION;
        sConfigurationPacket.sSpecifiedTransmitStartTime.tv_sec = sCurrentTime.tv_sec + 3;
        sConfigurationPacket.sSpecifiedTransmitStartTime.tv_usec = i * (u32TransmitWindowLength_us + 
                u32DeadTime_us);
        sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_sec = 0;
        sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_usec = u32TransmitWindowLength_us;
        sConfigurationPacket.i32DeadTime_us = u32DeadTime_us;
        sConfigurationPacket.uNumberOfRepeats = u32TransmitWindowsperClient;
        sConfigurationPacket.uNumClients = u32TotalClients;
        sConfigurationPacket.fWaitAfterStreamTransmitted_s = 1;
        sConfigurationPacket.i32ClientIndex = i;

        sendto(iSocketFileDescriptor, (const struct MetadataPacketMaster *)&sConfigurationPacket, \
            sizeof(struct MetadataPacketMaster),  
            MSG_CONFIRM, (const struct sockaddr *) &psCliAddrInit[i], 
                iSockAddressLength); 
    }
    
    

    //***** Wait For Data stream messages from client *****
    uint8_t * pu8TrailingPacketReceived = (uint8_t*) malloc(u32TotalClients*sizeof(uint8_t));
    memset(pu8TrailingPacketReceived,0,u32TotalClients*sizeof(uint8_t));
    int * piTotalSentPacketsPerClient = (int *) malloc(u32TotalClients*sizeof(int));
    memset(piTotalSentPacketsPerClient,0,u32TotalClients*sizeof(int));
    

    printf("Waiting for stream\n");
    int64_t i64ReceivedPacketsCount = 0;
    struct timeval sStopTime, sStartTime;
    gettimeofday(&sStartTime, NULL);
    while(1)//Keep waiting for data until trailing packets have been received
    {  
        iReceivedBytes = recvfrom(iSocketFileDescriptor, (char *)&psReceiveBuffer[i64ReceivedPacketsCount], 
                sizeof(struct UdpTestingPacket), MSG_WAITALL, ( struct sockaddr *) &sCliAddr, 
                &iSockAddressLength); 
        if(iReceivedBytes != sizeof(struct UdpTestingPacket))
        {
            printf("****** More than a single packet was received: %d *****",iReceivedBytes);
            return 1;
        }

        //This if-statement confirms end condition has been received - I think a better way to do this in the \
        future would be to just wait until a set time has passed and then send out of band messages to the clients \
        asking for meta data. This is not worth changing for now.
        if(psReceiveBuffer[i64ReceivedPacketsCount].sHeader.i32TrailingPacket != 0)
        {
            int iClientIndex = psReceiveBuffer[i64ReceivedPacketsCount].sHeader.i32ClientIndex;
            pu8TrailingPacketReceived[iClientIndex] = 1;
            piTotalSentPacketsPerClient[iClientIndex] = 
                    psReceiveBuffer[i64ReceivedPacketsCount].sHeader.i32PacketsSent;
            printf("Trailing packet received indicating client %d has finished transmitting.\n",
                    psReceiveBuffer[i64ReceivedPacketsCount].sHeader.i32ClientIndex);
            
            uint8_t u8End = 1;
            for (size_t i = 0; i < u32TotalClients; i++)
            {
                if(pu8TrailingPacketReceived[i] == 0){
                    u8End = 0;
                    break;
                }
            }
            
            if(u8End == 1){
                break;
            }
            continue;
        }

        if(i64ReceivedPacketsCount == 0)
        {
            gettimeofday(&sStartTime, NULL);
        }
        gettimeofday(&psRxTimes[i64ReceivedPacketsCount], NULL);
        i64ReceivedPacketsCount++;//Not counted in the case of a trailing packet
    }

    printf("All Messages Received\n");
    int64_t i64TotalSentPackets = 0;
    for (size_t i = 0; i < u32TotalClients; i++)
    {
        i64TotalSentPackets += piTotalSentPacketsPerClient[i];
    }
    
    sStopTime = psRxTimes[i64ReceivedPacketsCount-1]; //Set stop time equal to last received packet - not simply \
    getting system time here as trailing packets can take quite a while to arrive.

    //***** Analyse data, and calculate and display performance metrics *****
    calculate_metrics(sStopTime,sStartTime,psReceiveBuffer,
            psRxTimes,i64ReceivedPacketsCount,i64TotalSentPackets,pu8OutputFileName,u8NoTerminal);

    //Per for loop cleanup
    free(pu8TrailingPacketReceived);
    free(piTotalSentPacketsPerClient);
    free(psCliAddrInit);

    //Cleanup
    free(psReceiveBuffer);
    free(psRxTimes);
    close(iSocketFileDescriptor);

    return 0;
} 

int calculate_metrics(
        struct timeval sStopTime, 
        struct timeval sStartTime, 
        struct UdpTestingPacket * psReceiveBuffer, 
        struct timeval * psRxTimes, 
        int64_t i64ReceivedPacketsCount, 
        int64_t i64TotalSentPackets,
        char * pi8OutputFileName,
        uint8_t u8NoTerminal)
{
    FILE *pCsvFile;
    FILE *pTextFile;

    char * pi8OutputFileNameCsv = (char *) malloc(1 + strlen(pi8OutputFileName)+ strlen(".csv") );
    char * pi8OutputFileNameTxt = (char *) malloc(1 + strlen(pi8OutputFileName)+ strlen(".txt") );
    strcat(pi8OutputFileNameCsv,pi8OutputFileName);
    strcat(pi8OutputFileNameCsv,".csv");
    strcat(pi8OutputFileNameTxt,pi8OutputFileName);
    strcat(pi8OutputFileNameTxt,".txt");
    pCsvFile = fopen(pi8OutputFileNameCsv,"w");
    pTextFile = fopen(pi8OutputFileNameTxt,"w");
   
    float fTimeTaken_s = (sStopTime.tv_sec - sStartTime.tv_sec) + 
            ((float)(sStopTime.tv_usec - sStartTime.tv_usec))/1000000;
    double fDataRate_Gibps = ((i64ReceivedPacketsCount)*sizeof(struct UdpTestingPacket))
            *8.0/fTimeTaken_s/1024.0/1024.0/1024.0;

    double dRxTime_prev = (double)psRxTimes[0].tv_sec + ((double)psRxTimes[0].tv_usec)/1000000.0;
    double dTxTime_prev = (double)psReceiveBuffer[0].sHeader.sTransmitTime.tv_sec + 
            ((double)psReceiveBuffer[0].sHeader.sTransmitTime.tv_usec)/1000000.0;

    double dMinTxRxDiff=1,dMinTxTxDiff=1,dMinRxRxDiff=1;
    double dMaxTxRxDiff=-1,dMaxTxTxDiff=-1,dMaxRxRxDiff=-1;
    double dAvgTxRxDiff=0,dAvgTxTxDiff=0,dAvgRxRxDiff=0;

    int iWindowBoundaries=0;
    uint8_t u8OutOfOrder = 0;
    for (size_t i = 0; i < i64ReceivedPacketsCount; i++)
    {
        if(i != 0 && psReceiveBuffer[i-1].sHeader.i32PacketIndex > psReceiveBuffer[i].sHeader.i32PacketIndex 
                && psReceiveBuffer[i-1].sHeader.i32ClientIndex == psReceiveBuffer[i].sHeader.i32ClientIndex)
        {
            printf("Data received out of order\n");
            fprintf(pTextFile,"Data received out of order\n");
            u8OutOfOrder = 1;
        }

        double dTxTime = (double)psReceiveBuffer[i].sHeader.sTransmitTime.tv_sec 
                + ((double)psReceiveBuffer[i].sHeader.sTransmitTime.tv_usec)/1000000.0;
        double dRxTime = (double)psRxTimes[i].tv_sec + ((double)psRxTimes[i].tv_usec)/1000000.0;

        double dDiffRxTx = dRxTime-dTxTime;
        dAvgTxRxDiff+=dDiffRxTx;
        if(dDiffRxTx < dMinTxRxDiff && dDiffRxTx != 0)
        {
            dMinTxRxDiff = dDiffRxTx;
        }
        if(dDiffRxTx > dMaxTxRxDiff)
        {
            dMaxTxRxDiff = dDiffRxTx;
        }

        double dDiffRxRx = dRxTime-dRxTime_prev;
        dAvgRxRxDiff+=dDiffRxRx;
        if(dDiffRxRx < dMinRxRxDiff && dDiffRxRx != 0)
        {
            dMinRxRxDiff = dDiffRxRx;
        }
        if(dDiffRxRx > dMaxRxRxDiff)
        {
            dMaxRxRxDiff = dDiffRxRx;
        }

        double dDiffTxTx = dTxTime-dTxTime_prev;
        dAvgTxTxDiff+=dDiffTxTx;
        if(dDiffTxTx < dMinTxTxDiff && dDiffTxTx != 0)
        {
            dMinTxTxDiff = dDiffTxTx;
        }
        if(dDiffTxTx > dMaxTxTxDiff)
        {
            iWindowBoundaries++;
            dMaxTxTxDiff = dDiffTxTx;
        }

        //ONly print this if enabled - for long tests, significant time can be wasted here.
        if(u8NoTerminal == 0){
            printf("Packet %ld Client %d Window %d Client Packet ID %d TX %fs, RX %fs, Diff RX/TX %fs, Diff TX/TX %fs, \
                    Diff RX/RX %fs\n",
                    i, psReceiveBuffer[i].sHeader.i32ClientIndex, psReceiveBuffer[i].sHeader.i32TransmitWindowIndex, 
                    psReceiveBuffer[i].sHeader.i32PacketIndex, dTxTime, dRxTime, dDiffRxTx, dDiffTxTx, dDiffRxRx);
        }
        fprintf(pTextFile,"Packet %ld Client %d Window %d Client Packet ID %d TX %fs, RX %fs, Diff RX/TX %fs, \
                Diff TX/TX %fs, Diff RX/RX %fs\n",
                i, psReceiveBuffer[i].sHeader.i32ClientIndex, psReceiveBuffer[i].sHeader.i32TransmitWindowIndex, 
                psReceiveBuffer[i].sHeader.i32PacketIndex, dTxTime, dRxTime, dDiffRxTx, dDiffTxTx, dDiffRxRx);
        fprintf(pCsvFile,"%ld,%d,%d,%d,%f,%f\n",
                i, psReceiveBuffer[i].sHeader.i32ClientIndex, psReceiveBuffer[i].sHeader.i32TransmitWindowIndex,
                psReceiveBuffer[i].sHeader.i32PacketIndex, dTxTime, dRxTime);

        dRxTime_prev = dRxTime;
        dTxTime_prev = dTxTime;
    }
    dAvgTxRxDiff = dAvgTxRxDiff/(i64ReceivedPacketsCount-1);
    dAvgTxTxDiff = dAvgTxTxDiff/(i64ReceivedPacketsCount-1);
    dAvgRxRxDiff = dAvgRxRxDiff/(i64ReceivedPacketsCount-1);

    printf("\n Average Time Between Packets\n");
    fprintf(pTextFile,"\n Average Time Between Packets\n");
    printf("     |  Avg(s) |  Min(s) |  Max(s) |\n");
    fprintf(pTextFile,"     |  Avg(s) |  Min(s) |  Max(s) |\n");
    printf("TX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgTxRxDiff,dMinTxRxDiff,dMaxTxRxDiff);
    fprintf(pTextFile,"TX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgTxRxDiff,dMinTxRxDiff,dMaxTxRxDiff);
    printf("TX/TX|%9.6f|%9.6f|%9.6f|\n",dAvgTxTxDiff,dMinTxTxDiff,dMaxTxTxDiff);
    fprintf(pTextFile,"TX/TX|%9.6f|%9.6f|%9.6f|\n",dAvgTxTxDiff,dMinTxTxDiff,dMaxTxTxDiff);
    printf("RX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgRxRxDiff,dMinRxRxDiff,dMaxRxRxDiff);
    fprintf(pTextFile,"RX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgRxRxDiff,dMinRxRxDiff,dMaxRxRxDiff);
    printf("\n");
    fprintf(pTextFile,"\n");
    printf("It took %f seconds to receive %ld bytes of data (%ld packets)\n", 
           fTimeTaken_s,(i64ReceivedPacketsCount-1)*PACKET_SIZE_BYTES,i64ReceivedPacketsCount-1);
    fprintf(pTextFile,"It took %f seconds to receive %ld bytes of data (%ld packets)\n", 
           fTimeTaken_s,(i64ReceivedPacketsCount-1)*PACKET_SIZE_BYTES,i64ReceivedPacketsCount-1); 
    printf("\n");
    fprintf(pTextFile,"\n");

    if(u8OutOfOrder != 0)
    {
        printf("\n");
        printf("*********Data Received out of order - investigate this. ********\n");
        printf("\n");
        fprintf(pTextFile,"\n*********Data Received out of order - investigate this. ********\n\n");
    }
    else
    {
        printf("\n");
        printf("Data Received in order");
        printf("\n");
        fprintf(pTextFile,"\nData Received in order\n");
    }

    printf("\n");
    printf("%ld of %ld packets received. Drop rate = %.2f %%\n",
            i64ReceivedPacketsCount,i64TotalSentPackets,
            (1-((double)i64ReceivedPacketsCount)/((double)i64TotalSentPackets))*100);
    printf("\n");
    fprintf(pTextFile,"\n%ld of %ld packets received. Drop rate = %.2f %%\n\n",
            i64ReceivedPacketsCount,i64TotalSentPackets,
            (1-((double)i64ReceivedPacketsCount)/((double)i64TotalSentPackets))*100);

    double fDataRateAvg2_Gibps = ((double)sizeof(struct UdpTestingPacket))/dAvgTxTxDiff/1024.0/1024.0/1024.0*8;//*8 is \
    for bit to byte conversion
    printf("Data Rate According to Average Packet Tx Time Difference: %f Gibps\n",fDataRateAvg2_Gibps);
    fprintf(pTextFile,"Data Rate According to Average Packet Tx Time Difference: %f Gibps\n",fDataRateAvg2_Gibps);

    printf("\n");

    fclose(pCsvFile);
    fclose(pTextFile);
}


int parse_cmd_parameters(
        int argc, 
        char * argv[], 
        char ** pi8OutputFileName,
        uint32_t * u32TransmitWindowLength_us,
        uint32_t * u32DeadTime_us,
        uint32_t * u32TransmitWindowsperClient,
        uint32_t * u32TotalClients,
        uint8_t * u8NoTerminal)
{
    int opt; 
      
    // put ':' in the starting of the 
    // string so that program can  
    //distinguish between '?' and ':'  
    while((opt = getopt(argc, argv, ":t:d:hpn:o:w:")) != -1)  
    {  
        switch(opt)  
        {  
            case 'h':  
                printf(
                    "Program for testing network performance when scheduling packet sending across multiple "
                    "transmitters at specific time intervals.\n\n");
                    
                printf(
                    "This program creates the main receiving server. The udp_send program will create a "
                    "transmitter client that will connect to this server. All configuration information from this "
                    "program will be sent to the client.\n\n");

                printf(
                    "There will be NUM_TRANSMITTERS clients. Each tranmsitter will  transmit data for a specific "
                    "WINDOW_LENGTH in microseconds. Each client will transmit while the others wait. There will be "
                    "NUM_WINDOWS interleaved windows for each client. Between each transfer window there will be "
                    "DEAD_TIME microseconds where no client sends data.\n\n");

                printf("Options:\n");
                printf("    -d DEAD_TIME          The amount of dead time between windows.\n");
                printf("    -h                    Print this message and exit.\n");
                printf("    -n NUM_WINDOWS        The number of windows each client will transmit.\n");
                printf("    -o FILE               Write results to FILE.\n");
                printf("    -p                    Disable Print to terminal\n");
                printf("    -t NUM_TRANSMITTERS   The number of clients that will transmit to the host.\n");
                printf("    -w WINDOW_LENGTH      The length of a window in microseconds.\n");
                
                return 1;

            case 'd': 
                *u32DeadTime_us = atoi(optarg);
                printf("Deadtime set to %d us.\n",*u32DeadTime_us); 
                break;
            case 'n':  
                *u32TransmitWindowsperClient = atoi(optarg);
                printf("Each client will transmit %d windows.\n",*u32TransmitWindowsperClient);    
                break;  
            case 'o':  
                *pi8OutputFileName = optarg;
                printf("Output File name set to %s\n",*pi8OutputFileName); 
                break;  
            case 'p': 
                *u8NoTerminal = 1;
                printf("Full terminal output disabled.\n"); 
                break;
            case 't':
                *u32TotalClients = atoi(optarg);
                printf("Number of transmitters set to %d\n",*u32TotalClients);  
                break;  
            case 'w':
                *u32TransmitWindowLength_us = atoi(optarg);
                printf("Transmit window length set to %d us.\n",*u32TransmitWindowLength_us);  
                break;  
            case '?':
                printf("Unknown option: %c\n", optopt); 
                return 1;
        }  
    }  

    // optind is for the extra arguments 
    // which are not parsed 
    for(; optind < argc; optind++){      
        printf("extra arguments: %s\n", argv[optind]); 
        return 1; 
    } 

    return 0; 
}