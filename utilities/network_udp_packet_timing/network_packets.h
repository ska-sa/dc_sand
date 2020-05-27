
#include <time.h> //Gives access clock time
#include <stdint.h>

#define PACKET_SIZE_BYTES   4096 //4KiB

//UDP Port to use for sending streams of test data
#define UDP_TEST_PORT       8080

//Number of packets to be sent in a stream. TODO: Replace with a configurable command line parameter.
#define NUMBER_OF_PACKETS   1000

//Metadata Packet Client Codes
#define CLIENT_MESSAGE_EMPTY 0
#define CLIENT_MESSAGE_HELLO 1

//Metadata Packet Server Codes
#define SERVER_MESSAGE_CONFIGURATION 2

/** Struct that stores \ref UdpTestingPacket metadata. The header struct is seperate from the packet struct as this 
 *  allows fields to be added to the header without having to manually re-calculate the the size of the 
 *  \ref UdpTestingPacket data payload. The size of the data payload in \ref UdpTestingPacket can instead be set to 
 *  data[PACKET_SIZE_BYTES-sizeof(struct UdpTestingPacketHeader).
 */
struct UdpTestingPacketHeader{
    struct timeval sTransmitTime; //Time first packet in stream was sent
    int32_t i32PacketIndex; //Index of the current packet in stream
    int32_t i32TrailingPacket; //Sometimes UDP streams drop data making it difficult to know when to stop receiving \
    packets without having some sort of timeout/polling mechanism. Both of these incur additional costs/system calls. \
    For high bandwidth streaming this can result in additional packets being lost. For the purposes of this framework, \
    A few packets will be transmitted a long time after the end of the stream with this flag set to not zero. When the \
    receiver detects these signals, it know that sending is complete and to not pole any further.
};

/** Packet that will be transmitted to the server. Contains dummy data and useful header data. See 
 *  \ref UdpTestingPacketHeader for more detail on how the size is kept consistent.
 */
struct UdpTestingPacket{
    struct UdpTestingPacketHeader sHeader; // Header containing all useful data
    uint8_t u8Data[PACKET_SIZE_BYTES-sizeof(struct UdpTestingPacketHeader)]; //Dummy data to be part of the packet
};


/** Metadata packet that will be transmitted out of band from the server to the client for configuring the test
 */
struct MetadataPacketMaster{
    uint32_t u32MetadataPacketCode; //Code specifying the type of metadata within the packet
    struct timeval sSpecifiedTransmitTime; //Time the client must start transmitting data.
    float fWaitAfterStreamTransmitted_ms;
};

/** Metadata packet that will be transmitted out of band from the client to the server for configuring the test
 */
struct MetadataPacketClient{
    uint32_t u32MetadataPacketCode; //Code specifying the type of metadata within the packet
    float fTransmitTimeClient_ms;
};