
#include <time.h> //Gives access clock time
#include <stdint.h>

//Define the UDP stream packet size
#define PACKET_SIZE_BYTES   4096 //4KiB

//UDP Port to use for sending streams of test data
#define UDP_TEST_PORT       8081

//Number of packets to be sent in a stream. TODO: Replace with a configurable command line parameter.
#define MAXIMUM_NUMBER_OF_PACKETS   5000000

//Metadata packet client codes to fill in MetadataPacketClient.u32MetadataPacketCode
#define CLIENT_MESSAGE_EMPTY 0
#define CLIENT_MESSAGE_HELLO 1

//Metadata packet server codes to fill in MetadataPacketMaster.u32MetadataPacketCode
#define SERVER_MESSAGE_CONFIGURATION 2

/** Struct that stores \ref UdpTestingPacket metadata. The header struct is seperate from the packet struct as this 
 *  allows fields to be added to the header without having to manually re-calculate the the size of the 
 *  \ref UdpTestingPacket data payload. The size of the data payload in \ref UdpTestingPacket can instead be set to 
 *  data[PACKET_SIZE_BYTES-sizeof(struct UdpTestingPacketHeader).
 */
struct UdpTestingPacketHeader{
    struct timeval sTransmitTime; //Time first packet in stream was sent
    int64_t i64PacketIndex; //Index of the current packet in stream
    int64_t i64PacketsSent; //Slightly different from packet index - this can be kept constant in a trailing packet. \
    At the moment this field is left blank in non-trailing packets
    int32_t i32TrailingPacket; //Sometimes UDP streams drop data making it difficult to know when to stop receiving \
    packets without having some sort of timeout/polling mechanism. Both of these incur additional costs/system calls. \
    For high bandwidth streaming this can result in additional packets being lost. For the purposes of this framework, \
    A few packets will be transmitted a long time after the end of the stream with this flag set to not zero. When the \
    receiver detects these signals, it know that sending is complete and to not pole any further.
    int64_t i64TransmitWindowIndex;//A single client transmits over multiple windows. This value indicates the window \
    this packet was tranmitted in
    int32_t i32ClientIndex; //The index of this client within the collection of clients
    
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
    struct timeval sSpecifiedTransmitStartTime; //Time the client must start transmitting data.
    struct timeval sSpecifiedTransmitTimeLength; //Length of time client must transport data. (Not a clock time)
    float fWaitAfterStreamTransmitted_s; //Time to wait before sending trailing packets
    uint32_t uNumberOfRepeats; //Number of times to repeate the tranmit window
    uint32_t uNumClients; //Number of different clients transmitting data to the server.
    uint32_t i32ClientIndex; //The server tells the client what its number is among all the clients
    int32_t i32DeadTime_us; //The amount of space between clients where no data is being transferred
};

/** Metadata packet that will be transmitted out of band from the client to the server for configuring the test. At the
 * moment nothing useful is transmitted, however this struct can easily be extended.
 */
struct MetadataPacketClient{
    uint32_t u32MetadataPacketCode; //Code specifying the type of metadata within the packet
    float fTransmitTimeClient_ms; //Not used currently
};