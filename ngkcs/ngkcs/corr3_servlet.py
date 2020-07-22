import aiokatcp

class Corr3Servlet(aiokatcp.DeviceServer):
    VERSION = "corr3_servlet-0.1"
    BUILD_STATE = "corr3_servlet-0.1.0"  # What is the convention with these?
    
    def __init__(self, *,  # I'm forcing all these arguments to be named, I find it helps with readability.
                 name: str,
                 n_antennas: int,
                 n_channels: int,
                 host: str,
                 port: int,
                 **kwargs):
        self.name = name
        self.n_antennas = n_antennas
        self.n_channels = n_channels
        super(Corr3Servlet, self).__init__(host=host, port=port, **kwargs)

    async def request_beam_weights(self, ctx, data_stream: str, *weights):
        """Load weights for all inputs on a specified beam data-stream."""
        if len(weights) != self.n_antennas: 
            raise aiokatcp.connection.FailReply(f"{len(weights)} weights received, expected {self.n_antennas}") 
        
        if data_stream != "tied-array-channelised-voltage":
            return  # I wasn't quite clear whether this should be a problem. The way I read the ICD, no error should
                    # be returned, the request should just do nothing.

        for n, weight in enumerate(weights):
            #TODO: Pass the message along to all the individual nodes. Currently just sending an inform.
            ctx.inform(f"Antenna no {n} is being weighted by a factor of {weight.decode()}")
            
        
        #TODO: The ICD says "the explanation describes the current weights applied to the inputs of a specific beam".
        #      I should probably figure out what that looks like, and return appropriately.
        #      The test should probably also assert this.
        return "Beam weights set correctly."
        
        

