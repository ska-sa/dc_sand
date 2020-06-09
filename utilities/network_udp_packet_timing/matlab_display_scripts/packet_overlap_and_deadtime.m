%% Script that takes the results from a number of different tests with all
%% the same parameters except for deadtime and plots how the changing
%% deadtime affects the number of packets that overlap.
%%
%% Packet overlap is undesirable in these tests, and as such needs to be
%% characterised so that it can be compensated for.
%% @Author: Gareth Callanan

clear all
close all

%The different deadtime values that have been tested and written to file
DeadTimeArr_W5000_us = [0 100 200 300 500 750 1000 1500 2000 2500 3000 4000 5000];
DeadTimeArr_W10000_us = [0 100 500 1000 2000];

NumberOfOverlapsArr = [];
PercentageOverlapsArr = [];

%Choose which array to use
DeadTimeArr_us = DeadTimeArr_W10000_us;

%Go through each file and extract the number of overlaps that occur
for i = DeadTimeArr_us
    DeadTime_us = i;
    fileName = strcat("OverlapTest_W5000_D",string(DeadTime_us),".csv");
    fileName = strcat("LongWindow_N10_W10000_D",string(DeadTime_us),"_T3.csv");
    %fileName = "NetworkTest_W5000_D2000_T3_LONG.csv"
    M = csvread(fileName);

    WindowLength_us = str2double(extractBetween(fileName,"_W","_D"));
    DeadTime_us = str2double(extractBetween(fileName,"_D","_T"));
    NumberOfClients = str2double(extractBetween(fileName,"_T",".csv"));
    TotalPacketsCount = M(end,1);
    NumberOfWindows = M(end,3)+1;
    FirstTimeStamp_ms = M(1,5)*1000;

    ClientIndex = M(:,2);
    PacketChange = ClientIndex(2:end) - ClientIndex(1:end-1);
    PacketChange(PacketChange ~=0 ) = 1;
    RxTime_ms = (M(:,6))*1000 - FirstTimeStamp_ms;

    %The boundaries between windows are also counted as overlaps in the
    %above logic. This is incorrect and fixed in this line
    NumberOfOverlaps = sum(PacketChange) - NumberOfWindows*NumberOfClients;
    PercentageOverlap = NumberOfOverlaps/TotalPacketsCount*100;

    NumberOfOverlapsArr = [NumberOfOverlapsArr NumberOfOverlaps];
    PercentageOverlapsArr = [PercentageOverlapsArr PercentageOverlap];
end

%Plot the number of overlapping packets as a function of deadtime.
figure
plot(DeadTimeArr_us,NumberOfOverlapsArr)
ylabel(strcat("Overlapping Packets(out of ",string(TotalPacketsCount),')'));
xlabel("Deadtime(us)")
grid on

figure
plot(DeadTimeArr_us,PercentageOverlapsArr)
grid on
ylabel("Overlap(%)")
xlabel("Deadtime(us)")