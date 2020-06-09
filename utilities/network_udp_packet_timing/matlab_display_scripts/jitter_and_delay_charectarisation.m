%% Script that plots the maximum and minimum jitter per client per window.
%% A client is the same as a transmitter.  
%% @Author: Gareth Callanan

close all
clear all

fileName = strcat("StressTest_20200605_1316_N1000_W5000_D1500_T3.csv");
M = csvread(fileName);

Timestamp = string(1234);
Client1WindowTimes = [];
Client2WindowTimes = [];
Client3WindowTimes = [];

%Metadata encoded in the filename
WindowLength_us = str2double(extractBetween(fileName,"_W","_D"));
DeadTime_us = str2double(extractBetween(fileName,"_D","_T"));
NumberOfClients = str2double(extractBetween(fileName,"_T",".csv"));
TotalPacketsCount = M(end,1);
NumberOfWindows = M(end,3);
FirstTimeStamp_ms = M(1,5)*1000;

%Organise csv file column data
ClientIndex = M(:,2);
WindowIndex = M(:,3);
RxTime_ms = (M(:,6))*1000; % - FirstTimeStamp_ms
TxTime_ms = (M(:,5))*1000; % - FirstTimeStamp_ms
TxRxDiff_ms = (M(:,6)- M(:,5))*1000;

%Script seperates clients and windows, works out the min/mean/max tx
%difference per window and then stores all this aggregated data in an array
for i = 0:NumberOfWindows
    Client1TxRxDiff_ms = TxRxDiff_ms(ClientIndex == 0 & WindowIndex == i);
    Client2TxRxDiff_ms = TxRxDiff_ms(ClientIndex == 1 & WindowIndex == i);
    Client3TxRxDiff_ms = TxRxDiff_ms(ClientIndex == 2 & WindowIndex == i);
    try
        Client1WindowTimes = [Client1WindowTimes; [Timestamp i min(Client1TxRxDiff_ms) mean(Client1TxRxDiff_ms) max(Client1TxRxDiff_ms)]];
        Client2WindowTimes = [Client2WindowTimes; [Timestamp i min(Client2TxRxDiff_ms) mean(Client2TxRxDiff_ms) max(Client2TxRxDiff_ms)]];
        Client3WindowTimes = [Client3WindowTimes; [Timestamp i min(Client3TxRxDiff_ms) mean(Client3TxRxDiff_ms) max(Client3TxRxDiff_ms)]];
    catch ME
        disp(ME.identifier)
    end
end


%Plot the results - one plot per client
figure
hold on

for i = 1:3
    if(i == 1)
        ClientWindowTimes = Client1WindowTimes;
    else
        if(i == 2)
            ClientWindowTimes = Client2WindowTimes;
        else
            if(i == 3)
                ClientWindowTimes = Client3WindowTimes;
            end
        end
    end
    
    subplot(3,1,i)
    hold on
    title(strcat("TX Node ",string(i)));
    xlabel("Transfer Window")
    ylabel("Travel Time(ms)")
    
    absoluteMinimum = 0;min(str2double(ClientWindowTimes(:,3)));
    windowIndexVector = str2double(ClientWindowTimes(:,2));
    minVector = str2double(ClientWindowTimes(:,3)) - absoluteMinimum;
    meanVector = str2double(ClientWindowTimes(:,4)) - absoluteMinimum;
    maxVector = str2double(ClientWindowTimes(:,5)) - absoluteMinimum;
    
    plot(windowIndexVector,(minVector),'.-')%Plot Minimum Value
    plot(windowIndexVector,(meanVector),'.-')%Plot Minimum Value
    plot(windowIndexVector,(maxVector),'.-')%Plot Minimum Value
    legend("Min","Mean","Max");
    grid on
end