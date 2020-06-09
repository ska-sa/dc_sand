%% Script that displays the time the packets were transmitted and received.
%% The packets are colour coded by transmitter id, allowing the user to see
%% if the windows overlap.
%% @Author: Gareth Callanan

clear all
close all
format longEng

fileName = "OverlapTest_W5000_D300.csv";
M = csvread(fileName);

%Metadata has been encoded in the file name
WindowLength_us = str2double(extractBetween(fileName,"_W","_D"));
DeadTime_us = str2double(extractBetween(fileName,"_D","_T"));
NumberOfClients = str2double(extractBetween(fileName,"_T",".csv"));
TotalPacketsCount = M(end,1);
NumberOfWindows = M(end,3);
FirstTimeStamp_ms = M(1,5)*1000;

ClientIndex = M(:,2);
WindowIndex = M(:,3);
RxTime_ms = (M(:,6))*1000 - FirstTimeStamp_ms;
TxTime_ms = (M(:,5))*1000 - FirstTimeStamp_ms;

%The different clients(or transmitters) are sorted here to be plotted in
%different colours.
Client1TxTime_ms = TxTime_ms(ClientIndex == 0); %& (WindowIndex == 0 | WindowIndex == 1 | 1));
Client2TxTime_ms = TxTime_ms(ClientIndex == 1); %& (WindowIndex == 0 | WindowIndex == 1 | 1));
Client3TxTime_ms = TxTime_ms(ClientIndex == 2); %& (WindowIndex == 0 | WindowIndex == 1 | 1));

Client1RxTime_ms = RxTime_ms(ClientIndex == 0); %& (WindowIndex == 0 | WindowIndex == 1 | 1));
Client2RxTime_ms = RxTime_ms(ClientIndex == 1); %& (WindowIndex == 0 | WindowIndex == 1 | 1));
Client3RxTime_ms = RxTime_ms(ClientIndex == 2); %& (WindowIndex == 0 | WindowIndex == 1 | 1));

y1 = zeros(size(Client1RxTime_ms));
y2 = zeros(size(Client2RxTime_ms));
y3 = zeros(size(Client3RxTime_ms));

figure
hold on
grid on

%TX Times are plotted on the line y=1, the 0.002 offsets make it more 
%visually apparent when windows overlap
plot(Client1TxTime_ms,y1+1-0.002,'r.')
plot(Client2TxTime_ms,y2+1-0.00,'b.')
plot(Client3TxTime_ms,y3+1+0.002,'g.')

%RX Times are plotted on the line y=0, the 0.002 offsets make it more 
%visually apparent when windows overlap
h1 = plot(Client1RxTime_ms,y1-0.002,'r.')
h2 = plot(Client2RxTime_ms,y2+0.000,'b.')
h3 = plot(Client3RxTime_ms,y3+0.002,'g.')

%Only need to have legend for either RX time as TX plots have same colour.
legend([h1 h2 h3],'Client1','Client2','Client3')

xlabel("Time(ms)")
yticks([0 1])
ylim([-0.1 1.1])
xlim([0 30])
yticklabels({'Rx Node','Tx Node'})
