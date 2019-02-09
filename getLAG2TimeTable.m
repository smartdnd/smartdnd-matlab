function [timetable] = getLAG2TimeTable(filepath, signalname)
T1 = readtable(filepath);
A1 = table2array(T1);
Time1 = seconds(A1(:,1)./1000);
VarNames = {signalname,'LeftTurn','RightTurn','RoadBump'};
Signal1 = sqrt(A1(:,5).^2 + A1(:,6).^2 + A1(:,7).^2);
SignalLT = A1(:,2);
SignalRT = A1(:,3);
SignalRB = A1(:,4);
timetable = array2timetable([Signal1,SignalLT,SignalRT,SignalRB],'RowTimes',Time1,'VariableNames',VarNames);
end