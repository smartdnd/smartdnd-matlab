function [timetable] = SensorKineticsLinearAcc2TimeTable(filepath, signalname)
T1 = readtable(filepath);
A1 = table2array(T1);
Time1 = seconds(A1(:,1));
VarNames = {signalname};
Signal1 = sqrt(A1(:,2).^2 + A1(:,3).^2 + A1(:,4).^2);
timetable = array2timetable(Signal1,'RowTimes',Time1,'VariableNames',VarNames);
end

