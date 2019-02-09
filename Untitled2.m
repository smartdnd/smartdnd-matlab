% FrontLeft = getLAG2TimeTable( '../Data/1 - Peogeot/1541856722367-FrontLeft-laData.csv' ,'FrontLeft');
% FrontRight = getLAG2TimeTable('../Data/1 - Peogeot/1541857463925-FrontRight-laData.csv','FrontRight');
% BackLeft = getLAG2TimeTable(  '../Data/1 - Peogeot/1541857476500-BackLeft-laData.csv'  ,'BackLeft');
% BackRight = getLAG2TimeTable( '../Data/1 - Peogeot/1541857473424-BackRight-laData.csv' ,'BackRight');


% FrontLeft = SensorKineticsLinearAcc2TimeTable( '../Data/2 - Infinity/frontleft-lac.csv' ,'FrontLeft');
FrontRight = SensorKineticsLinearAcc2TimeTable('../Data/2 - Infinity/frontright-lac.csv','FrontRight');
% BackLeft = SensorKineticsLinearAcc2TimeTable(  '../Data/2 - Infinity/backleft-lac.csv'  ,'BackLeft');
% BackRight = SensorKineticsLinearAcc2TimeTable( '../Data/2 - Infinity/backright-lac.csv' ,'BackRight');

Markings = getLAG2TimeTable( '../Data/2 - Infinity/1543157423545-FrontLeft-laData.csv' ,'Markings');

% FrontRight = SensorKineticsLinearAcc2TimeTable('2-lac.csv','FrontRight');