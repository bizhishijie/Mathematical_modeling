%--------------------------------------------------------------------------
%  ��4��  ���ݵĵ����뵼��
%--------------------------------------------------------------------------
% CopyRight��xiezhh


%% examp4.2-1
data1 = dlmread('��������������.txt', ',', [2,0,4,5])
data2 = dlmread('��������������.txt', ',', [7,0,8,2])

%% examp4.2-2
fid = fopen('���ɼ�����.txt');
data1 = textscan(fid,'%d %f %f %d %s','HeaderLines',1,'CollectOutput',1)
data1{2}
fclose(fid);

%% examp4.2-3
fid = fopen('��ʦ��Ϣ����.txt','r');
A = textscan(fid, '%*s %s %*s %d %*s %d %*s %d %*s',...
    'delimiter', ' ', 'CollectOutput',1)
A{1,1}
A{1,2}

%% examp4.2-4
num1 = xlsread('����ͳ�Ƴɼ�.xls', 'A2:H4')
num2 = xlsread('����ͳ�Ƴɼ�.xls', 1, 'A2:H4')
num3 = xlsread('����ͳ�Ƴɼ�.xls', 'Sheet1', 'A2:H4')
[num4,text4] = xlsread('����ͳ�Ƴɼ�.xls', 'Sheet1', 'A2:H4')

%% examp4.2-5
VarName = {'id','Height','Weight','VitalCapacity','ObesityLevels'};
ds = dataset('File','���ɼ�����.txt','VarName',VarName)

%% examp4.2-6
VarName = {'x1','x2','x3','x4','x5','x6','x7','x8'};
ds = dataset('XLSFile','����ͳ�Ƴɼ�.xls','VarName',VarName)

%% examp4.2-7
T = readtable('ѧ����Ϣ����.txt','Delimiter',',','ReadRowNames',true)

%% examp4.2-8
T = readtable('����ͳ�Ƴɼ�.xls','ReadRowNames',true);
T.Properties.VariableNames = {'x1','x2','x3','x4','x5','x6','x7'}

%% examp4.3-1
x = 1:3; 
y = [1 2 3;4 5 6;7 8 9]; 
S = struct('Name',{'л�л�','xzh'},'Age',{20,10});
ds = dataset('XLSFile','����ͳ�Ƴɼ�.xls');
save('SaveDataToFile.mat'); 
save SaveDataToFile.mat;
save('SaveDataToFile1.mat','y','S');
save SaveDataToFile1.mat   y   S 

clear;
load SaveDataToFile.mat

%% examp4.3-2
x = rand(10);
[s,t] = xlswrite('10�����������.xls', x, 2, 'D6:M15')

%% examp4.3-3
x = {1,60101,6010101,'����',63,'';2,60101,6010102,'����',73,'';3,60101,...
    6010103,'������',0,'ȱ��'}
xlswrite('��������.xls', x, 'xiezhh', 'A3:F5')