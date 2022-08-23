%--------------------------------------------------------------------------
%  第4章  数据的导入与导出
%--------------------------------------------------------------------------
% CopyRight：xiezhh


%% examp4.2-1
data1 = dlmread('两段数据与文字.txt', ',', [2,0,4,5])
data2 = dlmread('两段数据与文字.txt', ',', [7,0,8,2])

%% examp4.2-2
fid = fopen('体测成绩数据.txt');
data1 = textscan(fid,'%d %f %f %d %s','HeaderLines',1,'CollectOutput',1)
data1{2}
fclose(fid);

%% examp4.2-3
fid = fopen('教师信息数据.txt','r');
A = textscan(fid, '%*s %s %*s %d %*s %d %*s %d %*s',...
    'delimiter', ' ', 'CollectOutput',1)
A{1,1}
A{1,2}

%% examp4.2-4
num1 = xlsread('概率统计成绩.xls', 'A2:H4')
num2 = xlsread('概率统计成绩.xls', 1, 'A2:H4')
num3 = xlsread('概率统计成绩.xls', 'Sheet1', 'A2:H4')
[num4,text4] = xlsread('概率统计成绩.xls', 'Sheet1', 'A2:H4')

%% examp4.2-5
VarName = {'id','Height','Weight','VitalCapacity','ObesityLevels'};
ds = dataset('File','体测成绩数据.txt','VarName',VarName)

%% examp4.2-6
VarName = {'x1','x2','x3','x4','x5','x6','x7','x8'};
ds = dataset('XLSFile','概率统计成绩.xls','VarName',VarName)

%% examp4.2-7
T = readtable('学生信息数据.txt','Delimiter',',','ReadRowNames',true)

%% examp4.2-8
T = readtable('概率统计成绩.xls','ReadRowNames',true);
T.Properties.VariableNames = {'x1','x2','x3','x4','x5','x6','x7'}

%% examp4.3-1
x = 1:3; 
y = [1 2 3;4 5 6;7 8 9]; 
S = struct('Name',{'谢中华','xzh'},'Age',{20,10});
ds = dataset('XLSFile','概率统计成绩.xls');
save('SaveDataToFile.mat'); 
save SaveDataToFile.mat;
save('SaveDataToFile1.mat','y','S');
save SaveDataToFile1.mat   y   S 

clear;
load SaveDataToFile.mat

%% examp4.3-2
x = rand(10);
[s,t] = xlswrite('10阶随机数矩阵.xls', x, 2, 'D6:M15')

%% examp4.3-3
x = {1,60101,6010101,'陈亮',63,'';2,60101,6010102,'李旭',73,'';3,60101,...
    6010103,'刘鹏飞',0,'缺考'}
xlswrite('测试数据.xls', x, 'xiezhh', 'A3:F5')