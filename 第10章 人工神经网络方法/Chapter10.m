%--------------------------------------------------------------------------
%  第10章  神经网络方法
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp10.5-1 BP网络数据拟合
HeadData = xlsread('头围.xls'); 
x = HeadData(:, 4)'; 
y = HeadData(:, 9)';
rng(0)
net = fitnet(3);
trainedNet = train(net,x,y);
view(trainedNet)
xnew = linspace(0,18,50);
ynew = trainedNet(xnew);
plot(x,y,'.',xnew,ynew,'k')
xlabel('年龄(x)');
ylabel('头围(y)');
trainedNet.IW{1}
trainedNet.LW{2,1}
trainedNet.b

%% examp10.6-1 SOM网络聚类
% 1. 读取数据
[data,TextData] = xlsread('2016各地区月平均气温.xls','A2:M32');
ObsLabel = TextData(:,1);
data = data';
% 2. 创建SOM网络进行聚类
net = selforgmap([3,1]);
trainedNet = train(net,data);
view(trainedNet)
plotsomtop(trainedNet)
y = trainedNet(data)
% 3. 查看聚类结果
classid = vec2ind(y);
ObsLabel(classid == 1)  % 查看第一类中包含的城市
ObsLabel(classid == 2)  % 查看第二类中包含的城市
ObsLabel(classid == 3)  % 查看第三类中包含的城市

%% examp10.7-1 BP网络模式识别
[data1,textdata1] = xlsread('神经元分类识别.xlsx','附录A');
[data2,textdata2] = xlsread('神经元分类识别.xlsx','附录B');
[data3,textdata3] = xlsread('神经元分类识别.xlsx','附录C');
trainData = data1(:,3:end)';
n1 = size(trainData,2);
trainGroup = textdata1(2:end,2);
[Gid,Gname] = grp2idx(trainGroup);
Gid = full(ind2vec(Gid'));
net = patternnet(41);
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 0/100;
sampleData = data2(:,3:end)';
n2 = size(sampleData,2);
testData = data3(:,3:end)';
n3 = size(testData,2);

m = 20;
trainResult = zeros(7,n1,m);
sampleResult = zeros(7,n2,m);
testResult = zeros(7,n3,m);
for i = 1:m
    trainedNet = train(net,trainData,Gid);
    trainResult(:,:,i) = trainedNet(trainData);
    sampleResult(:,:,i) = trainedNet(sampleData);
    testResult(:,:,i) = trainedNet(testData);
end
trainResult = mean(trainResult,3);
sampleResult = mean(sampleResult,3);
testResult = mean(testResult,3);

plotconfusion(Gid,trainResult)
testGroup = Gname(vec2ind(testResult))
sampleGroup = Gname(vec2ind(sampleResult))