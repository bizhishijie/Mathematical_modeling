%--------------------------------------------------------------------------
%  第11章  排队论方法
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp11.6-1 M/M/1模型
n_arrive = 0:6;
n_time = [14,27,27,18,9,4,1];
tm = [6,18,30,42,54,66,78,90,102,114,135,165,190];
n = [32,22,15,10,6,4,3,2,1,1,1,1,1];
lambda = sum(n_arrive.*n_time)/500
mu = 100/(sum(tm.*n)/60)
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,inf,inf])

%% examp11.6-2 M/M/1/N模型
N = 7;
lambda = 3;
mu = 4;
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,N,inf])

%% examp11.6-3 M/M/1/inf/m模型
m = 6;
lambda = 1/60;
mu = 1/30;
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,inf,m])

%% examp11.6-4 M/G/1模型
lambda = 5.5/48;
mu = 1/8;
VarT = 16;
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,inf,inf],VarT)

%% examp11.6-5 M/M/c模型
c = 3;             % 服务台数目
lambda = 0.9;      % 平均到达率
mu = 0.4;          % 平均服务率
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[c,inf,inf])   % 模型求解

%% examp11.6-6 M/M/c/N模型
c = 3;             % 服务台数目
N = 3;             % 系统容量
lambda = 1/2;      % 平均到达率
mu = 1/3;          % 平均服务率
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[c,N,inf])   % 模型求解

%% examp11.6-7 M/M/c/inf/m模型
c = 3;             % 服务台数目
m = 20;            % 顾客源数目
lambda = 1/60;     % 平均到达率
mu = 1/6;          % 平均服务率
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[c,inf,m])   % 模型求解


%% examp11.7-1 排队模型的随机模拟
lambda = 1/6;                        % 平均到达率
numCust = linspace(50,5000,100);     % 顾客总数向量，包含100个值
[Ls,Lq,Ws,Wq] = deal(zeros(100,50)); % 批量赋初值
% 对提前设定的顾客总数进行循环
for i = 1:numel(numCust)
    n = numCust(i);                  % 第i个顾客总数
    % 对每一个指定的顾客总数，重复50次模拟
    for j = 1:50
        x = exprnd(1/lambda,1,n);    % n个顾客到达间隔
        y = unifrnd(3,6,1,n);        % n个顾客的服务时间
        c = cumsum(x);               % n个顾客的到达时间
        b = zeros(1,n);              % n个顾客开始服务的时间
        e = b;                       % n个顾客结束服务的时间
        ws = b;                      % n个顾客的逗留时间
        wq = b;                      % n个顾客的排队等待时间
        ls = b;                      % 各时刻的队长
        % 通过循环计算每个时刻（或每位顾客）的相关指标
        for k = 1:numel(x)
            if k == 1
                b(k) = c(k);         % 计算第k个顾客开始服务的时间
            else
                b(k) = max(c(k),e(k-1));
            end
            e(k) = b(k) + y(k);      % 计算第k个顾客结束服务的时间
            ws(k) = e(k) - c(k);     % 计算第k个顾客的逗留时间
            wq(k) = b(k) - c(k);     % 计算第k个顾客排队等待的时间
            ls(k) = k - 1 - sum(e(1:k) <= c(k));  % 计算各时刻的队长
        end
        lq = max([ls-1;zeros(1,n)]); % 计算各时刻的等待队长
        
        Ls(i,j) = mean(ls(11:end));  % 计算第i个顾客总数的第j次模拟的平均队长
        Lq(i,j) = mean(lq(11:end));  % 计算第i个顾客总数的第j次模拟的平均等待队长
        Ws(i,j) = mean(ws(11:end));  % 计算第i个顾客总数的第j次模拟的平均逗留时间
        Wq(i,j) = mean(wq(11:end));  % 计算第i个顾客总数的第j次模拟的平均等待时间
    end
end
% 模拟结果的可视化
figure
subplot(2,1,1)
plot(numCust,mean(Ls,2))     % 绘制平均队长曲线
xlabel('到达的顾客总数'); ylabel('平均队长 Ls'); grid on
subplot(2,1,2)
plot(numCust,mean(Ws,2))     % 绘制平均逗留时间曲线
xlabel('到达的顾客总数'); ylabel('平均逗留时间 Ws'); grid on

% 理论解
lambda = 1/6;        % 平均到达率
mu = 1/((3+6)/2);    % 平均服务率
VarT = (6-3)^2/12;   % 服务时间的方差
% 调用自编QueuingSystem函数求理论解
[Ls2,Lq2,Ws2,Wq2,P0] = QueuingSystem(lambda,mu,[1,inf,inf],VarT)