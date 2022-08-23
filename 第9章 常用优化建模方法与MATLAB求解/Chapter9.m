%--------------------------------------------------------------------------
%  第9章  常用优化建模方法与MATLAB求解
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp9.2-1 线性规划
f = [-12,-14,-13];
A = [1.1, 1.2, 1.4; 0.5, 0.6, 0.6; 0.7, 0.8, 0.6];
b = [4600; 2100; 2500];
Aeq = [];
beq = [];
lb = [0; 0; 0];
[x,fval] = linprog(f,A,b,Aeq,beq,lb)

%% examp9.2-2 混合整数线性规划
f = [8;1];
intcon = 2;
A = [-1,-2;  -4,-1;  2,1];
b = [14; -30; 20];
Aeq = [2, -1];
beq = 3;
lb = [0; 0];
ub = [6; 9];
[x, fval] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub)

%% examp9.3-1 无约束最优化
fun = @(x)exp(x(1))*(4*x(1)^2+2*x(2)^2+4*x(1)*x(2)+2*x(2)+1);
x0 = [-1, 1];
[x,f] = fminsearch(fun,x0)
[x,f] = fminunc(fun,x0)

fun = @(x,y)exp(x).*(4*x.^2+2*y.^2+4*x.*y+2*y+1);
figure
ezmesh(fun, [0,1,-2,0]);
hold on;
plot3(x(1),x(2), f, 'r*', 'MarkerSize', 12);
view(-47,6)

%% examp9.3-2 一元函数极小值点
fun = @(x)exp(-0.1*x).*sin(x).^2-0.5*(x+0.1).*sin(x);
[x1,f1] = fminbnd(fun,-10,10)
[x2,f2] = fminbnd(fun,6,10)

figure
ezplot(fun,[-10,10])
hold on
plot(x1,f1,'ro')
plot(x2,f2,'r*')
grid on

%% examp9.3-3 非线性规划
fun = @(x)exp(x(1))*(4*x(1)^2+2*x(2)^2+4*x(1)*x(2)+2*x(2)+1);
A = [2,1; 3,5];
b = [4; 10];
Aeq = [1,-2];
beq = -1;
x0 = [1, 1];
lb = [0; 0];
ub = [inf; inf];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,@nlinconfun)

%% examp9.4-1 多目标规划（最大最小问题）
minimaxMyfun = @(x)sqrt([(x(1)-1.5)^2+(x(2)-6.8)^2; 
   (x(1)-6.0)^2+(x(2)-7.0)^2;
   (x(1)-8.9)^2+(x(2)-6.9)^2;
   (x(1)-3.5)^2+(x(2)-4.0)^2;
   (x(1)-7.4)^2+(x(2)-3.1)^2]);
x0 = [0.0; 0.0];
% 无约束情形
[x,fval] = fminimax(minimaxMyfun,x0)

% 有约束情形
Aeq = [1,-1];
beq = 2.5;
[x,fval] = fminimax(minimaxMyfun,x0,[],[],Aeq,beq)

%% examp9.4-2 多目标规划（多目标达到问题）
fun = @(x)[2*x(1)+5*x(2); 4*x(1)+x(2)];
goal = [20,12];
weight = [20,0];
A = [-1,-1];
b = -7;
lb = [0,0];
ub = [5,6];
x0 = [2,3];
[x,fval] = fgoalattain(fun,x0,goal,weight,A,b,[],[],lb,ub)

%% examp9.5-1 最小生成树
W = [0 2 3 5 4
    2 0 1 0 4
    3 1 0 4 0	
    5 0 4 0 2
    4 4 0 2 0];
G = graph(W,{'A','B','C','D','E'});
x = [0 -1 -1 1 1];
y = [0 1 -1 -1 1];
figure;
p = plot(G,'XData',x,'YData',y,...
    'EdgeLabel',G.Edges.Weight,...
    'EdgeColor','k');
T = minspantree(G);
highlight(p,T,'NodeColor','r',...
    'EdgeColor','b',...
    'LineStyle','--',...
    'LineWidth',2)
axis off

%% examp9.5-2 最短路
s = [1,1,1,2,2,3,3,3,3,4,5,5,6,6,7];
t = [2,3,4,3,5,4,5,6,7,7,6,8,7,8,8];
w = [2,8,1,6,1,7,5,1,2,9,3,8,4,6,3];
x = [0,1,1,1,3,3,3,4];
y = [1,2,1,0,2,1,0,1];
nodename = {'v1','v2','v3','v4','v5','v6','v7','v8'};
G = graph(s,t,w,nodename);
figure;
p = plot(G,'XData',x,'YData',y,...
    'EdgeLabel',G.Edges.Weight);
axis off

[path1,d] = shortestpath(G,1,8)
highlight(p,path1,'NodeColor','r',...
    'EdgeColor','b',...
    'LineStyle','--',...
    'LineWidth',2)

[TR,D] = shortestpathtree(G,1)

%% examp9.5-3 最短路（选址问题1）
W = [0 3 0 0 0 0 0
     3 0 2 0 18 2.5 0
     0 2 0 6 2 0 0
     0 0 6 0 3 0 0
     0 18 2 3 0 4 0
     0 2.5 0 0 4 0 1.5
     0 0 0 0 0 1.5 0];
nodename = {'v1','v2','v3','v4','v5','v6','v7'};
G = graph(W,nodename);
figure;
plot(G,'EdgeLabel',G.Edges.Weight,...
    'EdgeColor','k',...
    'LineWidth',2,...
    'MarkerSize',8,...
    'NodeFontSize',12,...
    'EdgeFontSize',12)

D = zeros(1,7);
for i = 1:7
    [~,Di] = shortestpathtree(G,i);
    D(i) = max(Di);
end
D

%% examp9.5-4 最短路（选址问题2）
s = [1,2,2,3,3,4,5,6];
t = [2,3,6,4,5,5,6,7];
w = [3,2,4,6,2,1,4,1.5];
a = [3,2,7,1,6,3,4];
nodename = {'v1(3)','v2(2)','v3(7)','v4(1)','v5(6)','v6(3)','v7(4)'};
G = graph(s,t,w,nodename);
figure;
plot(G,'EdgeLabel',G.Edges.Weight,...
    'EdgeColor','k',...
    'LineWidth',2,...
    'MarkerSize',8,...
    'NodeFontSize',12,...
    'EdgeFontSize',12)

D = zeros(1,7);
for i = 1:7
    [~,Di] = shortestpathtree(G,'all',i);
    D(i) = sum(a.*Di);
end
D

%% examp9.5-5 最大流
s = [1,1,2,2,3,3,4,4,4,5,6];
t = [2,4,3,5,5,6,3,6,7,7,7];
w = [6,6,2,3,2,2,3,1,5,4,4];
nodename = {'v1','v2','v3','v4','v5','v6','v7'};
G = digraph(s,t,w,nodename);

figure;
H = plot(G,'EdgeLabel',G.Edges.Weight,...
    'EdgeColor','k',...
    'LineWidth',2,...
    'MarkerSize',8,...
    'NodeFontSize',12,...
    'EdgeFontSize',12)

[mf,GF] = maxflow(G,1,7)
H.EdgeLabel = {};
highlight(H,GF,'EdgeColor','b',...
    'LineStyle','--',...
    'LineWidth',2);
st = GF.Edges.EndNodes;
labeledge(H,st(:,1),st(:,2),GF.Edges.Weight);


%% examp9.6-2 遗传算法
fun = @(x)exp(-0.1*x).*sin(x).^2-0.5*(x+0.1).*sin(x);   % 定义匿名函数
lb = -10;                          % 决策变量下界
ub = 10;                           % 决策变量上界
[x1,fval1] = ga(fun,1,[],[],[],[],lb,ub) % 调用ga函数进行求解

[x2,fval2] = fminbnd(fun,lb,ub)   % 调用fminbnd函数进行求解
x0 = 0;
[x3,fval3] = fmincon(fun,x0,[],[],[],[],lb,ub)   % 调用fmincon函数进行求解

%% examp9.6-3 模拟退火算法
fun = @(x)exp(-0.1*x).*sin(x).^2-0.5*(x+0.1).*sin(x);   % 定义匿名函数
x0 = 0;
lb = -10;                          % 决策变量下界
ub = 10;                           % 决策变量上界
[x,fval] = simulannealbnd(fun,x0,lb,ub) % 调用simulannealbnd函数进行求解

%% examp9.6-4 粒子群算法
fun = @(x)exp(-0.1*x).*sin(x).^2-0.5*(x+0.1).*sin(x);   % 定义匿名函数
nvars = 1;                         % 决策变量个数
lb = -10;                          % 决策变量下界
ub = 10;                           % 决策变量上界
[x,fval] = particleswarm(fun,nvars,lb,ub) % 调用particleswarm函数进行求解


%% examp9.6-5 几种优化算法的对比
rng(2);  % 控制随机数生成器
% 定义目标函数
fun1 = @(x)-(1+x(1)*sin(4*pi*x(1))+x(2)*sin(4*pi*x(2))+...
    sin(6*sqrt(x(1)^2+x(2)^2))/(6*sqrt(x(1)^2+x(2)^2+10^(-16))));
x0 = rand(1,2);
lb = [-1,-1];
ub = [1,1];
A = []; b = []; Aeq = []; beq = [];
[x1,f1] = fmincon(fun1,x0,A,b,Aeq,beq,lb,ub)  % 传统解法
[x2,f2] = ga(fun1,2,A,b,Aeq,beq,lb,ub)  % 遗传算法
[x3,f3] = simulannealbnd(fun1,x0,lb,ub)  % 模拟退火算法
[x4,f4] = particleswarm(fun1,2,lb,ub)  % 粒子群算法

% 绘图
[X,Y] = meshgrid(linspace(-1,1,200));  % 定义网格数据
% 重新定义函数
fun2 = @(x,y)1+x.*sin(4*pi*x)+y.*sin(4*pi*y)+...
    sin(6*sqrt(x.^2+y.^2))./(6*sqrt(x.^2+y.^2+10^(-16)));
Z = fun2(X,Y);  % 计算网格点处函数值
surf(X,Y,Z);  % 绘制曲面
shading interp  % 插值染色
camlight  % 添加光源
hold on;
% 绘制几种解法得出的最优点
h1 = plot3(x1(1),x1(2),fun2(x1(1),x1(2)),'r*'); % 红色星号
h2 = plot3(x2(1),x2(2),fun2(x2(1),x2(2)),'rp'); % 红色五角星
h3 = plot3(x3(1),x3(2),fun2(x3(1),x3(2)),'r>'); % 红色三角
h4 = plot3(x4(1),x4(2),fun2(x4(1),x4(2)),'ro'); % 红色三角
legend([h1,h2,h3,h4],{'内点法','遗传算法','模拟退火算法','粒子群算法'})
xlabel('x');ylabel('y');zlabel('z');

%% examp9.6-6 旅行商问题
% 初始化参数
m = 34; Alpha = 1; Beta = 5; Rho = 0.1; NC_max = 200; Q = 100;
% 读取城市坐标数据
T = readtable('TSP.xlsx');
CityName = T.NAME;  % 获取城市名称
lat = T.Lat; % 获取城市纬度坐标
lon = T.Lon; % 获取城市经度坐标
C = [lon,lat];  % 城市经纬度坐标矩阵
% 调用acotsp函数进行求解
[Shortest_Route,SL] = acotsp(C,NC_max,m,Alpha,Beta,Rho,Q);
subplot(1,2,1)
text(lon,lat,CityName);  % 标记城市名称
Str = CityName(Shortest_Route); % 返回最优路径
cellfun(@(x)fprintf('%s→',x),Str)  % 显示最优路径
fprintf('\n')