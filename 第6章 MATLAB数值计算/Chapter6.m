%--------------------------------------------------------------------------
%  第6章  MATLAB数值计算
%--------------------------------------------------------------------------
% CopyRight：xiezhh


%% examp6.1-1
h = 0.01;
x = 0:h:2*pi;
y = sin(x);

dy_dx1 = diff(y)./diff(x);
dy_dx2 = gradient(y,h);

figure;
plot(x,y);
hold on
plot(x(1:end-1),dy_dx1,'k:');
plot(x,dy_dx2,'r--');
legend('y = sin(x)','导函数曲线（diff）','导函数曲线（gradient）');
xlabel('x'); ylabel('正弦曲线及导函数曲线')

%% examp6.1-2
t1 = linspace(0,2*pi,60);
x1 = cos(t1); y1 = sin(t1);
s1 = abs(trapz(x1,y1))

t2 = linspace(0,2*pi,200);
x2 = cos(t2); y2 = sin(t2);
s2 = abs(trapz(x2,y2))

t3 = linspace(0,2*pi,2000);
x3 = cos(t3); y3 = sin(t3);
s3 = abs(trapz(x3,y3))

%% examp6.1-3
fun1 = @(x)exp(-x.^2);
s1 = integral(fun1,0,1)

fun2 = @(x,y)x.*sqrt(10-y.^2);
yfun1 = @(x)x-2;
yfun2 = @(x)2-sin(x);
s2 = integral2(fun2,-1,2,yfun1,yfun2)

fun3 = @(x,y,z)x.*y.*z;
yfun1 = @(x)x;
yfun2 = @(x)2*x;
zfun1 = @(x,y)x.*y;
zfun2 = @(x,y)2*x.*y;
s3 = integral3(fun3,1,2,yfun1,yfun2,zfun1,zfun2)

%% examp6.1-4
fxy = @(x,y)exp(-x.^2)./(x.^2+y.^2);
fy1 = @(y)integral(@(x)fxy(x,y),-1,1)^2;
fy2 = @(y)arrayfun(@(t)fy1(t),y);
fun1 = @(y)2*y.*exp(-y.^2).*fy2(y);
s1 = integral(fun1,0.2,1)

fun2 = @(x1,x2,x3,x4)exp(x1.*x2.*x3.*x4);
f_x1 = @(x1)integral3(@(x2,x3,x4)fun2(x1,x2,x3,x4),0,1,0,1,0,1);
f_x1 = @(x1)arrayfun(@(t)f_x1(t),x1);
s2 = integral(f_x1,0,1)

%% examp6.2-1
A = [1 -1 2 -3;-2 2 1 1;-1 1 8 -8];
b = [3;-1;6];
x = A\b

%% examp6.2-2
p = [2, -3, 5, -10];
x = roots(p)

%% examp6.2-3
fun = @(x)-x.*sin(5*exp(1-x.^2));
ezplot(fun,[-1 1]);
grid on;
[x,fval] = fzero(fun,[0.2,0.4])
hold on;
plot(x,fval,'ko');

%% examp6.2-4
fun = @(X)[X(1) - X(2) - exp(-X(1)); -X(1) + 2*X(2) - exp(-X(2))];
x0 = [1,1];
options = optimset('Display','iter'); %显示迭代过程
[x,fval] = fsolve(fun,x0,options)

%% examp6.2-5
xyt = [500    3300    21    9
       300     200    19   29
       800    1600    14   51
      1400    2200    13   17
      1700     700    11   46
      2300    2800    14   47
      2500    1900    10   14
      2900     900    11   46
      3200    3100    17   57
      3400     100    16   49];
x = xyt(:,1);
y = xyt(:,2);
Minutes = xyt(:,3);
Seconds = xyt(:,4);
T = Minutes + Seconds/60; 
modelfun = @(b) sqrt((x-b(1)).^2+(y-b(2)).^2+b(3).^2)/(60*b(4))+b(5)-T;
b0 = [1000 100 10 1 1];
options = optimoptions('fsolve','Display','none',...
    'Algorithm','Levenberg-Marquardt');
[Bval,Fval] = fsolve(modelfun,b0,options)

%% examp6.3-1
a = 14;b = 8;c = 10;
f = @(t,x)sqrt((c-x(1))^2+(b*t-x(2))^2); 
fun = @(t,x)[a*(c-x(1))/f(t,x);a*(b*t-x(2))/f(t,x)];
tspan = linspace(0,1.06,100);
x0 = [0;0];
[t,x] = ode45(fun,tspan,x0);

hpoint1 = line(0,0,'Color',[0 0 1],'Marker',...
    '.','MarkerSize',40);
hpoint2 = line(c,0,'MarkerFaceColor',[0 1 0],...
    'Marker','p','MarkerSize',15);
hline = line(0,0,'Color',[1 0 0],'linewidth',2);
line([c c],[0 c],'LineWidth',2);
hcat = text(-0.8,0,'猫','FontSize',12);
hmouse = text(c+0.3,0,'鼠','FontSize',12);
xlabel('X'); ylabel('Y');
axis([0 c+1 0 9.5]);

for i = 1:size(x,1)
    ymouse = t(i)*b;
    set(hpoint1,'xdata',x(i,1),'ydata',x(i,2));
    set(hpoint2,'xdata',c,'ydata',ymouse);
    set(hline,'xdata',x(1:i,1),'ydata',x(1:i,2));
    set(hcat,'Position',[x(i,1)-0.8,x(i,2),0]);
    set(hmouse,'Position',[c+0.3,ymouse,0]);
    pause(0.1);
    drawnow;
end

%% examp6.3-2
fun = @(t,y,mu)[y(2);mu*(1-y(1)^2)*y(2)-y(1)];
tspan = [0,30];%时间区间
y0 = [1 0];
ColorOrder = {'r','b','k'};
LineStyle = { '-','--',':'};
figure(1);ha1 = axes;hold on;
figure(2);ha2 = axes;hold on;
for mu = 1:3
    [t,y] = ode45(fun,tspan,y0,[],mu);
    plot(ha1,t,y(:,1),'color',ColorOrder{mu},'LineStyle',LineStyle{mu});
    plot(ha2,y(:,1),y(:,2),'color',ColorOrder{mu},'LineStyle',LineStyle{mu});
end
xlabel(ha1,'t'); ylabel(ha1,'x(t)');
legend(ha1,'\mu = 1','\mu = 2','\mu = 3');
hold off
xlabel(ha2,'位移'); ylabel(ha2,'速度');
legend(ha2,'\mu = 1','\mu = 2','\mu = 3');
hold off

%% examp6.3-3
fun = @(t,y,dy)[dy(1)-y(2);
                dy(2)*sin(y(4))+dy(4)^2+2*y(1)*y(3)-y(1)*dy(2)*y(4);
                dy(3)-y(4);
                y(1)*dy(2)*dy(4)+cos(dy(4))-3*y(2)*y(3)];

t0 = 0;         % 自变量的初值
y0 = [1;0;0;1]; % 状态变量初值向量y0
% fix_y0用来指定初值向量y0的元素是否可以改变。1表示对应元素不能改变，0为可以改变
fix_y0 = [1;1;1;1]; % 本例中y0的值都给出了，因此都不能改变，所有fix_y0全为1
dy0 = [0;3;1;0];    % 猜测一下一阶导数dy的初值dy0;
% 由于本例中一阶导数dy的初值dy0是猜测的，都可以改变，因此fix_dy0 全部为0
fix_dy0 = [0;0;0;0];
% 调用decic函数来决定y和dy的初值
[y02,dy02] = decic(fun,t0,y0,fix_y0,dy0,fix_dy0);

%求解微分方程
[t,y] = ode15i(fun,[0,5],y02,dy02); % y02和dy02由decic输出
% 结果图示
figure;
plot(t,y(:,1),'k-','linewidth',2);
hold on
plot(t,y(:,2),'k--','linewidth',2);
plot(t,y(:,3),'k-.','linewidth',2);
plot(t,y(:,4),'k:','linewidth',2);
% 图例,位置自动选择最佳位置
L = legend('y_1(t)','y_2(t)','y_3(t)','y_4(t)','Location','best');
set(L,'fontname','Times New Roman');
xlabel('t');ylabel('y(t)');

%% examp6.3-4
lags = [1,3];       % 延迟常数向量
history = [0,0,1];  % 小于初值时的历史函数
tspan = [0,8];      % 时间区间
% 方法一：调用dde23函数求解
sol = dde23(@ddefun,lags,history,tspan); 
% % 方法二：调用ddesd函数求解
% sol = ddesd(@ddefun,lags,history,tspan); 

% 画图呈现结果
plot(sol.x,sol.y(1,:),'k-','linewidth',2);
hold on
plot(sol.x,sol.y(2,:),'k-.','linewidth',2);
plot(sol.x,sol.y(3,:),'k-*','linewidth',1);
hold off
% 图例,位置自动选择最佳位置
L = legend('y_1(t)','y_2(t)','y_3(t)','Location','best');
set(L,'fontname','Times New Roman');   % 设置图例字体
xlabel('t');ylabel('y(t)');            % 添加坐标轴标签

%% examp6.3-6
% 微分方程组所对应的匿名函数
BvpOdeFun  = @(t,y)[y(2)
                    2*y(2)*cos(t)-y(1)*sin(4*t)-cos(3*t)];
% 边界条件所对应的匿名函数。
% 边界条件为 y1(0) = 1, y1(4) = 2，这里0,4分别对应y的下边界和上边界。
% 这里ylow(1)表示y1(0)，yup(1)表示y1(4)，类似的y2(0)和y2(4)分别用ylow(2)和yup(2)表示
BvpBcFun = @(ylow,yup)[ylow(1)-1; yup(1)-2];

T = linspace(0,4,10); % 为调用bvpinit生成初始化网格作准备
% 对状态变量y作出初始假设，由于y1(0) = 1,y1(4) = 2，可选取一个满足上述条件的函数
% y1(t) = 1+t/4来作为对y1(t)的初始假设，从而其导数1/4作为对y2(t)的初始假设
BvpYinit = @(t)[ 1+t/4; 1/4 ];
solinit = bvpinit(T,BvpYinit); % 调用bvpinit函数生成初始解

sol = bvp4c(BvpOdeFun,BvpBcFun,solinit); % 调用bvp4c求解,也可以换成bvp5c
tint = linspace(0,4,100);
Stint = deval(sol,tint); % 根据得到的sol利用deval函数求出[0,4]区间内更多其他的解

% 画图呈现结果
figure;
plot(tint,Stint(1,:),'k-','linewidth',2);
hold on
plot(tint,Stint(2,:),'k:','linewidth',2);
% 图例,位置自动选择最佳位置
L = legend('y_1(t)','y_2(t)','Location','best');
set(L,'fontname','Times New Roman');   % 设置图例字体
xlabel('t');ylabel('y(t)');            % 添加坐标轴标签

%% examp6.4-1
u1 = ones(1,49);
%  根据差分方程构造目标函数（方程组）
objfun = @(u)([u(2:end,:);u1]+[u1;u(1:end-1,:)]+...
       [u(:,2:end),u1']+[0*u1',u(:,1:end-1)])/4-u;
U0 = rand(49);
[Uin,Error] = fsolve(objfun,U0);  % 求解内点温度
U = zeros(size(U0)+2);
U(:,end) = 1;
U(1,:) = 1;
U(end,:) = 1;
U(2:end-1,2:end-1) = Uin;
[X,Y] = meshgrid(linspace(0,1,51));
surf(X,Y,U);
xlabel('X'); ylabel('Y'); zlabel('U(X,Y)');

%% examp6.4-2
U = zeros(100);  % 初值矩阵
t = (1:100)/100; x = t;  % t和x的划分向量
U(1,:) = sin(t);  % 下边界条件
U(end,:) = cos(t);  % 上边界条件
U(:,1) = x;  % 初值条件
b2 = 0.001; dx = 0.01;dt = 0.01;r = b2*dt/dx^2;  % 参数
% 差分方程求解
for j = 1:99
       U(2:99,j+1) = (1-2*r)*U(2:99,j)+r*(U(1:98,j)+U(3:100,j));
end
[T,X] = meshgrid(t);  % 网格矩阵
surf(T,X,U);  % 绘制面图
xlabel('T');  ylabel('X');  zlabel('U(T,X)');  % 坐标轴标签

%% examp6.4-3
u = zeros(301);                 % 定义零矩阵
dt = 1/300; c = 0.03;           % 参数
x = linspace(0,1,301); t = x';  % t和x的划分向量
u(:,1) = x.*(1-x)/10;           % 初值
u(1,:) = sin(t);                % 边值
v = sin(2*pi*x);                % 初速度
% 计算u(i,2)
u(2:300,2) = (1-c)*u(2:300,1) + ...
       1/2*c*(u(1:299,1)+u(3:301,1)) + v(2:300)'*dt;
h = plot(x,u(:,1));             % 绘制初始弦曲线
axis([-0.1,1.1,-1,1]);          % 设置坐标轴范围
xlabel('x');ylabel('U(x)');     % 坐标轴标签
% 用有限差分法求解方程，并动态展示求解结果
for j = 3:301
       u(2:300,j) = 2*(1-c)*u(2:300,j-1)+c*(u(3:301,j-1)+...
           u(1:299,j-1))-u(2:300,j-2);
       set(h,'YData',u(:,j));      % 更新弦上各点位移
       pause(0.1);                 % 暂停0.1秒
end
%text(0.8,0.8,['t = ',num2str((j-1)/300,'%3.2f')])

%% examp6.4-4
% 用有限元法求解波动方程
model = createpde(1);  % 创建包含一个方程的微分方程模型
geometryFromEdges(model,@squareg);  % 创建正方形求解区域
pdegplot(model,'EdgeLabels','on');  % 绘制求解区域
axis([-1.1,1.1,-1.1,1.1]);
axis equal;
applyBoundaryCondition(model,'Edge',[2,4],'g',0);  % 左右边值条件
applyBoundaryCondition(model,'Edge',[1,3],'g',0);  % 上下边值条件
Me = generateMesh(model,'Hmax',0.1,'GeometricOrder','linear');  % 划分网格
pdeplot(model);     % 显示网格图

% 初始条件
u0 = 0;
ut0 = 0;
% 方程参数
c = 0.01;
a = 0;
f = @framp;
d = 1;

tlist = linspace(0,20,61);  % 定义时间向量
u1 = hyperbolic(u0,ut0,tlist,model,c,a,f,d);  % 方程求解

% 结果可视化
p = Me.Nodes;  % 网格点坐标
t = Me.Elements;  % 三角网顶点编号
xyi = linspace(-1,1,30); % xy轴网格划分
% 三角网格转为矩形网格
uxy = tri2grid(p,t,u1(:,1),xyi,xyi); 
figure;
h = surf(xyi,xyi,uxy);  % 绘制面图
axis([-1.1,1.1,-1.1,1.1,-0.8,0.8]);
view(-30,70);  % 设置视点位置
colormap(jet);  % 设置颜色矩阵
shading interp; % 插值染色
%light('pos',[0.6,-0.6,20]);
camlight;  % 加入光源
lighting phong;  % 设置光照模式
xlabel('x');ylabel('y'),zlabel('u');
% 水波扩散的动态展示
for i = 1:numel(tlist)
    uxy = tri2grid(p,t,u1(:,i),xyi,xyi); 
    set(h,'ZData',uxy);  % 更新坐标
    drawnow;
    pause(0.1);
end

%% examp6.4-5
model2 = createpde(1);
geometryFromEdges(model2,@squareg);   % 创建正方形求解区域
pdegplot(model2,'EdgeLabels','on');   % 绘制求解区域，并显示边界标签
ylim([-1.1,1.1]);
axis equal;
% 设置第左右边界的边值条件（Dirichlet 边值条件）
applyBoundaryCondition(model2,'Edge',[2,4],'u',0);
% 设置上下边界的边值条件（Neumann 边值条件）
applyBoundaryCondition(model2,'Edge',[1,3],'g',0);
Me = generateMesh(model2,'Hmax',0.1,'GeometricOrder','linear');  % 划分网格
% 初始条件
u0 = 'atan(cos(pi/2*x))';
ut0 = '3*sin(pi*x).*exp(cos(pi*y))';
% 方程参数
c = 1;
a = 0;
f = 0;
d = 1;
tlist = linspace(0,6,41);                      % 定义时间向量
u1 = hyperbolic(u0,ut0,tlist,model2,c,a,f,d);  % 方程求解

XY = Me.Nodes';                                % 网格点坐标
Tri = Me.Elements';                            % 三角网顶点编号
figure;
h = trisurf(Tri,XY(:,1),XY(:,2),u1(:,1));      % 绘制三角网面图
axis([-1.1,1.1,-1.1,1.1,-3,3]);                % 设置坐标轴范围
xlabel('x');ylabel('y');zlabel('u(x,y)');
for i = 1:numel(tlist)
    set(h,'Vertices',[XY(:,1),XY(:,2),u1(:,i)]);  % 更新坐标
    drawnow;
    pause(0.1);
end

%% examp6.4-6
% 1. 创建包含一个方程的微分方程模型
model3 = createpde(1);

% 2. 创建圆形求解区域
geometryFromEdges(model3,@circleg);
pdegplot(model3,'EdgeLabels','on');   % 绘制求解区域，并显示边界标签 
ylim([-1.1,1.1]);
axis equal;

% 3. 设置边值条件（Dirichlet 边值条件）
NumEdges = model3.Geometry.NumEdges;  % 求解区域边界数
applyBoundaryCondition(model3,'Edge',1:NumEdges,'u',0);

% 4. 划分网格
generateMesh(model3,'Hmax',0.02,'GeometricOrder','linear'); 

% 5. 设置初始条件
p = model3.Mesh.Nodes;  % 三角网顶点坐标
u0 = zeros(size(p,2),1);
% 查找半径为0.4圆域内的点
ix = find(sqrt(p(1,:).^2 + p(2,:).^2) <= 0.4);
u0(ix) = ones(size(ix));

% 6. 设置方程参数
c = 1;
a = 0;
f = 0;
d = 1;

% 7. 方程求解
tlist = linspace(0,0.1,21);  % 定义时间向量
u = parabolic(u0,tlist,model3,c,a,f,d);  % 方程求解

% 8. 结果可视化
figure;
umax = max(max(u));  % 最大温度
umin = min(min(u));  % 最小温度
% 热扩散的动态展示
for i = 1:5%numel(tlist)
    pdeplot(model3,'xydata',u(:,i)); % 绘制温度分布图
    caxis([umin umax]);              % 设置坐标系颜色范围
    axis equal;
    axis([-1.1,1.1,-1.1,1.1]);
    xlabel('x');ylabel('y');
    drawnow;
    pause(0.1);
end

%% examp6.4-7
% 1. 创建包含一个方程的微分方程模型
model4 = createpde;

% 2. 从外部文件导入求解区域的几何描述
importGeometry(model4,'Block.stl'); 
h = pdegplot(model4,'FaceLabels','on'); % 绘制求解区域
h(1).FaceAlpha = 0.5; % 设置透明度值为0.5

% 3. 设置边值条件
% x = 0, x = 100, z = 0, z = 50所对应的边值条件
applyBoundaryCondition(model4,'Face',1:4,'u',0);
% y = 0所对应的边值条件
applyBoundaryCondition(model4,'Face',6,'g',-1);
% y = 20所对应的边值条件
applyBoundaryCondition(model4,'Face',5,'g',1);

% 4. 划分网格
generateMesh(model4);

% 5. 设置方程参数
c = 1;
a = 0;
f = 'log(1+x+y./(1+z))';

% 6. 方程求解
u = assempde(model4,c,a,f);

% 7. 结果可视化
p = model4.Mesh.Nodes;  % 三角网顶点坐标
% 对x,y,z坐标轴进行网格划分
xi = linspace(min(p(1,:)),max(p(1,:)),60);
yi = linspace(min(p(2,:)),max(p(2,:)),60);
zi = linspace(min(p(3,:)),max(p(3,:)),60);
[X,Y,Z] = meshgrid(xi,yi,zi);  % 生成矩形网格数据
% 根据求解结果创建PDEResults对象
result = createPDEResults(model4,u);
% 通过插值计算矩形网格点处函数值
V = interpolateSolution(result,X,Y,Z);
% 把函数值向量转为三维数组
V = reshape(V,size(X));

figure
colormap jet
sxyz = [1:15:60,60];  % 设置切片位置
% 绘制切片图
slice(X,Y,Z,V,xi(sxyz),yi(sxyz),zi(sxyz));
shading interp;  % 插值染色
alpha(0.5);  % 设置透明度
xlabel('x')
ylabel('y')
zlabel('z')
colorbar;  % 添加颜色条
axis equal;

%% examp6.4-8
% 1. 创建包含一个方程的微分方程模型
model5 = createpde;

% 2. 创建圆形求解区域
geometryFromEdges(model5,@circleg);

% 3. 设置边值条件
boundaryfun = @(region,state)region.x.^2;  % 定义边界函数
NumEdges = model5.Geometry.NumEdges;
applyBoundaryCondition(model5,'Edge',1:NumEdges,...
    'u',boundaryfun,'Vectorized','on');

% 4. 划分网格
% generateMesh(model5,'Hmax',0.1);
generateMesh(model5,'Hmax',0.1,'GeometricOrder','linear');

% 5. 设置方程参数
a = 0;
f = 0;
c = '1./sqrt(1+ux.^2+uy.^2)';

% 6. 方程求解
u = pdenonlin(model5,c,a,f);

% 7. 结果可视化
pdeplot(model5,'xydata',u,'zdata',u);

%% examp6.4-9
m = 0;                                     % 方程中的m参数
x = linspace(0,1,30);                      % 定义x向量
t = linspace(0,2,30);                      % 定义t向量
sol = pdepe(m,@pdefun,@pdeic,@pdebc,x,t);  % 方程求解
u1 = sol(:,:,1); u2 = sol(:,:,2);          % 分别提取u1和u2的结果
% 结果可视化
figure;
surf(x,t,u1);                              % 绘制u1关于x和t的三维曲面
title('u1(x,t)'); xlabel('Distance x'); ylabel('Time t')
figure;
surf(x,t,u2);                              % 绘制u2关于x和t的三维曲面
title('u2(x,t)'); xlabel('Distance x'); ylabel('Time t')
