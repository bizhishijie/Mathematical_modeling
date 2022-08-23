%--------------------------------------------------------------------------
%  第5章  MATLAB符号计算
%--------------------------------------------------------------------------
% CopyRight：xiezhh


%% examp5.1-1
a = sym('6.01');         % 定义符号常数
b = sym('b','real');     % 定义实数域上的符号变量
A = [1, 2; 3, 4];        % 定义数值矩阵
B = sym(A);              % 把数值矩阵转为符号矩阵
C = sym('c%d%d',[3,4])

syms  x  y               % 同时定义多个复数域上的符号变量
syms  z  positive        % 定义正实数域上的符号变量
syms  f(x,y)             % 定义符号函数
f(x,y) = x + y^2;        % 指定符号函数表达式
c = f(1, 2) 

zv = solve(z^2 == 1, z)  % 求方程z^2 = 1的解（只有大于零的解）

syms  z  clear           % 撤销对符号变量取值域的限定，将其恢复为复数域上的符号变量
zv = solve(z^2 == 1, z)

%% examp5.1-2
syms x                     % 定义符号变量
assume(x>0 & x<5);         % 对符号变量的取值域进行限定，0<x<5
assumeAlso(x,'integer');   % 对符号变量的取值域增加别的限定，x取整数
assumptions(x)             % 查看符号变量取值域的限定

result = solve(x^2>12)     % 求解不等式

syms  x  clear

%% examp5.1-3
syms a b c x y z          % 定义多个符号变量
f1 = a*x^2+b*x-c;         % 创建符号表达式f1
f2 = sin(x)*cos(y);       % 创建符号表达式f2
f3 = (x+y)/z;             % 创建符号表达式f3
f4 = [x+1, x^2; x^3, x^4] % 创建符号表达式矩阵f4

f5 = f4'                  % 符号表达式矩阵的共轭转置（'）

f6 = f4.'

%% examp5.1-4
syms x y                % 定义符号变量
f1 = abs(x) >= 0        % 创建符号表达式

f2 = x^2 + y^2 == 1     % 创建符号表达式

f3 = ~(y - sqrt(x) > 0) % 创建符号表达式

f4 = x > 0 | y < -1     % 创建符号表达式

f5 = x > 0 & y < -1     % 创建符号表达式

%% examp5.1-5
syms x
f = abs(x) >= 0;                % 创建符号表达式
result1 = isAlways(f)           % 判断不等式|x|>=0是否成立

result2 = isequaln(abs(x), x)   % 判断|x|是否等于x

assume(x>0);                    % 限定x>0
result3 = isequaln(abs(x), x)   % 重新判断|x|是否等于x

syms x clear                    % 撤销对符号变量取值域的限定

%% examp5.1-6
syms x y
f = factor(x^3-y^3)

fa = factor(sym('12345678901234567890'))

%% examp5.1-7
syms x y
f = (x+y)*(x^2+y^2+1);
collect(f,y)

%% examp5.1-8
syms x y a b
f = [cos(x+y); (a+b)*exp((a-b)^2)];
expand(f)

%% examp5.1-9
syms x
f1 = sqrt(4/x^2+4/x+1);
g1 = simplify(f1)                                % 按默认设置进行化简

g2 = simplify(f1,'IgnoreAnalyticConstraints',1)  % 忽略分析约束进行化简

pretty(g2)                                       % 把符号表达式显示为数学公式形式

f2 = cos(3*acos(x));
g3 = simplify(f2, 'Steps', 4)                    % 进行4步化简

%% examp5.1-10
syms f(x)                      % 定义符号函数
f(x) = log(sym(5.2))*exp(x);   % 指定符号函数表达式
y = f(3)                       % 计算符号函数在x = 3处的函数值

y1 = double(y)                 % 把符号数转为双精度数

y2 = vpa(y,10)                 % 以10位有效数字形式显示符号数

x = 3;                         % 指定x的值
y3 = eval(f)                   % 执行MATLAB运算，得到函数值

%% examp5.1-11
syms a b x
f = a*sin(x)+b;               % 定义符号表达式
f1 = subs(f,sin(x),'log(y)')  % 符号项替换

% 变量替换方式一
f2 = subs(f1,[a,b],[2,5])     % 同时替换变量a，b的值

% 变量替换方式二
f3 = subs(f1,{a,b},{2,5})     % 同时替换变量a，b的值

%% examp5.1-12
syms a b x
f = a*sin(x)+b;
y = subs(f, {a,b,x}, {2, 5, 1:3})  % 同时替换多个符号变量的值

double(y)                          % 将计算结果转为双精度值

%% examp5.1-13
syms a b x
f(x) = symfun(a*sin(x)+b, x);     % 把符号表达式转为符号函数
y = f(1:3)

%% examp5.1-14
syms a b c d x
f = a*(x+b)^c+d;                          % 定义符号表达式
g = subs(f,{a,b,c,d},{2,-1,sym(1/2),3});  % 同时替换多个变量
FunFromSym1 = matlabFunction(g)           % 将符号表达式转为匿名函数

y = FunFromSym1(10)                       % 调用匿名函数计算函数值

% 将符号表达式转为M文件函数FunFromSym2.m
matlabFunction(g,'file',[pwd,'\FunFromSym2.m'],...
    'vars',{'x'},'outputs',{'y'});
y = FunFromSym2(10)                       % 调用M函数计算函数值

%% examp5.1-15
syms f(x)               % 定义符号函数
f(x) = 1/log(abs(x));   % 指定符号函数表达式
ezplot(f,[-6,6]);       % 绘制函数图形

%% examp5.2-1
syms n a k x
xn = (-1)^n/(n+1)^2;
L1 = limit(xn,n,inf)

f1 = sin(a*x)/(a*x);
L2 = limit(f1,x,0,'left')

f2 = (1-2/x)^(k*x);
L3 = limit(f2,x,inf)

%% examp5.2-2
syms x y
f = sin(x)^2;
df = diff(f,x);
df_1 = subs(df,x,1)

ddf = diff(f,x,2)

Fxy = cos(x+sin(y))-sin(y);
dy_dx = -diff(Fxy,x)/diff(Fxy,y)

%% examp5.2-3
syms x
f = exp(x); 
g = taylor(f, x, 0, 'Order', 6)

%% examp5.2-4
syms k
f1 = (k-2)/2^k;
s1 = symsum(f1,k,3,inf)

f2 = [1/(2*k+1)^2,  (-1)^k/3^k];
s2 = symsum(f2,k,1,inf)

%% examp5.2-5
syms x1 x2
f = [x1+x2;x2*log(x1)];
v = [x1;x2];
jac = jacobian(f,v)

%% examp5.2-6
syms x y z a
F = int(x*log(a*x),x)

f1 = sqrt(1-x^2);
s1 = int(f1,x,-1,1)

f2 = exp(-x^2/2);
s2 = int(f2,x,-inf,inf)

f3 = (x+y)/z;
s3 = int(int(int(f3,z,x*y,2*x*y),y,x,2*x),x,1,2)
s4 = double(s3) 

%% examp5.3-1
syms x
Result1 = solve(x^3 - 2*x^2 + 4*x == 8, x)

Result2 = solve(sin(x) + cos(2*x) == 1, x)

[Result3,params,conditions] = solve(sin(x) + cos(2*x) == 1, x, 'ReturnConditions',true)

Result4 = solve(x + x*exp(x) == 10, x)

%% examp5.3-2
syms x y
[X,Y] = solve([1/x^3 + 1/y^3 == 28, 1/x + 1/y == 4], [x,y]) 

%% examp5.3-3
syms y(x)
Y = dsolve(diff(y,2) == x+y)

%% examp5.3-4
syms y(t)
Y = dsolve(diff(y) == 1 + y^2, y(0) == 1)

Y = dsolve(diff(y) == 1 + y^2, y(0) == 1, 'IgnoreAnalyticConstraints', false)

%% examp5.3-5
syms y(x)
Y = dsolve(x*diff(y,2)-3*diff(y) == x^2, [y(1) == 0, y(5) == 0])

h = ezplot(Y,[-1,6]);
set(h,'color','k','LineWidth',2,'LineStyle','--');
hold on;
plot([1 5],[0,0],'p','color','r','markersize',12); %画微分方程的两个边值点
text(1,1,'y(1) = 0'); %图上标注边值条件
text(4,1,'y(5) = 0');
title('');
hold off;

%% examp5.3-6
syms x(t)  y(t)
[X, Y] = dsolve(diff(x) == y, diff(y) == -x)