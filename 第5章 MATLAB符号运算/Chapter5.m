%--------------------------------------------------------------------------
%  ��5��  MATLAB���ż���
%--------------------------------------------------------------------------
% CopyRight��xiezhh


%% examp5.1-1
a = sym('6.01');         % ������ų���
b = sym('b','real');     % ����ʵ�����ϵķ��ű���
A = [1, 2; 3, 4];        % ������ֵ����
B = sym(A);              % ����ֵ����תΪ���ž���
C = sym('c%d%d',[3,4])

syms  x  y               % ͬʱ�������������ϵķ��ű���
syms  z  positive        % ������ʵ�����ϵķ��ű���
syms  f(x,y)             % ������ź���
f(x,y) = x + y^2;        % ָ�����ź������ʽ
c = f(1, 2) 

zv = solve(z^2 == 1, z)  % �󷽳�z^2 = 1�Ľ⣨ֻ�д�����Ľ⣩

syms  z  clear           % �����Է��ű���ȡֵ����޶�������ָ�Ϊ�������ϵķ��ű���
zv = solve(z^2 == 1, z)

%% examp5.1-2
syms x                     % ������ű���
assume(x>0 & x<5);         % �Է��ű�����ȡֵ������޶���0<x<5
assumeAlso(x,'integer');   % �Է��ű�����ȡֵ�����ӱ���޶���xȡ����
assumptions(x)             % �鿴���ű���ȡֵ����޶�

result = solve(x^2>12)     % ��ⲻ��ʽ

syms  x  clear

%% examp5.1-3
syms a b c x y z          % ���������ű���
f1 = a*x^2+b*x-c;         % �������ű��ʽf1
f2 = sin(x)*cos(y);       % �������ű��ʽf2
f3 = (x+y)/z;             % �������ű��ʽf3
f4 = [x+1, x^2; x^3, x^4] % �������ű��ʽ����f4

f5 = f4'                  % ���ű��ʽ����Ĺ���ת�ã�'��

f6 = f4.'

%% examp5.1-4
syms x y                % ������ű���
f1 = abs(x) >= 0        % �������ű��ʽ

f2 = x^2 + y^2 == 1     % �������ű��ʽ

f3 = ~(y - sqrt(x) > 0) % �������ű��ʽ

f4 = x > 0 | y < -1     % �������ű��ʽ

f5 = x > 0 & y < -1     % �������ű��ʽ

%% examp5.1-5
syms x
f = abs(x) >= 0;                % �������ű��ʽ
result1 = isAlways(f)           % �жϲ���ʽ|x|>=0�Ƿ����

result2 = isequaln(abs(x), x)   % �ж�|x|�Ƿ����x

assume(x>0);                    % �޶�x>0
result3 = isequaln(abs(x), x)   % �����ж�|x|�Ƿ����x

syms x clear                    % �����Է��ű���ȡֵ����޶�

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
g1 = simplify(f1)                                % ��Ĭ�����ý��л���

g2 = simplify(f1,'IgnoreAnalyticConstraints',1)  % ���Է���Լ�����л���

pretty(g2)                                       % �ѷ��ű��ʽ��ʾΪ��ѧ��ʽ��ʽ

f2 = cos(3*acos(x));
g3 = simplify(f2, 'Steps', 4)                    % ����4������

%% examp5.1-10
syms f(x)                      % ������ź���
f(x) = log(sym(5.2))*exp(x);   % ָ�����ź������ʽ
y = f(3)                       % ������ź�����x = 3���ĺ���ֵ

y1 = double(y)                 % �ѷ�����תΪ˫������

y2 = vpa(y,10)                 % ��10λ��Ч������ʽ��ʾ������

x = 3;                         % ָ��x��ֵ
y3 = eval(f)                   % ִ��MATLAB���㣬�õ�����ֵ

%% examp5.1-11
syms a b x
f = a*sin(x)+b;               % ������ű��ʽ
f1 = subs(f,sin(x),'log(y)')  % �������滻

% �����滻��ʽһ
f2 = subs(f1,[a,b],[2,5])     % ͬʱ�滻����a��b��ֵ

% �����滻��ʽ��
f3 = subs(f1,{a,b},{2,5})     % ͬʱ�滻����a��b��ֵ

%% examp5.1-12
syms a b x
f = a*sin(x)+b;
y = subs(f, {a,b,x}, {2, 5, 1:3})  % ͬʱ�滻������ű�����ֵ

double(y)                          % ��������תΪ˫����ֵ

%% examp5.1-13
syms a b x
f(x) = symfun(a*sin(x)+b, x);     % �ѷ��ű��ʽתΪ���ź���
y = f(1:3)

%% examp5.1-14
syms a b c d x
f = a*(x+b)^c+d;                          % ������ű��ʽ
g = subs(f,{a,b,c,d},{2,-1,sym(1/2),3});  % ͬʱ�滻�������
FunFromSym1 = matlabFunction(g)           % �����ű��ʽתΪ��������

y = FunFromSym1(10)                       % ���������������㺯��ֵ

% �����ű��ʽתΪM�ļ�����FunFromSym2.m
matlabFunction(g,'file',[pwd,'\FunFromSym2.m'],...
    'vars',{'x'},'outputs',{'y'});
y = FunFromSym2(10)                       % ����M�������㺯��ֵ

%% examp5.1-15
syms f(x)               % ������ź���
f(x) = 1/log(abs(x));   % ָ�����ź������ʽ
ezplot(f,[-6,6]);       % ���ƺ���ͼ��

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
plot([1 5],[0,0],'p','color','r','markersize',12); %��΢�ַ��̵�������ֵ��
text(1,1,'y(1) = 0'); %ͼ�ϱ�ע��ֵ����
text(4,1,'y(5) = 0');
title('');
hold off;

%% examp5.3-6
syms x(t)  y(t)
[X, Y] = dsolve(diff(x) == y, diff(y) == -x)