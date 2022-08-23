%--------------------------------------------------------------------------
%  ��6��  MATLAB��ֵ����
%--------------------------------------------------------------------------
% CopyRight��xiezhh


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
legend('y = sin(x)','���������ߣ�diff��','���������ߣ�gradient��');
xlabel('x'); ylabel('�������߼�����������')

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
options = optimset('Display','iter'); %��ʾ��������
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
hcat = text(-0.8,0,'è','FontSize',12);
hmouse = text(c+0.3,0,'��','FontSize',12);
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
tspan = [0,30];%ʱ������
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
xlabel(ha2,'λ��'); ylabel(ha2,'�ٶ�');
legend(ha2,'\mu = 1','\mu = 2','\mu = 3');
hold off

%% examp6.3-3
fun = @(t,y,dy)[dy(1)-y(2);
                dy(2)*sin(y(4))+dy(4)^2+2*y(1)*y(3)-y(1)*dy(2)*y(4);
                dy(3)-y(4);
                y(1)*dy(2)*dy(4)+cos(dy(4))-3*y(2)*y(3)];

t0 = 0;         % �Ա����ĳ�ֵ
y0 = [1;0;0;1]; % ״̬������ֵ����y0
% fix_y0����ָ����ֵ����y0��Ԫ���Ƿ���Ըı䡣1��ʾ��ӦԪ�ز��ܸı䣬0Ϊ���Ըı�
fix_y0 = [1;1;1;1]; % ������y0��ֵ�������ˣ���˶����ܸı䣬����fix_y0ȫΪ1
dy0 = [0;3;1;0];    % �²�һ��һ�׵���dy�ĳ�ֵdy0;
% ���ڱ�����һ�׵���dy�ĳ�ֵdy0�ǲ²�ģ������Ըı䣬���fix_dy0 ȫ��Ϊ0
fix_dy0 = [0;0;0;0];
% ����decic����������y��dy�ĳ�ֵ
[y02,dy02] = decic(fun,t0,y0,fix_y0,dy0,fix_dy0);

%���΢�ַ���
[t,y] = ode15i(fun,[0,5],y02,dy02); % y02��dy02��decic���
% ���ͼʾ
figure;
plot(t,y(:,1),'k-','linewidth',2);
hold on
plot(t,y(:,2),'k--','linewidth',2);
plot(t,y(:,3),'k-.','linewidth',2);
plot(t,y(:,4),'k:','linewidth',2);
% ͼ��,λ���Զ�ѡ�����λ��
L = legend('y_1(t)','y_2(t)','y_3(t)','y_4(t)','Location','best');
set(L,'fontname','Times New Roman');
xlabel('t');ylabel('y(t)');

%% examp6.3-4
lags = [1,3];       % �ӳٳ�������
history = [0,0,1];  % С�ڳ�ֵʱ����ʷ����
tspan = [0,8];      % ʱ������
% ����һ������dde23�������
sol = dde23(@ddefun,lags,history,tspan); 
% % ������������ddesd�������
% sol = ddesd(@ddefun,lags,history,tspan); 

% ��ͼ���ֽ��
plot(sol.x,sol.y(1,:),'k-','linewidth',2);
hold on
plot(sol.x,sol.y(2,:),'k-.','linewidth',2);
plot(sol.x,sol.y(3,:),'k-*','linewidth',1);
hold off
% ͼ��,λ���Զ�ѡ�����λ��
L = legend('y_1(t)','y_2(t)','y_3(t)','Location','best');
set(L,'fontname','Times New Roman');   % ����ͼ������
xlabel('t');ylabel('y(t)');            % ����������ǩ

%% examp6.3-6
% ΢�ַ���������Ӧ����������
BvpOdeFun  = @(t,y)[y(2)
                    2*y(2)*cos(t)-y(1)*sin(4*t)-cos(3*t)];
% �߽���������Ӧ������������
% �߽�����Ϊ y1(0) = 1, y1(4) = 2������0,4�ֱ��Ӧy���±߽���ϱ߽硣
% ����ylow(1)��ʾy1(0)��yup(1)��ʾy1(4)�����Ƶ�y2(0)��y2(4)�ֱ���ylow(2)��yup(2)��ʾ
BvpBcFun = @(ylow,yup)[ylow(1)-1; yup(1)-2];

T = linspace(0,4,10); % Ϊ����bvpinit���ɳ�ʼ��������׼��
% ��״̬����y������ʼ���裬����y1(0) = 1,y1(4) = 2����ѡȡһ���������������ĺ���
% y1(t) = 1+t/4����Ϊ��y1(t)�ĳ�ʼ���裬�Ӷ��䵼��1/4��Ϊ��y2(t)�ĳ�ʼ����
BvpYinit = @(t)[ 1+t/4; 1/4 ];
solinit = bvpinit(T,BvpYinit); % ����bvpinit�������ɳ�ʼ��

sol = bvp4c(BvpOdeFun,BvpBcFun,solinit); % ����bvp4c���,Ҳ���Ի���bvp5c
tint = linspace(0,4,100);
Stint = deval(sol,tint); % ���ݵõ���sol����deval�������[0,4]�����ڸ��������Ľ�

% ��ͼ���ֽ��
figure;
plot(tint,Stint(1,:),'k-','linewidth',2);
hold on
plot(tint,Stint(2,:),'k:','linewidth',2);
% ͼ��,λ���Զ�ѡ�����λ��
L = legend('y_1(t)','y_2(t)','Location','best');
set(L,'fontname','Times New Roman');   % ����ͼ������
xlabel('t');ylabel('y(t)');            % ����������ǩ

%% examp6.4-1
u1 = ones(1,49);
%  ���ݲ�ַ��̹���Ŀ�꺯���������飩
objfun = @(u)([u(2:end,:);u1]+[u1;u(1:end-1,:)]+...
       [u(:,2:end),u1']+[0*u1',u(:,1:end-1)])/4-u;
U0 = rand(49);
[Uin,Error] = fsolve(objfun,U0);  % ����ڵ��¶�
U = zeros(size(U0)+2);
U(:,end) = 1;
U(1,:) = 1;
U(end,:) = 1;
U(2:end-1,2:end-1) = Uin;
[X,Y] = meshgrid(linspace(0,1,51));
surf(X,Y,U);
xlabel('X'); ylabel('Y'); zlabel('U(X,Y)');

%% examp6.4-2
U = zeros(100);  % ��ֵ����
t = (1:100)/100; x = t;  % t��x�Ļ�������
U(1,:) = sin(t);  % �±߽�����
U(end,:) = cos(t);  % �ϱ߽�����
U(:,1) = x;  % ��ֵ����
b2 = 0.001; dx = 0.01;dt = 0.01;r = b2*dt/dx^2;  % ����
% ��ַ������
for j = 1:99
       U(2:99,j+1) = (1-2*r)*U(2:99,j)+r*(U(1:98,j)+U(3:100,j));
end
[T,X] = meshgrid(t);  % �������
surf(T,X,U);  % ������ͼ
xlabel('T');  ylabel('X');  zlabel('U(T,X)');  % �������ǩ

%% examp6.4-3
u = zeros(301);                 % ���������
dt = 1/300; c = 0.03;           % ����
x = linspace(0,1,301); t = x';  % t��x�Ļ�������
u(:,1) = x.*(1-x)/10;           % ��ֵ
u(1,:) = sin(t);                % ��ֵ
v = sin(2*pi*x);                % ���ٶ�
% ����u(i,2)
u(2:300,2) = (1-c)*u(2:300,1) + ...
       1/2*c*(u(1:299,1)+u(3:301,1)) + v(2:300)'*dt;
h = plot(x,u(:,1));             % ���Ƴ�ʼ������
axis([-0.1,1.1,-1,1]);          % ���������᷶Χ
xlabel('x');ylabel('U(x)');     % �������ǩ
% �����޲�ַ���ⷽ�̣�����̬չʾ�����
for j = 3:301
       u(2:300,j) = 2*(1-c)*u(2:300,j-1)+c*(u(3:301,j-1)+...
           u(1:299,j-1))-u(2:300,j-2);
       set(h,'YData',u(:,j));      % �������ϸ���λ��
       pause(0.1);                 % ��ͣ0.1��
end
%text(0.8,0.8,['t = ',num2str((j-1)/300,'%3.2f')])

%% examp6.4-4
% ������Ԫ����Ⲩ������
model = createpde(1);  % ��������һ�����̵�΢�ַ���ģ��
geometryFromEdges(model,@squareg);  % �����������������
pdegplot(model,'EdgeLabels','on');  % �����������
axis([-1.1,1.1,-1.1,1.1]);
axis equal;
applyBoundaryCondition(model,'Edge',[2,4],'g',0);  % ���ұ�ֵ����
applyBoundaryCondition(model,'Edge',[1,3],'g',0);  % ���±�ֵ����
Me = generateMesh(model,'Hmax',0.1,'GeometricOrder','linear');  % ��������
pdeplot(model);     % ��ʾ����ͼ

% ��ʼ����
u0 = 0;
ut0 = 0;
% ���̲���
c = 0.01;
a = 0;
f = @framp;
d = 1;

tlist = linspace(0,20,61);  % ����ʱ������
u1 = hyperbolic(u0,ut0,tlist,model,c,a,f,d);  % �������

% ������ӻ�
p = Me.Nodes;  % ���������
t = Me.Elements;  % ������������
xyi = linspace(-1,1,30); % xy�����񻮷�
% ��������תΪ��������
uxy = tri2grid(p,t,u1(:,1),xyi,xyi); 
figure;
h = surf(xyi,xyi,uxy);  % ������ͼ
axis([-1.1,1.1,-1.1,1.1,-0.8,0.8]);
view(-30,70);  % �����ӵ�λ��
colormap(jet);  % ������ɫ����
shading interp; % ��ֵȾɫ
%light('pos',[0.6,-0.6,20]);
camlight;  % �����Դ
lighting phong;  % ���ù���ģʽ
xlabel('x');ylabel('y'),zlabel('u');
% ˮ����ɢ�Ķ�̬չʾ
for i = 1:numel(tlist)
    uxy = tri2grid(p,t,u1(:,i),xyi,xyi); 
    set(h,'ZData',uxy);  % ��������
    drawnow;
    pause(0.1);
end

%% examp6.4-5
model2 = createpde(1);
geometryFromEdges(model2,@squareg);   % �����������������
pdegplot(model2,'EdgeLabels','on');   % ����������򣬲���ʾ�߽��ǩ
ylim([-1.1,1.1]);
axis equal;
% ���õ����ұ߽�ı�ֵ������Dirichlet ��ֵ������
applyBoundaryCondition(model2,'Edge',[2,4],'u',0);
% �������±߽�ı�ֵ������Neumann ��ֵ������
applyBoundaryCondition(model2,'Edge',[1,3],'g',0);
Me = generateMesh(model2,'Hmax',0.1,'GeometricOrder','linear');  % ��������
% ��ʼ����
u0 = 'atan(cos(pi/2*x))';
ut0 = '3*sin(pi*x).*exp(cos(pi*y))';
% ���̲���
c = 1;
a = 0;
f = 0;
d = 1;
tlist = linspace(0,6,41);                      % ����ʱ������
u1 = hyperbolic(u0,ut0,tlist,model2,c,a,f,d);  % �������

XY = Me.Nodes';                                % ���������
Tri = Me.Elements';                            % ������������
figure;
h = trisurf(Tri,XY(:,1),XY(:,2),u1(:,1));      % ������������ͼ
axis([-1.1,1.1,-1.1,1.1,-3,3]);                % ���������᷶Χ
xlabel('x');ylabel('y');zlabel('u(x,y)');
for i = 1:numel(tlist)
    set(h,'Vertices',[XY(:,1),XY(:,2),u1(:,i)]);  % ��������
    drawnow;
    pause(0.1);
end

%% examp6.4-6
% 1. ��������һ�����̵�΢�ַ���ģ��
model3 = createpde(1);

% 2. ����Բ���������
geometryFromEdges(model3,@circleg);
pdegplot(model3,'EdgeLabels','on');   % ����������򣬲���ʾ�߽��ǩ 
ylim([-1.1,1.1]);
axis equal;

% 3. ���ñ�ֵ������Dirichlet ��ֵ������
NumEdges = model3.Geometry.NumEdges;  % �������߽���
applyBoundaryCondition(model3,'Edge',1:NumEdges,'u',0);

% 4. ��������
generateMesh(model3,'Hmax',0.02,'GeometricOrder','linear'); 

% 5. ���ó�ʼ����
p = model3.Mesh.Nodes;  % ��������������
u0 = zeros(size(p,2),1);
% ���Ұ뾶Ϊ0.4Բ���ڵĵ�
ix = find(sqrt(p(1,:).^2 + p(2,:).^2) <= 0.4);
u0(ix) = ones(size(ix));

% 6. ���÷��̲���
c = 1;
a = 0;
f = 0;
d = 1;

% 7. �������
tlist = linspace(0,0.1,21);  % ����ʱ������
u = parabolic(u0,tlist,model3,c,a,f,d);  % �������

% 8. ������ӻ�
figure;
umax = max(max(u));  % ����¶�
umin = min(min(u));  % ��С�¶�
% ����ɢ�Ķ�̬չʾ
for i = 1:5%numel(tlist)
    pdeplot(model3,'xydata',u(:,i)); % �����¶ȷֲ�ͼ
    caxis([umin umax]);              % ��������ϵ��ɫ��Χ
    axis equal;
    axis([-1.1,1.1,-1.1,1.1]);
    xlabel('x');ylabel('y');
    drawnow;
    pause(0.1);
end

%% examp6.4-7
% 1. ��������һ�����̵�΢�ַ���ģ��
model4 = createpde;

% 2. ���ⲿ�ļ������������ļ�������
importGeometry(model4,'Block.stl'); 
h = pdegplot(model4,'FaceLabels','on'); % �����������
h(1).FaceAlpha = 0.5; % ����͸����ֵΪ0.5

% 3. ���ñ�ֵ����
% x = 0, x = 100, z = 0, z = 50����Ӧ�ı�ֵ����
applyBoundaryCondition(model4,'Face',1:4,'u',0);
% y = 0����Ӧ�ı�ֵ����
applyBoundaryCondition(model4,'Face',6,'g',-1);
% y = 20����Ӧ�ı�ֵ����
applyBoundaryCondition(model4,'Face',5,'g',1);

% 4. ��������
generateMesh(model4);

% 5. ���÷��̲���
c = 1;
a = 0;
f = 'log(1+x+y./(1+z))';

% 6. �������
u = assempde(model4,c,a,f);

% 7. ������ӻ�
p = model4.Mesh.Nodes;  % ��������������
% ��x,y,z������������񻮷�
xi = linspace(min(p(1,:)),max(p(1,:)),60);
yi = linspace(min(p(2,:)),max(p(2,:)),60);
zi = linspace(min(p(3,:)),max(p(3,:)),60);
[X,Y,Z] = meshgrid(xi,yi,zi);  % ���ɾ�����������
% �������������PDEResults����
result = createPDEResults(model4,u);
% ͨ����ֵ�����������㴦����ֵ
V = interpolateSolution(result,X,Y,Z);
% �Ѻ���ֵ����תΪ��ά����
V = reshape(V,size(X));

figure
colormap jet
sxyz = [1:15:60,60];  % ������Ƭλ��
% ������Ƭͼ
slice(X,Y,Z,V,xi(sxyz),yi(sxyz),zi(sxyz));
shading interp;  % ��ֵȾɫ
alpha(0.5);  % ����͸����
xlabel('x')
ylabel('y')
zlabel('z')
colorbar;  % �����ɫ��
axis equal;

%% examp6.4-8
% 1. ��������һ�����̵�΢�ַ���ģ��
model5 = createpde;

% 2. ����Բ���������
geometryFromEdges(model5,@circleg);

% 3. ���ñ�ֵ����
boundaryfun = @(region,state)region.x.^2;  % ����߽纯��
NumEdges = model5.Geometry.NumEdges;
applyBoundaryCondition(model5,'Edge',1:NumEdges,...
    'u',boundaryfun,'Vectorized','on');

% 4. ��������
% generateMesh(model5,'Hmax',0.1);
generateMesh(model5,'Hmax',0.1,'GeometricOrder','linear');

% 5. ���÷��̲���
a = 0;
f = 0;
c = '1./sqrt(1+ux.^2+uy.^2)';

% 6. �������
u = pdenonlin(model5,c,a,f);

% 7. ������ӻ�
pdeplot(model5,'xydata',u,'zdata',u);

%% examp6.4-9
m = 0;                                     % �����е�m����
x = linspace(0,1,30);                      % ����x����
t = linspace(0,2,30);                      % ����t����
sol = pdepe(m,@pdefun,@pdeic,@pdebc,x,t);  % �������
u1 = sol(:,:,1); u2 = sol(:,:,2);          % �ֱ���ȡu1��u2�Ľ��
% ������ӻ�
figure;
surf(x,t,u1);                              % ����u1����x��t����ά����
title('u1(x,t)'); xlabel('Distance x'); ylabel('Time t')
figure;
surf(x,t,u2);                              % ����u2����x��t����ά����
title('u2(x,t)'); xlabel('Distance x'); ylabel('Time t')
