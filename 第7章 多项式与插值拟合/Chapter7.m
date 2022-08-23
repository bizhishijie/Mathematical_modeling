%--------------------------------------------------------------------------
%  ��7��  ����ʽ���ֵ���
%--------------------------------------------------------------------------

%% examp7.1-1 ����ʽ���
%--------------------ɢ��ͼ------------------
[Data,Textdata] = xlsread('ʳƷ���ۼ۸�.xls');
x = Data(:,1);
y = Data(:,3);
timestr = Textdata(3:end,2);
plot(x,y,'k.','Markersize',15);
set(gca,'XTick',1:2:numel(x),'XTickLabel',timestr(1:2:end));
set(gca,'XTickLabelRotation',-30);
xlabel('ʱ��(x)');
ylabel('ʳƷ���ۼ۸����ָ��');
%-------------------4�׶���ʽ���--------------------
[p4,S4] = polyfit(x,y,4)
r = poly2sym(p4);
r = vpa(r,5)
 
%--------------------���߽׶���ʽ���---------------------
[p5,S5] = polyfit(x,y,5);
S5.normr
[p6,S6] = polyfit(x,y,6);
S6.normr
[p7,S7] = polyfit(x,y,7);
S7.normr
[p8,S8] = polyfit(x,y,8);
S8.normr
[p9,S9] = polyfit(x,y,9);
S9.normr

%-------------------���Ч��ͼ----------------------
figure;
plot(x,y,'k.','Markersize',15);
set(gca,'XTick',1:2:numel(x),'XTickLabel',timestr(1:2:end));
set(gca,'XTickLabelRotation',-30);
xlabel('ʱ��(x)');
ylabel('ʳƷ���ۼ۸����ָ��');
hold on;
yd4 = polyval(p4,x);
yd6 = polyval(p6,x);
yd8 = polyval(p8,x);
yd9 = polyval(p9,x);
plot(x,yd4,'r:+');
plot(x,yd6,'g--s');
plot(x,yd8,'b-.d');
plot(x,yd9,'m-p');
legend('ԭʼɢ��','4�ζ���ʽ���','6�ζ���ʽ���','8�ζ���ʽ���','9�ζ���ʽ���')

%% examp7.4-1 һά��ֵ
x0 = [0,3,5,7,9,11,12,13,14,15];
y01 = [0,1.8,2.2,2.7,3.0,3.1,2.9,2.5,2.0,1.6];
y02 = [0,1.2,1.7,2.0,2.1,2.0,1.8,1.2,1.0,1.6];
x = 0:0.1:15;
ysp1 = interp1(x0,y01,x,'spline');
ysp2 = interp1(x0,y02,x,'spline');
plot([x0,x0],[y01,y02],'o');
hold on;
plot(x,ysp1,'r',x,ysp2,'r');
xlabel('X')
ylabel('Y')
legend('��ֵ�ڵ�','����������ֵ','location','northwest') 


%% examp7.4-2 һά��ֵ
fun = @(x)sin(pi*x/2).*(x>=-1&x<1) + x.*exp(1-x.^2).*(x>=1 | x<-1);
%%----------------����[0,1]�ϵ�����������ֵ------------------
x01 = linspace(0,1,6);
y01 = fun(x01); 
x1 = linspace(0,1,20);
pp1 = csape(x01,[1,y01,0],'complete');
y1 = fnval(pp1,x1); 
%%----------------����[1,3]�ϵ�����������ֵ------------------
x02 = linspace(1,3,8);
y02 = fun(x02);   
x2 = linspace(1,3,30); 
pp2 = csape(x02,[0,y02,0.01],[1,2]);
y2 = fnval(pp2,x2);
%%-----------------------��ͼ---------------------
plot([x01,x02],[y01,y02],'ko');
hold on;
plot([x1,x2],fun([x1,x2]),'k','linewidth',2);
plot([x1,x2],[y1,y2],'--','linewidth',2);
xlabel('X');
ylabel('Y = f(x)');
legend('��ֵ�ڵ�','ԭ����ͼ��','����������ֵ');

%% examp7.4-3 ��ά����ڵ��ֵ
x = 100:100:500;
y = 100:100:400;
[X,Y] = meshgrid(x,y);
Z = [450  478  624  697  636
        420  478  630  712  698
        400  412  598  674  680
        310  334  552  626  662];
xd = 100:20:500;
yd = 100:20:400;
[Xd1,Yd1] = meshgrid(xd,yd);
[Xd2,Yd2] = ndgrid(xd,yd);

figure;  % �½�ͼ�δ���
% -------------- ����interp2����������������ֵ-------------------
Zd1 = interp2(X,Y,Z,Xd1,Yd1,'spline');
subplot(1,2,1);
surf(Xd1,Yd1,Zd1);
xlabel('X'); ylabel('Y'); zlabel('Z'); title('interp2')

% ---------����griddedInterpolant����������������ֵ--------------
F = griddedInterpolant({x,y},Z','spline');
Zd2 = F(Xd2,Yd2);
subplot(1,2,2);
surf(Xd2,Yd2,Zd2);
xlabel('X'); ylabel('Y'); zlabel('Z'); title('griddedInterpolant')

%% examp7.4-4 ��άɢ�ҽڵ��ֵ
xyz = xlsread('cumcm2011A.xls',1,'B4:D322');
Cd = xlsread('cumcm2011A.xls',2,'C4:C322');
x = xyz(:,1);
y = xyz(:,2);
z = xyz(:,3);
xd = linspace(min(x),max(x),60);
yd = linspace(min(y),max(y),60);
[Xd,Yd] = meshgrid(xd,yd);
% ------------����griddata������ɢ�ҽڵ��ֵ---------------
Zd1 = griddata(x,y,z,Xd,Yd);
Cd1 = griddata(x,y,Cd,Xd,Yd);
figure;
subplot(1,2,1);
surf(Xd,Yd,Zd1,Cd1);
shading interp;
xlabel('X'); ylabel('Y'); zlabel('Z'); title('griddata');
colorbar;

% ------------����scatteredInterpolant������ɢ�ҽڵ��ֵ------------
F1 = scatteredInterpolant(x,y,z,'linear','none');
Zd2 = F1(Xd,Yd);  % �����ֵ�㴦�ĺ��θ߶�
F2 = scatteredInterpolant(x,y,Cd,'linear','none');
Cd2 = F2(Xd,Yd);
subplot(1,2,2);
surf(Xd,Yd,Zd2,Cd2);
shading interp;
xlabel('X'); ylabel('Y'); zlabel('Z');title('scatteredInterpolant'); 
colorbar;

%% examp7.4-5 ��ά��ֵ
data = xlsread('�¶ȳ�.xlsx');
x = data(:,1);
y = data(:,2);
z = data(:,3);
v = data(:,4);
xi = linspace(min(x),max(x),60);
yi = linspace(min(y),max(y),60);
zi = linspace(min(z),max(z),60);
[Xd,Yd,Zd] = meshgrid(xi,yi,zi);
F = scatteredInterpolant(x,y,z,v);
Vd = F(Xd,Yd,Zd);

sxyz = [1:15:60,60];  % ������Ƭλ��
% ������Ƭͼ
slice(Xd,Yd,Zd,Vd,xi(sxyz),yi(sxyz),zi(sxyz));
shading interp;  % ��ֵȾɫ
alpha(0.5);  % ����͸����
xlabel('x');ylabel('y');zlabel('z')
axis equal
colorbar;  % �����ɫ��

figure;
MyIsosurface(Xd,Yd,Zd,Vd,800);