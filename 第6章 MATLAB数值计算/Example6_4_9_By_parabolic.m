%% ����parabolic�������ƫ΢�ַ����飨examp6.4-9��
% 1. ���������������̵�΢�ַ���ģ��
model = createpde(2);

% 2. �����������������
% 3Ϊ���α�ţ�4�Ǳ������������򶥵�x����Ϊ[0,1,1,0],y����Ϊ[0,0,1,1]
R1 = [3,4,0,1,1,0,0,0,1,1]'; 
geom = R1;
ns = char('R1')';             % �����������
sf = 'R1';                    % �����������Ĺ�ʽ
g = decsg(geom,sf,ns);        % ����������
geometryFromEdges(model,g);  % �����Զ����������
pdegplot(model,'EdgeLabels','on');   % ����������򣬲���ʾ�߽��ǩ 
ylim([-0.1,1.1]);
axis equal;

% 3. ���ñ�ֵ������Dirichlet ��ֵ������
% u1���ұ߽��Ӧ�ı�ֵ����
applyBoundaryCondition(model,'Edge',2,'u',1,'EquationIndex',1);
% u2����߽��Ӧ�ı�ֵ����
applyBoundaryCondition(model,'Edge',4,'u',0,'EquationIndex',2);

% 4. ��������
Me = generateMesh(model,'Hmax',0.1,'GeometricOrder','linear'); 

% 5. ���ó�ʼ����
p = model.Mesh.Nodes;  % ��������������
np = size(p,2);         % �������
u0 = [ones(np,1),zeros(np,1)];  % �������Ӧ�ĳ�ֵ
u0 = u0(:);

% 6. ���÷��̲���
c = [1/80;0;1/91;0];
a = [0;0];
f = char('exp(u(2)-u(1))-exp(u(1)-u(2))','exp(u(1)-u(2))-exp(u(2)-u(1))');
d = [1;1];

% 7. �������
tlist = linspace(0,2,31);  % ����ʱ������
u = parabolic(u0,tlist,model,c,a,f,d);  % �������

% 8. ������ӻ�
p = Me.Nodes;  % ���������
t = Me.Elements;  % ������������

xi = linspace(0,1,30);  % ����x��������
T = repmat(tlist',[1,30]);  % ����ʱ�����
U1 = zeros(31,30);
U2 = U1;
for i = 1:numel(tlist)
    Ui = tri2grid(p,t,u(1:np,i),xi,xi); % ��������תΪ��������
    U1(i,:) = Ui(1,:);                  % y = 0ʱ��u1
    Ui = tri2grid(p,t,u(np+1:end,i),xi,xi); 
    U2(i,:) = Ui(1,:);                  % y = 0ʱ��u2
end
figure;
surf(xi,tlist,U1);                      % u1����x��t������ͼ
title('u1(x,t)'); xlabel('Distance x'); ylabel('Time t');
figure;
surf(xi,tlist,U2);                      % u2����x��t������ͼ
title('u2(x,t)'); xlabel('Distance x'); ylabel('Time t');