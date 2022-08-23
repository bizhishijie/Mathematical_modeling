%% 调用parabolic函数求解偏微分方程组（examp6.4-9）
% 1. 创建包含两个方程的微分方程模型
model = createpde(2);

% 2. 创建正方形求解区域
% 3为矩形编号，4是边数，矩形区域顶点x坐标为[0,1,1,0],y坐标为[0,0,1,1]
R1 = [3,4,0,1,1,0,0,0,1,1]'; 
geom = R1;
ns = char('R1')';             % 求解区域名称
sf = 'R1';                    % 构造求解区域的公式
g = decsg(geom,sf,ns);        % 创建几何体
geometryFromEdges(model,g);  % 创建自定义求解区域
pdegplot(model,'EdgeLabels','on');   % 绘制求解区域，并显示边界标签 
ylim([-0.1,1.1]);
axis equal;

% 3. 设置边值条件（Dirichlet 边值条件）
% u1的右边界对应的边值条件
applyBoundaryCondition(model,'Edge',2,'u',1,'EquationIndex',1);
% u2的左边界对应的边值条件
applyBoundaryCondition(model,'Edge',4,'u',0,'EquationIndex',2);

% 4. 划分网格
Me = generateMesh(model,'Hmax',0.1,'GeometricOrder','linear'); 

% 5. 设置初始条件
p = model.Mesh.Nodes;  % 三角网顶点坐标
np = size(p,2);         % 顶点个数
u0 = [ones(np,1),zeros(np,1)];  % 各顶点对应的初值
u0 = u0(:);

% 6. 设置方程参数
c = [1/80;0;1/91;0];
a = [0;0];
f = char('exp(u(2)-u(1))-exp(u(1)-u(2))','exp(u(1)-u(2))-exp(u(2)-u(1))');
d = [1;1];

% 7. 方程求解
tlist = linspace(0,2,31);  % 定义时间向量
u = parabolic(u0,tlist,model,c,a,f,d);  % 方程求解

% 8. 结果可视化
p = Me.Nodes;  % 网格点坐标
t = Me.Elements;  % 三角网顶点编号

xi = linspace(0,1,30);  % 定义x坐标向量
T = repmat(tlist',[1,30]);  % 定义时间矩阵
U1 = zeros(31,30);
U2 = U1;
for i = 1:numel(tlist)
    Ui = tri2grid(p,t,u(1:np,i),xi,xi); % 三角网格转为矩形网格
    U1(i,:) = Ui(1,:);                  % y = 0时的u1
    Ui = tri2grid(p,t,u(np+1:end,i),xi,xi); 
    U2(i,:) = Ui(1,:);                  % y = 0时的u2
end
figure;
surf(xi,tlist,U1);                      % u1关于x和t的曲面图
title('u1(x,t)'); xlabel('Distance x'); ylabel('Time t');
figure;
surf(xi,tlist,U2);                      % u2关于x和t的曲面图
title('u2(x,t)'); xlabel('Distance x'); ylabel('Time t');