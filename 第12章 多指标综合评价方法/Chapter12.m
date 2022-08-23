%--------------------------------------------------------------------------
%  第12章  多指标综合评价方法
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp12.1-1
% ------------层次单排序与一致性检验---------------
%(1) 准则层对目标层
A = [1,1/2,4,3,3;2,1,7,5,5;1/4,1/7,1,1/2,1/3;1/3,1/5,2,1,1;1/3,1/5,3,1,1];
[V,L] = eig(A,'vector');
L = real(L);
[Lmax,id] = max(L)
CI = (Lmax-5)/(5-1)
RI = 1.12;
CR = CI/RI
W = V(:,id);
W = W/sum(W)

%（2） 方案层对准则层
B1 = [1,2,5;1/2,1,2;1/5,1/2,1];
[Lmax1,CI1,CR1,W1] = AHP(B1)

B2 = [1,1/3,1/8;3,1,1/3;8,3,1];
[Lmax2,CI2,CR2,W2] = AHP(B2)

B3 = [1,1,3;1,1,3;1/3,1/3,1];
[Lmax3,CI3,CR3,W3] = AHP(B3)

B4 = [1,3,4;1/3,1,1;1/4,1,1];
[Lmax4,CI4,CR4,W4] = AHP(B4)

B5 = [1,1,1/4;1,1,1/4;4,4,1];
[Lmax5,CI5,CR5,W5] = AHP(B5)

% ------------层次总排序与一致性检验及决策---------------
CIj = [CI1,CI2,CI3,CI4,CI5];   % 层次单排序的一致性指标
RIj = [0.52,0.52,0.52,0.52,0.52]; % 层次单排序的随机一致性指标
CR = CIj*W/(RIj*W)      % 层次总排序的一致性比率
Wj = [W1,W2,W3,W4,W5];  % 层次单排序的权重
Y = Wj*W                % 层次总排序的组合权重

%% examp12.2-1 一级模糊综合评价
A = [0.1,0.1,0.3,0.15,0.35];
R = [0.2,0.5,0.3,0;
     0.1,0.3,0.5,0.1;
     0,0.4,0.5,0.1;
     0,0.1,0.6,0.3;
     0.5,0.3,0.2,0];
% 选择加权求和算子
B1 = A*R

% 选择最小最大算子
A2 = repmat(A',[1,size(R,2)])
B2 = max(min(A2,R))
B2 = B2/sum(B2)

%% examp12.2-2 多级模糊综合评价
Mfun = @(x,y)max(min(repmat(x',[1,size(y,2)]),y));
A1 = [0.2,0.57,0.21,0.02];
R1 = [0.81,0.19,0,0;
    0.79,0.2,0.01,0;
    0.88,0.09,0.03,0;
    0,0.01,0.49,0.5];
B1 = Mfun(A1,R1)
B1 = B1/sum(B1)

A2 = [0.6,0.1,0.1,0.2];
R2 = [0.1,0.7,0.2,0;
    0.2,0.6,0.1,0.1;
    0,0.2,0.2,0.6;
    0,0.4,0.5,0.1];
B2 = Mfun(A2,R2)
B2 = B2/sum(B2)

A3 = [0.1,0.6,0.3];
R3 = [0,0.1,0.2,0.7;
    0.5,0.4,0.1,0;
    0.4,0.5,0.1,0];
B3 = Mfun(A3,R3)
B3 = B3/sum(B3)

A = [0.5,0.3,0.2];
R = [B1;B2;B3];
B = Mfun(A,R)
B = B/sum(B)