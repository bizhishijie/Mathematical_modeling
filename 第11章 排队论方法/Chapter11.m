%--------------------------------------------------------------------------
%  ��11��  �Ŷ��۷���
%--------------------------------------------------------------------------
% CopyRight��xiezhh

%% examp11.6-1 M/M/1ģ��
n_arrive = 0:6;
n_time = [14,27,27,18,9,4,1];
tm = [6,18,30,42,54,66,78,90,102,114,135,165,190];
n = [32,22,15,10,6,4,3,2,1,1,1,1,1];
lambda = sum(n_arrive.*n_time)/500
mu = 100/(sum(tm.*n)/60)
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,inf,inf])

%% examp11.6-2 M/M/1/Nģ��
N = 7;
lambda = 3;
mu = 4;
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,N,inf])

%% examp11.6-3 M/M/1/inf/mģ��
m = 6;
lambda = 1/60;
mu = 1/30;
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,inf,m])

%% examp11.6-4 M/G/1ģ��
lambda = 5.5/48;
mu = 1/8;
VarT = 16;
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[1,inf,inf],VarT)

%% examp11.6-5 M/M/cģ��
c = 3;             % ����̨��Ŀ
lambda = 0.9;      % ƽ��������
mu = 0.4;          % ƽ��������
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[c,inf,inf])   % ģ�����

%% examp11.6-6 M/M/c/Nģ��
c = 3;             % ����̨��Ŀ
N = 3;             % ϵͳ����
lambda = 1/2;      % ƽ��������
mu = 1/3;          % ƽ��������
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[c,N,inf])   % ģ�����

%% examp11.6-7 M/M/c/inf/mģ��
c = 3;             % ����̨��Ŀ
m = 20;            % �˿�Դ��Ŀ
lambda = 1/60;     % ƽ��������
mu = 1/6;          % ƽ��������
[Ls,Lq,Ws,Wq,P] = QueuingSystem(lambda,mu,[c,inf,m])   % ģ�����


%% examp11.7-1 �Ŷ�ģ�͵����ģ��
lambda = 1/6;                        % ƽ��������
numCust = linspace(50,5000,100);     % �˿���������������100��ֵ
[Ls,Lq,Ws,Wq] = deal(zeros(100,50)); % ��������ֵ
% ����ǰ�趨�Ĺ˿���������ѭ��
for i = 1:numel(numCust)
    n = numCust(i);                  % ��i���˿�����
    % ��ÿһ��ָ���Ĺ˿��������ظ�50��ģ��
    for j = 1:50
        x = exprnd(1/lambda,1,n);    % n���˿͵�����
        y = unifrnd(3,6,1,n);        % n���˿͵ķ���ʱ��
        c = cumsum(x);               % n���˿͵ĵ���ʱ��
        b = zeros(1,n);              % n���˿Ϳ�ʼ�����ʱ��
        e = b;                       % n���˿ͽ��������ʱ��
        ws = b;                      % n���˿͵Ķ���ʱ��
        wq = b;                      % n���˿͵��Ŷӵȴ�ʱ��
        ls = b;                      % ��ʱ�̵Ķӳ�
        % ͨ��ѭ������ÿ��ʱ�̣���ÿλ�˿ͣ������ָ��
        for k = 1:numel(x)
            if k == 1
                b(k) = c(k);         % �����k���˿Ϳ�ʼ�����ʱ��
            else
                b(k) = max(c(k),e(k-1));
            end
            e(k) = b(k) + y(k);      % �����k���˿ͽ��������ʱ��
            ws(k) = e(k) - c(k);     % �����k���˿͵Ķ���ʱ��
            wq(k) = b(k) - c(k);     % �����k���˿��Ŷӵȴ���ʱ��
            ls(k) = k - 1 - sum(e(1:k) <= c(k));  % �����ʱ�̵Ķӳ�
        end
        lq = max([ls-1;zeros(1,n)]); % �����ʱ�̵ĵȴ��ӳ�
        
        Ls(i,j) = mean(ls(11:end));  % �����i���˿������ĵ�j��ģ���ƽ���ӳ�
        Lq(i,j) = mean(lq(11:end));  % �����i���˿������ĵ�j��ģ���ƽ���ȴ��ӳ�
        Ws(i,j) = mean(ws(11:end));  % �����i���˿������ĵ�j��ģ���ƽ������ʱ��
        Wq(i,j) = mean(wq(11:end));  % �����i���˿������ĵ�j��ģ���ƽ���ȴ�ʱ��
    end
end
% ģ�����Ŀ��ӻ�
figure
subplot(2,1,1)
plot(numCust,mean(Ls,2))     % ����ƽ���ӳ�����
xlabel('����Ĺ˿�����'); ylabel('ƽ���ӳ� Ls'); grid on
subplot(2,1,2)
plot(numCust,mean(Ws,2))     % ����ƽ������ʱ������
xlabel('����Ĺ˿�����'); ylabel('ƽ������ʱ�� Ws'); grid on

% ���۽�
lambda = 1/6;        % ƽ��������
mu = 1/((3+6)/2);    % ƽ��������
VarT = (6-3)^2/12;   % ����ʱ��ķ���
% �����Ա�QueuingSystem���������۽�
[Ls2,Lq2,Ws2,Wq2,P0] = QueuingSystem(lambda,mu,[1,inf,inf],VarT)