function out = analyze_liang(X,dt_ms,Tm)
% X：E放电矩阵；dt_ms：步长(ms)；Tm：平均间隔(ms)；out：结果

%% 分箱
T = size(X,2); % 总步数
Nb = 0; % 完整bin数
if isfinite(Tm) && Tm > 0
    Nb = floor(T*dt_ms/Tm);
end
A = zeros(1,Nb); % 各bin的spike数
B = 100000; % 分块步数
for p = 1:B:T % 分块起点
    if Nb == 0, break; end
    j = p:min(p+B-1,T); % 当前时间步
    c = sum(X(:,j),1); % 各步spike数
    b = floor((j-1)*dt_ms/Tm)+1; % 所属bin
    use = b <= Nb & c > 0; % 有效放电步
    if any(use)
        [u,~,g] = unique(b(use)); % u：bin编号；g：分组
        A(u) = A(u)+accumarray(g(:),c(use)',[],@sum)';
    end
end

%% 雪崩
on = A > 0; % 非空bin
d = diff([false on false]); % 事件边界差分
t0 = find(d == 1); % 起始bin
t1 = find(d == -1)-1; % 结束bin
D = t1-t0+1; % 时长(bin)
cs = [0 cumsum(A)]; % 累计spike数
S = cs(t1+1)-cs(t0); % 雪崩大小
ok = t0 > 1 & t1 < Nb; % 完整事件标记

%% 统计
out = avalanche_stats(S(ok),D(ok),Tm);
out.bin_ms = Tm;
out.time_resolution_ms = dt_ms;
out.time_s = ((1:Nb)-0.5)*Tm/1000;
out.activity = A;
out.active = on;
out.recorded_ms = 0;
if Nb > 0, out.recorded_ms = Nb*Tm; end
out.discarded_tail_ms = T*dt_ms-out.recorded_ms;
out.events = struct('start_bin',t0,'end_bin',t1, ...
    'start_ms',(t0-1)*Tm,'end_ms',t1*Tm, ...
    'S',S,'D_bins',D,'D_ms',D*Tm,'complete',ok);
out.complete_count = nnz(ok);
out.censored_count = nnz(~ok);
end
