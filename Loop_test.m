function result = Loop_test(free_run_ms)
% Loop_test：15 ms 启动刺激，然后连续、无外部输入运行。
% 直接运行 Loop_test：撤销刺激后运行 900000 ms；Loop_test(1600) 可做短检查。
% 不调用或修改 main.m；沿用其 LIF、STDP、权重约束和阈值适应公式。
% 阈值下限为零；每 100000 ms 导出一张汇总 PNG，每次运行保存一个 MAT。
if nargin < 1
    free_run_ms = 900000;
end
%% 1. 读取 AD 训练检查点
experiment_dir = fileparts(mfilename('fullpath'));
checkpoint_file = fullfile(experiment_dir, ...
    'pcb_shuffle_ad_paper_cn1_run01_batch1200.mat');
checkpoint_data = load(checkpoint_file, 'checkpoint');
checkpoint = checkpoint_data.checkpoint;
state = checkpoint.state;

% W_AB 表示从 B 到 A 的连接；X 为 LGN 输入，E/I 为兴奋/抑制神经元。
W_EX = state.W_EX;
W_EE = state.W_EE;
W_EI = state.W_EI;
W_IX = state.W_IX;
W_IE = state.W_IE;
W_II = state.W_II;
theta_E = state.theta_E;
theta_I = state.theta_I;
theta_E = max(theta_E,0);
theta_I = max(theta_I,0);

N_E = size(W_EE, 1);
N_I = size(W_II, 1);
N_X = size(W_EX, 2) / 2; % 每个输入像素对应 ON、OFF 两个 LGN 通道。

fprintf('已读取 AD 检查点：%s\n', checkpoint_file);
fprintf('训练批次：%d | E：%d | I：%d | LGN：%d\n', ...
    checkpoint.batch, N_E, N_I, 2*N_X);

%% 2. 实验时间与原模型参数
dt = 1e-3;                       % s；原模型的 1 ms 时间步长
dt_ms = 1000*dt;
stimulus_ms = 15;
stimulus_rate_Hz = 300;
stimulus_neuron_count = 20;
seed = 20261123;
update_steps = 400;               % 仅更新权重/阈值，不重置膜电位或痕迹
group_ms = 100000;                % 从撤销启动刺激时起，每 100 s 统计一组
stimulus_steps = round(stimulus_ms/dt_ms);
total_steps = stimulus_steps + round(free_run_ms/dt_ms);
window_count = ceil(total_steps/update_steps);
group_end_ms = min(group_ms:group_ms:(ceil(free_run_ms/group_ms)*group_ms),free_run_ms);
group_end_steps = stimulus_steps+round(group_end_ms/dt_ms);
group_snapshots = cell(numel(group_end_ms),1);
next_group = 1;

exp_m = exp(-dt/10e-3);
exp_i = exp(-dt/20e-3);
exp_l = exp(-dt/50e-3);
V_min = -10;
A_p = 1; A_d = 2; A_i = 0.5;
invtau_i = 1/20e-3; invtau_l = 1/50e-3;
rho_E = 2; rho_I = 4;             % Hz
eta_EE = 1e-4*state.learning_scale;
eta_EI = 9e-2*state.learning_scale;
eta_IE = 4e-2*state.learning_scale;
eta_II = 6e-2*state.learning_scale;
eta_tE = state.learning_scale;
eta_tI = state.learning_scale;
L1_EE = 10; L1_EI = 120; L1_IE = 240; L1_II = 120;

protocol = struct('checkpoint_file',checkpoint_file, ...
    'checkpoint_batch',checkpoint.batch,'dt_ms',dt_ms, ...
    'stimulus_ms',stimulus_ms,'stimulus_rate_Hz',stimulus_rate_Hz, ...
    'stimulus_neuron_count',stimulus_neuron_count,'seed',seed, ...
    'free_run_ms',free_run_ms,'total_ms',total_steps*dt_ms, ...
    'update_window_ms',update_steps*dt_ms,'parallel_trajectories',1, ...
    'LGN_input',false,'background_noise_std',0, ...
    'reset_between_windows',false,'reset_voltage_after_spike',0, ...
    'threshold_adaptation',true,'threshold_min',0, ...
    'group_ms',group_ms,'rate_bin_ms',1000,'learning_scale',state.learning_scale);
protocol.poisson_discretization = ...
    '1-ms Bernoulli approximation: independent P(spike)=rate*dt; at most one spike per neuron per step.';
protocol.spike_time_convention = ...
    'time_ms=(step_index-1)*dt_ms; stimulus in [0,15), free run in [15,total_ms).';

%% 3. 随机选 20 个 E 神经元，生成各自独立的启动 spike
old_rng = rng;
rng_cleanup = onCleanup(@() rng(old_rng));
rng(seed,'twister');
stimulus.E_ids = sort(randperm(N_E,stimulus_neuron_count))';
% shortcut: 原模型每毫秒最多发放一次，此处采用离散 Poisson 近似；
% 若要分辨同一毫秒内的多个到达事件，需要更小的时间步长并重新验证动力学。
stimulus.forced_spikes = rand(stimulus_neuron_count,stimulus_steps) < ...
    stimulus_rate_Hz*dt;
stimulus.bin_start_ms = (0:stimulus_steps-1)*dt_ms;
%% 4. 初始化膜电位和突触痕迹
% 检查点没有保存 V/U，因此从零开始；后面跨越更新窗口时均连续保留。
V_E = zeros(N_E,1); V_I = zeros(N_I,1);
U_Ei = zeros(N_E,1); U_El = zeros(N_E,1); U_Ii = zeros(N_I,1);
history.time_ms = zeros(window_count,1);
history.window_ms = zeros(window_count,1);
history.rate_E_Hz = zeros(window_count,1);
history.rate_I_Hz = zeros(window_count,1);
history.mean_theta_E = zeros(window_count,1);
history.mean_theta_I = zeros(window_count,1);
history.min_theta_E = zeros(window_count,1);
history.min_theta_I = zeros(window_count,1);
spike_step_blocks = cell(window_count,1);
spike_id_blocks = cell(window_count,1);
forced_extra_spike_count = 0;
run_clock = tic; report_clock = tic;
fprintf('启动：20 E，300 Hz，15 ms；随后无输入运行 %d ms。\n',free_run_ms);
fprintf('检查点学习率系数：%g；无 LGN 输入，无高斯背景噪声。\n',state.learning_scale);

%% 5. 网络动力学
for window = 1:window_count
    first_step = (window-1)*update_steps+1;
    last_step = min(window*update_steps,total_steps);
    steps_in_window = last_step-first_step+1;
    T = steps_in_window*dt;
    R_E = zeros(N_E,1); R_I = zeros(N_I,1);
    dW_EE = zeros(N_E,N_E); dW_EI = zeros(N_E,N_I);
    dW_IE = zeros(N_I,N_E); dW_II = zeros(N_I,N_I);
    window_spikes = false(N_E+N_I,steps_in_window);

    for step = first_step:last_step
        S_E = V_E > theta_E;
        S_I = V_I > theta_I;
        if step <= stimulus_steps
            forced_ids = stimulus.E_ids(stimulus.forced_spikes(:,step));
            forced_extra_spike_count = forced_extra_spike_count + ...
                sum(~S_E(forced_ids));
            S_E(forced_ids) = true;
        end
        % 仅发生 spike 的神经元复位；这里没有 patch 切换或周期性清零。
        V_E(S_E) = 0; V_I(S_I) = 0;
        R_E = R_E+S_E; R_I = R_I+S_I;
        window_spikes(:,step-first_step+1) = [S_E;S_I];

        % 与 main.m 相同的递推；LGN spike 和背景噪声均严格为零。
        V_E = V_E*exp_m + W_EE*sparse(double(S_E)) - W_EI*sparse(double(S_I));
        V_I = V_I*exp_m + W_IE*sparse(double(S_E)) - W_II*sparse(double(S_I));
        V_E = max(V_E,V_min); V_I = max(V_I,V_min);

        U_Ei = U_Ei*exp_i; U_El = U_El*exp_l; U_Ii = U_Ii*exp_i;
        % 下列按活跃行/列累加，等价于 main.m 的外积公式，避免计算全零项。
        e = find(S_E); i = find(S_I);
        if ~isempty(e)
            dW_EE(e,:) = dW_EE(e,:) + A_p*U_El(e)*U_Ei';
            dW_EE(:,e) = dW_EE(:,e) - A_d*U_Ei;
            dW_EI(e,:) = dW_EI(e,:) + A_i*U_Ii';
            dW_IE(:,e) = dW_IE(:,e) + A_i*U_Ii;
        end
        if ~isempty(i)
            dW_EI(:,i) = dW_EI(:,i) + A_i*U_Ei;
            dW_IE(i,:) = dW_IE(i,:) + A_i*U_Ei';
            dW_II(:,i) = dW_II(:,i) + A_i*U_Ii;
            dW_II(i,:) = dW_II(i,:) + A_i*U_Ii';
            dW_II(i,i) = dW_II(i,i) + A_i;
            if ~isempty(e)
                dW_EI(e,i) = dW_EI(e,i) + A_i;
                dW_IE(i,e) = dW_IE(i,e) + A_i;
            end
        end
        U_Ei(S_E) = U_Ei(S_E)+invtau_i;
        U_El(S_E) = U_El(S_E)+invtau_l;
        U_Ii(S_I) = U_Ii(S_I)+invtau_i;
        if next_group <= numel(group_end_steps) && step == group_end_steps(next_group)
            group_snapshots{next_group} = struct('EX',W_EX,'EE',W_EE,'EI',W_EI, ...
                'IX',W_IX,'IE',W_IE,'II',W_II);
            next_group = next_group+1;
        end
    end

    % 一条连续轨迹，N_b=1；最后不足 400 ms 的窗口按实际 T 计算目标计数。
    W_EE = constrain_weights(W_EE+eta_EE*dW_EE,L1_EE,true);
    W_EI = constrain_weights(W_EI+eta_EI*dW_EI,L1_EI,false);
    W_IE = constrain_weights(W_IE+eta_IE*dW_IE,L1_IE,false);
    W_II = constrain_weights(W_II+eta_II*dW_II,L1_II,true);
    % main.m 的正向阈值增量上限使用初始 eta_t=1，随后不随学习率减半。
    theta_E = max(theta_E + min((R_E-rho_E*T)*eta_tE,rho_E*T),0);
    theta_I = max(theta_I + min((R_I-rho_I*T)*eta_tI,rho_I*T),0);
    % 组末若恰好也是更新时刻，保存本次可塑性更新后的权重。
    if next_group > 1 && last_step == group_end_steps(next_group-1)
        group_snapshots{next_group-1} = struct('EX',W_EX,'EE',W_EE,'EI',W_EI, ...
            'IX',W_IX,'IE',W_IE,'II',W_II);
    end
    [neuron_id,local_step] = find(window_spikes);
    spike_step_blocks{window} = uint32(first_step-1+local_step);
    spike_id_blocks{window} = uint16(neuron_id);
    history.time_ms(window) = last_step*dt_ms;
    history.window_ms(window) = steps_in_window*dt_ms;
    history.rate_E_Hz(window) = mean(R_E)/T;
    history.rate_I_Hz(window) = mean(R_I)/T;
    history.mean_theta_E(window) = mean(theta_E);
    history.mean_theta_I(window) = mean(theta_I);
    history.min_theta_E(window) = min(theta_E);
    history.min_theta_I(window) = min(theta_I);
    if toc(report_clock) >= 20 || window == window_count
        fprintf('进度 %d / %d ms；E %.3f Hz，I %.3f Hz；耗时 %.1f s\n', ...
            last_step*dt_ms,total_steps*dt_ms,mean(R_E)/T,mean(R_I)/T,toc(run_clock));
        report_clock = tic;
        drawnow limitrate;
    end
end

%% 6. 保存完整 spike、初末状态，以及每 100 s 的分组统计和图片
spikes.step_index = vertcat(spike_step_blocks{:});
spikes.neuron_id = vertcat(spike_id_blocks{:});
spikes.id_definition = sprintf('E: 1-%d; I: %d-%d',N_E,N_E+1,N_E+N_I);
% 不另存 time_ms：用 double(spikes.step_index-1)*protocol.dt_ms 即可得到。
final_state = struct('W_EX',W_EX,'W_EE',W_EE,'W_EI',W_EI, ...
    'W_IX',W_IX,'W_IE',W_IE,'W_II',W_II, ...
    'theta_E',theta_E,'theta_I',theta_I,'learning_scale',state.learning_scale, ...
    'V_E',V_E,'V_I',V_I,'U_Ei',U_Ei,'U_El',U_El,'U_Ii',U_Ii);
stimulus.forced_extra_spike_count = forced_extra_spike_count;
result = struct('protocol',protocol,'stimulus',stimulus, ...
    'initial_state',state,'final_state',final_state,'spikes',spikes,'history',history, ...
    'runtime_seconds',toc(run_clock));
output_dir = fullfile(experiment_dir,'results','Loop_test', ...
    char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
if ~isfolder(output_dir), mkdir(output_dir); end
result.output_file = fullfile(output_dir,'result.mat');
result.groups = summarize_groups(spikes,group_snapshots,group_end_ms,protocol,N_E,output_dir);
save(result.output_file,'result','-v7.3');
fprintf('已保存：%s\n',result.output_file);
model_dir = fullfile(experiment_dir,'..','Ruslim network','bio-spiking-v1-learning-main');
export_group_figures(result,model_dir,N_X);
end

function groups = summarize_groups(spikes,snapshots,end_ms,protocol,N_E,output_dir)
% 各组的 spike/发放率只取本组；权重取该组末时刻的快照。
E = spikes.neuron_id <= N_E;
E_time_ms = (double(spikes.step_index(E))-1)*protocol.dt_ms-protocol.stimulus_ms;
E_ids = double(spikes.neuron_id(E));
groups = struct([]);
start_ms = 0;
for k = 1:numel(end_ms)
    selected = E_time_ms >= start_ms & E_time_ms < end_ms(k);
    times = E_time_ms(selected);
    ids = E_ids(selected);
    counts = accumarray(ids,1,[N_E 1]);
    edges = [start_ms:protocol.rate_bin_ms:(end_ms(k)-1), end_ms(k)];
    groups(k).start_ms = start_ms;
    groups(k).end_ms = end_ms(k);
    groups(k).W = snapshots{k};
    groups(k).E_spike_count = counts;
    groups(k).E_rate_Hz = counts/((end_ms(k)-start_ms)/1000);
    groups(k).E_spike_time_ms = times;
    groups(k).E_spike_id = uint16(ids);
    groups(k).rate_time_s = (edges(1:end-1)+edges(2:end))/2000;
    groups(k).E_population_rate_Hz = histcounts(times,edges)./(diff(edges)/1000)/N_E;
    groups(k).figure_file = fullfile(output_dir, ...
        sprintf('group_%02d_%06d_%06dms.png',k,start_ms,end_ms(k)));
    start_ms = end_ms(k);
end
end

function export_group_figures(result,model_dir,N_X)
% 复用 Ruslim 的 display_matrix 和 scm 绘制感受野；退出时恢复 MATLAB 路径。
old_path = path;
path_cleanup = onCleanup(@() path(old_path));
addpath(model_dir);
initial = result.initial_state;
initial_W = struct('EX',initial.W_EX,'EE',initial.W_EE,'EI',initial.W_EI, ...
    'IX',initial.W_IX,'IE',initial.W_IE,'II',initial.W_II);
names = {'EX','EE','EI','IX','IE','II'};
groups = result.groups;
N_E = numel(initial.theta_E);
weight_edges = cell(size(names));
for j = 1:numel(names)
    top = max(connection_values(initial_W,names{j}));
    for k = 1:numel(groups)
        top = max(top,max(connection_values(groups(k).W,names{j})));
    end
    if top == 0, top = 1; end
    weight_edges{j} = linspace(0,top,41);
end
rate_top = 1; neuron_rate_top = 1;
for k = 1:numel(groups)
    rate_top = max(rate_top,max(groups(k).E_population_rate_Hz));
    neuron_rate_top = max(neuron_rate_top,max(groups(k).E_rate_Hz));
end
for k = 1:numel(groups)
    g = groups(k);
    f = figure('Visible','off','Color','w','Theme','light', ...
        'Position',[100 100 1600 1250]);
    layout = tiledlayout(f,4,3,'TileSpacing','compact','Padding','compact');
    for j = 1:numel(names)
        ax = nexttile(layout,j);
        histogram(ax,connection_values(initial_W,names{j}),weight_edges{j}, ...
            'DisplayStyle','stairs','Normalization','probability', ...
            'EdgeColor',[0.6 0.6 0.6],'LineWidth',1.2);
        hold(ax,'on');
        histogram(ax,connection_values(g.W,names{j}),weight_edges{j}, ...
            'DisplayStyle','stairs','Normalization','probability', ...
            'EdgeColor',[0 0.35 0.8],'LineWidth',1.3);
        title(ax,['W^{' names{j} '}']);
        xlabel(ax,'Weight'); ylabel(ax,'Probability per bin');
        xlim(ax,[weight_edges{j}(1) weight_edges{j}(end)]);
        ylim(ax,[0 1]);
        if j == 1, legend(ax,{'Initial','Group end'},'Location','northeast'); end
    end

    ax = nexttile(layout,7,[1 2]);
    plot(ax,g.E_spike_time_ms/1000,double(g.E_spike_id),'.', ...
        'Color',[0 0.35 0.8],'MarkerSize',3);
    title(ax,'E spike raster');
    xlabel(ax,'Time after stimulus withdrawal (s)'); ylabel(ax,'E neuron ID');
    xlim(ax,[g.start_ms g.end_ms]/1000); ylim(ax,[0.5 N_E+0.5]);
    if isempty(g.E_spike_id)
        text(ax,0.5,0.5,'No E spikes in this interval', ...
            'Units','normalized','HorizontalAlignment','center');
    end

    ax = nexttile(layout,9,[2 1]);
    display_matrix((g.W.EX(:,1:N_X)-g.W.EX(:,N_X+1:end))',ax);
    colormap(ax,scm(256));
    title(ax,{'E receptive fields: ON - OFF','Each field normalized'});

    ax = nexttile(layout,10);
    plot(ax,g.rate_time_s,g.E_population_rate_Hz,'Color',[0 0.35 0.8],'LineWidth',1.2);
    title(ax,'E population firing rate (1-s bins)');
    xlabel(ax,'Time after withdrawal (s)'); ylabel(ax,'Mean rate per neuron (Hz)');
    xlim(ax,[g.start_ms g.end_ms]/1000); ylim(ax,[0 1.05*rate_top]);

    ax = nexttile(layout,11);
    histogram(ax,g.E_rate_Hz,linspace(0,1.05*neuron_rate_top,21), ...
        'FaceColor',[0 0.35 0.8],'EdgeColor','none');
    title(ax,'E mean-rate distribution');
    xlabel(ax,'Rate over this interval (Hz)'); ylabel(ax,'Number of E neurons');
    xlim(ax,[0 1.05*neuron_rate_top]); ylim(ax,[0 N_E]);
    sgtitle(layout,sprintf('%.3g-%.3g s after stimulus withdrawal', ...
        g.start_ms/1000,g.end_ms/1000));
    exportgraphics(f,g.figure_file,'Resolution',180);
    close(f);
    fprintf('已保存第 %d 组图片：%s\n',k,g.figure_file);
end
end

function values = connection_values(W,name)
values = W.(name);
if ismember(name,{'EE','II'})
    values = values(~eye(size(values,1)));
else
    values = values(:);
end
end

function W = constrain_weights(W,limit,remove_self)
% main.m 的非负化、去自连接、行 L1 上限；零行保持零，避免除零。
W = max(W,0);
if remove_self, W(1:size(W,1)+1:end) = 0; end
row_sum = sum(W,2);
W = max(W + (min(row_sum,limit)-row_sum)/size(W,2),0);
row_sum = sum(W,2);
nonzero = row_sum > 0;
W(nonzero,:) = W(nonzero,:) ./ row_sum(nonzero) .* min(row_sum(nonzero),limit);
end
