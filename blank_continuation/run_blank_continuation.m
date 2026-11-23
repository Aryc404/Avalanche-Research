function output_dir = run_blank_continuation()
% 自然图像训练终点 -> 无LGN输入续训；背景噪声及原学习规则继续工作。
batches = 1200;
save_offsets = [0 100 400 800 1200];
seed = 3;

experiment_dir = fileparts(mfilename('fullpath'));
model_dir = fullfile(experiment_dir,'..','..','Ruslim network','bio-spiking-v1-learning-main');
source_file = fullfile(model_dir,'results','natural_input','training_data.mat');
baseline = load(source_file,'W','theta','input_config');
assert(strcmp(baseline.input_config.type,'natural'), 'Expected the natural-input endpoint.');
output_dir = fullfile(experiment_dir,'results',char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
assert(~isfolder(output_dir), 'Choose a new output directory.');
mkdir(output_dir);

protocol = struct('source_file',source_file,'batches',batches, ...
    'save_offsets',save_offsets,'seed',seed,'output_dir',output_dir);
protocol.learning_rate_names = {'EX','EE','EI','IX','IE','II','threshold_E','threshold_I'};
protocol.note = '100 parallel 400-ms trials per batch, reset each batch; zero LGN, background noise retained.';
protocol.main_source = fileread(fullfile(model_dir,'main.m'));
snapshot = struct('batch',1200,'offset',0,'W',baseline.W,'theta',baseline.theta);
save(fullfile(output_dir,'batch_0000.mat'),'snapshot');

original_dir = pwd;
previous_config = [];
if isappdata(groot,'ruslim_input_config')
    previous_config = getappdata(groot,'ruslim_input_config');
end
threads = maxNumCompThreads(1);
cleanup = onCleanup(@() restore_state(original_dir,previous_config,threads));
config = struct('type','blank','training_seed',seed,'noise_seed',2,'resume',protocol);
setappdata(groot,'ruslim_input_config',config);
cd(model_dir);
history = continue_network();
save(fullfile(output_dir,'history.mat'),'history','protocol');
trajectory = table((1:batches)',history.rate_E,history.rate_I, ...
    history.threshold_E,history.threshold_I, ...
    'VariableNames',{'extra_batch','E_rate_Hz','I_rate_Hz','E_mean_threshold','I_mean_threshold'});
writetable(trajectory,fullfile(output_dir,'trajectory.csv'));

names = fieldnames(baseline.W);
states = cell(numel(save_offsets),1);
for k=1:numel(save_offsets)
    d=load(fullfile(output_dir,sprintf('batch_%04d.mat',save_offsets(k))),'snapshot');
    states{k}=d.snapshot;
end
% No LGN spikes means no feedforward STDP; only normalization roundoff is allowed.
assert(norm(states{end}.W.EX-baseline.W.EX,'fro')<1e-9 && ...
       norm(states{end}.W.IX-baseline.W.IX,'fro')<1e-9, 'Feedforward weights changed without LGN input.');

summary = zeros(numel(save_offsets)*numel(names),6);
row = 0;
f=figure('Color','w','Theme','light','Visible','off','Position',[100 100 1300 700]);
tiledlayout(2,3);
for j=1:numel(names)
    nexttile; hold on;
    values=cellfun(@(s) connection_values(s.W,names{j}),states,'UniformOutput',false);
    edges=linspace(0,max(vertcat(values{:}))+eps,41);
    for k=1:numel(states)
        x=values{k}; w=states{k}.W.(names{j});
        assert(all(isfinite(w(:))), 'Nonfinite weights in %s.',names{j});
        histogram(x,edges,'DisplayStyle','stairs','Normalization','probability', ...
            'LineWidth',1.2,'DisplayName',sprintf('+%d batches',save_offsets(k)));
        row=row+1;
        summary(row,:)=[save_offsets(k),j,mean(x),std(x),mean(x==0), ...
            norm(w-baseline.W.(names{j}),'fro')/norm(baseline.W.(names{j}),'fro')];
    end
    title(names{j}); xlabel('Weight'); ylabel('Probability per bin');
    if j==1, legend('Location','northeast'); end
end
sgtitle('Zero LGN input; background noise retained');
exportgraphics(f,fullfile(output_dir,'weight_distributions.png'),'Resolution',180);
savefig(f,fullfile(output_dir,'weight_distributions.fig')); close(f);
weight_summary=array2table(summary,'VariableNames', ...
    {'extra_batch','connection_index','mean_weight','std_weight','zero_fraction','relative_change'});
weight_summary.connection=string(names(weight_summary.connection_index));
weight_summary.connection_index=[];
writetable(weight_summary,fullfile(output_dir,'weight_summary.csv'));

f=figure('Color','w','Theme','light','Visible','off','Position',[100 100 1100 650]);
tiledlayout(2,2);
nexttile; plot(1:batches,[history.rate_E history.rate_I]);
xlabel('Extra training batches'); ylabel('Mean rate (Hz)'); legend('E','I');
nexttile; plot(1:batches,[history.threshold_E history.threshold_I]);
xlabel('Extra training batches'); ylabel('Mean firing threshold'); legend('E','I');
for column=[6 5]
    nexttile; hold on;
    for j=1:numel(names)
        rows=summary(:,2)==j;
        plot(summary(rows,1),summary(rows,column),'-o','DisplayName',names{j});
    end
    xlabel('Extra training batches');
    if column==6, ylabel('Relative weight change'); else, ylabel('Zero-weight fraction (no self-edges)'); end
    legend('Location','best');
end
exportgraphics(f,fullfile(output_dir,'network_changes.png'),'Resolution',180);
savefig(f,fullfile(output_dir,'network_changes.fig')); close(f);
fprintf('\nContinuation results: %s\n',output_dir);
end

function history = continue_network()
main;
history=continuation;
end

function x = connection_values(W,name)
w=W.(name);
if strcmp(name,'EE') || strcmp(name,'II')
    x=w(~eye(size(w,1)));
else
    x=w(:);
end
end

function restore_state(original_dir,previous_config,threads)
cd(original_dir);
maxNumCompThreads(threads);
if isempty(previous_config)
    rmappdata(groot,'ruslim_input_config');
else
    setappdata(groot,'ruslim_input_config',previous_config);
end
end
