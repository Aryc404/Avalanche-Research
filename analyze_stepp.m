function out = analyze_stepp(spikes,W_EE,dt_ms)
% Stepp：连接追踪、branching ratio、ISI-DFA。
if nargin < 3, dt_ms = 1; end
spikes = logical(spikes);
[neuron_count,time_count] = size(spikes);
A = W_EE > 0;
A(1:neuron_count+1:end) = false;
[event_neuron,event_step] = find(spikes);
event_count = numel(event_step);
lags = 2:4;

sizes = zeros(event_count,1);
start_step = zeros(event_count,1,'uint32');
start_neuron = zeros(event_count,1,'uint16');
last_step = zeros(event_count,1,'uint32');
root_mark = zeros(event_count,1,'uint32');
memberships = cell(neuron_count,5);
history_spikers = cell(1,5);
history_event = zeros(neuron_count,5,'uint32');
root_count = 0; event_index = 0;

preview_count = min(200,nnz((event_step-1)*dt_ms < 100));
preview_memberships = cell(preview_count,1);
preview_edges = cell(preview_count,1);
branching = NaN(1,time_count);
pre_counts = NaN(1,time_count);
post_counts = NaN(1,time_count);

for t = 1:time_count
    slot = mod(t-1,5)+1;
    previous = history_spikers{slot};
    if ~isempty(previous)
        memberships(previous,slot) = {[]};
        history_event(previous,slot) = 0;
    end
    active = find(spikes(:,t));
    history_spikers{slot} = active;

    for neuron = active'
        event_index = event_index+1;
        parents_for_preview = cell(1,numel(lags));
        for j = 1:numel(lags)
            parent_time = t-lags(j);
            if parent_time < 1, continue; end
            parent_slot = mod(parent_time-1,5)+1;
            candidates = history_spikers{parent_slot};
            parents = candidates(A(neuron,candidates));
            for parent = parents'
                inherited = memberships{parent,parent_slot};
                root_mark(inherited) = event_index;
            end
            if event_index <= preview_count
                parents_for_preview{j} = history_event(parents,parent_slot);
            end
        end

        labels = uint32(find(root_mark(1:root_count) == event_index));
        if isempty(labels)
            root_count = root_count+1;
            labels = uint32(root_count);
            start_step(root_count) = t;
            start_neuron(root_count) = neuron;
        end
        sizes(labels) = sizes(labels)+1;
        last_step(labels) = t;
        memberships{neuron,slot} = labels;
        history_event(neuron,slot) = event_index;
        if event_index <= preview_count
            preview_memberships{event_index} = labels;
            parents = uint32(reshape(vertcat(parents_for_preview{:}),[],1));
            preview_edges{event_index} = [parents,repmat(uint32(event_index),numel(parents),1)];
        end
    end

    if t >= 5 && t <= time_count-4 && ~isempty(active)
        pre_active = any(spikes(:,t-lags),2);
        post_active = any(spikes(:,t+lags),2);
        pre_counts(t) = sum(A(active,pre_active),'all');
        post_counts(t) = sum(A(post_active,active),'all');
        if pre_counts(t) > 0
            branching(t) = post_counts(t)/pre_counts(t);
        end
    end
end

sizes = sizes(1:root_count);
start_step = start_step(1:root_count);
start_neuron = start_neuron(1:root_count);
last_step = last_step(1:root_count);
right_censored = double(last_step)+max(lags) > time_count;
out.parameters = struct('phi',1,'Delta',3,'lags',lags,'dt_ms',dt_ms);
out.G = A';
out.tracking = struct('size',sizes,'start_step',start_step, ...
    'start_neuron',start_neuron,'last_step',last_step, ...
    'right_censored',right_censored,'complete_sizes',sizes(~right_censored), ...
    'total_count',root_count,'complete_count',nnz(~right_censored));
out.size_distribution = size_distribution(sizes(~right_censored));
out.branching = struct('time_ms',(0:time_count-1)*dt_ms,'value',branching, ...
    'pre_count',pre_counts,'post_count',post_counts, ...
    'valid_count',nnz(isfinite(branching)),'mean',mean(branching,'omitnan'));
out.preview = struct('step_index',uint32(event_step(1:preview_count)), ...
    'neuron_id',uint16(event_neuron(1:preview_count)));
out.preview.time_ms = (double(out.preview.step_index)-1)*dt_ms;
out.preview.memberships = preview_memberships;
out.preview.parent_edges = zeros(0,2,'uint32');
if preview_count > 0
    out.preview.parent_edges = reshape(vertcat(preview_edges{:}),[],2);
end
out.dfa = neuron_dfa(spikes,dt_ms);
end

function distribution = size_distribution(sizes)
[S,~,index] = unique(sizes);
counts = zeros(size(S));
if ~isempty(S), counts = accumarray(index,1,[numel(S) 1]); end
P = counts/max(numel(sizes),1);
distribution = struct('S',S,'P',P,'counts',counts, ...
    'slope',NaN,'intercept',NaN,'r_squared',NaN,'fit_valid',false, ...
    'fit_min',NaN,'fit_max',NaN,'n_avalanches',numel(sizes));
if numel(S) >= 3
    x = log10(S); y = log10(P);
    coefficients = polyfit(x,y,1);
    fitted = polyval(coefficients,x);
    total = sum((y-mean(y)).^2);
    distribution.slope = coefficients(1);
    distribution.intercept = coefficients(2);
    if total > 0, distribution.r_squared = 1-sum((y-fitted).^2)/total; end
    distribution.fit_valid = true;
    distribution.fit_min = min(S);
    distribution.fit_max = max(S);
end
end

function dfa = neuron_dfa(spikes,dt_ms)
neuron_count = size(spikes,1);
dfa.ISI_ms = cell(neuron_count,1);
dfa.scales = cell(neuron_count,1);
dfa.F = cell(neuron_count,1);
dfa.F_squared = cell(neuron_count,1);
dfa.ISI_count = zeros(neuron_count,1);
dfa.alpha = NaN(neuron_count,1);
dfa.intercept = NaN(neuron_count,1);
dfa.r_squared = NaN(neuron_count,1);
dfa.scale_span_decades = NaN(neuron_count,1);
dfa.limited_span = true(neuron_count,1);
for neuron = 1:neuron_count
    ISI = diff(find(spikes(neuron,:)))'*dt_ms;
    dfa.ISI_ms{neuron} = ISI;
    dfa.ISI_count(neuron) = numel(ISI);
    max_scale = floor(numel(ISI)/4);
    if max_scale < 4, continue; end
    scales = unique(round(logspace(log10(4),log10(max_scale),12)));
    profile = cumsum(ISI-mean(ISI));
    F = zeros(size(scales));
    for k = 1:numel(scales)
        l = scales(k);
        M = floor(numel(ISI)/l);
        z = reshape(profile(1:M*l),l,M);
        x = (1:l)'; x = x-mean(x);
        slopes = (x'*z)/(x'*x);
        trends = mean(z,1)+x*slopes;
        residual = z-trends;
        F(k) = sqrt(mean(residual(:).^2));
    end
    dfa.scales{neuron} = scales;
    dfa.F{neuron} = F;
    dfa.F_squared{neuron} = F.^2;
    usable = isfinite(F) & F > 0;
    if nnz(usable) < 4 || all(ISI == ISI(1)), continue; end
    x = log10(scales(usable)); y = log10(F(usable));
    coefficients = polyfit(x,y,1);
    fitted = polyval(coefficients,x);
    total = sum((y-mean(y)).^2);
    dfa.alpha(neuron) = coefficients(1);
    dfa.intercept(neuron) = coefficients(2);
    if total > 0, dfa.r_squared(neuron) = 1-sum((y-fitted).^2)/total; end
    dfa.scale_span_decades(neuron) = max(x)-min(x);
    dfa.limited_span(neuron) = max(x)-min(x) < 1;
end
dfa.valid_count = nnz(isfinite(dfa.alpha));
dfa.adequate_span_count = nnz(isfinite(dfa.alpha) & ~dfa.limited_span);
end
