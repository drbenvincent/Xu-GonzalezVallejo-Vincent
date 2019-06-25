% Take the raw data from Ping and process it to make it easy to work with.

block_order =  readtable('order_information.csv', 'ReadRowNames', true);
block_order = rows2vars(block_order);
block_order.OriginalVariableNames = [];

%% deal with DESCRIPTION protocol data file
T = readtable('original_expt2_s2d.csv');
type = 'description';
T = removevars(T, {'gender', 'RT_first', 'pick_larger_position',...
    'RT_second', 'adjusting_order', 'choices1', 'condition'});
T = ranamevar(T, 'Var1', 'trial');
T = ranamevar(T, 'samount', 'A');
T = ranamevar(T, 'lamount', 'B');
T = ranamevar(T, 'sdelay', 'DA');
T = ranamevar(T, 'ldealy', 'DB');
T = ranamevar(T, 'choice2', 'R');
T.R = 1-T.R; % flip encoding of response

participant_numbers = unique(T.Participant);
for p = 1:numel(participant_numbers)
    id = participant_numbers(p);
    pt = T(T.Participant==id , :);
    process_participant_table(pt, id, type, block_order)
end



%% deal with EXPERIENCE protocol data file
T = readtable('original_expt2_s2e.csv');
type = 'experience';
T = removevars(T, {'gender', 'RT_first', 'RT_second', 'pick_larger_position',...
    'RT_second', 'adjusting_order', 'choices1', 'condition'});
T = ranamevar(T, 'Var1', 'trial');
T = ranamevar(T, 'samount', 'A');
T = ranamevar(T, 'lamount', 'B');
T = ranamevar(T, 'sdelay', 'DA');
T = ranamevar(T, 'ldealy', 'DB');
T = ranamevar(T, 'choice2', 'R');
%T.R = 1-T.R; % flip encoding of response

% encoding of responses seems messed up for experience-loss ~~~~~~~~
T.R(T.B<0) = 1-T.R(T.B<0);
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

participant_numbers = unique(T.Participant);
for p = 1:numel(participant_numbers)
    id = participant_numbers(p);    
    pt = T(T.Participant==id , :);
    process_participant_table(pt, id, type, block_order);
end



%%
function table = ranamevar(table, old_var_name, new_var_name)
table.(new_var_name) = table.(old_var_name);
table.(old_var_name) = [];
end

function process_participant_table(T, id, type, block_order)

% skip participant 1001 as they were a test participant
if id==1001, return, end
    
T = sortrows(T, 'trial');
T.trial = T.trial-min(T.trial)+1;
if T.B(1) >0
	sign_condition = 'gain';
else 
	sign_condition = 'loss';
end
	

% Exclusion criteria:
% remove those with >= 90% response consistency
proportion_delayed = sum(T.R==1) / numel(T.R);
if proportion_delayed >= 0.9 || proportion_delayed <= 0.1
	fprintf('PARTICIPANT %d EXCLUDED: (%s, %s)\n', id, type, sign_condition)
	return
end

% add a new column called "index" which is the order in which the delay
% values were delayed
block_order_for_participant = T.block_order(1);
lookup = block_order{block_order_for_participant,[2:end]};
for t=1:size(T,1)
   index(t,1) = find(T.DB(t) == lookup);
end
T.index = index;

% save
fname = [num2str(id) '.csv'];
folder = [type '_' sign_condition];
fpathname = fullfile(folder, fname);
ensureFolderExists(folder);
writetable(T, fpathname);
end